# TRIBE Decoder — Project Context Document
# For Claude Code terminal session
# Last updated: April 13, 2026

---

## What This Project Is

TRIBE Decoder is a web application that reverse-engineers what emotional reaction a piece of text content is designed to trigger in the reader. It uses Meta FAIR's TRIBE v2 brain encoding model (d'Ascoli et al. 2026) — trained on 500+ hours of fMRI data from 700+ subjects — to predict cortical activation from text input, then maps that activation onto neuroscience-grounded reaction profiles.

**The core insight:** instead of using TRIBE v2 forward (stimulus → brain response for research), we use it in reverse — as a manipulation detector. Feed in a news article, get back which brain circuits it's targeting and why.

**The interpretation layer is fully rule-based (no LLM).** Every scoring rule cites published neuroscience literature.

---

## Current Architecture (What Exists Now)

```
Local MacBook (localhost:5001)
    Flask proxy (app.py)
        ↓ HTTP
    Google Colab (T4 GPU, manual session)
        TRIBE v2 inference pipeline
        Schaefer 2018 atlas mapping
        Reaction profile scoring
        Flask API on port 5000
        ngrok tunnel (temporary URL, changes every session)
```

### Problems With Current Setup
- Colab sessions expire, URL changes every time
- Must manually re-run 4 cells each session
- Only accessible locally (localhost:5001)
- T4 has ~14.5GB VRAM, TRIBE v2 uses ~13.5GB leaving ~1GB for WhisperX (causes OOM)
- ngrok free tier = 1 tunnel limit, sessions crash frequently

---

## Target Architecture (What To Build)

```
Public URL (Railway)
    Flask frontend + proxy (app.py)
        ↓ HTTPS
    Modal (serverless GPU — A10G 24GB)
        TRIBE v2 inference (permanent deployment)
        Model weights on Modal Volume (cached, no re-download)
```

---

## File Structure

```
tribe-decoder/
├── app.py                          # Local Flask proxy server (port 5001)
├── requirements_local.txt          # Local deps: flask, flask-cors, requests
├── templates/
│   └── index.html                  # Full frontend (single file, ~1100 lines)
├── tribe_decoder_colab.py          # Colab cells 1-7 (original, being replaced)
├── tribe_decoder_colab_additions.py # Colab cells 4b + new Cell 6 (additions)
└── README.md
```

### Files To Create for Migration
```
tribe-decoder/
├── modal_app.py                    # Modal serverless GPU deployment (TO CREATE)
├── requirements_modal.txt          # Modal deps (TO CREATE)
├── requirements_server.txt         # Railway Flask deps (TO CREATE)
└── Procfile                        # Railway start command (TO CREATE)
```

---

## The Inference Pipeline (What Runs on GPU)

### Input
Plain text string (max ~6000 chars, ~1 min of speech equivalent)

### Pipeline Steps
1. **Text → TTS → WhisperX**: TRIBE v2 internally converts text to speech, then uses WhisperX to get word-level timing alignment. This is where OOM happens on T4 (WhisperX needs ~2GB, TRIBE v2 takes ~13.5GB of 14.5GB)
2. **TRIBE v2 inference**: LLaMA 3.2-3B (text encoder) + transformer → predicted BOLD signal on fsaverage5 cortical mesh. Output shape: `(n_timesteps, 20484)` — one prediction per TR (1 second) across 20,484 cortical vertices
3. **Schaefer 200-parcel atlas mapping**: Maps vertex activations to 200 named brain parcels using the Schaefer 2018 atlas (downloaded from Yeo Lab CBIG GitHub)
4. **Z-scoring**: Normalizes parcel activations relative to each other
5. **Yeo 7-network aggregation**: Groups parcels into 7 functional networks (DMN, Limbic, SAL, DAN, FPN, SMN, VIS)
6. **Reaction profile scoring**: Rule-based, literature-cited scoring of 6 emotional profiles
7. **Timeseries extraction**: Per-TR network activation curves
8. **Segment breakdown**: Maps sentences to TR ranges, scores dominant profile per sentence

### Output JSON Structure
```json
{
  "success": true,
  "network_activations": {"Default": 0.63, "Limbic": 0.0, ...},
  "network_raw_z": {"Default": 1.2, ...},
  "network_profiles": {...},
  "top_rois": [{"parcel": "...", "z_score": 2.14, "network": "Vis", "roi_name": "...", "description": "..."}],
  "reaction_profiles": [{"id": "urgency_action", "label": "...", "score": 0.79, ...}],
  "interpretation": {
    "primary_target": "Urgency / Action Mobilization",
    "manipulation_index": 5.7,
    "intended_outcome": "...",
    "who_benefits": "...",
    "inoculation": "..."
  },
  "raw_stats": {"n_timesteps": 83, "n_vertices": 20484, ...},
  "network_timeseries": {"Default": [0.1, 0.3, ...], ...},
  "segments_breakdown": [{"text": "...", "ts_start": 0, "ts_end": 25, "dominant_profile": "urgency_action", ...}]
}
```

---

## The 6 Reaction Profiles

All scoring is rule-based from network activations. Each cites specific literature.

| Profile ID | Label | Key Networks | Key ROIs | Literature |
|-----------|-------|-------------|---------|------------|
| `fear_threat` | Fear / Threat Response | Limbic + SAL | TempPole, OFC | LeDoux (1996); Seeley et al. (2007) |
| `self_relevance` | Personal Identity Targeting | Default (DMN) | mPFC, pCunPCC | Buckner et al. (2008); Northoff (2004) |
| `social_tribal` | Tribal / Social Alarm | SAL | TPJ, STS | Saxe & Kanwisher (2003); Greene et al. (2001) |
| `reward_validation` | Reward / Validation Seeking | Limbic | OFC, vmPFC, Hipp | Wallis (2007); Lieberman (2013) |
| `analytical_bypass` | Critical Thinking Bypass | Cont/FPN (inverted) | dlPFC | Bechara et al. (2000); Kahneman (2011) |
| `urgency_action` | Urgency / Action Mobilization | SAL + SomMot | — | Seeley et al. (2007); Pessoa (2017) |

### Manipulation Index
Combines top emotional profile score × analytical bypass score, scaled 0-10.
`manip_index = min(10, top_emotional × 5 + bypass_score × 5)`

---

## The Yeo 7 Networks

| Network Key | Full Name | Abbreviation | Protective? |
|------------|----------|-------------|------------|
| Default | Default Mode Network | DMN | No |
| Limbic | Limbic Network | LMB | No |
| SalVentAttn | Salience / Ventral Attention | SAL | No |
| DorsAttn | Dorsal Attention Network | DAN | No |
| Cont | Frontoparietal Control | FPN | YES — high is good |
| SomMot | Somatomotor Network | SMN | No |
| Vis | Visual Network | VIS | No |

---

## The Frontend (index.html)

Single HTML file, ~1100 lines. Stack: vanilla JS, JetBrains Mono + Crimson Pro fonts, dark navy aesthetic (`--bg: #09090f`).

### Panels (Normal Mode)
1. **Colab Endpoint** — collapsible, shows connection status
2. **Content to Analyze** — collapsible textarea, shows preview when collapsed
3. **Loading** — animated step display while inference runs
4. **Primary Card** — targeted reaction + manipulation index (big number)
5. **Network Activations** — horizontal bars for 7 networks
6. **Top Activated Parcels** — table of top Schaefer parcels with z-scores
7. **Cortical Network Map** — SVG brain schematic, lateral + medial views, circle size = activation
8. **Network Activation Timeseries** — SVG line chart, 1 TR = 1 second, clickable legend
9. **Sentence-Level Analysis** — per-sentence dominant profile with colored left bar
10. **Reaction Profiles** — 6 scored profiles with mechanisms + literature
11. **Interpretation** — intended outcome, who benefits, inoculation strategy
12. **Raw Stats** — collapsible, shows raw TRIBE v2 output numbers

### Mode Toggle (Header)
- **NORMAL** — single text analysis
- **COMPARISON** — 2-3 sources, runs sequentially (not parallel, to avoid VRAM OOM)

### Comparison Mode
- Starts with 2 source cards (Source A, Source B), editable names
- "+ ADD THIRD SOURCE" button adds optional Source C
- Runs sources sequentially via `for` loop (NOT Promise.all — causes OOM)
- Results: summary cards, delta table, sentence framing diff, brain map comparison

---

## Known Bugs To Fix

### High Priority
1. **Comparison mode loading display**: The step text updates but the step circle doesn't animate correctly through sequential runs
2. **Comparison error handling**: When one source fails mid-sequence, the error message sometimes doesn't show which source failed
3. **OOM on long text**: Very long inputs (>4000 chars) can still OOM even with the WhisperX CPU fix — need better length management

### Medium Priority
4. **ROI table labels**: Most parcels fall back to network-level descriptions instead of specific ROI names (e.g. "Somatomotor Network" instead of "Primary Motor Cortex"). This is because the top activated parcels for geopolitical content tend to be generic SMN/VIS parcels that don't match any specific ROI keyword
5. **Limbic always 0%**: Limbic network is persistently near-zero across all test cases. Likely a parcel mapping issue — the Schaefer 200 atlas has limited limbic coverage vs other atlases

### Low Priority
6. **Brain map text overlap**: When multiple circles are large, text labels overlap on the SVG
7. **Config panel**: The ngrok URL field will need to be replaced with a Modal endpoint URL field after migration

---

## Migration Tasks (Ordered)

### 1. Modal Backend (`modal_app.py`)
```python
# Key structure needed:
import modal

app = modal.App("tribe-decoder")

# Volume for model weights (download once, persist forever)
model_volume = modal.Volume.from_name("tribe-decoder-models", create_if_missing=True)

@app.function(
    gpu="A10G",           # 24GB VRAM — no more OOM
    volumes={"/models": model_volume},
    timeout=600,
    image=modal.Image.debian_slim().pip_install([...])
)
def analyze(text: str) -> dict:
    # Full inference pipeline here
    ...

@app.local_entrypoint()
def main():
    result = analyze.remote("test text")
```

### 2. Update app.py for Railway
- Remove config file logic (no more ngrok URL storage)
- Read `MODAL_ENDPOINT` from environment variable
- Add basic API key check (optional but recommended)
- Remove the `/config` POST endpoint

### 3. Railway Deployment Files
- `Procfile`: `web: gunicorn app:app`
- `requirements_server.txt`: flask, flask-cors, requests, gunicorn
- Environment variables: `MODAL_ENDPOINT`, optionally `API_KEY`

### 4. Frontend Updates
- Remove ngrok config panel
- Add Modal endpoint status indicator
- The rest of the UI stays exactly the same

---

## Dependencies

### Current Colab (GPU side)
```
tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git
nibabel
nilearn
flask
flask-cors
pyngrok
langdetect==1.0.9   # pinned — newer versions break
numpy==2.2.6        # pinned — tribev2 requires exactly this version
torch (CUDA)
```

### Known Dependency Issues
- **numpy==2.2.6**: Must be force-reinstalled after tribev2 install because Colab's system numpy conflicts. On Modal this won't be an issue since we control the entire image
- **langdetect**: Must be patched — `LangDetectException.__init__()` signature changed, tribev2 uses old signature
- **WhisperX**: Needs ~2GB VRAM for speech transcription. On T4 this causes OOM. On A10G (24GB) this is fine

### Local (app.py side)
```
flask
flask-cors
requests
```

---

## Environment / Credentials Needed

- **HuggingFace token**: Required for LLaMA 3.2-3B (gated model). Must request access at huggingface.co/meta-llama/Llama-3.2-3B
- **ngrok authtoken**: Currently hardcoded in Colab cell (REPLACE THIS — it's been exposed in conversation history)
- **Modal token**: Will be needed after migration (`modal setup`)

---

## Test Cases That Have Worked

Successfully analyzed:
- Iran / Strait of Hormuz news article (~1150 chars, 83 TRs) — manipulation index 5.7, Urgency + Identity dominant
- Same article headline only (~95 chars, 8 TRs) — manipulation index 6.8, Urgency dominant
- Short text works but gives less reliable signal due to few TRs

### Expected Output Characteristics
- Somatomotor (SMN) tends to be dominant for action/conflict content
- FPN (analytical) at 40-60% is normal for news content
- Limbic at 0% is a known issue (see Known Bugs)
- Manipulation index 5-7 is typical for mainstream news; 7.5+ for explicitly emotional content

---

## What Claude Code Should Know About the Codebase Style

- **Frontend**: No frameworks, no build step, vanilla JS. All in one `index.html` file
- **Python**: No type hints needed, keep it readable, student-style comments
- **CSS**: CSS custom properties (`--bg`, `--cyan`, etc.), all colors defined in `:root`
- **No LLM in the interpretation layer**: The reaction profiles are purely rule-based. Do not add Claude API calls to the interpretation pipeline — the whole point is that the output is grounded in neuroscience, not LLM vibes
- **The science matters**: Don't simplify or remove the literature citations. They're load-bearing for the credibility of the tool

---

## Quick Reference: Key Variable Names

### Colab/Modal (Python)
- `model` — TribeModel instance
- `all_labels` — (20484,) int array, Schaefer parcel index per vertex
- `all_parcel_names` — dict {parcel_idx: name_string}
- `network_vertex_masks` — dict {network_name: bool_array(20484,)}
- `NETWORK_PROFILES` — dict of network metadata
- `ROI_FUNCTIONS` — dict {keyword: (roi_name, description)}
- `preds` — (n_timesteps, 20484) float32 predicted BOLD
- `mean_bold` — (20484,) temporal mean
- `parcel_z` — dict {parcel_name: z_score}
- `network_mean_z` — dict {network_name: mean_z_score}
- `network_display` — dict {network_name: 0-1 normalized score}

### Frontend (JavaScript)
- `NET_COLORS` — dict of network → hex color
- `NET_LABELS` — dict of network → display name
- `NET_ABBR` — dict of network → abbreviation
- `PROFILE_COLORS` — dict of profile_id → hex color
- `activeSourceCount` — 2 or 3 (comparison mode)
- `tsHidden` — dict tracking which timeseries lines are toggled off
- `currentMode` — 'normal' or 'compare'

---

## Claude Code Suggested First Commands

```bash
# 1. Install Claude Code if not already done
npm install -g @anthropic-ai/claude-code

# 2. Navigate to project
cd ~/Desktop/brain_modeling/tribe-decoder

# 3. Start Claude Code
claude

# 4. First ask Claude Code to:
# "Create modal_app.py that deploys the TRIBE v2 inference pipeline 
#  as a serverless GPU function on Modal, using an A10G GPU and a 
#  persistent Volume for model weights. Read CONTEXT.md for full details."
```

---

## References

- TRIBE v2 paper: d'Ascoli et al. (2026) "A foundation model of vision, audition, and language for in-silico neuroscience"
- TRIBE v2 code: github.com/facebookresearch/tribev2
- TRIBE v2 weights: huggingface.co/facebook/tribev2 (CC BY-NC 4.0)
- Schaefer atlas: Schaefer et al. (2018) Cerebral Cortex, doi:10.1093/cercor/bhx179
- Yeo networks: Yeo et al. (2011) J Neurophysiol
