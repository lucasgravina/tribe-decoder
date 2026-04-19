# TRIBE Decoder — Project Context
> Last updated: April 19, 2026. For use when continuing in a new chat session.

---

## What It Is

A web tool that takes any text (news article, speech, etc.) and runs it through Meta FAIR's TRIBE v2 brain encoding model to predict which neural circuits the text would activate. The output shows which emotional/cognitive circuits are being targeted, scored as reaction profiles and a 0–10 Emotional Activation Score.

**Core insight:** TRIBE v2 was designed to predict BOLD fMRI responses from stimuli. We use it as a manipulation detector — feed in a news article, get back which brain circuits it's targeting. The interpretation layer is fully rule-based, no LLM — every score is deterministic math citing published neuroscience.

---

## Current Architecture (fully deployed, working)

```
User browser
    ↓
Flask proxy (Railway PaaS)     app.py
    ↓  POST /analyze
Modal serverless (A10G GPU)    modal_app.py  — TRIBE v2 inference
```

- **Frontend/proxy**: Flask on Railway. Reads `MODAL_ENDPOINT` env var (set to Modal deployment URL). Optionally enforces `ACCESS_KEY` header. `gunicorn --timeout 660 --workers 2`.
- **Backend**: Modal `@app.cls` on A10G (24GB VRAM). Model stays in GPU memory between requests (`scaledown_window=300`). Cold start ~60s, warm requests ~30–90s.
- **Model weights**: Cached in Modal persistent Volume (`tribe-decoder-models`) at `/models/tribe_cache/`. Downloaded once via `modal run modal_app.py::download_models`.
- **Current access key**: fetch calls in index.html use `X-Access-Key: LUCAS` header.

**Local dev:**
```bash
export MODAL_ENDPOINT=https://lgravina--tribe-decoder-web.modal.run
python app.py   # opens at localhost:5001
```

**Redeploy backend:** `modal deploy modal_app.py`

---

## Inference Pipeline (modal_app.py)

Steps run on GPU inside `TribeInference.run_analysis()`:

1. Text written to temp `.txt` file (TRIBE requires file input — it runs TTS internally)
2. `model.get_events_dataframe(text_path)` → word-level timing via TTS + WhisperX
3. `model.predict(events=df)` → `preds` shape `(n_timesteps, 20484)` predicted BOLD on fsaverage5. 1 TR = 1 second.
4. Temporal mean → `mean_bold` shape `(20484,)`
5. Map vertices → Schaefer 200-parcel atlas (lh/rh `.annot` files from Yeo Lab CBIG)
6. Z-score across parcels → `parcel_z`
7. Aggregate parcels → Yeo 7-network mean z-scores → `network_mean_z`
8. Min-max normalize to 0–1 for display → `network_display`
9. Score 6 reaction profiles (deterministic math, no LLM)
10. Build interpretation (primary target, inoculation text)
11. Per-TR network timeseries (z-scored) for line chart
12. Per-sentence segment breakdown (maps sentences to TR windows)

**Key normalization formula:**
```python
def net(name):
    z = network_mean_z.get(name, 0.0)
    return min(1.0, max(0.0, z / 0.5))
# CLT gives network_mean_z typically ±0.3–0.5 across ~30 parcels.
# Cap at 0.5 → moderate activation hits 1.0; neutral text stays near 0.
```

**Emotional Activation Score (0–10):**
```python
manipulation_index = min(10.0, top_emotional * 5 + bypass_score * 5)
# top_emotional = max score across non-inverted profiles
# bypass_score  = emotional_load / (fpn + emotional_load + 0.01)
# emotional_load = (SAL + LMB + SomMot) / 3
```

**Text limit:** 6000 chars (trimmed with warning if exceeded).

---

## Yeo 7 Networks

| Key | Abbr | Full Name | Role | Protective? |
|-----|------|-----------|------|-------------|
| Default | DMN | Default Mode | Self-reference, identity, autobiographical memory | No |
| Limbic | LMB | Limbic | Emotional valuation, OFC, temporal pole (NOT amygdala) | No |
| SalVentAttn | SAL | Salience/Ventral Attention | Threat detection, social alarm, anterior insula + ACC, TPJ | No |
| Cont | FPN | Frontoparietal Control | Analytical reasoning, working memory — **high = protective** | YES |
| SomMot | SMN | Somatomotor | Motor planning, embodied simulation, urgency-to-act | No |
| DorsAttn | DAN | Dorsal Attention | Sustained attention, task engagement | No |
| Vis | VIS | Visual | Imagery, concrete visualization | No |

**Important:** Yeo 7 "Limbic" ≠ amygdala. It covers OFC and temporal pole. SAL is the primary emotional/threat hub containing anterior insula, ACC, and TPJ.

---

## 6 Reaction Profiles

Scored deterministically in `_score_reaction_profiles()` — all cite published literature:

| ID | Label | Primary Driver |
|----|-------|---------------|
| `fear_threat` | Fear / Threat Response | SAL×0.65 + LMB×0.25 + TempPole/OFC ROI boosts |
| `self_relevance` | Personal Identity Targeting | DMN + mPFC/PCC ROI boosts |
| `social_tribal` | Tribal / Social Alarm | SAL×0.3 + TPJ ROI×0.5 + STS×0.2 |
| `reward_validation` | Reward / Validation Seeking | LMB + OFC ROI + DMN×0.2 |
| `analytical_bypass` | Critical Thinking Bypass | emotional_load / (fpn + emotional_load + 0.01) |
| `urgency_action` | Urgency / Action Impulse | SomMot + SAL + DAN×0.2 |

Profile scores are 0–1. Top 2 shown by default; "SEE ALL" expands to all 6.

---

## API Response Shape

```json
{
  "success": true,
  "network_activations": { "Default": 0.72, "SalVentAttn": 0.91, ... },
  "network_raw_z": { "Default": 0.31, ... },
  "top_rois": [{ "parcel": "...", "z_score": 1.4, "network": "SAL",
                 "roi_name": "...", "description": "..." }],
  "reaction_profiles": [{ "id": "fear_threat", "label": "...", "score": 0.82,
                          "networks_driving": [...], "mechanism": "...", "literature": "..." }],
  "interpretation": {
    "primary_target": "Fear / Threat Response",
    "manipulation_index": 7.4,
    "intended_outcome": "...",
    "inoculation": "..."
  },
  "raw_stats": { "n_timesteps": 18, "n_vertices": 20484, ... },
  "network_timeseries": { "Default": [0.1, -0.3, ...], ... },
  "segments_breakdown": [{ "sentence": "...", "dominant_profile": "fear_threat",
                           "profile_scores": {...}, ... }]
}
```

---

## Frontend (templates/index.html)

Single HTML/CSS/JS file, ~2800 lines. No framework, no build step, vanilla JS.

### Pages
- **Cover** (`#coverPage`): text input, Normal / Compare mode toggle, analyze button
- **Analysis** (`#analysisPage`): contains both `#normalMode` and `#compareMode`

### Normal Mode Layout (top → bottom)
1. **Primary card** — left: overview summary paragraph + PRIMARY INFLUENCE STYLE label + inoculation defense tip; right: big Emotional Activation Score (0–10) + band label (LOW/MODERATE/ELEVATED/HIGH) + "reflects neural activation patterns" note
2. **Top Influence Profiles panel** — top 2 profile cards, "SEE ALL" expands to 6
3. **Cortical Network Map** — SVG schematic, Yeo 7 networks, lateral + medial views, circle size = activation
4. **Network Activation Timeseries** — SVG line chart, 1 TR = 1s, hover to explore, click to open sentence
5. **Sentence-Level Breakdown** — chips color-coded by dominant profile
6. **Technical Details** (collapsed toggle) — Yeo 7 bars, top ROI table, raw stats

### Compare Mode Layout (top → bottom)
1. **Key Insights** (`#compareL1`) — always visible, rendered by `renderCompareL1()`
2. **Cognitive Engagement Profile** — always visible panel (delta table across sources)
3. **Detailed Analysis** (`#compareL2`, **collapsed by default**) — network breakdowns, per-source panels; requires click to open
4. **Technical View** (`#compareL3`, collapsed) — brain map comparison

### CSS variables (dark mono aesthetic)
```css
--bg: #09090f        /* near-black background */
--surface: #0d0f1c
--border: #1c2040
--text: #c4cce0
--text-dim: #5a6080
--cyan: #3af0c0      /* FPN / analytical */
--amber: #f0a83a     /* emotional / warnings */
--red: #f05a5a
--mono: 'JetBrains Mono'
--serif: 'Crimson Pro'
```

### Key JS functions

| Function | ~Line | Purpose |
|----------|-------|---------|
| `renderNormal(data)` | 1461 | Main results renderer for normal mode |
| `generateOverviewSummary()` | 1403 | Generates prose overview paragraph |
| `renderProfiles(calibProfiles)` | 1797 | Renders influence profile cards |
| `renderRoiTable(rois)` | 1570 | "SubRegion (Network)" format ROI table |
| `renderNetworkBars(act)` | 1554 | Yeo 7 activation bars |
| `renderBrainMap(act)` | — | SVG circle sizes/colors |
| `renderTimeSeries(ts, nTR)` | — | Timeseries SVG line chart |
| `renderSegments(segs)` | — | Sentence breakdown chips |
| `renderComparison(sources, results)` | 1986 | Orchestrates all compare renderers |
| `renderCompareL1(sources, results)` | 2491 | Key insights cards |
| `renderCompareSummary(sources, results)` | 2053 | Summary cards |
| `renderCompareSourcePanels(sources, results)` | 2103 | Per-source detail panels |
| `renderDeltaTable(sources, results)` | 2243 | Cognitive engagement delta table |
| `renderBrainCompare(sources, results)` | — | Side-by-side brain maps |
| `applyCalibration(profiles)` | — | Adjusts scores if user has neural profile saved |
| `switchMode(mode)` | — | Toggles normal/compare, ensures analysis page visible |

### Personalization system
Users can fill out a survey (sensitivity traits). Stored in `localStorage` as `tribeUserProfile`. Applied via `applyCalibration()` to weight profile scores. Calibration bar shown at top of results. `_resultsMode` is `'personalized'` or `'population'`.

### Compare source inputs
Two source cards (`#srcCard0`, `#srcCard1`) are hardcoded in HTML (not dynamically created). Each has `#srcText0`/`#srcText1` textareas. `buildCompareInputs()` is a no-op. This was done to fix a blank comparison page bug caused by dynamic DOM creation.

---

## File Structure

```
tribe-decoder/
├── app.py                    Flask proxy frontend (Railway)
├── modal_app.py              Modal GPU backend (TRIBE v2 inference + all scoring)
├── templates/index.html      Entire frontend (~2800 lines)
├── Procfile                  web: gunicorn app:app --timeout 660 --workers 2
├── nixpacks.toml             [phases.install] cmds = ["pip install -r requirements_server.txt"]
├── requirements_server.txt   Flask, gunicorn, requests, flask-cors
├── requirements_modal.txt    modal, nibabel, nilearn, fastapi
├── requirements_local.txt    local dev deps
└── tribe_decoder_colab.py    Original Colab notebook version (legacy, not used)
```

---

## Dependency Notes

- **numpy==2.2.6**: Pinned — tribev2 contract. Force-reinstalled in Modal image.
- **langdetect==1.0.9**: Pinned — newer versions changed `LangDetectException.__init__()` signature, tribev2 uses old positional call. Patched in `TribeInference.setup()`.
- **tribev2**: Installed via `uv` from `git+https://github.com/facebookresearch/tribev2.git`
- **WhisperX**: Needs VRAM. Fine on A10G (24GB). Was OOM on T4 (14.5GB).

---

## Known Issues / Open Questions

- **Strait of Hormuz scoring**: AP wire ~2.2 vs cable news ~2.6 — user noted scores seem too close together (cable news should score higher). Root cause not yet investigated.
- **Chrome extension**: Was discussed but deferred. Idea: content script grabs article text → background script calls Railway endpoint → popup shows results. No approval needed to develop locally; Chrome Web Store review required for public distribution.

---

## Git Log (recent)

```
2ffbd60  Revert normal page results card to original layout
0639eba  Move Cognitive Engagement Profile to main comparison view, collapse Detailed Analysis by default
5901207  Redesign results card and clean up comparison page (comparison changes kept, normal reverted)
7003b85  (prior large UI overhaul session)
```

---

## Neuroscience References

- d'Ascoli et al. (2026) — TRIBE v2 paper (Meta FAIR)
- Yeo et al. (2011) J Neurophysiol — 7-network atlas
- Schaefer et al. (2018) Cerebral Cortex doi:10.1093/cercor/bhx179 — 200-parcel atlas
- Buckner et al. (2008) Ann NY Acad Sci — Default Mode Network
- Seeley et al. (2007) J Neurosci — Salience network
- Saxe & Kanwisher (2003) NeuroImage — TPJ / theory of mind
- LeDoux (1996) The Emotional Brain — fear circuitry
- Miller & Cohen (2001) Ann Rev Neurosci — FPN / prefrontal control
- Wallis (2007) Nat Rev Neurosci — OFC reward/value
- Pessoa (2017) Trends Cogn Sci — somatomotor/emotion interaction
