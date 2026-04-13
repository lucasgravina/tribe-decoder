# TRIBE Decoder

Reverse-engineers what emotional reaction a piece of content is engineered to trigger in you.

Uses Meta FAIR's **TRIBE v2** (d'Ascoli et al. 2026) — a foundation model trained on 500+ hours
of fMRI data from 700+ subjects — to predict cortical activation from text input. Maps predicted
BOLD signal onto the **Schaefer 2018 200-parcel 7-network atlas** (Schaefer et al. 2018) to
identify which functional brain networks are being targeted, then interprets the pattern against
neuroscience-grounded reaction profiles.

**The interpretation layer is fully rule-based — no LLM.** Every scoring rule cites the
neuroscience literature it's derived from.

---

## Architecture

```
Text input
    ↓
[Colab — GPU]
  TRIBE v2 inference
    LLaMA 3.2 (text encoder)
    → predicted BOLD: (n_timesteps, 20484 vertices) on fsaverage5
  Schaefer 200-parcel atlas mapping
    → parcel activations → z-scores
  Yeo 7-network aggregation
  Reaction profile scoring (rule-based)
    ↓ JSON
[Local — ngrok tunnel]
    ↓
[Browser — localhost:5001]
  Results display
```

---

## Prerequisites

### 1. HuggingFace account + LLaMA 3.2 access
TRIBE v2's text encoder is LLaMA 3.2-3B, which is a gated model.

1. Create a HuggingFace account: https://huggingface.co/join
2. Request access to LLaMA 3.2: https://huggingface.co/meta-llama/Llama-3.2-3B
   (Usually approved within minutes)
3. Create a read access token: https://huggingface.co/settings/tokens

### 2. ngrok account (free)
Used to expose the Colab Flask server to your local machine.

1. Sign up: https://ngrok.com
2. Get your authtoken: https://dashboard.ngrok.com/get-started/your-authtoken

---

## Setup

### Part 1: Colab (the inference backend)

1. Go to [colab.google.com](https://colab.google.com)
2. Create a new notebook
3. **Set runtime to GPU**: Runtime → Change runtime type → T4 GPU
4. Copy sections from `tribe_decoder_colab.py` into cells, one section per cell
   (sections are marked with `## CELL N`)
5. In Cell 7, paste your ngrok authtoken where it says `YOUR_NGROK_AUTHTOKEN_HERE`
6. Run cells 1–7 in order
   - Cell 1: install deps (~3-5 min first time)
   - Cell 2: HuggingFace login (paste your HF token)
   - Cell 3: download TRIBE v2 checkpoint (~1GB + backbone models)
   - Cell 4: download Schaefer atlas files
   - Cell 5-6: load profiles and functions
   - Cell 7: starts server + prints your ngrok URL
7. **Copy the ngrok URL printed by Cell 7** — you'll need it for the local app

**Keep Cell 7 running.** If the cell stops, the tunnel dies.

### Part 2: Local app

```bash
# Install local dependencies
pip install -r requirements_local.txt

# Start the local server
python app.py

# Open the app
open http://localhost:5001
```

Paste your ngrok URL into the config field and click CONNECT.
When the status indicator turns green, you're ready.

---

## Usage

Paste any text into the input area and click **RUN TRIBE V2 ANALYSIS**.

The app will:
1. Send text to Colab via ngrok
2. Run TRIBE v2 inference (30–120 sec on T4 GPU)
3. Map predicted BOLD → Schaefer 200-parcel atlas
4. Score 6 reaction profiles from network activations
5. Return a structured analysis showing:
   - **Primary targeted reaction** + manipulation index
   - **Yeo 7-network activation bars** (relative z-scored BOLD)
   - **Top activated brain regions** (Schaefer parcels, named by function)
   - **Reaction profile scores** with mechanisms + literature citations
   - **Interpretation**: intended outcome, who benefits, inoculation strategy
   - **Raw TRIBE v2 output stats**

---

## Reaction Profiles

The 6 profiles scored, with their neurological basis:

| Profile | Key Networks | Key ROIs | Literature |
|---------|-------------|----------|------------|
| Fear / Threat Response | Limbic + SalVentAttn | TempPole, OFC | LeDoux (1996); Seeley et al. (2007) |
| Personal Identity Targeting | Default (DMN) | mPFC, pCunPCC | Buckner et al. (2008); Northoff (2004) |
| Tribal / Social Alarm | SalVentAttn | TPJ, STS | Saxe & Kanwisher (2003); Greene et al. (2001) |
| Reward / Validation Seeking | Limbic | OFC, vmPFC, Hipp | Wallis (2007); Lieberman (2013) |
| Critical Thinking Bypass | Cont (inverted) | dlPFC | Bechara et al. (2000); Kahneman (2011) |
| Urgency / Action Mobilization | SalVentAttn + SomMot | — | Seeley et al. (2007); Pessoa (2017) |

---

## Notes on accuracy

TRIBE v2 was trained on naturalistic stimuli (films, podcasts, videos). Text-only input requires
an internal TTS step, meaning the model processes speech generated from your text. For very short
inputs (< 50 words), predictions will be less reliable due to limited temporal signal.

The Schaefer atlas mapping and reaction profiles are applied to the average-subject predictions
(TRIBE v2 predicts for an average brain, not your individual brain).

The interpretation is rule-based and should be treated as hypothesis-generating, not definitive.

---

## License

TRIBE v2 is licensed CC BY-NC 4.0 (Meta FAIR).
This codebase is provided for research and educational use.

## Citation

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and
          Brookes, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and
          Banville, Hubert and King, Jean-R{\'e}mi},
  year={2026}
}
```
