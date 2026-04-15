# ═══════════════════════════════════════════════════════════════════════════════
# TRIBE DECODER — Modal Serverless GPU Backend
# Deploys the TRIBE v2 brain encoding inference pipeline as a persistent
# serverless API on Modal, using an A10G GPU (24GB VRAM).
#
# Architecture:
#   web() function  →  FastAPI ASGI app (no GPU, cheap routing)
#       ↓ .remote()
#   TribeInference  →  A10G container (model lives here between requests)
#       /models Volume  →  atlas files + HF weights (downloaded once, forever)
#
# One-time setup:
#   1. pip install modal
#   2. modal setup                         (authenticate)
#   3. modal secret create huggingface-secret HF_TOKEN=hf_...
#   4. modal run modal_app.py::download_models   (populates volume — ~10 min)
#   5. modal deploy modal_app.py           (go live)
#   6. modal run modal_app.py              (smoke test via local_entrypoint)
#
# After deploy, Modal prints a URL like:
#   https://<your-workspace>--tribe-decoder-web.modal.run
# Paste that into the Railway env var MODAL_ENDPOINT.
# ═══════════════════════════════════════════════════════════════════════════════

import modal

# ── Container image ────────────────────────────────────────────────────────────
# This image is built once and reused. The A10G has CUDA 12.x.
# Order matters:
#   1. System libs (git for pip install from GitHub, ffmpeg for WhisperX TTS)
#   2. tribev2 via uv (fast resolver, pulls torch+CUDA, WhisperX, LLaMA deps)
#   3. numpy re-pinned (tribev2 installer may float it — we need exactly 2.2.6)
#   4. langdetect re-pinned (newer versions break tribev2's LangDetectException call)
#   5. nibabel/nilearn for atlas loading
#   6. fastapi for the web layer
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",           # required by WhisperX audio pipeline
        "libsndfile1",      # soundfile backend for audio I/O
        "libgl1",           # OpenGL libs pulled by some nilearn deps
    )
    .run_commands(
        "pip install uv --quiet",
        # tribev2 from Meta FAIR GitHub — includes WhisperX + LLaMA 3.2-3B deps
        'uv pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git" '
        "--system --quiet",
        # numpy pinned — tribev2 contract. Force-reinstall in case tribev2 floated it.
        "uv pip install 'numpy==2.2.6' --system --force-reinstall --quiet",
        # langdetect pinned — tribev2 calls LangDetectException(code, message) positionally;
        # newer langdetect changed the signature and raises TypeError at import time.
        "uv pip install 'langdetect==1.0.9' --system --force-reinstall --quiet",
    )
    .pip_install(
        "nibabel>=3.2.0",           # Schaefer atlas .annot file loading
        "nilearn>=0.10.0",          # surface data utilities
        "huggingface-hub>=0.23.0",  # HF model download
        "fastapi[standard]>=0.110.0",
    )
)

# ── Persistent Volume — model weights cached across all deployments ─────────────
# /models/atlas/       — lh/rh Schaefer 200-parcel 7-network .annot files
# /models/tribe_cache/ — TRIBE v2 checkpoint + LLaMA 3.2-3B weights (~15GB total)
model_volume = modal.Volume.from_name("tribe-decoder-models", create_if_missing=True)

VOLUME_PATH = "/models"
ATLAS_DIR   = f"{VOLUME_PATH}/atlas"
TRIBE_CACHE = f"{VOLUME_PATH}/tribe_cache"

# ── App ────────────────────────────────────────────────────────────────────────
app = modal.App("tribe-decoder")


# ─────────────────────────────────────────────────────────────────────────────
# ONE-TIME SETUP: Download atlas + model weights into the Volume
# Run with: modal run modal_app.py::download_models
# This takes ~10 minutes on first run. Subsequent deploys are instant.
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={VOLUME_PATH: model_volume},
    timeout=3600,   # 1 hour ceiling — LLaMA 3.2-3B download can be slow
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_models():
    """
    Download Schaefer atlas + TRIBE v2 weights to the persistent Volume.
    Run once before deploying. Safe to re-run — all downloads are idempotent.
    """
    import os
    import urllib.request
    from tribev2 import TribeModel

    # ── Schaefer 2018 atlas ──────────────────────────────────────────────────
    # 200-parcel 7-network parcellation on fsaverage5.
    # Source: Yeo Lab CBIG (authoritative Schaefer distribution).
    # Ref: Schaefer et al. (2018) Cerebral Cortex doi:10.1093/cercor/bhx179
    os.makedirs(ATLAS_DIR, exist_ok=True)
    CBIG_BASE = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/FreeSurfer5.3/fsaverage5/label/"
    )
    for hemi in ["lh", "rh"]:
        fname = f"{hemi}.Schaefer2018_200Parcels_7Networks_order.annot"
        dst = os.path.join(ATLAS_DIR, fname)
        if not os.path.exists(dst):
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(CBIG_BASE + fname, dst)
        else:
            print(f"  ✓ {fname} (cached)")

    # ── TRIBE v2 weights ─────────────────────────────────────────────────────
    # TribeModel.from_pretrained() downloads the TRIBE checkpoint + LLaMA 3.2-3B.
    # Redirect HF_HOME so the backbone weights land on the volume (not /root/.cache
    # which is ephemeral). Without this, LLaMA re-downloads every cold start.
    os.makedirs(TRIBE_CACHE, exist_ok=True)
    os.environ["HF_HOME"]      = TRIBE_CACHE
    os.environ["HF_HUB_CACHE"] = TRIBE_CACHE

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    print("Downloading TRIBE v2 weights (this may take several minutes)...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=TRIBE_CACHE)
    print("Model downloaded successfully.")

    # Commit volume changes so they're visible to all future containers
    model_volume.commit()
    print("\nVolume committed. Ready to deploy.")


# ─────────────────────────────────────────────────────────────────────────────
# GPU CLASS — loads model once per warm container, serves many requests
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    image=image,
    gpu="A10G",                         # 24GB VRAM — no more OOM
    volumes={VOLUME_PATH: model_volume},
    timeout=600,                        # 10 min per request ceiling
    scaledown_window=300,               # keep container warm for 5 min
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class TribeInference:
    """
    Stateful inference container. The @modal.enter() block runs once when the
    container starts; the loaded model stays in GPU memory across all requests
    that hit this warm container.
    """

    @modal.enter()
    def setup(self):
        """
        Load atlas + model into memory at container startup.
        Called once per warm container (not once per request).
        """
        import os
        import numpy as np
        import nibabel as nib

        # ── Patch langdetect ──────────────────────────────────────────────────
        # tribev2 calls LangDetectException(code, message) with positional args.
        # langdetect 1.0.9 uses keyword-only 'message' parameter — patch it.
        try:
            from langdetect.lang_detect_exception import LangDetectException
            def _patched_init(self, code, message=""):
                self.code = code
                self.message = message
            LangDetectException.__init__ = _patched_init
        except Exception:
            pass

        # ── Schaefer atlas ────────────────────────────────────────────────────
        def _decode(x):
            return x.decode("utf-8") if isinstance(x, bytes) else x

        lh_labels, _, lh_names = nib.freesurfer.read_annot(
            os.path.join(ATLAS_DIR, "lh.Schaefer2018_200Parcels_7Networks_order.annot")
        )
        rh_labels, _, rh_names = nib.freesurfer.read_annot(
            os.path.join(ATLAS_DIR, "rh.Schaefer2018_200Parcels_7Networks_order.annot")
        )
        lh_names = [_decode(n) for n in lh_names]
        rh_names = [_decode(n) for n in rh_names]

        # fsaverage5: 10,242 verts per hemi. TRIBE output: LH first, then RH.
        # RH parcel indices offset by 100 so they don't collide with LH (1-100).
        rh_labels_global = np.where(rh_labels > 0, rh_labels + 100, 0)
        self.all_labels = np.concatenate([lh_labels, rh_labels_global])  # (20484,)

        self.all_parcel_names = {}
        for i, name in enumerate(lh_names):
            if i > 0:
                self.all_parcel_names[i] = name
        for i, name in enumerate(rh_names):
            if i > 0:
                self.all_parcel_names[i + 100] = name

        # Precompute per-network vertex masks for timeseries extraction
        self.network_vertex_masks = {}
        for net_key in NETWORK_PROFILES:
            mask = np.zeros(20484, dtype=bool)
            for idx, name in self.all_parcel_names.items():
                if _extract_network(name) == net_key:
                    mask |= (self.all_labels == idx)
            self.network_vertex_masks[net_key] = mask

        print(f"Atlas ready: {len(self.all_parcel_names)} parcels loaded")

        # ── TRIBE v2 model ────────────────────────────────────────────────────
        from tribev2 import TribeModel

        # Point HF cache at the volume so backbone weights are found without re-download
        os.environ["HF_HOME"]      = TRIBE_CACHE
        os.environ["HF_HUB_CACHE"] = TRIBE_CACHE

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

        print("Loading TRIBE v2...")
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2", cache_folder=TRIBE_CACHE
        )
        print("TRIBE v2 ready.")

    @modal.method()
    def run_analysis(self, text: str) -> dict:
        """Full pipeline: text → TRIBE v2 → reaction profiles → JSON result."""
        import tempfile, os, traceback as tb
        import numpy as np

        # Write text to temp file — TRIBE requires file-based input (TTS reads it)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            text_path = f.name

        try:
            # ── Step 1-2: TRIBE v2 inference ──────────────────────────────────
            # get_events_dataframe: internal TTS → WhisperX → word-level timing DataFrame
            # predict: LLaMA 3.2-3B text encoder + transformer → predicted BOLD
            print(f"  Building events dataframe ({len(text)} chars)...")
            df = self.model.get_events_dataframe(text_path=text_path)
            print(f"  Events: {len(df)} segments. Running TRIBE v2 inference...")
            preds, segments = self.model.predict(events=df)
            # preds: (n_timesteps, 20484) float32 predicted BOLD on fsaverage5
            # 1 TR = 1 second, hemodynamic lag already compensated by TRIBE
            if hasattr(preds, "numpy"):
                preds = preds.numpy()
            print(f"  Prediction shape: {preds.shape}")

            # ── Step 3: Temporal mean → overall activation pattern ─────────────
            mean_bold = preds.mean(axis=0)  # (20484,)
            assert mean_bold.shape[0] == 20484

            # ── Step 4: Vertex → Schaefer 200-parcel mapping ──────────────────
            parcel_activations = {}
            for parcel_idx in range(1, 201):
                mask = (self.all_labels == parcel_idx)
                if mask.sum() == 0:
                    continue
                parcel_activations[parcel_idx] = float(mean_bold[mask].mean())

            # ── Step 5: Z-score across parcels ────────────────────────────────
            # Relative comparison: which parcels are MORE active than average.
            vals = np.array(list(parcel_activations.values()))
            z_mean, z_std = vals.mean(), vals.std()
            parcel_z = {}
            for idx, act in parcel_activations.items():
                name = self.all_parcel_names.get(idx, f"parcel_{idx}")
                parcel_z[name] = float((act - z_mean) / (z_std + 1e-8))

            # ── Step 6: Yeo 7-network aggregation ─────────────────────────────
            network_buckets = {net: [] for net in NETWORK_PROFILES}
            for name, z in parcel_z.items():
                net = _extract_network(name)
                if net in network_buckets:
                    network_buckets[net].append(z)

            network_mean_z = {
                net: float(np.mean(zs))
                for net, zs in network_buckets.items()
                if zs
            }

            # Min-max normalize to 0-1 for display
            nv = np.array(list(network_mean_z.values()))
            nmin, nmax = nv.min(), nv.max()
            network_display = {
                k: round(float((v - nmin) / (nmax - nmin + 1e-8)), 4)
                for k, v in network_mean_z.items()
            }

            # ── Step 7: Top parcels by z-score ────────────────────────────────
            sorted_parcels = sorted(parcel_z.items(), key=lambda x: x[1], reverse=True)
            top_rois = []
            for name, z in sorted_parcels[:12]:
                roi_name, roi_desc = _get_roi_function(name)
                top_rois.append({
                    "parcel": name,
                    "z_score": round(z, 3),
                    "network": _extract_network(name),
                    "roi_name": roi_name,
                    "description": roi_desc,
                })

            # ── Step 8: Reaction profiles + interpretation ─────────────────────
            profiles = _score_reaction_profiles(network_mean_z, network_display, parcel_z)
            interpretation = _build_interpretation(profiles, network_display)

            # ── Step 9: Per-TR network activation timeseries ───────────────────
            # For each Yeo network, compute mean activation at each TR.
            # Used to draw the timeseries line chart in the frontend.
            network_timeseries = {}
            for net_name, vert_mask in self.network_vertex_masks.items():
                if vert_mask.sum() > 0:
                    ts = preds[:, vert_mask].mean(axis=1)  # (n_timesteps,)
                    # Z-score the timeseries for display (relative fluctuations)
                    ts_mean, ts_std = ts.mean(), ts.std()
                    ts_z = (ts - ts_mean) / (ts_std + 1e-8)
                    network_timeseries[net_name] = [round(float(v), 4) for v in ts_z]

            # ── Step 10: Per-sentence breakdown ───────────────────────────────
            # Maps sentences from the original text to TR windows, scores each.
            segments_breakdown = _build_segments_breakdown(
                text, preds, df, self.all_labels, self.all_parcel_names,
                network_mean_z, network_display
            )

            return {
                "success": True,
                "network_activations": network_display,
                "network_raw_z": {k: round(v, 4) for k, v in network_mean_z.items()},
                "network_profiles": NETWORK_PROFILES,
                "top_rois": top_rois,
                "reaction_profiles": profiles,
                "interpretation": interpretation,
                "raw_stats": {
                    "n_timesteps": int(preds.shape[0]),
                    "n_vertices": int(preds.shape[1]),
                    "n_parcels_mapped": len(parcel_activations),
                    "mean_predicted_bold": round(float(mean_bold.mean()), 6),
                    "std_predicted_bold": round(float(mean_bold.std()), 6),
                    "min_predicted_bold": round(float(mean_bold.min()), 6),
                    "max_predicted_bold": round(float(mean_bold.max()), 6),
                },
                "network_timeseries": network_timeseries,
                "segments_breakdown": segments_breakdown,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": tb.format_exc(),
            }
        finally:
            os.unlink(text_path)


# ─────────────────────────────────────────────────────────────────────────────
# WEB ENDPOINT — lightweight FastAPI ASGI app (no GPU)
# Routes /health and /analyze. Dispatches inference to TribeInference via .remote()
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

fastapi_app = FastAPI(title="TRIBE Decoder API")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Railway frontend makes cross-origin requests
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str

@fastapi_app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "TRIBE v2",
        "atlas": "Schaefer 200-parcel 7-network",
    }

@fastapi_app.post("/analyze")
async def analyze(body: AnalyzeRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    trimmed = len(text) > 6000
    if trimmed:
        text = text[:6000]

    result = TribeInference().run_analysis.remote(text)

    if trimmed:
        result["warning"] = "Text trimmed to 6000 characters for inference speed."

    return result

@app.function(image=image, timeout=600)
@modal.asgi_app()
def web():
    """Entry point for the Modal web deployment. Single URL for all endpoints."""
    return fastapi_app


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL ENTRYPOINT — smoke test from CLI: modal run modal_app.py
# ─────────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    test_text = (
        "Tensions in the Strait of Hormuz have reached a critical juncture. "
        "Iran has threatened to close the waterway in response to new sanctions, "
        "a move that would disrupt 20 percent of global oil supply. "
        "Western allies are preparing a naval response. "
        "Analysts warn the window to act is closing fast."
    )
    print(f"Sending {len(test_text)} chars to TRIBE v2...")
    result = TribeInference().run_analysis.remote(test_text)

    if result["success"]:
        print("\n── Result ──────────────────────────────────────────────")
        interp = result["interpretation"]
        print(f"  Primary target:     {interp['primary_target']}")
        print(f"  Manipulation index: {interp['manipulation_index']}/10")
        print(f"  Network activations:")
        for net, score in result["network_activations"].items():
            print(f"    {net:<15} {score:.3f}")
        print(f"  Timeseries TRs:     {len(next(iter(result['network_timeseries'].values())))}")
        print(f"  Segments:           {len(result['segments_breakdown'])}")
    else:
        print(f"\nERROR: {result['error']}")
        print(result.get("traceback", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# NEUROSCIENCE LAYER
# All constants and scoring functions below are traceable to published literature.
# No LLM — every score is deterministic from network activation values.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Yeo 7-network functional definitions ──────────────────────────────────────
# Source: Yeo et al. (2011) J Neurophysiol; Thomas Yeo lab network reviews
NETWORK_PROFILES = {
    "Default": {
        "full_name": "Default Mode Network",
        "abbreviation": "DMN",
        "core_function": "Self-referential thought, autobiographical memory, social cognition, future simulation",
        "reaction_targeted": "Personal Identity / Self-Relevance",
        "high_activation_interpretation": (
            "Content is activating your brain's 'self-referential' circuitry — "
            "the DMN fires when you think about yourself, your beliefs, what others "
            "think of you, or your personal future. High DMN = content has successfully "
            "made itself feel personally relevant to you."
        ),
        "references": "Buckner et al. (2008) Ann NY Acad Sci; Andrews-Hanna et al. (2010) Neuron",
        "protective": False,
    },
    "Limbic": {
        "full_name": "Limbic Network",
        "abbreviation": "LMB",
        "core_function": "Emotional valuation, visceral responses, value-based decision-making",
        "reaction_targeted": "Visceral Emotional Response",
        "high_activation_interpretation": (
            "Limbic network activation signals that content is engaging emotional "
            "valuation circuits — regions involved in computing the emotional 'charge' "
            "of stimuli. Includes temporal pole (emotional memory/familiarity) and OFC "
            "(reward/threat value). Expect fear, craving, disgust, or strong valenced affect."
        ),
        "references": "Olson et al. (2007) Cereb Cortex; Wallis (2007) Nat Rev Neurosci",
        "protective": False,
    },
    "SalVentAttn": {
        "full_name": "Salience / Ventral Attention Network",
        "abbreviation": "SAL",
        "core_function": "Alerting to salient events, social alarm, TPJ-mediated moral judgment, unexpected stimuli",
        "reaction_targeted": "Urgency / Social Alarm",
        "high_activation_interpretation": (
            "Content is triggering your salience detection system — the network that "
            "evolved to flag threats, moral violations, and socially important events. "
            "TPJ activation within this network specifically governs in-group/out-group "
            "distinctions and theory of mind (understanding others' mental states as threatening)."
        ),
        "references": "Seeley et al. (2007) J Neurosci; Saxe & Kanwisher (2003) NeuroImage",
        "protective": False,
    },
    "DorsAttn": {
        "full_name": "Dorsal Attention Network",
        "abbreviation": "DAN",
        "core_function": "Top-down sustained attention, task engagement, spatial attention control",
        "reaction_targeted": "Sustained Engagement / Attention Capture",
        "high_activation_interpretation": (
            "Content is demanding top-down attentional resources — your brain is being "
            "pulled into sustained engagement. This isn't inherently manipulative; it can "
            "reflect genuine cognitive interest. Context-dependent interpretation."
        ),
        "references": "Corbetta & Shulman (2002) Nat Rev Neurosci",
        "protective": False,
    },
    "Cont": {
        "full_name": "Frontoparietal Control Network",
        "abbreviation": "FPN",
        "core_function": "Analytical reasoning, working memory, executive control, deliberate evaluation",
        "reaction_targeted": "Critical / Analytical Engagement",
        "high_activation_interpretation": (
            "Frontoparietal control network is PROTECTIVE — it governs deliberate, "
            "analytical reasoning. HIGH activation here means content is engaging your "
            "evaluative faculties. LOW activation is the pattern to notice: less "
            "deliberate evaluation, more emotional or reflexive processing."
        ),
        "references": "Miller & Cohen (2001) Ann Rev Neurosci; Dosenbach et al. (2007) PNAS",
        "protective": True,  # Inverted: low Cont = high manipulation concern
    },
    "SomMot": {
        "full_name": "Somatomotor Network",
        "abbreviation": "SMN",
        "core_function": "Motor planning, embodied simulation, action readiness",
        "reaction_targeted": "Action Impulse / Urgency to Act",
        "high_activation_interpretation": (
            "Somatomotor activation reflects embodied simulation — your brain preparing "
            "for motor action in response to content. Creates the physical feeling of "
            "urgency: 'something must be done.' Classic in mobilization rhetoric."
        ),
        "references": "Huth et al. (2016) Nature; Pessoa (2017) Trends Cogn Sci",
        "protective": False,
    },
    "Vis": {
        "full_name": "Visual Network",
        "abbreviation": "VIS",
        "core_function": "Visual processing and imagery",
        "reaction_targeted": "Vivid Imagery / Visualization",
        "high_activation_interpretation": (
            "Visual cortex activation by text alone signals strong imagery — content "
            "evokes concrete visual scenes. Vivid imagery increases emotional impact and "
            "reduces abstract/analytical processing."
        ),
        "references": "Kosslyn et al. (2006) Nat Rev Neurosci",
        "protective": False,
    },
}

# ── Specific ROI keyword → functional description ──────────────────────────────
# Matched against Schaefer parcel names (e.g., "7Networks_LH_SalVentAttn_TPJ_1")
# References: established ROI function literature
ROI_FUNCTIONS = {
    "TPJ":      ("Temporoparietal Junction",           "Theory of mind; moral judgment; in-group/out-group detection. Saxe & Kanwisher (2003)"),
    "PFC":      ("Prefrontal Cortex [Default]",        "Self-referential processing; belief importance to identity. Buckner et al. (2008)"),
    "pCunPCC":  ("Posterior Cingulate / Precuneus",    "Autobiographical memory retrieval; self-relevance; rumination. Andrews-Hanna (2010)"),
    "TempPole": ("Temporal Pole",                      "Social/emotional memory; familiar person/situation recognition. Olson et al. (2007)"),
    "OFC":      ("Orbitofrontal Cortex",               "Reward/punishment value computation; drives approach/avoidance. Wallis (2007)"),
    "Insula":   ("Anterior Insula",                    "Interoception; visceral disgust; moral norm violations. Craig (2009)"),
    "Hipp":     ("Hippocampus",                        "Episodic memory retrieval; nostalgia; contextual familiarity. Squire et al. (2004)"),
    "STS":      ("Superior Temporal Sulcus",           "Biological motion; social perception; voice/speech. Allison et al. (2000)"),
    "IFG":      ("Inferior Frontal Gyrus [Broca]",     "Language comprehension; authority framing via rhetorical structure. Bookheimer (2002)"),
    "Rsp":      ("Retrosplenial Cortex",               "Scene/context familiarity; spatial-temporal grounding. Vann et al. (2009)"),
    "IPL":      ("Inferior Parietal Lobule",           "Attention; numerical/causal reasoning; embodied simulation. Culham & Kanwisher (2001)"),
    "TPOJ":     ("Temporo-Parieto-Occipital Junction", "Biological motion; social agency detection. Beauchamp et al. (2002)"),
    "Cingul":   ("Cingulate Cortex",                   "Conflict monitoring; error detection; cognitive control. Botvinick et al. (2004)"),
}


def _get_roi_function(parcel_name: str):
    """Match parcel name to known ROI functional description."""
    for keyword, (roi_name, description) in ROI_FUNCTIONS.items():
        if keyword in parcel_name:
            return roi_name, description
    # Fallback: network-level description
    parts = parcel_name.split("_")
    network = parts[2] if len(parts) >= 3 else "Unknown"
    prof = NETWORK_PROFILES.get(network, {})
    return prof.get("full_name", "Cortical region"), prof.get("core_function", "")


def _extract_network(parcel_name: str):
    """Extract Yeo network key from Schaefer parcel name.
    Format: 7Networks_{Hemi}_{Network}_{SubRegion}_{Index}
    """
    parts = parcel_name.split("_")
    return parts[2] if len(parts) >= 3 else None


def _score_reaction_profiles(network_mean_z: dict, network_display: dict, parcel_z: dict) -> list:
    """
    Score each emotional reaction profile from network activations.
    All scoring rules are traceable to published neuroscience literature.
    No LLM — just activation math.
    """
    import numpy as np

    def net(name):
        # Use z-score normalized to [0,1]: z=2.0 → 1.0, z=0 → 0.0, z<0 → 0.0
        # This means neutral text (all z-scores near 0) scores near 0,
        # while strongly activated networks (z≥2) reach 1.0.
        z = network_mean_z.get(name, 0.0)
        return min(1.0, max(0.0, z / 2.0))

    def net_z(name):
        return network_mean_z.get(name, 0.0)

    def roi_mean_z(keyword):
        matching = [z for name, z in parcel_z.items() if keyword in name]
        return float(np.mean(matching)) if matching else 0.0

    profiles = []

    # ── 1. Fear / Threat Response ──────────────────────────────────────────────
    # Limbic + SalVentAttn co-activation = threat detection circuit.
    # TempPole (emotional memory familiarity) and OFC (threat value) as ROI boosters.
    # Ref: LeDoux (1996) The Emotional Brain; Seeley et al. (2007) J Neurosci
    fear_base  = net("Limbic") * 0.45 + net("SalVentAttn") * 0.45
    fear_score = min(1.0, fear_base
                     + max(0, roi_mean_z("TempPole")) * 0.05
                     + max(0, roi_mean_z("OFC")) * 0.05)
    profiles.append({
        "id": "fear_threat",
        "label": "Fear / Threat Response",
        "score": round(fear_score, 3),
        "networks_driving": ["Limbic", "SalVentAttn"],
        "mechanism": (
            "Limbic network + salience network co-activation is the neurological signature "
            "of threat detection. The same circuitry that evolved to detect predators is "
            "being engaged by symbolic content. Temporal pole activation suggests emotional "
            "memory familiarity is being leveraged — the threat feels 'known' and real."
        ),
        "literature": "LeDoux (1996) The Emotional Brain; Seeley et al. (2007) J Neurosci",
    })

    # ── 2. Personal Identity Targeting ────────────────────────────────────────
    # High DMN = self-referential processing. mPFC + PCC/precuneus are core hubs.
    # Ref: Buckner et al. (2008) Ann NY Acad Sci; Northoff & Bermpohl (2004) Trends Cogn Sci
    identity_score = min(1.0, net("Default")
                         + max(0, roi_mean_z("PFC")) * 0.05
                         + max(0, roi_mean_z("pCunPCC")) * 0.05)
    profiles.append({
        "id": "self_relevance",
        "label": "Personal Identity Targeting",
        "score": round(identity_score, 3),
        "networks_driving": ["Default"],
        "mechanism": (
            "Default Mode Network is the brain's 'self-referential' system — it activates "
            "when you think about yourself, your beliefs, your social identity, your future. "
            "High DMN activation means content has triggered self-referential processing: "
            "you are reading this as being *about you* or *relevant to who you are*. "
            "mPFC and PCC (posterior cingulate) specifically index personal relevance and "
            "autobiographical memory integration."
        ),
        "literature": "Buckner et al. (2008) Ann NY Acad Sci; Northoff & Bermpohl (2004) Trends Cogn Sci",
    })

    # ── 3. Tribal / Social Alarm ───────────────────────────────────────────────
    # TPJ is the key ROI — governs theory of mind and moral judgment.
    # SalVentAttn network houses TPJ. STS for social/agency perception.
    # Ref: Saxe & Kanwisher (2003) NeuroImage; Greene et al. (2001) Science
    tpj_z = roi_mean_z("TPJ")
    sts_z = roi_mean_z("STS")
    tribal_score = min(1.0, net("SalVentAttn") * 0.3
                       + max(0, tpj_z) * 0.5
                       + max(0, sts_z) * 0.2)
    profiles.append({
        "id": "social_tribal",
        "label": "Tribal / Social Alarm",
        "score": round(tribal_score, 3),
        "networks_driving": ["SalVentAttn"],
        "key_roi": "TPJ",
        "mechanism": (
            "The temporoparietal junction (TPJ) is the brain's dedicated circuit for "
            "thinking about other people's mental states — their beliefs, intentions, "
            "group membership. High TPJ activation means content is activating "
            "social categorization: 'us vs. them', moral judgment of out-groups, "
            "or concern about social standing. Superior temporal sulcus supports "
            "biological motion and agency detection — perceiving groups as agents."
        ),
        "literature": "Saxe & Kanwisher (2003) NeuroImage; Greene et al. (2001) Science",
    })

    # ── 4. Reward / Validation Seeking ────────────────────────────────────────
    # OFC and vmPFC = reward value; Limbic-OFC circuit drives approach.
    # Hippocampal activation can signal nostalgia as a reward mechanism.
    # Ref: Wallis (2007) Nat Rev Neurosci; Lieberman (2013) Social
    ofc_z = roi_mean_z("OFC")
    reward_score = min(1.0, net("Limbic") * 0.25
                       + max(0, ofc_z) * 0.6
                       + max(0, roi_mean_z("Hipp")) * 0.15)
    profiles.append({
        "id": "reward_validation",
        "label": "Reward / Validation Seeking",
        "score": round(reward_score, 3),
        "networks_driving": ["Limbic"],
        "key_roi": "OFC / vmPFC",
        "mechanism": (
            "Orbitofrontal cortex computes the reward value of stimuli and drives "
            "approach behavior. When OFC activates for content, it's computing 'this "
            "is good for me / my group / my beliefs.' Content targeting this circuit "
            "offers social validation, status confirmation, or in-group belonging as "
            "an implicit reward for continued engagement or belief adoption."
        ),
        "literature": "Wallis (2007) Nat Rev Neurosci; Lieberman (2013) Social: Why Our Brains Are Wired to Connect",
    })

    # ── 5. Critical Thinking Bypass (protective — inverted) ────────────────────
    # LOW frontoparietal (Cont) = reasoning suppressed. Emotional load competes
    # with FPN for processing resources (dual-process theory, Bechara et al.).
    # Ref: Bechara et al. (2000) Cognition; Kahneman (2011) Thinking Fast and Slow
    fpn_activation  = net("Cont")
    emotional_load  = (net("Limbic") + net("SalVentAttn")) / 2
    bypass_score    = min(1.0, (1.0 - fpn_activation) * 0.5 + emotional_load * 0.5)
    profiles.append({
        "id": "analytical_bypass",
        "label": "Critical Thinking Bypass",
        "score": round(bypass_score, 3),
        "networks_driving": ["Cont"],  # inverted: low Cont = high bypass
        "inverted": True,
        "mechanism": (
            "Frontoparietal control network governs deliberate, analytical evaluation — "
            "the slow, effortful reasoning of System 2 cognition. Emotionally loaded "
            "content suppresses this network as emotional processing competes for "
            "prefrontal resources (Bechara et al.). Low Cont activation + high Limbic/SAL "
            "is a pattern consistent with 'hot cognition' — where emotional processing "
            "tends to precede analytical evaluation, leaving less capacity for deliberate "
            "scrutiny. This does not mean the content is dishonest, but it is worth "
            "slowing down before acting or sharing."
        ),
        "literature": "Bechara et al. (2000) Cognition; Kahneman (2011) Thinking Fast and Slow",
    })

    # ── 6. Urgency / Action Mobilization ──────────────────────────────────────
    # SAL flags urgency; SomMot prepares embodied motor action.
    # Together = felt sense that 'something must be done now'.
    # Ref: Seeley et al. (2007) J Neurosci; Pessoa (2017) Trends Cogn Sci
    urgency_score = net("SalVentAttn") * 0.45 + net("SomMot") * 0.45
    profiles.append({
        "id": "urgency_action",
        "label": "Urgency / Action Mobilization",
        "score": round(urgency_score, 3),
        "networks_driving": ["SalVentAttn", "SomMot"],
        "mechanism": (
            "Salience network flags urgency while somatomotor regions prepare for motor "
            "action — together they create the felt sense that 'something must be done "
            "now.' This is the neurological correlate of mobilization rhetoric. The body "
            "prepares to act even before the content has been analytically evaluated. "
            "Time-pressure framing, countdown language, and crisis narratives specifically "
            "target this circuit."
        ),
        "literature": "Seeley et al. (2007) J Neurosci; Pessoa (2017) Trends Cogn Sci",
    })

    profiles.sort(key=lambda x: x["score"], reverse=True)
    return profiles


def _build_interpretation(profiles: list, network_display: dict) -> dict:
    """
    Build final human-readable interpretation from scored profiles.
    Rule-based mapping — traceable to activation pattern, no LLM generation.
    """
    primary   = profiles[0] if profiles else None
    secondary = profiles[1] if len(profiles) > 1 else None

    # ── Manipulation index ─────────────────────────────────────────────────────
    # Combines emotional targeting intensity × analytical suppression (scale 0-10).
    # High emotional score + high bypass → high manipulation index.
    bypass_profile = next((p for p in profiles if p["id"] == "analytical_bypass"), None)
    top_emotional  = max((p["score"] for p in profiles if not p.get("inverted")), default=0)
    bypass_score   = bypass_profile["score"] if bypass_profile else 0.5
    manip_index    = round(min(10.0, top_emotional * 5.0 + bypass_score * 5.0), 1)

    # ── Primary target label ───────────────────────────────────────────────────
    if primary and secondary and secondary["score"] > 0.65 * primary["score"]:
        primary_label = f"{primary['label']} + {secondary['label']}"
    elif primary:
        primary_label = primary["label"]
    else:
        primary_label = "Neutral / Unclear"

    # ── Intended behavioral outcome (rule-based from top profile IDs) ──────────
    top_ids = [p["id"] for p in profiles[:3]]
    outcomes = {
        frozenset(["fear_threat", "urgency_action"]): (
            "Likely to produce: immediate clicking or sharing driven by anxiety, "
            "action before critical evaluation — the urgency feels real and time-sensitive"
        ),
        frozenset(["fear_threat", "social_tribal"]): (
            "Likely to produce: adoption of in-group position on the threat, sharing as "
            "group loyalty signal, heightened distrust of the framed out-group"
        ),
        frozenset(["self_relevance", "reward_validation"]): (
            "Likely to produce: feeling understood and validated, seeking more confirming "
            "content, belief adoption because it fits existing self-concept"
        ),
        frozenset(["social_tribal", "reward_validation"]): (
            "Likely to produce: reinforced group identity, in-group sharing for social "
            "signaling, increased moral distance from the framed out-group"
        ),
        frozenset(["fear_threat", "self_relevance"]): (
            "Likely to produce: personalizing the threat to your own identity, feeling "
            "specifically targeted, seeking safety through deeper engagement with the source"
        ),
        frozenset(["urgency_action", "analytical_bypass"]): (
            "Likely to produce: acting or sharing before deliberate evaluation — "
            "emotional urgency tends to outpace critical reflection in this pattern"
        ),
    }
    intended_outcome = "Likely to produce: emotional engagement, perspective adoption, sharing or action"
    for key_set, outcome in outcomes.items():
        if key_set.issubset(set(top_ids)):
            intended_outcome = outcome
            break

    # ── Who benefits ───────────────────────────────────────────────────────────
    if manip_index >= 7.5:
        who_benefits = "Most likely the content source. High emotional activation combined with reduced analytical engagement — worth examining who benefits from your reaction before acting or sharing."
    elif manip_index >= 5.0:
        who_benefits = "Possibly the source. Moderate emotional framing that may or may not be intentional — legitimate reporting can produce similar patterns. Worth a second read."
    elif manip_index >= 2.5:
        who_benefits = "Unclear. Low-to-moderate emotional activation with analytical engagement still present — no strong signal of one-sided framing."
    else:
        who_benefits = "Likely you, the reader. Low emotional activation signature — content appears primarily informational."

    # ── Inoculation ────────────────────────────────────────────────────────────
    # Based on inoculation theory (McGuire 1964) and prebunking research
    # (Lewandowsky & van der Linden 2021).
    inoculation_map = {
        "fear_threat": (
            "Name the threat explicitly out loud. Then ask: what specific action does "
            "this content suggest I take, and would that action actually reduce the threat? "
            "Vague, unresolvable threats that don't translate into concrete protective action "
            "tend to maintain anxious engagement without giving you a clear path forward."
        ),
        "self_relevance": (
            "Notice when content feels personally about you — especially with second-person "
            "framing ('you', 'your', 'people like you'). Ask who wrote this and how they "
            "could know it's relevant to your specific situation. DMN activation is easily "
            "triggered by generic content that uses identity-adjacent language."
        ),
        "social_tribal": (
            "When you feel moral certainty about an out-group, that's TPJ doing social "
            "categorization. Slow it down: what does someone on the other side say this "
            "content is doing? Tribal alarm compresses complex social reality into a binary. "
            "The compression is the technique."
        ),
        "reward_validation": (
            "If content makes you feel seen or confirmed — your beliefs validated, your "
            "group defended — check whether it's giving you new information or just "
            "confirming what you already believe. Content that only validates doesn't update "
            "your model of the world. That's a signal it's optimizing for your continued "
            "engagement, not your understanding."
        ),
        "analytical_bypass": (
            "If you want to share this before finishing it, or before looking anything up, "
            "notice that impulse. High-bypass content tends to front-load emotional impact "
            "so the felt reaction precedes deliberate evaluation. Finishing it, and waiting "
            "a beat before sharing, is usually enough to reactivate critical judgment."
        ),
        "urgency_action": (
            "When content produces a strong felt urgency, it's worth pausing to check "
            "whether the urgency maps to a concrete action that will actually help. "
            "Content that generates urgency without a clear resolution path can keep you "
            "engaged without moving you forward. Ask: what exactly would change if I "
            "waited 24 hours before doing anything about this?"
        ),
    }
    primary_id  = primary["id"] if primary else "fear_threat"
    inoculation = inoculation_map.get(
        primary_id,
        "Notice your emotional state while reading. Strong reactions are often the target, not a byproduct.",
    )

    return {
        "primary_target":    primary_label,
        "manipulation_index": manip_index,
        "intended_outcome":  intended_outcome,
        "who_benefits":      who_benefits,
        "inoculation":       inoculation,
    }


def _build_segments_breakdown(
    text: str,
    preds,          # (n_timesteps, 20484) np.ndarray
    df,             # events DataFrame from model.get_events_dataframe()
    all_labels,     # (20484,) parcel index per vertex
    all_parcel_names: dict,
    network_mean_z_full: dict,
    network_display_full: dict,
) -> list:
    """
    Split text into sentences and assign each a TR window and dominant profile.

    Approach:
      1. Split text into sentences on ., !, ?
      2. Estimate TR range for each sentence proportional to its character length.
         (More accurate than arbitrary splits; avoids needing to parse word-level df.)
      3. Slice preds[ts_start:ts_end], compute parcel z-scores for that window,
         aggregate to networks, score profiles → dominant profile for that sentence.

    The result gives the frontend its per-sentence colored profile bars.
    """
    import re
    import numpy as np

    n_trs = preds.shape[0]
    if n_trs == 0:
        return []

    # Split into sentences (keep delimiter with the sentence)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        return []

    # Proportional TR allocation by character count
    char_counts = [len(s) for s in sentences]
    total_chars  = sum(char_counts)

    # Build TR boundaries for each sentence
    boundaries = []
    cursor = 0
    for i, chars in enumerate(char_counts):
        frac     = chars / total_chars
        start_tr = cursor
        end_tr   = cursor + max(1, round(frac * n_trs))
        if i == len(char_counts) - 1:
            end_tr = n_trs  # last sentence gets remainder
        end_tr = min(end_tr, n_trs)
        boundaries.append((start_tr, end_tr))
        cursor = end_tr

    breakdown = []
    for sentence, (ts_start, ts_end) in zip(sentences, boundaries):
        if ts_end <= ts_start:
            continue

        # Mean BOLD for this sentence's TR window
        window_bold = preds[ts_start:ts_end].mean(axis=0)  # (20484,)

        # Parcel activations for this window
        parcel_acts = {}
        for parcel_idx in range(1, 201):
            mask = (all_labels == parcel_idx)
            if mask.sum() == 0:
                continue
            parcel_acts[parcel_idx] = float(window_bold[mask].mean())

        if not parcel_acts:
            continue

        # Z-score within this window
        vals   = np.array(list(parcel_acts.values()))
        z_mean = vals.mean()
        z_std  = vals.std()
        parcel_z_win = {
            all_parcel_names.get(idx, f"parcel_{idx}"): float((act - z_mean) / (z_std + 1e-8))
            for idx, act in parcel_acts.items()
        }

        # Network aggregation
        buckets = {net: [] for net in NETWORK_PROFILES}
        for name, z in parcel_z_win.items():
            net = _extract_network(name)
            if net in buckets:
                buckets[net].append(z)

        net_mean_z_win = {
            net: float(np.mean(zs)) for net, zs in buckets.items() if zs
        }
        nv = np.array(list(net_mean_z_win.values()))
        if nv.max() == nv.min():
            net_display_win = {k: 0.5 for k in net_mean_z_win}
        else:
            nmin, nmax = nv.min(), nv.max()
            net_display_win = {
                k: round(float((v - nmin) / (nmax - nmin + 1e-8)), 4)
                for k, v in net_mean_z_win.items()
            }

        # Score profiles for this window
        window_profiles = _score_reaction_profiles(net_mean_z_win, net_display_win, parcel_z_win)
        dominant = window_profiles[0] if window_profiles else {"id": "unknown", "label": "Unknown", "score": 0}

        breakdown.append({
            "text":             sentence,
            "ts_start":         ts_start,
            "ts_end":           ts_end,
            "dominant_profile": dominant["id"],
            "dominant_label":   dominant["label"],
            "dominant_score":   dominant["score"],
            "network_activations": net_display_win,
        })

    return breakdown
