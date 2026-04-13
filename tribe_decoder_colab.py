# ═══════════════════════════════════════════════════════════════════════════════
# TRIBE DECODER — Colab Backend
# Reverse-engineers the emotional reaction a piece of text is designed to trigger.
# Uses Meta FAIR's TRIBE v2 fMRI brain encoding model (d'Ascoli et al. 2026).
#
# Instructions:
#   1. Open Google Colab (colab.google.com), set runtime to GPU (T4 or better)
#   2. Paste each section (marked ## CELL N) into a separate Colab cell
#   3. Run cells in order. Cells 1-5 are one-time setup.
#   4. Cell 7 (the server) must stay running — it's your API endpoint.
#
# Prerequisites:
#   - Free ngrok account: https://ngrok.com (get your authtoken)
#   - HuggingFace account with LLaMA 3.2 access approved:
#     https://huggingface.co/meta-llama/Llama-3.2-3B
# ═══════════════════════════════════════════════════════════════════════════════


## CELL 1 — Install uv + TRIBE v2 (run once, ~3-5 min)
# ───────────────────────────────────────────────────────────────────────────────
"""
!pip install uv --quiet
!uv pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git" --system --quiet
!uv pip install nibabel nilearn flask flask-cors pyngrok --system --quiet
"""


## CELL 2 — Authenticate with HuggingFace (required for LLaMA 3.2 backbone)
# ───────────────────────────────────────────────────────────────────────────────
# TRIBE v2 uses LLaMA 3.2-3B as its text encoder. This model is gated on HF.
# You must first request access at: https://huggingface.co/meta-llama/Llama-3.2-3B
# Then run this cell and paste your HF read token when prompted.
"""
from huggingface_hub import login
login()
"""


## CELL 3 — Load TRIBE v2 Model (~1GB checkpoint + backbone models on first run)
# ───────────────────────────────────────────────────────────────────────────────
"""
from tribev2 import TribeModel
import torch

print("Loading TRIBE v2 from HuggingFace...")
print("First run downloads the checkpoint and backbone models. This may take several minutes.")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./tribe_cache")
print(f"Model loaded. Device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'N/A'}")
"""


## CELL 4 — Download Schaefer 2018 Atlas for fsaverage5
# ───────────────────────────────────────────────────────────────────────────────
# Schaefer 2018: 200-parcel, 7-network parcellation on fsaverage5 surface.
# Authoritative source: Yeo Lab CBIG (https://github.com/ThomasYeoLab/CBIG)
# Reference: Schaefer et al. (2018) Cerebral Cortex, doi:10.1093/cercor/bhx179
#
# Why Schaefer 200 / 7-network?
#   - 200 parcels gives sufficient spatial specificity without over-fragmentation
#   - 7 Yeo networks are the field standard for functional network decomposition
#   - TRIBE v2 paper (Table 2) validates recovery of these exact network landmarks
"""
import os
import urllib.request
import numpy as np
import nibabel as nib

ATLAS_DIR = "./atlas"
os.makedirs(ATLAS_DIR, exist_ok=True)

# Yeo Lab CBIG GitHub — authoritative Schaefer parcellation source
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

# Load parcellation labels for each hemisphere
# fsaverage5: exactly 10,242 vertices per hemisphere
lh_labels, _, lh_names = nib.freesurfer.read_annot(
    os.path.join(ATLAS_DIR, "lh.Schaefer2018_200Parcels_7Networks_order.annot")
)
rh_labels, _, rh_names = nib.freesurfer.read_annot(
    os.path.join(ATLAS_DIR, "rh.Schaefer2018_200Parcels_7Networks_order.annot")
)

def _decode(x):
    return x.decode("utf-8") if isinstance(x, bytes) else x

lh_names = [_decode(n) for n in lh_names]
rh_names = [_decode(n) for n in rh_names]

N_VERTS_PER_HEMI = 10242
assert lh_labels.shape[0] == N_VERTS_PER_HEMI, "Unexpected vertex count"
assert rh_labels.shape[0] == N_VERTS_PER_HEMI, "Unexpected vertex count"

# TRIBE v2 output: (n_timesteps, 20484) — LH vertices first, then RH
# Parcel indexing: 0 = medial wall (unlabeled)
#   LH parcels: 1-100 (in lh_labels)
#   RH parcels: 101-200 (in rh_labels, stored as 1-100, we offset by 100)
rh_labels_global = np.where(rh_labels > 0, rh_labels + 100, 0)
all_labels = np.concatenate([lh_labels, rh_labels_global])  # (20484,)

# Build lookup: parcel index 1-200 → name string
all_parcel_names = {}
for i, name in enumerate(lh_names):
    if i > 0:  # 0 = "unknown" / medial wall
        all_parcel_names[i] = name
for i, name in enumerate(rh_names):
    if i > 0:
        all_parcel_names[i + 100] = name

print(f"\nAtlas ready: {len(all_parcel_names)} parcels on fsaverage5")
print(f"Example parcel names:")
for idx, name in list(all_parcel_names.items())[:4]:
    print(f"  [{idx:3d}] {name}")
"""


## CELL 5 — Neuroscience Profiles (rule-based, literature-grounded)
# ───────────────────────────────────────────────────────────────────────────────
# These profiles map Yeo 7-network activation patterns to emotional/cognitive
# reactions. Every claim here is traceable to published neuroscience literature.
# This is the interpretive layer — it translates real fMRI predictions into
# human-readable reaction descriptions. It does NOT use an LLM.
"""
# ── Yeo 7-Network functional definitions ─────────────────────────────────────
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
            "evaluative faculties. LOW activation is the warning signal: content is "
            "designed to bypass critical thinking."
        ),
        "references": "Miller & Cohen (2001) Ann Rev Neurosci; Dosenbach et al. (2007) PNAS",
        "protective": True,  # Inverted — low is the concern
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

# ── Specific ROI keyword → functional description ─────────────────────────────
# Matched against Schaefer parcel names (e.g., "7Networks_LH_SalVentAttn_TPJ_1")
# References: established ROI function literature
ROI_FUNCTIONS = {
    "TPJ":     ("Temporoparietal Junction", "Theory of mind; moral judgment; in-group/out-group detection. Saxe & Kanwisher (2003)"),
    "PFC":     ("Prefrontal Cortex [Default]", "Self-referential processing; belief importance to identity. Buckner et al. (2008)"),
    "pCunPCC": ("Posterior Cingulate / Precuneus", "Autobiographical memory retrieval; self-relevance; rumination. Andrews-Hanna (2010)"),
    "TempPole":("Temporal Pole", "Social/emotional memory; familiar person/situation recognition. Olson et al. (2007)"),
    "OFC":     ("Orbitofrontal Cortex", "Reward/punishment value computation; drives approach/avoidance. Wallis (2007)"),
    "Insula":  ("Anterior Insula", "Interoception; visceral disgust; moral norm violations. Craig (2009)"),
    "Hipp":    ("Hippocampus", "Episodic memory retrieval; nostalgia; contextual familiarity. Squire et al. (2004)"),
    "STS":     ("Superior Temporal Sulcus", "Biological motion; social perception; voice/speech. Allison et al. (2000)"),
    "IFG":     ("Inferior Frontal Gyrus [Broca]", "Language comprehension; authority framing via rhetorical structure. Bookheimer (2002)"),
    "Rsp":     ("Retrosplenial Cortex", "Scene/context familiarity; spatial-temporal grounding. Vann et al. (2009)"),
    "IPL":     ("Inferior Parietal Lobule", "Attention; numerical/causal reasoning; embodied simulation. Culham & Kanwisher (2001)"),
    "TPOJ":    ("Temporo-Parieto-Occipital Junction", "Biological motion; social agency detection. Beauchamp et al. (2002)"),
    "Cingul":  ("Cingulate Cortex", "Conflict monitoring; error detection; cognitive control. Botvinick et al. (2004)"),
}

def get_roi_function(parcel_name):
    """Match parcel name to known ROI functional description."""
    for keyword, (roi_name, description) in ROI_FUNCTIONS.items():
        if keyword in parcel_name:
            return roi_name, description
    # Fallback: return network-level description
    parts = parcel_name.split("_")
    network = parts[2] if len(parts) >= 3 else "Unknown"
    prof = NETWORK_PROFILES.get(network, {})
    return prof.get("full_name", "Cortical region"), prof.get("core_function", "")

def extract_network(parcel_name):
    """Extract Yeo network identifier from Schaefer parcel name.
    Format: 7Networks_{Hemi}_{Network}_{SubRegion}_{Index}
    """
    parts = parcel_name.split("_")
    if len(parts) >= 3:
        return parts[2]
    return None

print("Neuroscience profiles loaded.")
print(f"  {len(NETWORK_PROFILES)} network profiles")
print(f"  {len(ROI_FUNCTIONS)} specific ROI mappings")
"""


## CELL 6 — Core Analysis Functions
# ───────────────────────────────────────────────────────────────────────────────
"""
import tempfile
import traceback as tb

def run_tribe_inference(text: str) -> dict:
    """
    Full pipeline: text → TRIBE v2 → ROI activations → reaction profiles.

    Pipeline steps:
      1. Write text to temp file (TRIBE requires file-based input)
      2. TribeModel.get_events_dataframe() converts text→speech (internal TTS),
         extracts word-level timings, builds events DataFrame
      3. TribeModel.predict() runs inference → (n_timesteps, 20484) predicted BOLD
         on fsaverage5, 1 TR per second, hemodynamic lag already compensated
      4. Temporal mean → (20484,) mean predicted BOLD across the stimulus
      5. Vertex → Schaefer parcel mapping → 200 parcel activations
      6. Z-score normalization across parcels (relative activation)
      7. Aggregate to 7 Yeo network activations
      8. Score reaction profiles from network + specific ROI data
      9. Build rule-based interpretation (no LLM)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        text_path = f.name

    try:
        # ── Step 2-3: TRIBE v2 inference ───────────────────────────────────────
        print("  Building events dataframe (TTS + word timing alignment)...")
        df = model.get_events_dataframe(text_path=text_path)
        print(f"  Events: {len(df)} segments")

        print("  Running TRIBE v2 inference...")
        preds, segments = model.predict(events=df)
        # preds: (n_timesteps, n_vertices), float32 predicted BOLD signal
        print(f"  Prediction shape: {preds.shape}")

        if hasattr(preds, "numpy"):
            preds = preds.numpy()  # torch tensor → numpy if needed

        # ── Step 4: Temporal aggregation ──────────────────────────────────────
        # Mean across time gives overall activation pattern for this content.
        # Standard approach in fMRI encoding literature for non-time-resolved analysis.
        mean_bold = preds.mean(axis=0)  # (20484,)
        assert mean_bold.shape[0] == 20484, f"Unexpected vertex count: {mean_bold.shape[0]}"

        # ── Step 5: Vertex → parcel mapping ───────────────────────────────────
        parcel_activations = {}
        for parcel_idx in range(1, 201):
            mask = (all_labels == parcel_idx)
            n_verts = mask.sum()
            if n_verts == 0:
                continue
            parcel_activations[parcel_idx] = float(mean_bold[mask].mean())

        # ── Step 6: Z-score across parcels ────────────────────────────────────
        # Normalizes for relative comparison — which regions are MORE active
        # than average for this content, not absolute BOLD values.
        vals = np.array(list(parcel_activations.values()))
        z_mean, z_std = vals.mean(), vals.std()

        parcel_z = {}
        for idx, act in parcel_activations.items():
            name = all_parcel_names.get(idx, f"parcel_{idx}")
            parcel_z[name] = float((act - z_mean) / (z_std + 1e-8))

        # ── Step 7: Network-level aggregation ─────────────────────────────────
        network_buckets = {net: [] for net in NETWORK_PROFILES}
        for name, z in parcel_z.items():
            net = extract_network(name)
            if net in network_buckets:
                network_buckets[net].append(z)

        network_mean_z = {}
        for net, zs in network_buckets.items():
            if zs:
                network_mean_z[net] = float(np.mean(zs))

        # Min-max normalize for display (0-1 scale)
        nv = np.array(list(network_mean_z.values()))
        nmin, nmax = nv.min(), nv.max()
        network_display = {
            k: round(float((v - nmin) / (nmax - nmin + 1e-8)), 4)
            for k, v in network_mean_z.items()
        }

        # ── Step 8: Top/bottom parcels ────────────────────────────────────────
        sorted_parcels = sorted(parcel_z.items(), key=lambda x: x[1], reverse=True)

        top_rois = []
        for name, z in sorted_parcels[:12]:
            roi_name, roi_desc = get_roi_function(name)
            top_rois.append({
                "parcel": name,
                "z_score": round(z, 3),
                "network": extract_network(name),
                "roi_name": roi_name,
                "description": roi_desc,
            })

        # ── Step 9: Reaction profiles ──────────────────────────────────────────
        profiles = score_reaction_profiles(network_mean_z, network_display, parcel_z)
        interpretation = build_interpretation(profiles, network_display)

        return {
            "success": True,
            "network_activations": network_display,
            "network_raw_z": {k: round(v, 4) for k, v in network_mean_z.items()},
            "network_profiles": {k: v for k, v in NETWORK_PROFILES.items()},
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
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": tb.format_exc(),
        }
    finally:
        os.unlink(text_path)


def score_reaction_profiles(network_mean_z: dict, network_display: dict, parcel_z: dict) -> list:
    """
    Score each emotional reaction profile from network activations.
    All scoring rules are traceable to published neuroscience literature.
    No LLM, no vibes — just activation math.
    """

    def net(name):
        """Get 0-1 normalized activation for a network."""
        return network_display.get(name, 0.0)

    def net_z(name):
        """Get raw z-score for a network."""
        return network_mean_z.get(name, 0.0)

    def roi_mean_z(keyword):
        """Mean z-score across all parcels matching keyword."""
        matching = [z for name, z in parcel_z.items() if keyword in name]
        return float(np.mean(matching)) if matching else 0.0

    profiles = []

    # ── 1. Fear / Threat Response ─────────────────────────────────────────────
    # Limbic + SalVentAttn co-activation = threat detection circuit
    # TempPole (emotional memory) and OFC (threat value) as ROI boosters
    fear_base = net("Limbic") * 0.45 + net("SalVentAttn") * 0.45
    temppole_boost = max(0, roi_mean_z("TempPole")) * 0.05
    ofc_boost = max(0, roi_mean_z("OFC")) * 0.05
    fear_score = min(1.0, fear_base + temppole_boost + ofc_boost)

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

    # ── 2. Personal Identity Targeting ───────────────────────────────────────
    # High DMN = self-referential processing engaged
    # mPFC + PCC/precuneus are core DMN hubs for self-referential thought
    dmn_base = net("Default")
    pfc_default_boost = max(0, roi_mean_z("PFC")) * 0.05  # Default network PFC only
    pcc_boost = max(0, roi_mean_z("pCunPCC")) * 0.05
    identity_score = min(1.0, dmn_base + pfc_default_boost + pcc_boost)

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

    # ── 3. Tribal / Social Alarm ──────────────────────────────────────────────
    # TPJ is the key ROI — governs theory of mind and moral judgment
    # SalVentAttn network houses TPJ for social alarm function
    tpj_z = roi_mean_z("TPJ")
    sts_z = roi_mean_z("STS")
    tribal_base = net("SalVentAttn") * 0.3
    tpj_contribution = max(0, tpj_z) * 0.5
    sts_contribution = max(0, sts_z) * 0.2
    tribal_score = min(1.0, tribal_base + tpj_contribution + sts_contribution)

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
    # OFC and vmPFC = reward value computation and social reward
    # Limbic-OFC circuit drives approach toward validating content
    ofc_z = roi_mean_z("OFC")
    reward_base = net("Limbic") * 0.25
    ofc_contribution = max(0, ofc_z) * 0.6
    # Hippocampal activation can also signal nostalgia as a reward
    hipp_contribution = max(0, roi_mean_z("Hipp")) * 0.15
    reward_score = min(1.0, reward_base + ofc_contribution + hipp_contribution)

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

    # ── 5. Critical Thinking Bypass (protective, inverted) ────────────────────
    # LOW frontoparietal (Cont) = reasoning suppressed
    # The bypass score is 1 - analytical engagement
    # Reference: high emotion suppresses prefrontal executive control
    fpn_activation = net("Cont")
    # When emotional networks (Limbic, SalVentAttn) are very high, they compete
    # with FPN for processing resources (dual-process theory)
    emotional_load = (net("Limbic") + net("SalVentAttn")) / 2
    # Bypass is stronger when emotional load is high AND analytical is low
    bypass_score = min(1.0, (1.0 - fpn_activation) * 0.5 + emotional_load * 0.5)

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
            "= content is designed to be felt and acted on rather than evaluated. "
            "This is the neurological basis of 'hot cognition' — decisions made under "
            "emotional load with reduced deliberative capacity."
        ),
        "literature": "Bechara et al. (2000) Cognition; Kahneman (2011) Thinking Fast and Slow",
    })

    # ── 6. Urgency / Action Mobilization ─────────────────────────────────────
    # SAL + SomMot = salience flag + embodied action preparation
    urgency_score = net("SalVentAttn") * 0.5 + net("SomMot") * 0.5

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


def build_interpretation(profiles: list, network_display: dict) -> dict:
    """
    Build final human-readable interpretation from scored profiles.
    Rule-based mapping — traceable to activation pattern, no LLM generation.
    """
    primary = profiles[0] if profiles else None
    secondary = profiles[1] if len(profiles) > 1 else None

    # ── Manipulation index ────────────────────────────────────────────────────
    # Combines: emotional targeting intensity × analytical suppression
    # Scale 0-10. High emotional score + high bypass = high manipulation index.
    bypass_profile = next((p for p in profiles if p["id"] == "analytical_bypass"), None)
    top_emotional = max((p["score"] for p in profiles if not p.get("inverted")), default=0)
    bypass_score = bypass_profile["score"] if bypass_profile else 0.5

    manip_index = round(min(10.0, top_emotional * 5.0 + bypass_score * 5.0), 1)

    # ── Primary target label ──────────────────────────────────────────────────
    if primary and secondary and secondary["score"] > 0.65 * primary["score"]:
        primary_label = f"{primary['label']} + {secondary['label']}"
    elif primary:
        primary_label = primary["label"]
    else:
        primary_label = "Neutral / Unclear"

    # ── Intended behavioral outcome (rule-based from top profile IDs) ─────────
    top_ids = [p["id"] for p in profiles[:3]]

    outcomes = {
        frozenset(["fear_threat", "urgency_action"]): (
            "Click through immediately, share to warn others (from anxiety), "
            "take impulsive action before critical evaluation"
        ),
        frozenset(["fear_threat", "social_tribal"]): (
            "Adopt in-group position on the threat, share to signal group loyalty, "
            "heighten distrust of out-group framed as responsible"
        ),
        frozenset(["self_relevance", "reward_validation"]): (
            "Feel understood and validated, seek more confirming content, "
            "adopt belief because it fits self-concept"
        ),
        frozenset(["social_tribal", "reward_validation"]): (
            "Strengthen group identity, share for social signaling within in-group, "
            "punish or ostracize out-group"
        ),
        frozenset(["fear_threat", "self_relevance"]): (
            "Personalize the threat to your own life/identity, feel specifically targeted, "
            "seek safety by engaging more deeply with the source"
        ),
        frozenset(["urgency_action", "analytical_bypass"]): (
            "Act impulsively without deliberation, share before reading fully, "
            "circumvent your own critical evaluation process"
        ),
    }

    intended_outcome = "Engage emotionally, adopt the framed perspective, share or act"
    for key_set, outcome in outcomes.items():
        if key_set.issubset(set(top_ids)):
            intended_outcome = outcome
            break

    # ── Who benefits ──────────────────────────────────────────────────────────
    if manip_index >= 7.5:
        who_benefits = "The content creator. High emotional targeting + analytical suppression = engineered persuasion."
    elif manip_index >= 5.0:
        who_benefits = "Possibly the source. Moderate emotional framing — may be legitimate, worth examining critically."
    elif manip_index >= 2.5:
        who_benefits = "Unclear. Low-moderate emotional activation, analytical engagement present."
    else:
        who_benefits = "Likely neutral. Low manipulation signature — content appears informational."

    # ── Inoculation ───────────────────────────────────────────────────────────
    # Specific to the primary reaction profile. Based on inoculation theory
    # (McGuire 1964) and prebunking research (Lewandowsky & van der Linden 2021)
    inoculation_map = {
        "fear_threat": (
            "Name the threat explicitly out loud. Then ask: what specific action does "
            "this content want me to take, and does that action actually reduce the threat? "
            "Vague, unresolvable threats that don't translate into concrete protective action "
            "are designed to maintain anxious engagement — not to help you."
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
            "that impulse is the mechanism. High-bypass content is optimized for reflexive "
            "sharing — the emotional payload is front-loaded precisely so you act before "
            "the analytical system catches up."
        ),
        "urgency_action": (
            "Urgency is almost always manufactured. Real emergencies give you enough "
            "information to act; manufactured urgency withholds resolution to keep you "
            "dependent on the source for the next update. Ask: what exactly would change "
            "if I waited 24 hours before doing anything about this?"
        ),
    }

    primary_id = primary["id"] if primary else "fear_threat"
    inoculation = inoculation_map.get(
        primary_id,
        "Notice your emotional state while reading. Strong reactions are often the target, not a byproduct."
    )

    return {
        "primary_target": primary_label,
        "manipulation_index": manip_index,
        "intended_outcome": intended_outcome,
        "who_benefits": who_benefits,
        "inoculation": inoculation,
    }


print("Analysis functions ready.")
"""


## CELL 7 — Flask API Server + ngrok Tunnel (keep this cell running)
# ───────────────────────────────────────────────────────────────────────────────
# This cell starts the API server and exposes it publicly via ngrok.
# Copy the printed URL into the local web app's config field.
# Keep this cell running for the duration of your session.
"""
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

# ── Replace with your ngrok authtoken ────────────────────────────────────────
# Get it at: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"
# ─────────────────────────────────────────────────────────────────────────────

ngrok.set_auth_token(NGROK_AUTHTOKEN)

flask_app = Flask(__name__)
CORS(flask_app)

@flask_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "TRIBE v2", "atlas": "Schaefer 200-parcel 7-network"})

@flask_app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"success": False, "error": "No text provided"}), 400

    text = data["text"].strip()
    if len(text) > 6000:
        # Trim to ~6000 chars (~1 min of speech). Longer = slower TTS+inference.
        text = text[:6000]
        trimmed = True
    else:
        trimmed = False

    print(f"\n[REQUEST] {len(text)} chars")
    result = run_tribe_inference(text)
    if trimmed:
        result["warning"] = "Text was trimmed to 6000 characters for inference speed."
    print(f"[DONE] success={result.get('success')}")
    return jsonify(result)

def _run_flask():
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=_run_flask, daemon=True)
flask_thread.start()

tunnel = ngrok.connect(5000)
public_url = tunnel.public_url

print()
print("=" * 65)
print("  TRIBE DECODER — API LIVE")
print("=" * 65)
print(f"  Endpoint : {public_url}")
print(f"  Health   : {public_url}/health")
print(f"  Analyze  : POST {public_url}/analyze")
print("=" * 65)
print()
print("  → Copy the Endpoint URL into the local web app config.")
print("  → Keep this cell running. Interrupting it kills the tunnel.")
print()
"""
