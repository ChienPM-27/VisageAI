"""
fusion.py — Geometric feature vectorization + DINOv2 feature fusion.

v0.4 change: removed 6 absolute-pixel dimensions (jaw_width_px, icd_px,
ipd_px, brow_height_left, brow_height_right, cheekbone_width_px).
These were resolution-dependent and redundant with their ratio counterparts
already present in the vector.

Final vector: 43 geo + 384 DINOv2 CLS = 427-d (resolution-invariant)
"""
import numpy as np


# ─── Geometric feature manifest (order is fixed — do not reorder) ──────────
# 43 features, all scale-invariant ratios or angular measurements.
GEO_FEATURE_NAMES = [
    # ── Canthal Tilt (v0.2) ──────────────────────────────────────────────
    "canthal_tilt_left_deg",
    "canthal_tilt_right_deg",
    "canthal_tilt_mean_deg",
    # ── fWHR + Jaw (v0.2) ────────────────────────────────────────────────
    "fwhr",
    "jaw_right_angle",
    "jaw_left_angle",
    "jaw_mean_angle",
    "jaw_to_face_ratio",
    # ── Vertical Proportions (v0.2) ───────────────────────────────────────
    "thirds_mid_ratio",
    "thirds_lower_ratio",
    "thirds_mid_to_lower",
    # ── Horizontal Proportions / Fifths (v0.2) ────────────────────────────
    "fifths_seg0_ratio",
    "fifths_seg1_ratio",
    "fifths_seg2_ratio",
    "fifths_seg3_ratio",
    "fifths_seg4_ratio",
    "fifths_outer_pair_diff",
    "fifths_inner_pair_diff",
    # ── Facial Ratios (v0.2) ──────────────────────────────────────────────
    "ratios_nose_to_eye",
    "ratios_mouth_to_eye",
    "ratios_nose_to_intercanthal",
    "eye_spacing_ratio",
    # ── Symmetry (v0.2) ───────────────────────────────────────────────────
    "landmark_symmetry_score",
    # ── Jaw Shape (v0.3) — ratios only ───────────────────────────────────
    "jaw_to_bizygomatic",
    "jaw_to_face_height",
    "chin_angle_deg",
    "jaw_curvature_ratio",
    # ── Global Face Shape (v0.3) ──────────────────────────────────────────
    "face_aspect_ratio",
    "midface_ratio",
    "cheekbone_to_face_height",
    # ── Lips / Philtrum (v0.3) ────────────────────────────────────────────
    "philtrum_ratio",
    "lip_upper_to_lower_ratio",
    # ── Nose (v0.3) ───────────────────────────────────────────────────────
    "nose_to_intercanthal_ratio",
    "nose_to_bizygomatic_ratio",
    "nose_length_to_midface",
    "nose_length_to_total",
    # ── Eye / Brow Spacing (v0.3) — ratios only ──────────────────────────
    "icd_to_bizygomatic",
    "ipd_to_bizygomatic",
    "brow_to_eye_left",
    "brow_to_eye_right",
    "eye_openness_left",
    "eye_openness_right",
    "eye_openness_mean",
]

GEO_DIM = len(GEO_FEATURE_NAMES)  # 43
assert GEO_DIM == 43, f"Expected 43 geo features, got {GEO_DIM}"


def geometry_to_vector(metrics: dict) -> tuple:
    """
    Flattens a geometry metrics dict (from compute_all_metrics) into a
    1-D numpy array of resolution-invariant floats.

    Returns:
        vector        : np.ndarray of shape (43,) — NaN/Inf replaced by 0.0
        feature_names : list[str]  (same order as GEO_FEATURE_NAMES)
    """
    ct   = metrics.get("canthal_tilt", {})
    fwhr = metrics.get("fwhr", {})
    jaw  = metrics.get("jawline", {})
    t3   = metrics.get("facial_thirds", {})
    f5   = metrics.get("facial_fifths", {})
    segs = f5.get("segments", [])
    rat  = metrics.get("facial_ratios", {})
    # v0.3
    jw   = metrics.get("jaw_width", {})
    ca   = metrics.get("chin_angle", {})
    far  = metrics.get("face_aspect_ratio", {})
    mfr  = metrics.get("midface_ratio", {})
    phil = metrics.get("philtrum_ratio", {})
    lip  = metrics.get("lip_thickness_ratio", {})
    nwr  = metrics.get("nose_width_ratio", {})
    nlr  = metrics.get("nose_length_ratio", {})
    icd  = metrics.get("intercanthal_distance", {})
    ipd  = metrics.get("interpupillary_distance", {})
    brow = metrics.get("eyebrow_height", {})
    eyo  = metrics.get("eye_openness", {})
    jcur = metrics.get("jaw_curvature", {})
    cbw  = metrics.get("cheekbone_width", {})

    values = [
        # canthal tilt
        ct.get("left_deg", 0.0),
        ct.get("right_deg", 0.0),
        ct.get("mean_deg", 0.0),
        # fwhr + jaw ratios
        fwhr.get("value", 0.0),
        jaw.get("right_angle_deg", 0.0),
        jaw.get("left_angle_deg", 0.0),
        jaw.get("mean_angle_deg", 0.0),
        jaw.get("jaw_to_face_ratio", 0.0),
        # vertical
        t3.get("mid_ratio", 0.0),
        t3.get("lower_ratio", 0.0),
        t3.get("mid_to_lower", 0.0),
        # fifths
        segs[0]["ratio"] if len(segs) > 0 else 0.0,
        segs[1]["ratio"] if len(segs) > 1 else 0.0,
        segs[2]["ratio"] if len(segs) > 2 else 0.0,
        segs[3]["ratio"] if len(segs) > 3 else 0.0,
        segs[4]["ratio"] if len(segs) > 4 else 0.0,
        f5.get("outer_pair_diff", 0.0),
        f5.get("inner_pair_diff", 0.0),
        # facial ratios
        rat.get("nose_to_eye_ratio", 0.0),
        rat.get("mouth_to_eye_ratio", 0.0),
        rat.get("nose_to_intercanthal", 0.0),
        metrics.get("eye_spacing_ratio", 0.0),
        # symmetry
        metrics.get("landmark_symmetry_score", 0.0),
        # jaw shape (ratios)
        jw.get("jaw_to_bizygomatic", 0.0),
        jw.get("jaw_to_face_height", 0.0),
        ca.get("chin_angle_deg", 0.0),
        jcur.get("curvature_ratio", 0.0),
        # global face shape
        far.get("value", 0.0),
        mfr.get("midface_ratio", 0.0),
        cbw.get("cheekbone_to_face_height", 0.0),
        # lips / philtrum
        phil.get("philtrum_ratio", 0.0),
        lip.get("upper_to_lower_ratio", 0.0),
        # nose
        nwr.get("nose_to_intercanthal_ratio", 0.0),
        nwr.get("nose_to_bizygomatic_ratio", 0.0),
        nlr.get("nose_to_midface_ratio", 0.0),
        nlr.get("nose_to_total_face_ratio", 0.0),
        # eye/brow (ratios only)
        icd.get("icd_to_bizygomatic", 0.0),
        ipd.get("ipd_to_bizygomatic", 0.0),
        brow.get("left_brow_to_eye_ratio", 0.0),
        brow.get("right_brow_to_eye_ratio", 0.0),
        eyo.get("left_openness_ratio", 0.0),
        eyo.get("right_openness_ratio", 0.0),
        eyo.get("mean_openness_ratio", 0.0),
    ]

    assert len(values) == GEO_DIM, \
        f"Mismatch: {len(values)} values vs {GEO_DIM} names"

    vec = np.array(values, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, GEO_FEATURE_NAMES


def fuse_features(geo_vector: np.ndarray, dino_embedding: np.ndarray) -> dict:
    """
    Concatenates geometric feature vector with DINOv2 embedding.

    v0.4 baseline fusion: simple concatenation.
    MLP regression head (src/training/train_mlp.py) maps this to a score.

    Args:
        geo_vector     : np.ndarray (43,)  — from geometry_to_vector()
        dino_embedding : np.ndarray (384,) — CLS token from DINOv2 ViT-S/14

    Returns:
        dict with fused vector and dimensional metadata.
    """
    fused = np.concatenate([geo_vector, dino_embedding]).astype(np.float32)
    return {
        "vector":            fused,
        "total_dims":        int(fused.shape[0]),
        "geo_dims":          int(geo_vector.shape[0]),
        "dino_dims":         int(dino_embedding.shape[0]),
        "geo_feature_names": GEO_FEATURE_NAMES,
        "note": (
            "v0.4: 43 resolution-invariant geo features + 384-d DINOv2 CLS = 427-d. "
            "Absolute px dimensions removed. MLP regressor in src/training/train_mlp.py."
        ),
    }
