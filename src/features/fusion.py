import numpy as np


# Names of geometric scalars extracted from metrics dict (order matters for the vector)
GEO_FEATURE_NAMES = [
    # ── Existing features (v0.2) ────────────────────────────────
    "canthal_tilt_left_deg",
    "canthal_tilt_right_deg",
    "canthal_tilt_mean_deg",
    "fwhr",
    "thirds_mid_ratio",
    "thirds_lower_ratio",
    "thirds_mid_to_lower",
    "fifths_seg0_ratio",
    "fifths_seg1_ratio",
    "fifths_seg2_ratio",
    "fifths_seg3_ratio",
    "fifths_seg4_ratio",
    "fifths_outer_pair_diff",
    "fifths_inner_pair_diff",
    "ratios_nose_to_eye",
    "ratios_mouth_to_eye",
    "ratios_nose_to_intercanthal",
    "eye_spacing_ratio",
    "jaw_right_angle",
    "jaw_left_angle",
    "jaw_mean_angle",
    "jaw_to_face_ratio",
    "landmark_symmetry_score",
    # ── New features (v0.3) ─────────────────────────────────────
    "jaw_width_px",
    "jaw_to_bizygomatic",
    "jaw_to_face_height",
    "chin_angle_deg",
    "face_aspect_ratio",
    "midface_ratio",
    "philtrum_ratio",
    "lip_upper_to_lower_ratio",
    "nose_to_intercanthal_ratio",
    "nose_to_bizygomatic_ratio",
    "nose_length_to_midface",
    "nose_length_to_total",
    "icd_px",
    "icd_to_bizygomatic",
    "ipd_px",
    "ipd_to_bizygomatic",
    "brow_height_left",
    "brow_height_right",
    "brow_to_eye_left",
    "brow_to_eye_right",
    "eye_openness_left",
    "eye_openness_right",
    "eye_openness_mean",
    "jaw_curvature_ratio",
    "cheekbone_width_px",
    "cheekbone_to_face_height",
]


def geometry_to_vector(metrics: dict) -> tuple:
    """
    Flattens a geometry metrics dict (from compute_all_metrics) into a 1-D numpy
    array of floats suitable for downstream ML fusion.

    Returns:
        vector        : numpy array of shape (N_geo,) — NaN/Inf replaced by 0.0
        feature_names : list of strings describing each dimension
    """
    ct    = metrics.get("canthal_tilt", {})
    fwhr  = metrics.get("fwhr", {})
    t3    = metrics.get("facial_thirds", {})
    f5    = metrics.get("facial_fifths", {})
    segs  = f5.get("segments", [])
    rat   = metrics.get("facial_ratios", {})
    jaw   = metrics.get("jawline", {})
    # new
    jw    = metrics.get("jaw_width", {})
    ca    = metrics.get("chin_angle", {})
    far   = metrics.get("face_aspect_ratio", {})
    mfr   = metrics.get("midface_ratio", {})
    phil  = metrics.get("philtrum_ratio", {})
    lip   = metrics.get("lip_thickness_ratio", {})
    nwr   = metrics.get("nose_width_ratio", {})
    nlr   = metrics.get("nose_length_ratio", {})
    icd   = metrics.get("intercanthal_distance", {})
    ipd   = metrics.get("interpupillary_distance", {})
    brow  = metrics.get("eyebrow_height", {})
    eyo   = metrics.get("eye_openness", {})
    jcur  = metrics.get("jaw_curvature", {})
    cbw   = metrics.get("cheekbone_width", {})

    values = [
        # existing
        ct.get("left_deg", 0.0),
        ct.get("right_deg", 0.0),
        ct.get("mean_deg", 0.0),
        fwhr.get("value", 0.0),
        t3.get("mid_ratio", 0.0),
        t3.get("lower_ratio", 0.0),
        t3.get("mid_to_lower", 0.0),
        segs[0]["ratio"] if len(segs) > 0 else 0.0,
        segs[1]["ratio"] if len(segs) > 1 else 0.0,
        segs[2]["ratio"] if len(segs) > 2 else 0.0,
        segs[3]["ratio"] if len(segs) > 3 else 0.0,
        segs[4]["ratio"] if len(segs) > 4 else 0.0,
        f5.get("outer_pair_diff", 0.0),
        f5.get("inner_pair_diff", 0.0),
        rat.get("nose_to_eye_ratio", 0.0),
        rat.get("mouth_to_eye_ratio", 0.0),
        rat.get("nose_to_intercanthal", 0.0),
        metrics.get("eye_spacing_ratio", 0.0),
        jaw.get("right_angle_deg", 0.0),
        jaw.get("left_angle_deg", 0.0),
        jaw.get("mean_angle_deg", 0.0),
        jaw.get("jaw_to_face_ratio", 0.0),
        metrics.get("landmark_symmetry_score", 0.0),
        # new
        jw.get("jaw_width_px", 0.0),
        jw.get("jaw_to_bizygomatic", 0.0),
        jw.get("jaw_to_face_height", 0.0),
        ca.get("chin_angle_deg", 0.0),
        far.get("value", 0.0),
        mfr.get("midface_ratio", 0.0),
        phil.get("philtrum_ratio", 0.0),
        lip.get("upper_to_lower_ratio", 0.0),
        nwr.get("nose_to_intercanthal_ratio", 0.0),
        nwr.get("nose_to_bizygomatic_ratio", 0.0),
        nlr.get("nose_to_midface_ratio", 0.0),
        nlr.get("nose_to_total_face_ratio", 0.0),
        icd.get("icd_px", 0.0),
        icd.get("icd_to_bizygomatic", 0.0),
        ipd.get("ipd_px", 0.0),
        ipd.get("ipd_to_bizygomatic", 0.0),
        brow.get("left_brow_height_px", 0.0),
        brow.get("right_brow_height_px", 0.0),
        brow.get("left_brow_to_eye_ratio", 0.0),
        brow.get("right_brow_to_eye_ratio", 0.0),
        eyo.get("left_openness_ratio", 0.0),
        eyo.get("right_openness_ratio", 0.0),
        eyo.get("mean_openness_ratio", 0.0),
        jcur.get("curvature_ratio", 0.0),
        cbw.get("cheekbone_width_px", 0.0),
        cbw.get("cheekbone_to_face_height", 0.0),
    ]

    assert len(values) == len(GEO_FEATURE_NAMES), \
        f"Mismatch: {len(values)} values vs {len(GEO_FEATURE_NAMES)} names"

    vector = np.array(values, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector, GEO_FEATURE_NAMES


def fuse_features(geo_vector: np.ndarray, dino_embedding: np.ndarray) -> dict:
    """
    Concatenates geometric feature vector with DINOv2 embedding.

    v0.2/v0.3 baseline fusion: simple concatenation (no learned weights).
    MLP regression head planned for v0.4 once labelled data (SCUT-FBP5500)
    is integrated.

    Args:
        geo_vector     : numpy array (N_geo,)   — from geometry_to_vector()
        dino_embedding : numpy array (N_dino,)  — CLS or mean-pool token from DINOv2

    Returns:
        dict with fused vector and dimensional metadata
    """
    fused = np.concatenate([geo_vector, dino_embedding], axis=0).astype(np.float32)
    return {
        "vector":            fused,
        "total_dims":        int(fused.shape[0]),
        "geo_dims":          int(geo_vector.shape[0]),
        "dino_dims":         int(dino_embedding.shape[0]),
        "geo_feature_names": GEO_FEATURE_NAMES,
        "note": (
            "v0.3 baseline: simple concatenation of 46 geo features + DINOv2. "
            "MLP regression head planned for v0.4 after SCUT-FBP5500 labelling."
        )
    }
