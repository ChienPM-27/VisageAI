"""
scoring.py — All scoring strategies for VisageAI.

Three scorers, each with a different basis:

1. GeometricScorer  — Rule-based. Deducts points for deviations from
   Neoclassical Canons (Farkas 1994, Baudouin 2004...). No training needed.
   Always available; used as fallback when ML model is absent.

2. MLScorer         — Data-driven. Loads face_rater_v1.pt (MLP trained on
   SCUT-FBP5500, 5500 human-rated faces). Requires running the full training
   pipeline first (extract_features.py → train_mlp.py).
   This is the PRIMARY scorer once the model is trained.

3. AnchorScorer     — Metric-learning. Computes cosine similarity between
   the query face and representative cluster medoids for each attractiveness
   tier extracted from SCUT-FBP5500. Requires train_anchors.py output.
   Provides a complementary "which tier am I closest to?" interpretation.

Usage in main.py:
    - Try MLScorer first (best accuracy, data-driven).
    - Try AnchorScorer for tier interpretation.
    - Always compute GeometricScorer (objective baseline, interpretable).
"""

import os
import json
import pickle
import numpy as np

import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1. GeometricScorer (rule-based, always available)
# ─────────────────────────────────────────────────────────────────────────────

class GeometricScorer:
    """
    Rule-based scorer: deducts points for deviations from aesthetic canons.
    Total weight budget = 100 pts. Score ∈ [0.0, 10.0].

    Weight allocation (sums to 100):
        Symmetry            18 — Baudouin & Tiberghien (2004)
        Canthal Tilt        15 — Kunjur et al. (2006)
        Face Aspect Ratio   12 — neoclassical ovoid canon
        Facial Fifths       10 — Farkas (1994)
        Facial Thirds        8 — Farkas (1994)
        Midface Ratio        8 — Farkas (1994)
        Nose Width Ratio     7 — nasal-intercanthal norm
        IPD Ratio            7 — Dodgson (2004)
        Nose Length Ratio    5 — vertical proportion
        Eye Openness         5 — Price et al. (2011)
        Philtrum Ratio       3 — Farkas (1994)
        Lip Thickness        2 — upper:lower ratio
        TOTAL              100
    """

    FEATURES = {
        "symmetry":          (18, 0.92,  1.00),
        "canthal_tilt":      (15, 3.0,   9.0),
        "face_aspect_ratio": (12, 1.25,  1.55),
        "facial_fifths":     (10, 0.18,  0.22),
        "facial_thirds":     (8,  0.90,  1.10),
        "midface_ratio":     (8,  0.41,  0.45),
        "nose_width":        (7,  0.90,  1.10),
        "ipd_ratio":         (7,  0.44,  0.48),
        "nose_length":       (5,  0.53,  0.62),
        "eye_openness":      (5,  0.27,  0.36),
        "philtrum_ratio":    (3,  0.28,  0.38),
        "lip_thickness":     (2,  0.50,  0.75),
    }

    def __init__(self):
        total_w = sum(v[0] for v in self.FEATURES.values())
        assert total_w == 100, f"Weights must sum to 100, got {total_w}"

    @staticmethod
    def _penalty(value, ideal_min, ideal_max, weight):
        if value < ideal_min:
            dev = ideal_min - value
        elif value > ideal_max:
            dev = value - ideal_max
        else:
            return 0.0
        span = max(ideal_max - ideal_min, 1e-6)
        return weight * min(dev / span, 1.0)

    @staticmethod
    def _get(d, *keys, default=0.0):
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k)
            if d is None:
                return default
        return float(d) if isinstance(d, (int, float)) else default

    def score(self, metrics: dict) -> dict:
        g, p, F = self._get, self._penalty, self.FEATURES
        penalties = {}

        penalties["symmetry"]          = p(g(metrics, "landmark_symmetry_score", default=1.0),
                                           *F["symmetry"][1:], F["symmetry"][0])
        tilt = g(metrics, "canthal_tilt", "mean_deg", default=5.0)
        w, lo, hi = F["canthal_tilt"]
        penalties["canthal_tilt"]      = (w * min(abs(tilt) / 6.0, 1.0)
                                          if tilt < 0 else p(tilt, lo, hi, w))
        penalties["face_aspect_ratio"] = p(g(metrics, "face_aspect_ratio", "value", default=1.4),
                                           *F["face_aspect_ratio"][1:], F["face_aspect_ratio"][0])
        fifths = g(metrics, "facial_fifths", "segments", default=[])
        w_f, lo_f, hi_f = F["facial_fifths"]
        penalties["facial_fifths"]     = (sum(p(s.get("ratio", 0.2), lo_f, hi_f, w_f / 5)
                                             for s in fifths)
                                          if isinstance(fifths, list) and len(fifths) == 5
                                          else 0.0)
        penalties["facial_thirds"]     = p(g(metrics, "facial_thirds", "mid_to_lower", default=1.0),
                                           *F["facial_thirds"][1:], F["facial_thirds"][0])
        penalties["midface_ratio"]     = p(g(metrics, "midface_ratio", "midface_ratio", default=0.43),
                                           *F["midface_ratio"][1:], F["midface_ratio"][0])
        penalties["nose_width"]        = p(g(metrics, "nose_width_ratio", "nose_to_intercanthal_ratio", default=1.0),
                                           *F["nose_width"][1:], F["nose_width"][0])
        penalties["ipd_ratio"]         = p(g(metrics, "interpupillary_distance", "ipd_to_bizygomatic", default=0.46),
                                           *F["ipd_ratio"][1:], F["ipd_ratio"][0])
        penalties["nose_length"]       = p(g(metrics, "nose_length_ratio", "nose_to_midface_ratio", default=0.57),
                                           *F["nose_length"][1:], F["nose_length"][0])
        penalties["eye_openness"]      = p(g(metrics, "eye_openness", "mean_openness_ratio", default=0.31),
                                           *F["eye_openness"][1:], F["eye_openness"][0])
        penalties["philtrum_ratio"]    = p(g(metrics, "philtrum_ratio", "philtrum_ratio", default=0.33),
                                           *F["philtrum_ratio"][1:], F["philtrum_ratio"][0])
        penalties["lip_thickness"]     = p(g(metrics, "lip_thickness_ratio", "upper_to_lower_ratio", default=0.625),
                                           *F["lip_thickness"][1:], F["lip_thickness"][0])

        total_penalty = min(sum(penalties.values()), 100.0)
        score_10 = round((100.0 - total_penalty) / 10.0, 2)
        breakdown = dict(sorted(
            {k: round(v, 2) for k, v in penalties.items() if v > 0.01}.items(),
            key=lambda x: x[1], reverse=True
        ))
        return {
            "score_out_of_10":     score_10,
            "total_penalty_pts":   round(total_penalty, 2),
            "penalties_breakdown": breakdown,
            "method": "Rule-based (12 features, weights=100). "
                      "Refs: Farkas1994, Baudouin2004, Carre2008, Price2011.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. MLScorer (data-driven, requires trained model)
# ─────────────────────────────────────────────────────────────────────────────

class MLScorer:
    """
    MLP regressor trained on SCUT-FBP5500 (5500 human-rated faces).
    Maps a 427-d fused feature vector (43 geo + 384 DINOv2 CLS) to a score
    in [0, 10] that reflects human aesthetic ratings.

    Requires:
        weights/face_rater_v1.pt   — trained model weights
        weights/scaler.pkl         — StandardScaler fitted on training set
        weights/model_meta.json    — metadata (input_dim, val_pearson, etc.)

    Raises FileNotFoundError if any of the above are missing.
    Run the full training pipeline first:
        python src/scripts/extract_features.py
        python src/training/train_mlp.py
    """

    def __init__(self,
                 model_path:  str = "weights/face_rater_v1.pt",
                 scaler_path: str = "weights/scaler.pkl",
                 meta_path:   str = "weights/model_meta.json"):

        for path in (model_path, scaler_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"ML model artefact not found: {path}\n"
                    "Run: python src/scripts/extract_features.py\n"
                    "     python src/training/train_mlp.py"
                )

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        input_dim = 427
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
            input_dim = self.meta.get("input_dim", 427)

        # Import here to avoid circular imports at module load time
        from models.face_rater import FaceRatingMLP
        self.model = FaceRatingMLP(input_dim=input_dim)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        self.input_dim = input_dim

    def score(self, fused_vector: np.ndarray) -> dict:
        """
        Args:
            fused_vector: 1-D numpy array of shape (input_dim,)
                          from fuse_features()["vector"]
        Returns:
            dict with score_out_of_10, raw_output, method, model_metrics
        """
        if fused_vector.shape[0] != self.input_dim:
            raise ValueError(
                f"Vector dim mismatch: got {fused_vector.shape[0]}, "
                f"expected {self.input_dim}. Did you change the feature set?"
            )

        vec = self.scaler.transform(fused_vector.reshape(1, -1))[0]
        t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw = float(self.model(t).item())

        score = round(float(np.clip(raw, 0.0, 10.0)), 2)
        return {
            "score_out_of_10": score,
            "raw_output":      round(raw, 4),
            "method":          "MLP regressor (SCUT-FBP5500, 5500 human-rated faces). "
                               "Input: 427-d (43 geo + 384 DINOv2 CLS).",
            "model_metrics": {
                "val_pearson":  self.meta.get("val_pearson"),
                "test_pearson": self.meta.get("test_pearson"),
                "test_mae":     self.meta.get("test_mae"),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. AnchorScorer (metric-learning, requires trained anchors)
# ─────────────────────────────────────────────────────────────────────────────

class AnchorScorer:
    """
    Anchor-based scorer using cosine similarity to tier medoids.

    Each "anchor" is a real face from SCUT-FBP5500 that is representative
    of its attractiveness tier (elite / above-avg / average / below-avg).
    Anchors are computed by train_anchors.py using K-Means + medoid selection
    on the training split only (no test leakage).

    Scoring logic:
        1. Standardise the query vector with the training scaler.
        2. For each tier, compute mean cosine similarity to that tier's anchors.
        3. Apply softmax over the 4 tier similarities → probability weights.
        4. Score = weighted sum of tier midpoints (calibrated to SCUT distribution).

    Requires:
        weights/anchors.json  — anchor vectors from train_anchors.py
        weights/scaler.pkl    — same StandardScaler used in training

    Raises FileNotFoundError if weights/anchors.json is missing.
    Run: python src/training/train_anchors.py
    """

    # Tier midpoints on [0, 10] scale — calibrated to SCUT quantile tiers
    TIER_MIDPOINTS = {
        "tier_1_elite":         9.0,
        "tier_2_above_average": 7.5,
        "tier_3_average":       5.5,
        "tier_4_below_average": 3.0,
    }

    def __init__(self,
                 anchors_path: str = "weights/anchors.json",
                 scaler_path:  str = "weights/scaler.pkl"):

        if not os.path.exists(anchors_path):
            raise FileNotFoundError(
                f"Anchors not found: {anchors_path}\n"
                "Run: python src/training/train_anchors.py"
            )

        with open(anchors_path) as f:
            raw = json.load(f)

        # Build tier_name -> (N_anchors, D) numpy arrays (unscaled)
        self.anchors: dict[str, np.ndarray] = {}
        for tier, records in raw.items():
            vecs = np.array([r["feature_vector"] for r in records], dtype=np.float32)
            self.anchors[tier] = vecs
            # Also use stored scores to calibrate midpoints if available
            scores = [r.get("score_0_10") for r in records if r.get("score_0_10")]
            if scores:
                self.TIER_MIDPOINTS[tier] = float(np.mean(scores))

        self.scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Mean cosine similarity between vector a and rows of matrix b."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return float(np.mean(b_norm @ a_norm))

    def score(self, fused_vector: np.ndarray) -> dict:
        """
        Args:
            fused_vector: 1-D numpy array from fuse_features()["vector"]
        Returns:
            dict with score_out_of_10, tier_similarities, closest_tier, method
        """
        vec = fused_vector.copy()
        if self.scaler is not None:
            vec = self.scaler.transform(vec.reshape(1, -1))[0]
            anchor_vecs = {
                t: self.scaler.transform(a) for t, a in self.anchors.items()
            }
        else:
            anchor_vecs = self.anchors

        # Cosine similarity to each tier
        sims = {}
        for tier, anc in anchor_vecs.items():
            if anc.shape[0] > 0:
                sims[tier] = self._cosine_sim(vec, anc)

        # Softmax over similarities → weights
        sim_vals = np.array(list(sims.values()), dtype=np.float64)
        exp_sims = np.exp((sim_vals - sim_vals.max()) * 5.0)  # temperature=5
        weights  = exp_sims / exp_sims.sum()

        # Weighted score
        weighted_score = sum(
            w * self.TIER_MIDPOINTS.get(tier, 5.0)
            for w, tier in zip(weights, sims.keys())
        )
        score = round(float(np.clip(weighted_score, 0.0, 10.0)), 2)
        closest_tier = max(sims, key=sims.get)

        return {
            "score_out_of_10":   score,
            "closest_tier":      closest_tier,
            "tier_similarities": {t: round(float(v), 4) for t, v in sims.items()},
            "method":            "Anchor-based (cosine similarity to SCUT-FBP5500 tier medoids). "
                                 "Each tier: K-Means medoid on training split only.",
        }
