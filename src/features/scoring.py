"""
scoring.py — Geometric Scoring with normalized weights.

Total weight budget = 100 points. Penalties are normalized so the sum
of all maximum per-feature penalties equals exactly 100.
This guarantees that score ∈ [0.0, 10.0] is always meaningful.

Penalty formula per feature:
    penalty = weight × clamp(deviation / range_span, 0, 1)
where:
    deviation = max(0, value - ideal_max)  or  max(0, ideal_min - value)
    range_span = ideal_max - ideal_min
    weight = max contribution this feature can deduct from 100 pts

Scientific references embedded in ideal ranges:
    - Farkas (1994): Anthropometry of the Head and Face
    - Baudouin & Tiberghien (2004): Symmetry and facial attractiveness
    - Carré et al. (2008): fWHR and social perception
    - Price et al. (2011): Palpebral fissure (eye openness) norms
    - Dodgson (2004): Interpupillary distance norms
"""


class GeometricScorer:
    """
    Rule-based scorer evaluating facial aesthetics via deviation from
    established Neoclassical Canons. Returns a score [0.0, 10.0].

    Weight allocation (sums to 100):
        Symmetry            18 — most perceptible asymmetry impact
        Canthal Tilt        15 — strongest predictor of attractiveness
        Face Aspect Ratio   12 — overall shape / ovoid ideal
        Facial Fifths       10 — lateral eye spacing balance
        Facial Thirds        8 — vertical proportion balance
        Midface Ratio        8 — hunter-eye / midface dominance
        Nose Width Ratio     7 — nose to inter-canthal proportion
        IPD Ratio            7 — inter-pupillary spacing
        Nose Length Ratio    5 — vertical nose proportion
        Eye Openness         5 — palpebral fissure openness
        Philtrum Ratio       3 — upper lip to philtrum length
        Lip Thickness        2 — upper:lower lip ratio
        TOTAL              100
    """

    # Feature registry: { key: (weight, ideal_min, ideal_max) }
    # Each weight = max points that feature can deduct.
    FEATURES = {
        "symmetry":          (18, 0.92,  1.00),   # Baudouin 2004: >0.92 imperceptible
        "canthal_tilt":      (15, 3.0,   9.0),    # degrees, slightly positive = attractive
        "face_aspect_ratio": (12, 1.25,  1.55),   # ovoid canonical face
        "facial_fifths":     (10, 0.18,  0.22),   # each of 5 segments ≈ 0.20
        "facial_thirds":     (8,  0.90,  1.10),   # mid/lower ratio ≈ 1.0
        "midface_ratio":     (8,  0.41,  0.45),   # Farkas 1994
        "nose_width":        (7,  0.90,  1.10),   # nose-width / ICD ratio
        "ipd_ratio":         (7,  0.44,  0.48),   # IPD / bizygomatic, Dodgson 2004
        "nose_length":       (5,  0.53,  0.62),   # nose / midface height
        "eye_openness":      (5,  0.27,  0.36),   # Price 2011: palpebral fissure ratio
        "philtrum_ratio":    (3,  0.28,  0.38),   # Farkas 1994 philtrum/lower-face
        "lip_thickness":     (2,  0.50,  0.75),   # upper:lower lip thickness ratio
    }

    def __init__(self):
        # Validate weights sum to 100
        total_w = sum(v[0] for v in self.FEATURES.values())
        assert total_w == 100, f"Weights must sum to 100, got {total_w}"

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _penalty(value, ideal_min, ideal_max, weight):
        """
        Soft-clamp penalty. Returns 0 when value is inside [ideal_min, ideal_max].
        Returns exactly `weight` when deviation equals the range span.
        Capped at `weight` (max 100% deduction per feature).
        """
        if value < ideal_min:
            dev = ideal_min - value
        elif value > ideal_max:
            dev = value - ideal_max
        else:
            return 0.0
        span = max(ideal_max - ideal_min, 1e-6)
        # clamp at 1.0 so max penalty = weight (no overcounting)
        return weight * min(dev / span, 1.0)

    @staticmethod
    def _get(d, *keys, default=0.0):
        """Safe nested dict accessor. Returns `default` only when key is absent."""
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k)
            if d is None:
                return default
        # d is the final value — return it if it's a real number
        if isinstance(d, (int, float)):
            return float(d)
        return default

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def score(self, metrics: dict) -> dict:
        """
        Compute overall geometric score from metrics dict.

        Args:
            metrics: output of compute_all_metrics()

        Returns:
            dict with keys:
                score_out_of_10    : float [0, 10]
                total_penalty_pts  : float [0, 100]
                penalties_breakdown: { feature: pts_lost }
                method             : description string
        """
        g = self._get
        p = self._penalty
        F = self.FEATURES
        penalties = {}

        # 1. Symmetry — use as 0..1 score, invert to penalty
        sym = g(metrics, "landmark_symmetry_score", default=1.0)
        w, lo, hi = F["symmetry"]
        # Symmetry 1.0 = perfect. Penalty scales from 0 at 1.0 to full weight at lo.
        sym_penalty = p(sym, lo, hi, w)
        penalties["symmetry"] = sym_penalty

        # 2. Canthal Tilt — negative tilt is more severe than positive overshoot
        tilt = g(metrics, "canthal_tilt", "mean_deg", default=5.0)
        w, lo, hi = F["canthal_tilt"]
        if tilt < 0:
            # Negative tilt: linear penalty, more severe. Max out at -6°.
            severity = min(abs(tilt) / 6.0, 1.0)
            penalties["canthal_tilt"] = w * severity
        else:
            penalties["canthal_tilt"] = p(tilt, lo, hi, w)

        # 3. Face Aspect Ratio
        far = g(metrics, "face_aspect_ratio", "value", default=1.4)
        w, lo, hi = F["face_aspect_ratio"]
        penalties["face_aspect_ratio"] = p(far, lo, hi, w)

        # 4. Facial Fifths — 5 equal segments, each penalised individually
        #    Total weight = 10, so each segment has weight 2.
        fifths = g(metrics, "facial_fifths", "segments", default=[])
        w_total, lo, hi = F["facial_fifths"]
        if isinstance(fifths, list) and len(fifths) == 5:
            per_seg_w = w_total / 5
            penalties["facial_fifths"] = sum(
                p(seg.get("ratio", 0.2), lo, hi, per_seg_w) for seg in fifths
            )
        else:
            penalties["facial_fifths"] = 0.0

        # 5. Facial Thirds — mid/lower ratio
        mid_to_lower = g(metrics, "facial_thirds", "mid_to_lower", default=1.0)
        w, lo, hi = F["facial_thirds"]
        penalties["facial_thirds"] = p(mid_to_lower, lo, hi, w)

        # 6. Midface Ratio
        mfr = g(metrics, "midface_ratio", "midface_ratio", default=0.43)
        w, lo, hi = F["midface_ratio"]
        penalties["midface_ratio"] = p(mfr, lo, hi, w)

        # 7. Nose Width (nose / intercanthal)
        nwr = g(metrics, "nose_width_ratio", "nose_to_intercanthal_ratio", default=1.0)
        w, lo, hi = F["nose_width"]
        penalties["nose_width"] = p(nwr, lo, hi, w)

        # 8. IPD Ratio (IPD / bizygomatic)
        ipd = g(metrics, "interpupillary_distance", "ipd_to_bizygomatic", default=0.46)
        w, lo, hi = F["ipd_ratio"]
        penalties["ipd_ratio"] = p(ipd, lo, hi, w)

        # 9. Nose Length (nose / midface)
        nlr = g(metrics, "nose_length_ratio", "nose_to_midface_ratio", default=0.57)
        w, lo, hi = F["nose_length"]
        penalties["nose_length"] = p(nlr, lo, hi, w)

        # 10. Eye Openness
        eyo = g(metrics, "eye_openness", "mean_openness_ratio", default=0.31)
        w, lo, hi = F["eye_openness"]
        penalties["eye_openness"] = p(eyo, lo, hi, w)

        # 11. Philtrum Ratio
        phil = g(metrics, "philtrum_ratio", "philtrum_ratio", default=0.33)
        w, lo, hi = F["philtrum_ratio"]
        penalties["philtrum_ratio"] = p(phil, lo, hi, w)

        # 12. Lip Thickness
        lip = g(metrics, "lip_thickness_ratio", "upper_to_lower_ratio", default=0.625)
        w, lo, hi = F["lip_thickness"]
        penalties["lip_thickness"] = p(lip, lo, hi, w)

        # ── Final score ──────────────────────────────────────────────────────
        total_penalty = min(sum(penalties.values()), 100.0)   # hard cap at 100
        score_10 = round((100.0 - total_penalty) / 10.0, 2)

        # Only report penalties that actually lost points
        breakdown = {k: round(v, 2) for k, v in penalties.items() if v > 0.01}
        # Sort by severity descending for readability
        breakdown = dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))

        return {
            "score_out_of_10":    score_10,
            "total_penalty_pts":  round(total_penalty, 2),
            "penalties_breakdown": breakdown,
            "method": (
                "Rule-based geometric scoring: 12 features, "
                "normalized weights summing to 100pts. "
                "Refs: Farkas 1994, Baudouin 2004, Carré 2008, Price 2011."
            ),
        }
