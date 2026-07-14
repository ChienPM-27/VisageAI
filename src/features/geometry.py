import numpy as np

# ============================================================
# LANDMARK INDEX REFERENCE (MediaPipe 468-point Face Mesh)
# ============================================================
# fWHR Width:  234 (right zygomatic arch / bizygomatic) ↔ 454 (left zygomatic arch / bizygomatic)
#              NOTE: These are the closest MediaPipe approximations to the
#              anatomical zygion (outermost point of the zygomatic arch).
#              See Lefevre et al. (2012) for fWHR definition.
# fWHR Height: 9   (glabella — point between eyebrows) ↔ 0 (upper lip — stomion approx)
#              Matches Carré et al. (2008) mid-face height definition.
#
# Jaw Contour (17 points, bilateral):
#   Right (image-right, person's left) : 132,58,172,136,150,149,176,148,152
#   Left  (image-left,  person's right): 152,377,400,378,379,365,397,288,361
# ============================================================

# Jaw contour landmark indices (image right → chin → image left)
JAW_CONTOUR_RIGHT = [132, 58, 172, 136, 150, 149, 176, 148, 152]
JAW_CONTOUR_LEFT  = [152, 377, 400, 378, 379, 365, 397, 288, 361]


def calculate_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def compute_canthal_tilt(landmarks_pixel):
    """
    Canthal tilt: angle of the inter-canthal axis relative to horizontal.

    Reference: Kunjur et al. (2006) — facial aesthetic units.
    Landmarks:
      Image-left eye (person's right): outer = 33, inner = 133
      Image-right eye (person's left): inner = 362, outer = 263
    Positive tilt = outer canthus higher than inner canthus (upswept).
    """
    dx_left = landmarks_pixel[133][0] - landmarks_pixel[33][0]
    dy_left = landmarks_pixel[133][1] - landmarks_pixel[33][1]
    left_tilt = np.degrees(np.arctan2(dy_left, dx_left))

    dx_right = landmarks_pixel[263][0] - landmarks_pixel[362][0]
    dy_right = landmarks_pixel[362][1] - landmarks_pixel[263][1]
    right_tilt = np.degrees(np.arctan2(dy_right, dx_right))

    return left_tilt, right_tilt


def compute_fwhr(landmarks_pixel):
    """
    Facial Width-to-Height Ratio (fWHR).

    Reference: Carré et al. (2008), Lefevre et al. (2012).
    Width  : 234 ↔ 454  (bizygomatic — zygomatic arch approximation in MediaPipe)
    Height : 9   ↔ 0    (glabella to upper lip / upper stomion)
    """
    width  = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])
    height = calculate_distance(landmarks_pixel[9][:2],   landmarks_pixel[0][:2])
    if height == 0:
        return {"value": 0.0, "width_px": 0.0, "height_px": 0.0,
                "width_landmarks": "234-454 (bizygomatic approx)",
                "height_landmarks": "9-0 (glabella to upper lip)"}
    return {
        "value":            float(width / height),
        "width_px":         float(width),
        "height_px":        float(height),
        "width_landmarks":  "234-454 (bizygomatic / zygomatic arch approx)",
        "height_landmarks": "9-0 (glabella to upper lip / stomion approx)"
    }


def compute_facial_thirds(landmarks_pixel):
    """
    Mid-face and lower-face proportion (two-segment version).

    NOTE: MediaPipe does NOT include a hairline/trichion landmark — it only
    tracks skin surface.  Using landmark 10 as a trichion proxy systematically
    under-estimates the forehead and produces a biased upper-third value.
    This function therefore omits the upper-third and reports only:
      Middle (mid-face): glabella (9) → subnasale (2)
      Lower  (lower-face): subnasale (2) → menton (152)

    Ideal mid:lower ratio is approximately 1:1 (Golden Canon).
    Reference: Farkas (1994) Anthropometry of the Head and Face.
    """
    mid_dist   = calculate_distance(landmarks_pixel[9][:2],   landmarks_pixel[2][:2])
    lower_dist = calculate_distance(landmarks_pixel[2][:2],   landmarks_pixel[152][:2])
    total = mid_dist + lower_dist
    if total == 0:
        return {"mid_ratio": 0.0, "lower_ratio": 0.0, "mid_to_lower": 0.0,
                "note": "Hairline/trichion excluded — MediaPipe has no hairline landmark"}
    return {
        "mid_ratio":    float(mid_dist / total),
        "lower_ratio":  float(lower_dist / total),
        "mid_to_lower": float(mid_dist / lower_dist) if lower_dist > 0 else 0.0,
        "ideal_mid_to_lower": 1.0,
        "note": "Hairline/trichion excluded — MediaPipe has no hairline landmark"
    }


def compute_facial_fifths(landmarks_pixel):
    """
    Facial fifths: the face is divided into 5 vertical columns of equal width.

    Reference: Powell & Humphreys (1984) Proportions of the Aesthetic Face.
    Landmarks (left-to-right in image coordinates):
      Ear-right (image) : 234
      Outer canthus R   : 33
      Inner canthus R   : 133
      Inner canthus L   : 362
      Outer canthus L   : 263
      Ear-left  (image) : 454
    Ideal: all 5 segments are equal (~0.20 each).
    """
    pts = [
        landmarks_pixel[234][0],  # right ear / face edge
        landmarks_pixel[33][0],   # outer canthus image-right eye
        landmarks_pixel[133][0],  # inner canthus image-right eye
        landmarks_pixel[362][0],  # inner canthus image-left eye
        landmarks_pixel[263][0],  # outer canthus image-left eye
        landmarks_pixel[454][0],  # left ear / face edge
    ]
    total_width = pts[5] - pts[0]
    if total_width <= 0:
        return {"segments": [], "note": "Could not compute facial fifths"}

    segments = []
    labels = ["Ear→OC_R", "OC_R→IC_R", "IC_R→IC_L", "IC_L→OC_L", "OC_L→Ear"]
    for i in range(5):
        seg_w   = pts[i + 1] - pts[i]
        seg_rat = float(seg_w / total_width)
        segments.append({"label": labels[i], "ratio": round(seg_rat, 4)})

    # Symmetry: compare left-side fifth pairs
    outer_diff = abs(segments[0]["ratio"] - segments[4]["ratio"])
    inner_diff = abs(segments[1]["ratio"] - segments[3]["ratio"])
    return {
        "segments": segments,
        "ideal_each": 0.20,
        "outer_pair_diff": round(float(outer_diff), 4),
        "inner_pair_diff": round(float(inner_diff), 4)
    }


def compute_facial_ratios(landmarks_pixel):
    """
    Key facial proportion ratios.

    Eye width     : distance between outer and inner canthus (per eye), averaged.
    Nose width    : landmark 129 (right alar base) ↔ 358 (left alar base).
    Mouth width   : landmark 61  (right mouth corner) ↔ 291 (left mouth corner).
    Ideal ratios  : nose_width ≈ eye_width (intercanthal distance rule).
                    mouth_width ≈ 1.5 × eye_width (neoclassical canon).
    """
    # Eye widths (outer−inner canthus per eye)
    left_eye_w  = calculate_distance(landmarks_pixel[33][:2],  landmarks_pixel[133][:2])
    right_eye_w = calculate_distance(landmarks_pixel[263][:2], landmarks_pixel[362][:2])
    mean_eye_w  = (left_eye_w + right_eye_w) / 2.0

    # Intercanthal distance (inner-inner)
    intercanthal = calculate_distance(landmarks_pixel[133][:2], landmarks_pixel[362][:2])

    # Nose width (alar base)
    nose_w  = calculate_distance(landmarks_pixel[129][:2], landmarks_pixel[358][:2])

    # Mouth width
    mouth_w = calculate_distance(landmarks_pixel[61][:2], landmarks_pixel[291][:2])

    safe = lambda num, den: float(num / den) if den > 0 else 0.0

    return {
        "eye_width_px":          {"left": float(left_eye_w), "right": float(right_eye_w), "mean": float(mean_eye_w)},
        "intercanthal_px":       float(intercanthal),
        "nose_width_px":         float(nose_w),
        "mouth_width_px":        float(mouth_w),
        "nose_to_eye_ratio":     safe(nose_w,  mean_eye_w),
        "mouth_to_eye_ratio":    safe(mouth_w, mean_eye_w),
        "nose_to_intercanthal":  safe(nose_w,  intercanthal),
        "ideal_nose_to_eye":     1.0,
        "ideal_mouth_to_eye":    1.5,
    }


def compute_eye_spacing(landmarks_pixel):
    """
    Intercanthal distance / bizygomatic width ratio.
    Ideal ≈ 0.46 (Farkas 1994).
    """
    intercanthal = calculate_distance(landmarks_pixel[133][:2], landmarks_pixel[362][:2])
    face_width   = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])
    return float(intercanthal / face_width) if face_width > 0 else 0.0


def fit_jaw_contour(landmarks_pixel):
    """
    Jaw angle via polyline fitting on the full jaw contour.

    Instead of using only 3 points (error-prone), we fit a degree-1 polynomial
    (linear regression) independently on the right and left jaw halves using all
    available contour points.  The slope of that line gives the jaw angle relative
    to horizontal, which is more robust to individual landmark noise.

    Right half (image): JAW_CONTOUR_RIGHT (chin excluded)
    Left  half (image): JAW_CONTOUR_LEFT  (chin excluded)

    Returns angles in degrees. Positive right angle = jaw slopes downward left.
    """
    chin = landmarks_pixel[152][:2]

    # Right jaw half (image): points from right face edge down to chin
    right_pts = np.array([landmarks_pixel[i][:2] for i in JAW_CONTOUR_RIGHT[:-1]])  # exclude chin
    left_pts  = np.array([landmarks_pixel[i][:2] for i in JAW_CONTOUR_LEFT[1:]])    # exclude chin

    def fit_angle(pts):
        if len(pts) < 2:
            return 0.0
        xs, ys = pts[:, 0], pts[:, 1]
        if xs.max() - xs.min() < 1:
            return 90.0
        slope = float(np.polyfit(xs, ys, deg=1)[0])
        return float(np.degrees(np.arctan(slope)))

    right_angle = fit_angle(right_pts)
    left_angle  = fit_angle(left_pts)

    jaw_width    = calculate_distance(landmarks_pixel[172][:2], landmarks_pixel[397][:2])
    face_width   = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])
    jaw_to_face  = float(jaw_width / face_width) if face_width > 0 else 0.0

    return {
        "right_angle_deg":    round(right_angle, 2),
        "left_angle_deg":     round(left_angle, 2),
        "mean_angle_deg":     round((right_angle + left_angle) / 2.0, 2),
        "jaw_to_face_ratio":  round(jaw_to_face, 4),
        "method":             "polyfit-deg1 on jaw contour (17 pts)"
    }


def compute_landmark_symmetry(landmarks_pixel):
    """
    Geometric landmark symmetry score [0, 1].

    Computes the mean relative asymmetry of bilateral landmark pairs about the
    vertical midline (estimated from landmarks 10–152).

    NOTE: This score reflects ONLY the geometric symmetry of detected 2-D
    landmarks.  It does NOT account for texture, lighting, expression, or
    out-of-plane pose asymmetry.  Always report as 'landmark_symmetry_score'.

    Pairs: outer eyes (33,263), inner eyes (133,362), cheeks (234,454),
           jaw corners (172,397), mouth corners (61,291).
    """
    top    = landmarks_pixel[10][:2]
    bottom = landmarks_pixel[152][:2]
    line_vec = bottom - top
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0.0
    line_unit = line_vec / line_len

    pairs = [(33, 263), (133, 362), (234, 454), (172, 397), (61, 291)]
    diffs = []
    for p1, p2 in pairs:
        v1 = landmarks_pixel[p1][:2] - top
        v2 = landmarks_pixel[p2][:2] - top
        d1 = abs(float(np.cross(line_unit, v1)))
        d2 = abs(float(np.cross(line_unit, v2)))
        avg = (d1 + d2) / 2.0
        if avg > 0:
            diffs.append(abs(d1 - d2) / avg)

    if not diffs:
        return 0.0
    return float(max(0.0, 1.0 - np.mean(diffs) * 5.0))


# ==============================================================
# NEW FEATURES (v0.3)
# ==============================================================

def compute_jaw_width(landmarks_pixel):
    """
    Jaw width: distance between gonion points (jaw corners).
    Landmarks: 172 (right gonion, image-right) <-> 397 (left gonion, image-left).
    Reference: Farkas (1994) Anthropometry of the Head and Face.
    Ideal: jaw_width / bizygomatic_width ≈ 0.70-0.80 for balanced faces.
    """
    jaw_w  = calculate_distance(landmarks_pixel[172][:2], landmarks_pixel[397][:2])
    biz_w  = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])
    face_h = calculate_distance(landmarks_pixel[9][:2],   landmarks_pixel[152][:2])
    return {
        "jaw_width_px":          round(float(jaw_w), 2),
        "bizygomatic_width_px":  round(float(biz_w), 2),
        "jaw_to_bizygomatic":    round(float(jaw_w / biz_w)  if biz_w  > 0 else 0, 4),
        "jaw_to_face_height":    round(float(jaw_w / face_h) if face_h > 0 else 0, 4),
        "landmarks":             "172 (right gonion) <-> 397 (left gonion)"
    }


def compute_chin_angle(landmarks_pixel):
    """
    Chin angle: interior angle at the menton (chin tip) formed by the two
    gonion-to-menton vectors.  Smaller angle = sharper / more pointed chin.

    Landmarks: 172 (right gonion) — 152 (menton) — 397 (left gonion).
    Typical values: 80-120° (wider = rounder chin, narrower = V-shaped chin).
    Reference: Naini et al. (2012) Facial Aesthetics: Concepts and Clinical Diagnosis.
    """
    chin        = landmarks_pixel[152][:2]
    right_jaw   = landmarks_pixel[172][:2]
    left_jaw    = landmarks_pixel[397][:2]

    v1 = right_jaw - chin
    v2 = left_jaw  - chin
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    return {
        "chin_angle_deg": round(angle_deg, 2),
        "landmarks":      "172 (gonion-R) — 152 (menton) — 397 (gonion-L)",
        "note":           "< 100deg = pointed/V-chin, > 120deg = round/square chin"
    }


def compute_face_aspect_ratio(landmarks_pixel):
    """
    Face Aspect Ratio (FAR): face height / face width.
    Height: nasion (168, nasal bridge) to menton (152, chin bottom).
    Width:  bizygomatic (234 <-> 454).

    FAR > 1 = longer face (oval/oblong), FAR < 1 = wider face (round/square).
    Typical range: 1.2 – 1.6 for oval faces (often considered aesthetically ideal).
    Reference: Baudouin & Tiberghien (2004) Symmetry, averageness, and feature size
                in the facial attractiveness of women.
    """
    height = calculate_distance(landmarks_pixel[168][:2], landmarks_pixel[152][:2])
    width  = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])
    return {
        "value":              round(float(height / width) if width > 0 else 0, 4),
        "face_height_px":     round(float(height), 2),
        "face_width_px":      round(float(width),  2),
        "height_landmarks":   "168 (nasion) to 152 (menton)",
        "width_landmarks":    "234 (bizyg-R) to 454 (bizyg-L)",
        "ideal_range":        "1.2 – 1.6 (oval face)"
    }


def compute_midface_ratio(landmarks_pixel):
    """
    Midface ratio: midface height as a proportion of total face height.
    Midface: nasion (168) to subnasale (2).
    Total face: nasion (168) to menton (152).

    Ideal midface ratio ≈ 0.43 (Farkas canon).
    Low midface = face appears bottom-heavy; high midface = upper dominance.
    Reference: Farkas (1994); Ioi et al. (2012) IJOMS.
    """
    midface_h = calculate_distance(landmarks_pixel[168][:2], landmarks_pixel[2][:2])
    total_h   = calculate_distance(landmarks_pixel[168][:2], landmarks_pixel[152][:2])
    return {
        "midface_ratio":     round(float(midface_h / total_h) if total_h > 0 else 0, 4),
        "midface_height_px": round(float(midface_h), 2),
        "total_height_px":   round(float(total_h),   2),
        "landmarks":         "168 (nasion) -> 2 (subnasale) / 152 (menton)",
        "ideal":             "~0.43 (Farkas canon)"
    }


def compute_philtrum_ratio(landmarks_pixel):
    """
    Philtrum ratio: philtrum length relative to lower-face height.
    Philtrum: subnasale (2) to upper lip (0).
    Lower face: subnasale (2) to menton (152).

    Ideal philtrum ratio ≈ 0.33 (Farkas), meaning the philtrum is ~1/3 of
    the lower face.  Short philtrum (<0.25) or long philtrum (>0.45) are
    often flagged in aesthetic analysis.
    Reference: Farkas (1994); Heidekrueger et al. (2017) JPRAS.
    """
    philtrum_len = calculate_distance(landmarks_pixel[2][:2],   landmarks_pixel[0][:2])
    lower_face_h = calculate_distance(landmarks_pixel[2][:2],   landmarks_pixel[152][:2])
    return {
        "philtrum_ratio":    round(float(philtrum_len / lower_face_h) if lower_face_h > 0 else 0, 4),
        "philtrum_px":       round(float(philtrum_len), 2),
        "lower_face_px":     round(float(lower_face_h), 2),
        "landmarks":         "2 (subnasale) -> 0 (upper lip) / 152 (menton)",
        "ideal":             "~0.33 (Farkas canon)"
    }


def compute_lip_thickness_ratio(landmarks_pixel):
    """
    Lip thickness ratio: upper lip height / lower lip height.
    Upper lip height: top of upper lip (0) to top of lower lip (13, inner vermillion border).
    Lower lip height: bottom of upper lip (13) to bottom of lower lip (14 -> 17).

    Approach: use vertical extents.
    Upper lip: lm 0 (cupid's bow top) to lm 13 (upper vermillion inner) — vertical distance.
    Lower lip: lm 14 (lower vermillion inner) to lm 17 (mentolabial bottom) — vertical distance.

    Ideal upper:lower ratio ≈ 1:1.6 (golden ratio analogue).
    Reference: Penna et al. (2015) Aesthetic Surgery Journal.
    """
    upper_h = calculate_distance(landmarks_pixel[0][:2],  landmarks_pixel[13][:2])
    lower_h = calculate_distance(landmarks_pixel[14][:2], landmarks_pixel[17][:2])
    return {
        "upper_lip_height_px":    round(float(upper_h), 2),
        "lower_lip_height_px":    round(float(lower_h), 2),
        "upper_to_lower_ratio":   round(float(upper_h / lower_h) if lower_h > 0 else 0, 4),
        "landmarks_upper":        "0 (top cupid bow) -> 13 (inner upper lip)",
        "landmarks_lower":        "14 (inner lower lip) -> 17 (lower lip base)",
        "ideal_upper_to_lower":   0.625,
        "note":                   "Ideal ~0.625 (1:1.6 golden ratio — lower lip fuller)"
    }


def compute_nose_width_ratio(landmarks_pixel):
    """
    Nose width ratio: alar base width relative to intercanthal distance.
    Alar base: landmark 129 (right alar) <-> 358 (left alar).
    Intercanthal distance: inner eye corners 133 <-> 362.

    Neoclassical canon: nose width ≈ intercanthal distance (ratio ≈ 1.0).
    Ratio > 1.1 = wide nose; < 0.9 = narrow nose.
    Reference: Farkas (1994); Jayaratne & Zwahlen (2014) Orthod Craniofac Res.
    """
    nose_w  = calculate_distance(landmarks_pixel[129][:2], landmarks_pixel[358][:2])
    ic_dist = calculate_distance(landmarks_pixel[133][:2], landmarks_pixel[362][:2])
    biz_w   = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])
    return {
        "nose_width_px":               round(float(nose_w),  2),
        "nose_to_intercanthal_ratio":  round(float(nose_w / ic_dist) if ic_dist > 0 else 0, 4),
        "nose_to_bizygomatic_ratio":   round(float(nose_w / biz_w)   if biz_w   > 0 else 0, 4),
        "landmarks_alar":              "129 (right alar base) <-> 358 (left alar base)",
        "ideal_nose_to_ic":            1.0,
        "note":                        "Neoclassical canon: nose width = intercanthal distance"
    }


def compute_nose_length_ratio(landmarks_pixel):
    """
    Nose length ratio: nose length relative to midface height.
    Nose length: nasion/rhinion (168, nasal bridge root) to nose tip/pronasale (4).
    Midface height: nasion (168) to subnasale (2).

    Ideal ratio: nose occupies ~55-60% of midface height.
    Reference: Farkas (1994); Rohrich & Pessa (2008) Plast Reconstr Surg.
    """
    nose_len  = calculate_distance(landmarks_pixel[168][:2], landmarks_pixel[4][:2])
    midface_h = calculate_distance(landmarks_pixel[168][:2], landmarks_pixel[2][:2])
    total_h   = calculate_distance(landmarks_pixel[168][:2], landmarks_pixel[152][:2])
    return {
        "nose_length_px":             round(float(nose_len),  2),
        "nose_to_midface_ratio":      round(float(nose_len / midface_h) if midface_h > 0 else 0, 4),
        "nose_to_total_face_ratio":   round(float(nose_len / total_h)   if total_h   > 0 else 0, 4),
        "landmarks":                  "168 (nasion) -> 4 (pronasale)",
        "ideal_nose_to_midface":      "0.55 – 0.60"
    }


def compute_intercanthal_distance(landmarks_pixel):
    """
    Intercanthal distance (ICD): distance between the two medial (inner) canthi.
    Landmarks: 133 (inner canthus, image-left eye) <-> 362 (inner canthus, image-right eye).

    Also reports ratio to bizygomatic width.
    Ideal ICD/bizygomatic ≈ 0.28-0.34 (Farkas 1994).
    Reference: Farkas (1994); Whitaker et al. (1981) Plast Reconstr Surg.
    """
    icd   = calculate_distance(landmarks_pixel[133][:2], landmarks_pixel[362][:2])
    biz_w = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])
    return {
        "icd_px":                 round(float(icd),  2),
        "icd_to_bizygomatic":     round(float(icd / biz_w) if biz_w > 0 else 0, 4),
        "landmarks":              "133 (medial canthus-R) <-> 362 (medial canthus-L)",
        "ideal_icd_to_biz":       "0.28 – 0.34 (Farkas 1994)"
    }


def compute_interpupillary_distance(landmarks_pixel):
    """
    Interpupillary distance (IPD): distance between the two pupil centres.
    Requires MediaPipe refine_landmarks=True (478-point mesh).
    Iris landmarks: 468 (right iris centre, image-left) <-> 473 (left iris centre, image-right).

    Falls back to estimating pupil as midpoint of inner+outer canthus if iris
    landmarks are unavailable (<= 467 points).

    Ideal IPD/bizygomatic ≈ 0.46 (roughly matches intercanthal + one eye width).
    Reference: Dodgson (2004) Variation and Extrema of Human Interpupillary Distance.
    """
    n_lm  = len(landmarks_pixel)
    biz_w = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])

    if n_lm >= 478:
        # Iris centre landmarks available
        right_pupil = landmarks_pixel[468][:2]
        left_pupil  = landmarks_pixel[473][:2]
        method = "iris_center (landmarks 468, 473)"
    else:
        # Fallback: midpoint of inner + outer canthus per eye
        right_pupil = (landmarks_pixel[33][:2] + landmarks_pixel[133][:2]) / 2.0
        left_pupil  = (landmarks_pixel[263][:2] + landmarks_pixel[362][:2]) / 2.0
        method = "canthus_midpoint_fallback (landmarks 33+133, 263+362)"

    ipd = calculate_distance(right_pupil, left_pupil)
    return {
        "ipd_px":             round(float(ipd),  2),
        "ipd_to_bizygomatic": round(float(ipd / biz_w) if biz_w > 0 else 0, 4),
        "method":             method,
        "ideal_ipd_to_biz":   "~0.46"
    }


def compute_eyebrow_height(landmarks_pixel):
    """
    Eyebrow height: vertical distance from eyebrow arch peak to the upper eyelid.
    Measures how high the brow sits above the eye, which influences perceived
    expressiveness and youthfulness.

    Image-left eye (person's right):
      Brow peak:  105  |  Upper eyelid top: 159
    Image-right eye (person's left):
      Brow peak:  334  |  Upper eyelid top: 386

    Normalised by eye width (outer-inner canthus distance) for scale invariance.
    Ideal brow-to-eye distance ≈ 0.4-0.6× eye width (Packiriswamy et al. 2012).
    Reference: Packiriswamy et al. (2012) Aesthetic Surgery Journal.
    """
    # Image-left eye
    brow_l  = landmarks_pixel[105][:2]
    lid_l   = landmarks_pixel[159][:2]
    eye_w_l = calculate_distance(landmarks_pixel[33][:2], landmarks_pixel[133][:2])
    h_l     = abs(float(lid_l[1] - brow_l[1]))  # vertical distance in image coords

    # Image-right eye
    brow_r  = landmarks_pixel[334][:2]
    lid_r   = landmarks_pixel[386][:2]
    eye_w_r = calculate_distance(landmarks_pixel[263][:2], landmarks_pixel[362][:2])
    h_r     = abs(float(lid_r[1] - brow_r[1]))

    mean_eye_w = (eye_w_l + eye_w_r) / 2.0
    return {
        "left_brow_height_px":    round(h_l, 2),
        "right_brow_height_px":   round(h_r, 2),
        "mean_brow_height_px":    round((h_l + h_r) / 2.0, 2),
        "left_brow_to_eye_ratio": round(h_l / eye_w_l if eye_w_l > 0 else 0, 4),
        "right_brow_to_eye_ratio":round(h_r / eye_w_r if eye_w_r > 0 else 0, 4),
        "landmarks_left":         "105 (brow peak) -> 159 (upper eyelid)",
        "landmarks_right":        "334 (brow peak) -> 386 (upper eyelid)",
        "ideal_brow_to_eye":      "0.4 – 0.6"
    }


def compute_eye_openness(landmarks_pixel):
    """
    Eye openness (palpebral fissure ratio): vertical eye aperture / horizontal eye width.
    Measures how open the eye appears — higher = more open / almond-shaped.

    Image-left eye (person's right):
      Vertical aperture: top eyelid (159) to bottom eyelid (145).
      Horizontal width:  outer canthus (33) to inner canthus (133).
    Image-right eye (person's left):
      Vertical aperture: top eyelid (386) to bottom eyelid (374).
      Horizontal width:  inner canthus (362) to outer canthus (263).

    Typical palpebral fissure ratio ≈ 0.28-0.35.
    Reference: Price et al. (2011) Ophthal Plast Reconstr Surg.
    """
    # Left eye (image-left)
    v_l   = calculate_distance(landmarks_pixel[159][:2], landmarks_pixel[145][:2])
    w_l   = calculate_distance(landmarks_pixel[33][:2],  landmarks_pixel[133][:2])

    # Right eye (image-right)
    v_r   = calculate_distance(landmarks_pixel[386][:2], landmarks_pixel[374][:2])
    w_r   = calculate_distance(landmarks_pixel[263][:2], landmarks_pixel[362][:2])

    ratio_l = float(v_l / w_l) if w_l > 0 else 0.0
    ratio_r = float(v_r / w_r) if w_r > 0 else 0.0
    return {
        "left_aperture_px":     round(v_l, 2),
        "right_aperture_px":    round(v_r, 2),
        "left_openness_ratio":  round(ratio_l, 4),
        "right_openness_ratio": round(ratio_r, 4),
        "mean_openness_ratio":  round((ratio_l + ratio_r) / 2.0, 4),
        "landmarks_left":       "159 (upper lid) / 145 (lower lid) / 33-133 (width)",
        "landmarks_right":      "386 (upper lid) / 374 (lower lid) / 362-263 (width)",
        "ideal_range":          "0.28 – 0.35"
    }


def compute_jaw_curvature(landmarks_pixel):
    """
    Jaw curvature: ratio of jaw arc length to jaw chord length.
    Arc: sum of consecutive distances along the full jaw contour (34 points total).
    Chord: straight-line distance from right gonion (172) to left gonion (397).

    Ratio = 1.0 → perfectly straight jaw (boxy/square jaw).
    Ratio > 1.0 → curved jaw (the higher, the more tapered / V-shaped).

    This metric complements the jaw_angle; while angle measures slope,
    curvature measures how round/tapered the overall jaw shape is.
    Reference: Adapted from Koscinski (2013) PLOS ONE facial shape analysis.
    """
    contour_indices = JAW_CONTOUR_RIGHT + JAW_CONTOUR_LEFT[1:]  # avoid duplicating chin
    pts = np.array([landmarks_pixel[i][:2] for i in contour_indices], dtype=float)

    # Arc length
    arc_len = float(sum(
        calculate_distance(pts[j], pts[j + 1]) for j in range(len(pts) - 1)
    ))
    # Chord length (gonion to gonion)
    chord_len = calculate_distance(landmarks_pixel[172][:2], landmarks_pixel[397][:2])

    ratio = float(arc_len / chord_len) if chord_len > 0 else 0.0
    return {
        "curvature_ratio":  round(ratio, 4),
        "arc_length_px":    round(arc_len, 2),
        "chord_length_px":  round(chord_len, 2),
        "n_contour_points": len(contour_indices),
        "note":             "1.0 = square jaw, higher = more tapered/V-shaped"
    }


def compute_cheekbone_width(landmarks_pixel):
    """
    Cheekbone (zygomatic) width: bizygomatic distance — the widest part of the face.
    Landmarks: 234 (right zygion approx) <-> 454 (left zygion approx).

    Also reports relative to face height for a proportional measure.
    High cheekbones correlate with facial attractiveness in multiple studies.
    Reference: Gangestad & Thornhill (1997); Penton-Voak et al. (2001).
    """
    biz_w  = calculate_distance(landmarks_pixel[234][:2], landmarks_pixel[454][:2])
    face_h = calculate_distance(landmarks_pixel[168][:2], landmarks_pixel[152][:2])
    return {
        "cheekbone_width_px":         round(float(biz_w),  2),
        "cheekbone_to_face_height":   round(float(biz_w / face_h) if face_h > 0 else 0, 4),
        "landmarks":                  "234 (right zygion approx) <-> 454 (left zygion approx)",
        "note":                       "Wider relative to face height = prominent cheekbones"
    }


# ==============================================================

def compute_all_metrics(landmarks_pixel):
    """
    Computes all geometric metrics. Returns a structured dict.
    """
    left_tilt, right_tilt = compute_canthal_tilt(landmarks_pixel)

    return {
        # ── Existing features ──────────────────────────────────
        "canthal_tilt": {
            "left_deg":  round(float(left_tilt),  2),
            "right_deg": round(float(right_tilt), 2),
            "mean_deg":  round(float((left_tilt + right_tilt) / 2.0), 2)
        },
        "fwhr":                    compute_fwhr(landmarks_pixel),
        "facial_thirds":           compute_facial_thirds(landmarks_pixel),
        "facial_fifths":           compute_facial_fifths(landmarks_pixel),
        "facial_ratios":           compute_facial_ratios(landmarks_pixel),
        "eye_spacing_ratio":       round(compute_eye_spacing(landmarks_pixel), 4),
        "jawline":                 fit_jaw_contour(landmarks_pixel),
        "landmark_symmetry_score": round(compute_landmark_symmetry(landmarks_pixel), 4),
        # ── New features (v0.3) ────────────────────────────────
        "jaw_width":               compute_jaw_width(landmarks_pixel),
        "chin_angle":              compute_chin_angle(landmarks_pixel),
        "face_aspect_ratio":       compute_face_aspect_ratio(landmarks_pixel),
        "midface_ratio":           compute_midface_ratio(landmarks_pixel),
        "philtrum_ratio":          compute_philtrum_ratio(landmarks_pixel),
        "lip_thickness_ratio":     compute_lip_thickness_ratio(landmarks_pixel),
        "nose_width_ratio":        compute_nose_width_ratio(landmarks_pixel),
        "nose_length_ratio":       compute_nose_length_ratio(landmarks_pixel),
        "intercanthal_distance":   compute_intercanthal_distance(landmarks_pixel),
        "interpupillary_distance": compute_interpupillary_distance(landmarks_pixel),
        "eyebrow_height":          compute_eyebrow_height(landmarks_pixel),
        "eye_openness":            compute_eye_openness(landmarks_pixel),
        "jaw_curvature":           compute_jaw_curvature(landmarks_pixel),
        "cheekbone_width":         compute_cheekbone_width(landmarks_pixel),
    }
