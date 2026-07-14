import cv2
import numpy as np

# =============================================================
# 3D model points of facial landmarks in generic face coords (mm).
# Used by solvePnP for head pose estimation.
# Reference: Guo et al. (2020) "PFLD: A Practical Facial Landmark Detector"
# Points: nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
# =============================================================
MODEL_3D_POINTS = np.array([
    [0.0,    0.0,    0.0],    # Nose tip (landmark 1)
    [0.0,   -63.6,  -12.5],   # Chin (landmark 152)
    [-43.3,  32.7,  -26.0],   # Left eye outer (landmark 33)  — image-right eye
    [43.3,   32.7,  -26.0],   # Right eye outer (landmark 263) — image-left eye
    [-28.9, -28.9,  -24.1],   # Left mouth corner (landmark 61)
    [28.9,  -28.9,  -24.1],   # Right mouth corner (landmark 291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices for the 6 reference points
MODEL_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(landmarks_pixel, image_shape):
    """
    Estimates head pose (Yaw, Pitch, Roll) in degrees using solvePnP.

    Camera model: pinhole with focal length ≈ image_width (standard approximation
    for 35mm-equivalent shots; acceptable at baseline when no EXIF data is available).
    Principal point at image centre.

    Args:
        landmarks_pixel : numpy array (N, 3) in pixel coordinates.
        image_shape     : (height, width, channels) tuple.
    Returns:
        dict with yaw, pitch, roll (degrees), rvec, tvec.
    """
    h, w = image_shape[:2]
    focal_length = float(w)          # standard pinhole approximation
    cx, cy = w / 2.0, h / 2.0

    camera_matrix = np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,            1 ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points = np.array(
        [landmarks_pixel[i][:2] for i in MODEL_LANDMARK_INDICES],
        dtype=np.float64
    )

    success, rvec, tvec = cv2.solvePnP(
        MODEL_3D_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return {"yaw": None, "pitch": None, "roll": None, "pose_valid": False}

    # Convert rotation vector to rotation matrix, then to Euler angles
    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix to Euler angles (ZYX convention)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll  = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = 0.0

    return {
        "yaw":        round(float(yaw),   2),
        "pitch":      round(float(pitch), 2),
        "roll":       round(float(roll),  2),
        "pose_valid": True
    }


def compute_blur_score(image):
    """
    Laplacian variance blur estimator.
    Higher = sharper image. Below 100 is considered blurry.
    Reference: Pertuz et al. (2013) "Analysis of focus measure operators for shape-from-focus".
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "score":    round(lap_var, 2),
        "is_sharp": lap_var >= 100.0,
        "threshold": 100.0
    }


def compute_resolution_check(image):
    """Check that the shorter dimension is ≥ 224 px (minimum for DINOv2 input)."""
    h, w = image.shape[:2]
    min_dim = min(h, w)
    return {
        "width":  w,
        "height": h,
        "min_dim": min_dim,
        "ok": min_dim >= 224
    }


def compute_landmark_visibility(face_landmarks_raw):
    """
    Estimates overall landmark detection confidence from MediaPipe visibility scores.
    MediaPipe assigns a visibility value [0,1] per landmark indicating occlusion likelihood.
    Returns mean visibility across all landmarks.
    """
    visibilities = [lm.visibility for lm in face_landmarks_raw.landmark]
    mean_vis = float(np.mean(visibilities))
    min_vis  = float(np.min(visibilities))
    return {
        "mean_visibility": round(mean_vis, 4),
        "min_visibility":  round(min_vis,  4),
        "confidence":      round(mean_vis, 4)
    }


def compute_alignment_angle(landmarks_pixel):
    """
    Returns the alignment rotation angle applied (degrees).
    Computed from the slope between the two eye centres.
    Negative = image was rotated counter-clockwise to align.
    """
    left_eye  = (landmarks_pixel[33][:2]  + landmarks_pixel[133][:2]) / 2.0
    right_eye = (landmarks_pixel[263][:2] + landmarks_pixel[362][:2]) / 2.0
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    return round(float(np.degrees(np.arctan2(dY, dX))), 2)


def assess_quality(image, landmarks_pixel, face_landmarks_raw, image_shape):
    """
    Full quality assessment pipeline.
    Returns a dict with all quality metrics and a 'geometry_valid' flag.

    geometry_valid = False when:
      - abs(yaw)  > 15° (out-of-plane rotation degrades geometry accuracy)
      - abs(pitch)> 20°
      - blur_score < 50 (severely blurry)
    """
    pose       = estimate_head_pose(landmarks_pixel, image_shape)
    blur       = compute_blur_score(image)
    resolution = compute_resolution_check(image)
    visibility = compute_landmark_visibility(face_landmarks_raw)
    align_ang  = compute_alignment_angle(landmarks_pixel)

    # Geometry validity gate
    yaw_ok   = pose["pose_valid"] and abs(pose["yaw"])   <= 15.0
    pitch_ok = pose["pose_valid"] and abs(pose["pitch"]) <= 20.0
    blur_ok  = blur["score"] >= 50.0

    geometry_valid = yaw_ok and pitch_ok and blur_ok

    warnings = []
    if not yaw_ok:
        warnings.append(f"Yaw={pose['yaw']}° > 15° threshold — geometry metrics unreliable")
    if not pitch_ok:
        warnings.append(f"Pitch={pose['pitch']}° > 20° threshold — geometry metrics unreliable")
    if not blur_ok:
        warnings.append(f"Blur score={blur['score']} < 50 — image too blurry")
    if not resolution["ok"]:
        warnings.append(f"Min dimension={resolution['min_dim']}px < 224px")

    return {
        "face_detected":        True,
        "alignment_angle_deg":  align_ang,
        "head_pose":            pose,
        "blur":                 blur,
        "resolution":           resolution,
        "landmark_visibility":  visibility,
        "geometry_valid":       geometry_valid,
        "warnings":             warnings
    }
