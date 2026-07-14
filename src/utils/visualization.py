import cv2
import numpy as np


def draw_hud_text(image, text, position, font_scale=0.45, text_color=(255, 255, 255),
                  bg_color=(30, 30, 30), alpha=0.75):
    """Draws text with a semi-transparent background box."""
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_line_with_label(image, pt1, pt2, label, color, font_scale=0.38, thickness=1):
    """Draws a measurement line and a small label at its midpoint."""
    cv2.line(image, tuple(pt1), tuple(pt2), color, thickness, cv2.LINE_AA)
    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    draw_hud_text(image, label, (mid[0] + 4, mid[1] - 4),
                  font_scale=font_scale, bg_color=(50, 50, 50))


# Colour palette
C_GOLD    = (30,  165, 255)   # amber / gold  (BGR)
C_GREEN   = (80,  200, 120)   # emerald
C_CYAN    = (230, 200,  50)   # teal/cyan
C_RED     = (50,   60, 220)   # alert red
C_GREY    = (160, 160, 160)   # reference grey
C_PURPLE  = (200,  80, 200)   # purple
C_BLUE    = (220, 120,  50)   # soft blue
C_WHITE   = (255, 255, 255)


def draw_quality_gate_warning(image, warnings):
    """Draws a red banner at the top of the image if geometry is invalid."""
    if not warnings:
        return
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (30, 30, 200), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(image, f"⚠ GEOMETRY INVALID: {warnings[0][:70]}",
                (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_facial_thirds(image, landmarks_pixel, thirds):
    """Draws mid-face and lower-face horizontal dividers and ratios."""
    h, w = image.shape[:2]
    for idx, label, color in [
        (9,   "Glabella",   C_GREY),
        (2,   "Subnasale",  C_GREY),
        (152, "Menton",     C_GREY),
    ]:
        y = int(landmarks_pixel[idx][1])
        cv2.line(image, (10, y), (w - 10, y), color, 1, cv2.LINE_AA)
        draw_hud_text(image, label, (14, y - 4), font_scale=0.35, bg_color=(40, 40, 40))

    mid_r   = thirds.get("mid_ratio", 0)
    lower_r = thirds.get("lower_ratio", 0)
    m2l     = thirds.get("mid_to_lower", 0)
    draw_hud_text(image,
        f"Mid:{mid_r:.2f} Low:{lower_r:.2f} M/L:{m2l:.2f}(ideal 1.0)",
        (10, 22), bg_color=(40, 40, 80))


def draw_fwhr(image, landmarks_pixel, fwhr):
    """Draws fWHR width and height measurement lines."""
    pt_lz  = landmarks_pixel[234][:2].astype(int)
    pt_rz  = landmarks_pixel[454][:2].astype(int)
    pt_gl  = landmarks_pixel[9][:2].astype(int)
    pt_lip = landmarks_pixel[0][:2].astype(int)

    draw_line_with_label(image, pt_lz, pt_rz,
        f"W (biz.) {fwhr.get('width_px', 0):.0f}px", C_GREEN)
    draw_line_with_label(image, pt_gl, pt_lip,
        f"H (gla-lip) {fwhr.get('height_px', 0):.0f}px", C_CYAN)
    draw_hud_text(image, f"fWHR: {fwhr.get('value', 0):.2f}", (10, 48), bg_color=(40, 80, 40))


def draw_canthal_tilt(image, landmarks_pixel, tilt):
    """Draws inter-canthal axis lines and angle labels."""
    lo = landmarks_pixel[33][:2].astype(int)   # outer image-left eye
    li = landmarks_pixel[133][:2].astype(int)  # inner
    ri = landmarks_pixel[362][:2].astype(int)  # inner image-right eye
    ro = landmarks_pixel[263][:2].astype(int)  # outer

    cv2.line(image, tuple(li), tuple(lo), C_GOLD, 2, cv2.LINE_AA)
    cv2.line(image, tuple(ri), tuple(ro), C_GOLD, 2, cv2.LINE_AA)

    draw_hud_text(image,
        f"Canthal  L:{tilt.get('left_deg', 0):.1f}°  R:{tilt.get('right_deg', 0):.1f}°",
        (10, 74), bg_color=(80, 60, 0))


def draw_jawline(image, landmarks_pixel, jaw):
    """Draws jaw contour fit line and angle label."""
    from features.geometry import JAW_CONTOUR_RIGHT, JAW_CONTOUR_LEFT

    right_pts = [landmarks_pixel[i][:2].astype(int) for i in JAW_CONTOUR_RIGHT]
    left_pts  = [landmarks_pixel[i][:2].astype(int) for i in JAW_CONTOUR_LEFT]

    for pts in [right_pts, left_pts]:
        for j in range(len(pts) - 1):
            cv2.line(image, tuple(pts[j]), tuple(pts[j + 1]), C_PURPLE, 2, cv2.LINE_AA)

    draw_hud_text(image,
        f"Jaw  R:{jaw.get('right_angle_deg', 0):.1f}°  L:{jaw.get('left_angle_deg', 0):.1f}°  "
        f"J/F:{jaw.get('jaw_to_face_ratio', 0):.2f}",
        (10, 100), bg_color=(60, 0, 60))


def draw_facial_fifths(image, landmarks_pixel, fifths):
    """Draws 5 vertical lines dividing the face into fifths."""
    h = image.shape[0]
    indices = [234, 33, 133, 362, 263, 454]
    colors  = [C_GREY, C_CYAN, C_GREEN, C_GREEN, C_CYAN, C_GREY]

    segs = fifths.get("segments", [])
    for k, idx in enumerate(indices):
        x = int(landmarks_pixel[idx][0])
        cv2.line(image, (x, 0), (x, h), colors[k % len(colors)], 1, cv2.LINE_AA)

    if segs:
        ratios_str = "  ".join([f"{s['ratio']:.2f}" for s in segs])
        draw_hud_text(image, f"Fifths: {ratios_str} (ideal:0.20 each)",
                      (10, 126), bg_color=(0, 60, 60))


def draw_facial_ratios(image, ratios):
    """Draws ratio panel in lower HUD."""
    draw_hud_text(image,
        f"Nose/Eye:{ratios.get('nose_to_eye_ratio', 0):.2f}(≈1.0)  "
        f"Mouth/Eye:{ratios.get('mouth_to_eye_ratio', 0):.2f}(≈1.5)  "
        f"Nose/IC:{ratios.get('nose_to_intercanthal', 0):.2f}(≈1.0)",
        (10, 152), bg_color=(40, 40, 60))


def draw_symmetry(image, score):
    """Draws landmark symmetry score."""
    col = C_GREEN if score >= 0.85 else C_GOLD if score >= 0.70 else C_RED
    draw_hud_text(image,
        f"Landmark Symmetry: {score * 10:.1f}/10  ({score:.3f})",
        (10, 178), bg_color=(30, 60, 30))


def draw_key_landmarks(image, landmarks_pixel):
    """Dots for key structural landmarks."""
    key_indices = [
        33, 133, 263, 362,        # eye corners
        70, 105, 336, 296,        # eyebrows
        1, 2, 4, 129, 358,        # nose
        61, 291, 0, 17,           # mouth
        234, 454, 172, 397, 152,  # jaw / cheek
    ]
    for idx in key_indices:
        pt = landmarks_pixel[idx][:2].astype(int)
        cv2.circle(image, tuple(pt), 2, C_GOLD, -1)


def draw_aesthetic_metrics(image, landmarks_pixel, metrics, quality=None):
    """
    Master function: annotates the face image with all geometric measurements,
    quality warnings, and a HUD panel.

    Args:
        image            : BGR numpy array.
        landmarks_pixel  : numpy array (N, 3) in pixel coords.
        metrics          : output of compute_all_metrics().
        quality          : output of assess_quality(), or None.
    Returns:
        annotated BGR image.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # 1. Quality gate warning banner
    if quality and not quality.get("geometry_valid", True):
        draw_quality_gate_warning(annotated, quality.get("warnings", []))

    # 2. Facial thirds (2-segment, no hairline)
    draw_facial_thirds(annotated, landmarks_pixel, metrics.get("facial_thirds", {}))

    # 3. fWHR
    draw_fwhr(annotated, landmarks_pixel, metrics.get("fwhr", {}))

    # 4. Canthal tilt
    draw_canthal_tilt(annotated, landmarks_pixel, metrics.get("canthal_tilt", {}))

    # 5. Jaw contour
    draw_jawline(annotated, landmarks_pixel, metrics.get("jawline", {}))

    # 6. Facial fifths (vertical lines)
    draw_facial_fifths(annotated, landmarks_pixel, metrics.get("facial_fifths", {}))

    # 7. Facial ratios text
    draw_facial_ratios(annotated, metrics.get("facial_ratios", {}))

    # 8. Landmark symmetry score
    draw_symmetry(annotated, metrics.get("landmark_symmetry_score", 0.0))

    # 9. Head pose info (if quality available)
    if quality:
        pose = quality.get("head_pose", {})
        blur = quality.get("blur", {})
        draw_hud_text(annotated,
            f"Pose  Y:{pose.get('yaw', '?')}°  P:{pose.get('pitch', '?')}°  "
            f"R:{pose.get('roll', '?')}°  |  Blur:{blur.get('score', '?'):.0f}",
            (10, h - 14), bg_color=(20, 20, 20))

    # 10. Key landmark dots
    draw_key_landmarks(annotated, landmarks_pixel)

    return annotated
