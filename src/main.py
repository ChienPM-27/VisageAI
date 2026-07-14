import os
import sys
import argparse
import json
import cv2
import numpy as np

from features.landmarks import FaceMeshExtractor
from features.geometry import compute_all_metrics
from features.quality import assess_quality
from features.fusion import geometry_to_vector, fuse_features
from models.backbone import DINOv2Extractor
from utils.visualization import draw_aesthetic_metrics


def build_json_safe(obj):
    """Recursively convert numpy types to Python native types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: build_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [build_json_safe(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description="AI Face Rating Pipeline v0.3"
    )
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output files")
    parser.add_argument("--model",      type=str, default="dinov2_vits14",
                        help="DINOv2 model name")
    parser.add_argument("--pool",       type=str, default="both",
                        choices=["cls", "mean", "both"],
                        help="DINOv2 pooling strategy: cls | mean | both")
    parser.add_argument("--skip_dl",    action="store_true",
                        help="Skip DINOv2 extraction (geometry only)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load image ─────────────────────────────────────────────────────────
    if not os.path.exists(args.image):
        print(f"Error: image not found at {args.image}")
        return
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: failed to load image at {args.image}")
        return
    print(f"Loaded image: {args.image} ({image.shape[1]}×{image.shape[0]})")

    # ── 2. Detect landmarks & align ───────────────────────────────────────────
    print("Extracting face landmarks…")
    extractor = FaceMeshExtractor()
    raw_result = extractor.extract_landmarks(image)

    if raw_result is None:
        print("Error: no face detected.")
        out = {"error": "No face detected",
               "face_quality": {"face_detected": False}}
        with open(os.path.join(args.output_dir, "report.json"), "w") as f:
            json.dump(out, f, indent=4)
        return

    landmarks_pixel_orig, _, raw_face_landmarks = raw_result  # unpack 3-tuple

    print("Aligning face…")
    aligned_image, _ = extractor.align_face(image, landmarks_pixel_orig)

    print("Re-extracting landmarks on aligned image…")
    aligned_result = extractor.extract_landmarks(aligned_image)
    if aligned_result is None:
        print("Warning: face lost after alignment — using original.")
        landmarks_pixel = landmarks_pixel_orig
        raw_face_lm     = raw_face_landmarks
    else:
        landmarks_pixel, _, raw_face_lm = aligned_result

    # ── 3. Quality Assessment ─────────────────────────────────────────────────
    print("Assessing image quality & head pose…")
    quality = assess_quality(
        aligned_image, landmarks_pixel, raw_face_lm, aligned_image.shape
    )

    if quality["warnings"]:
        for w in quality["warnings"]:
            print(f"  ⚠ {w}")
    if not quality["geometry_valid"]:
        print("  → Geometry metrics will be skipped (pose/blur out of range).")

    # ── 4. Compute Geometric Metrics ──────────────────────────────────────────
    metrics = {}
    if quality["geometry_valid"]:
        print("Computing geometric metrics…")
        metrics = compute_all_metrics(landmarks_pixel)
    else:
        print("Geometry skipped due to quality gate.")

    # ── 5. DINOv2 Embedding ───────────────────────────────────────────────────
    dino_result = {}
    fusion_result = {}
    embedding_for_fusion = None

    if not args.skip_dl:
        dino_ext = DINOv2Extractor(model_name=args.model)
        raw_dino = dino_ext.extract_features(aligned_image, pool_strategy=args.pool)

        # Keep only JSON-serialisable data (drop raw numpy arrays from output)
        dino_result = {
            k: {kk: vv for kk, vv in v.items() if kk != "embedding"}
            if isinstance(v, dict) else v
            for k, v in raw_dino.items()
        }

        # Pick primary embedding for fusion (prefer CLS token)
        if "cls_token" in raw_dino:
            embedding_for_fusion = raw_dino["cls_token"]["embedding"]
        elif "mean_pool" in raw_dino:
            embedding_for_fusion = raw_dino["mean_pool"]["embedding"]

        print(f"DINOv2 extracted — strategy: {args.pool}")
    else:
        print("DINOv2 extraction skipped.")
        dino_result = {"status": "skipped"}

    # ── 6. Feature Fusion ─────────────────────────────────────────────────────
    if metrics and embedding_for_fusion is not None:
        geo_vec, geo_names = geometry_to_vector(metrics)
        fused = fuse_features(geo_vec, embedding_for_fusion)
        fusion_result = {
            "total_dims":  fused["total_dims"],
            "geo_dims":    fused["geo_dims"],
            "dino_dims":   fused["dino_dims"],
            "note":        fused["note"],
        }
        print(f"Feature fusion complete: {fused['total_dims']}-d vector "
              f"({fused['geo_dims']} geo + {fused['dino_dims']} DINOv2)")
    elif not metrics:
        fusion_result = {"status": "skipped — geometry invalid"}
    else:
        fusion_result = {"status": "skipped — DINOv2 not run"}

    # ── 7. Visualise ──────────────────────────────────────────────────────────
    print("Generating visual annotations…")
    annotated = draw_aesthetic_metrics(aligned_image, landmarks_pixel, metrics, quality)

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    stem = os.path.splitext(os.path.basename(args.image))[0]
    aligned_path   = os.path.join(args.output_dir, f"{stem}_aligned.jpg")
    annotated_path = os.path.join(args.output_dir, f"{stem}_annotated.jpg")
    report_path    = os.path.join(args.output_dir, f"{stem}_report.json")

    cv2.imwrite(aligned_path,   aligned_image)
    cv2.imwrite(annotated_path, annotated)

    report = {
        "pipeline_version": "v0.3",
        "input_image":      args.image,
        "face_quality":     quality,
        "geometric_metrics": build_json_safe(metrics),
        "dinov2_features":  build_json_safe(dino_result),
        "fused_feature":    build_json_safe(fusion_result),
        "output_files": {
            "aligned_image":   aligned_path,
            "annotated_image": annotated_path,
        }
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("Pipeline v0.2 completed successfully!")
    print(f"  Aligned     -> {aligned_path}")
    print(f"  Annotated   -> {annotated_path}")
    print(f"  Report JSON -> {report_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
