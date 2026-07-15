"""
predict.py — Standalone inference script (train → serve).

This script closes the loop between the training pipeline and the product:
    extract_features.py → train_mlp.py → predict.py

Usage:
    python src/scripts/predict.py --image path/to/face.jpg
    python src/scripts/predict.py --image face.jpg --skip_dl   # geometry only

Output:
    Prints all available scores to console.
    Saves a compact JSON report to --output_dir.

Scorer priority:
    1. MLScorer     — best accuracy (learned from 5500 human-rated faces)
    2. AnchorScorer — tier interpretation (which attractiveness cluster)
    3. GeometricScorer — always available, rule-based baseline

If the ML model has not been trained yet, scorers 1 and 2 are skipped
with a clear message and only the geometric score is returned.
"""

import os
import sys
import argparse
import json
import cv2

# Allow running from project root as: python src/scripts/predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.landmarks import FaceMeshExtractor
from features.geometry import compute_all_metrics
from features.quality import assess_quality
from features.fusion import geometry_to_vector, fuse_features
from features.scoring import GeometricScorer, MLScorer, AnchorScorer
from models.backbone import DINOv2Extractor


def build_json_safe(obj):
    """Recursively strip non-serialisable objects (numpy arrays etc.)."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: build_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [build_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def predict(image_path: str,
            skip_dl: bool = False,
            model_name: str = "dinov2_vits14",
            pool: str = "cls",
            output_dir: str = "output",
            weights_dir: str = "weights") -> dict:
    """
    Full inference pipeline for a single image.

    Returns a dict with:
        scores     : { geometric, ml (if available), anchor (if available) }
        quality    : head pose, blur, geometry gate
        fusion     : fused feature metadata
        scorers_used: list of active scorers
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load image ──────────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    print(f"Image loaded: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # ── Landmarks + alignment ────────────────────────────────────────────────
    extractor = FaceMeshExtractor()
    raw = extractor.extract_landmarks(image)
    if raw is None:
        raise RuntimeError("No face detected in image.")
    lm_pixel, _, raw_face_lm = raw
    aligned, _ = extractor.align_face(image, lm_pixel)

    aligned_raw = extractor.extract_landmarks(aligned)
    if aligned_raw:
        lm_pixel, _, raw_face_lm = aligned_raw

    # ── Quality gate ─────────────────────────────────────────────────────────
    quality = assess_quality(aligned, lm_pixel, raw_face_lm, aligned.shape)
    if quality["warnings"]:
        for w in quality["warnings"]:
            print(f"  ⚠ {w}")

    # ── Geometric features + scorer ──────────────────────────────────────────
    scores = {}
    metrics = {}
    raw_fused = None

    if quality["geometry_valid"]:
        metrics = compute_all_metrics(lm_pixel)
        scores["geometric"] = GeometricScorer().score(metrics)
        geo_s = scores["geometric"]["score_out_of_10"]
        print(f"  Geometric Score  : {geo_s}/10.0  (rule-based baseline)")

    # ── DINOv2 + Fusion ──────────────────────────────────────────────────────
    if not skip_dl and metrics:
        dino = DINOv2Extractor(model_name=model_name)
        dino_res = dino.extract_features(aligned, pool_strategy=pool)
        emb = (dino_res.get("cls_token", {}).get("embedding") or
               dino_res.get("mean_pool", {}).get("embedding"))
        if emb is not None:
            geo_vec, _ = geometry_to_vector(metrics)
            fused = fuse_features(geo_vec, emb)
            raw_fused = fused["vector"]
            print(f"  Fused vector     : {fused['total_dims']}-d "
                  f"({fused['geo_dims']} geo + {fused['dino_dims']} DINOv2)")

    # ── ML Scorer ────────────────────────────────────────────────────────────
    scorers_used = ["geometric"] if scores else []
    if raw_fused is not None:
        model_path  = os.path.join(weights_dir, "face_rater_v1.pt")
        scaler_path = os.path.join(weights_dir, "scaler.pkl")
        meta_path   = os.path.join(weights_dir, "model_meta.json")
        try:
            ml = MLScorer(model_path=model_path,
                          scaler_path=scaler_path,
                          meta_path=meta_path)
            scores["ml"] = ml.score(raw_fused)
            scorers_used.append("ml")
            ml_s = scores["ml"]["score_out_of_10"]
            pr   = scores["ml"]["model_metrics"].get("test_pearson", "?")
            print(f"  ML Score         : {ml_s}/10.0  "
                  f"(test Pearson r={pr} on SCUT-FBP5500)")
        except FileNotFoundError as e:
            print(f"  ML model not found: {e}")
            print("  → Run python src/scripts/extract_features.py")
            print("         python src/training/train_mlp.py")

        anchors_path = os.path.join(weights_dir, "anchors.json")
        try:
            anc = AnchorScorer(anchors_path=anchors_path,
                               scaler_path=scaler_path)
            scores["anchor"] = anc.score(raw_fused)
            scorers_used.append("anchor")
            anc_s = scores["anchor"]["score_out_of_10"]
            tier  = scores["anchor"]["closest_tier"]
            print(f"  Anchor Score     : {anc_s}/10.0  (closest: {tier})")
        except FileNotFoundError:
            pass

    # ── Summary ──────────────────────────────────────────────────────────────
    primary_score = (scores.get("ml") or
                     scores.get("anchor") or
                     scores.get("geometric", {}))
    final = primary_score.get("score_out_of_10", "N/A")
    primary_method = ("ML model" if "ml" in scores else
                      "Anchor similarity" if "anchor" in scores else
                      "Geometric rules")
    print(f"\n{'='*50}")
    print(f"  FINAL SCORE  : {final} / 10.0")
    print(f"  Basis        : {primary_method}")
    print(f"  Scorers used : {', '.join(scorers_used)}")
    print(f"{'='*50}\n")

    result = {
        "image":        image_path,
        "final_score":  final,
        "primary_basis": primary_method,
        "scores":       build_json_safe(scores),
        "quality":      build_json_safe(quality),
        "scorers_used": scorers_used,
    }

    # Save report
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_predict.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Report saved : {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="VisageAI — Standalone face rating inference"
    )
    parser.add_argument("--image",      required=True,  help="Path to face image")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--weights_dir",default="weights")
    parser.add_argument("--model",      default="dinov2_vits14")
    parser.add_argument("--pool",       default="cls", choices=["cls", "mean", "both"])
    parser.add_argument("--skip_dl",    action="store_true",
                        help="Skip DINOv2 (geometric score only, no ML)")
    args = parser.parse_args()

    predict(
        image_path=args.image,
        skip_dl=args.skip_dl,
        model_name=args.model,
        pool=args.pool,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
    )


if __name__ == "__main__":
    main()
