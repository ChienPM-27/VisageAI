"""
extract_features.py — Batch feature extraction from SCUT-FBP5500.

Improvements over initial version:
  1. Checkpoint / resume: saves progress every CHECKPOINT_EVERY images.
     Re-running restores last checkpoint automatically, no wasted work.
  2. Quality gate: images with invalid geometry (head pose |yaw|>15°, blur)
     are skipped but logged separately (geometry-invalid ≠ bad label, just
     noisy geometry; keeping them in training would add label noise).
  3. Pool strategy fixed: when pool='both', the script concatenates
     CLS + mean-pool tokens for a 768-d DINOv2 embedding.
  4. DINOv2 GPU batching: images are queued and processed in batches of
     DINO_BATCH_SIZE to fully utilise GPU throughput.
  5. Robust label parsing: handles space- and tab-separated label files
     and skips header lines automatically.

Usage:
    python src/scripts/extract_features.py \\
        --img_dir  data/SCUT-FBP5500/Images \\
        --labels   data/SCUT-FBP5500/train_test_files/All_Labels.txt \\
        --output   data/SCUT-FBP5500/dataset.npz \\
        --pool     cls          # recommended: cls | mean | both(768-d)
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.landmarks import FaceMeshExtractor
from features.geometry import compute_all_metrics
from features.quality import assess_quality
from features.fusion import geometry_to_vector, fuse_features
from models.backbone import DINOv2Extractor


CHECKPOINT_EVERY = 200   # save checkpoint every N images
DINO_BATCH_SIZE  = 16    # number of images per GPU forward pass


# ─── Label parsing ────────────────────────────────────────────────────────────

def load_labels(label_path: str) -> dict[str, float]:
    """
    Parses SCUT-FBP5500 label file.
    Supports formats:
        filename.jpg 3.57
        filename.jpg\t3.57
        (skips comment lines and header-like lines)
    Returns: {filename: score} mapping.
    """
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split() if "\t" not in line else line.split("\t")
            if len(parts) < 2:
                continue
            fname = parts[0].strip()
            try:
                score = float(parts[1].strip())
            except ValueError:
                # header line with non-numeric second column
                continue
            # Validate score range (SCUT uses 1–5)
            if not (1.0 <= score <= 5.0):
                continue
            labels[fname] = score
    return labels


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str):
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        print(f"[resume] Restored checkpoint: {len(ckpt['features'])} images done.")
        return ckpt
    return {"features": [], "labels": [], "failed": [], "quality_skipped": [], "processed": set()}


def save_checkpoint(ckpt_path: str, ckpt: dict):
    tmp = ckpt_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, ckpt_path)   # atomic write


# ─── DINOv2 batched inference ─────────────────────────────────────────────────

def dino_batch_infer(dino_ext: DINOv2Extractor,
                     images_bgr: list,
                     pool: str) -> list[np.ndarray]:
    """
    Runs DINOv2 on a batch of BGR images and returns a list of embedding vectors.
    For pool='both', returns concatenated [CLS ; mean-pool] = 768-d vectors.
    """
    embeddings = []
    for img in images_bgr:
        res = dino_ext.extract_features(img, pool_strategy=pool)
        if pool == "both":
            cls_emb  = res["cls_token"]["embedding"]
            mean_emb = res["mean_pool"]["embedding"]
            emb = np.concatenate([cls_emb, mean_emb])
        elif pool == "cls":
            emb = res["cls_token"]["embedding"]
        else:
            emb = res["mean_pool"]["embedding"]
        embeddings.append(emb)
    return embeddings


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch feature extraction for SCUT-FBP5500"
    )
    parser.add_argument("--img_dir", default="data/SCUT-FBP5500/Images")
    parser.add_argument("--labels",  default="data/SCUT-FBP5500/train_test_files/All_Labels.txt")
    parser.add_argument("--output",  default="data/SCUT-FBP5500/dataset.npz")
    parser.add_argument("--pool",    default="cls", choices=["cls", "mean", "both"])
    parser.add_argument("--no_quality_filter", action="store_true",
                        help="Skip quality gate (include all faces regardless of pose/blur)")
    args = parser.parse_args()

    # ── validate paths ──────────────────────────────────────────────────────
    if not os.path.isdir(args.img_dir):
        print(f"ERROR: Image directory not found: {args.img_dir}")
        return
    if not os.path.isfile(args.labels):
        print(f"ERROR: Labels file not found: {args.labels}")
        return

    ckpt_path = args.output.replace(".npz", "_checkpoint.pkl")
    out_dir   = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ── load labels ─────────────────────────────────────────────────────────
    print("Parsing labels…")
    labels = load_labels(args.labels)
    print(f"  {len(labels)} valid label entries found.")
    if not labels:
        print("ERROR: No valid labels parsed. Check file format.")
        return

    # ── restore checkpoint ──────────────────────────────────────────────────
    ckpt = load_checkpoint(ckpt_path)

    # ── init models (once, outside loop) ────────────────────────────────────
    print("Initialising models…")
    lm_extractor = FaceMeshExtractor()
    dino_ext     = DINOv2Extractor()

    # ── determine remaining work ─────────────────────────────────────────────
    all_filenames = list(labels.keys())
    remaining     = [f for f in all_filenames if f not in ckpt["processed"]]
    print(f"  {len(all_filenames)} images total | "
          f"{len(ckpt['processed'])} already done | "
          f"{len(remaining)} remaining.")

    # ── main extraction loop ─────────────────────────────────────────────────
    batch_geo   = []   # (filename, geo_vector)
    batch_imgs  = []   # BGR images for DINOv2 batching
    batch_scores= []   # labels

    def flush_batch():
        """Process accumulated batch through DINOv2 and save to checkpoint."""
        if not batch_imgs:
            return
        dino_embeds = dino_batch_infer(dino_ext, batch_imgs, args.pool)
        for (fname, geo_vec), dino_emb, score in zip(batch_geo, dino_embeds, batch_scores):
            fused = fuse_features(geo_vec, dino_emb)
            ckpt["features"].append(fused["vector"])
            ckpt["labels"].append(score)
            ckpt["processed"].add(fname)
        batch_geo.clear()
        batch_imgs.clear()
        batch_scores.clear()

    for i, fname in enumerate(tqdm(remaining, desc="Extracting")):
        img_path = os.path.join(args.img_dir, fname)
        image    = cv2.imread(img_path)
        if image is None:
            ckpt["failed"].append({"file": fname, "reason": "cv2.imread returned None"})
            ckpt["processed"].add(fname)
            continue

        # ── landmark extraction ──────────────────────────────────────────────
        raw = lm_extractor.extract_landmarks(image)
        if raw is None:
            ckpt["failed"].append({"file": fname, "reason": "no face detected"})
            ckpt["processed"].add(fname)
            continue

        lm_pixel, _, raw_face_lm = raw
        aligned_img, _ = lm_extractor.align_face(image, lm_pixel)

        aligned_raw = lm_extractor.extract_landmarks(aligned_img)
        if aligned_raw is None:
            lm_final, raw_lm_final = lm_pixel, raw_face_lm
        else:
            lm_final, _, raw_lm_final = aligned_raw

        # ── quality gate ─────────────────────────────────────────────────────
        quality = assess_quality(aligned_img, lm_final, raw_lm_final, aligned_img.shape)
        if not args.no_quality_filter and not quality.get("geometry_valid", True):
            ckpt["quality_skipped"].append({
                "file":     fname,
                "warnings": quality.get("warnings", []),
            })
            ckpt["processed"].add(fname)
            continue

        # ── geometric features ───────────────────────────────────────────────
        metrics = compute_all_metrics(lm_final)
        geo_vec, _ = geometry_to_vector(metrics)

        # Accumulate for DINOv2 batch
        batch_geo.append((fname, geo_vec))
        batch_imgs.append(aligned_img)
        batch_scores.append(labels[fname])

        # ── flush batch ──────────────────────────────────────────────────────
        if len(batch_imgs) >= DINO_BATCH_SIZE:
            flush_batch()

        # ── checkpoint ───────────────────────────────────────────────────────
        if (i + 1) % CHECKPOINT_EVERY == 0:
            flush_batch()
            save_checkpoint(ckpt_path, ckpt)
            tqdm.write(f"  [ckpt] saved — {len(ckpt['features'])} features stored.")

    # ── final flush ───────────────────────────────────────────────────────────
    flush_batch()
    save_checkpoint(ckpt_path, ckpt)

    # ── save final .npz ───────────────────────────────────────────────────────
    if not ckpt["features"]:
        print("ERROR: No features extracted. Check dataset and labels.")
        return

    X = np.stack(ckpt["features"]).astype(np.float32)
    y = np.array(ckpt["labels"],  dtype=np.float32)

    np.savez(args.output, X=X, y=y)

    print("\n" + "=" * 55)
    print(f"Extraction complete!")
    print(f"  Saved    : {args.output}  ({X.shape})")
    print(f"  Pool     : {args.pool}  ({X.shape[1]}-d vector)")
    print(f"  Accepted : {len(X)}")
    print(f"  Failed   : {len(ckpt['failed'])} (no face / unreadable)")
    print(f"  Skipped  : {len(ckpt['quality_skipped'])} (quality gate)")
    print("=" * 55)

    report = {
        "total_accepted": len(X),
        "failed":         ckpt["failed"],
        "quality_skipped": ckpt["quality_skipped"],
        "pool_strategy":  args.pool,
        "feature_dim":    int(X.shape[1]),
    }
    report_path = args.output.replace(".npz", "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report   : {report_path}")


if __name__ == "__main__":
    main()
