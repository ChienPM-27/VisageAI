"""
train_anchors.py — Anchor Face extraction (Metric Learning).

Finds representative faces (medoids) for each attractiveness tier
from the SCUT-FBP5500 training split.

v0.5 fixes:
  1. Loads split_indices.json from train_mlp.py → uses TRAIN samples only.
     Anchors computed on test samples would leak test distribution into
     AnchorScorer inference, invalidating the held-out test evaluation.
  2. StandardScaler applied BEFORE K-Means (prevents high-magnitude features
     like angles in degrees dominating the distance metric).
  3. Medoid instead of centroid: selects the REAL sample closest to each
     cluster centre, so every anchor corresponds to an actual face.
  4. Tier thresholds derived from dataset quantiles (not hardcoded) to stay
     balanced even with SCUT-FBP5500's skewed score distribution.
  5. Removed the ghost reference to 'AnchorScorer in scoring.py' — 
     AnchorScorer is now fully implemented in src/features/scoring.py.

Usage:
    # Must run AFTER train_mlp.py (needs split_indices.json + scaler.pkl)
    python src/training/train_anchors.py \\
        --dataset  data/SCUT-FBP5500/dataset.npz \\
        --out_dir  weights \\
        --n_anchors 5
"""

import argparse
import os
import json
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin


def find_medoids(X_scaled: np.ndarray,
                 centroids: np.ndarray) -> np.ndarray:
    """
    For each centroid, find the index in X_scaled of the closest real sample.
    Returns array of indices into the (potentially filtered) feature matrix.
    """
    return pairwise_distances_argmin(centroids, X_scaled)


def get_tier_masks(y_10: np.ndarray) -> dict[str, np.ndarray]:
    """
    Returns boolean masks based on dataset quantiles so each tier is
    proportionally balanced even with skewed distributions.

    Tiers (% of TRAINING set):
        Elite         : top 10%
        Above average : 10–35%
        Average       : 35–65%
        Below average : bottom 35%
    """
    p10 = np.percentile(y_10, 90)
    p35 = np.percentile(y_10, 65)
    p65 = np.percentile(y_10, 35)

    return {
        "tier_1_elite":         y_10 >= p10,
        "tier_2_above_average": (y_10 >= p35) & (y_10 < p10),
        "tier_3_average":       (y_10 >= p65) & (y_10 < p35),
        "tier_4_below_average": y_10 < p65,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract Anchor Faces from TRAINING split only"
    )
    parser.add_argument("--dataset",    default="data/SCUT-FBP5500/dataset.npz")
    parser.add_argument("--out_dir",    default="weights")
    parser.add_argument("--n_anchors",  type=int, default=5,
                        help="Number of anchor faces per tier")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}\n"
              f"Run src/scripts/extract_features.py first.")
        return

    # ── Load full dataset ────────────────────────────────────────────────────
    print("Loading dataset…")
    data = np.load(args.dataset)
    X    = data["X"].astype(np.float32)
    y    = data["y"].astype(np.float32)
    y_10 = ((y - 1.0) / 4.0) * 10.0
    print(f"  {len(X)} total samples | "
          f"score [{y_10.min():.2f}, {y_10.max():.2f}] mean={y_10.mean():.2f}")

    # ── Load train split indices (avoid test leakage) ────────────────────────
    split_path = os.path.join(args.out_dir, "split_indices.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        train_idx = np.array(split["train"])
        print(f"  Using training split only: {len(train_idx)} samples "
              f"(split_indices.json loaded)")
    else:
        print("  WARNING: split_indices.json not found.")
        print("  Run train_mlp.py first to generate the split.")
        print("  Falling back to full dataset (may include test samples).")
        train_idx = np.arange(len(X))

    X_train  = X[train_idx]
    y_train  = y_10[train_idx]

    # ── Load or fit StandardScaler ───────────────────────────────────────────
    scaler_path = os.path.join(args.out_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"  Loaded scaler from {scaler_path}")
    else:
        print("  Fitting StandardScaler (no existing scaler found)…")
        scaler = StandardScaler()
        scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)

    # ── Tier masks (on training set only) ───────────────────────────────────
    tiers = get_tier_masks(y_train)
    for name, mask in tiers.items():
        n = mask.sum()
        if n > 0:
            print(f"  {name}: {n} samples "
                  f"| score [{y_train[mask].min():.2f}, {y_train[mask].max():.2f}]")

    # ── Cluster and find medoids ─────────────────────────────────────────────
    anchors_out = {}
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nClustering ({args.n_anchors} anchors per tier, train-only)…")
    for tier_name, mask in tiers.items():
        group_X_sc  = X_train_scaled[mask]     # scaled, for KMeans
        group_X_raw = X_train[mask]             # unscaled, for saving
        group_y     = y_train[mask]

        if len(group_X_sc) < args.n_anchors:
            print(f"  [{tier_name}] Only {len(group_X_sc)} samples — using all.")
            medoid_local = np.arange(len(group_X_sc))
        else:
            kmeans = KMeans(
                n_clusters=args.n_anchors,
                random_state=42,
                n_init=20,
                max_iter=500,
            )
            kmeans.fit(group_X_sc)
            medoid_local = find_medoids(group_X_sc, kmeans.cluster_centers_)

        records = []
        for li in medoid_local:
            records.append({
                "train_local_index": int(li),
                "score_0_10":        float(round(group_y[li], 3)),
                "feature_vector":    group_X_raw[li].tolist(),  # unscaled, for AnchorScorer
            })
        anchors_out[tier_name] = records
        scores_sel = [r["score_0_10"] for r in records]
        print(f"  [{tier_name}] {len(records)} medoids "
              f"| scores: {[round(s, 2) for s in scores_sel]}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, "anchors.json")
    with open(out_path, "w") as f:
        json.dump(anchors_out, f, indent=2)

    print(f"\nAnchors saved to: {out_path}")
    print("Each anchor = real face from TRAINING split (medoid, not synthetic centroid).")
    print("AnchorScorer in src/features/scoring.py loads this file for inference.")


if __name__ == "__main__":
    main()
