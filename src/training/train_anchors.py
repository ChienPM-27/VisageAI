"""
train_anchors.py — Anchor Face extraction using Metric Learning principles.

Improvements over initial version:
  1. StandardScaler applied before KMeans: prevents high-magnitude features
     (angles in degrees vs ratios in 0-1) from dominating the distance metric.
  2. Medoid instead of centroid: uses the *real sample closest to the cluster
     centre* as the anchor. This means every anchor corresponds to an actual
     face in the dataset, not an unattainable mathematical average.
  3. Tier thresholds calibrated to SCUT-FBP5500 actual score distribution.
     The dataset's mean score is ≈3.0 (60th percentile ≈ 3.2, top 5% ≈ 3.8).
     Thresholds are now derived from dataset quantiles, not hardcoded values.
  4. Saves both the anchor vectors AND the source filenames/labels so you can
     visually inspect which faces were selected as archetypes.
  5. Removed unused `cosine_similarity` import.

Usage:
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
    Returns array of indices into the original (unscaled) dataset.
    """
    return pairwise_distances_argmin(centroids, X_scaled)


def get_tier_masks(y_10: np.ndarray) -> dict[str, np.ndarray]:
    """
    Returns boolean masks for each tier, based on dataset quantiles.
    Using quantile-based thresholds makes tiers balanced even if the score
    distribution is skewed (which it is in SCUT-FBP5500).

    Tiers (% of dataset):
        Elite         : top 10%
        Above average : 10–35%
        Average       : 35–65%
        Below average : bottom 35%
    """
    p10  = np.percentile(y_10, 90)   # top 10%
    p35  = np.percentile(y_10, 65)   # top 35%
    p65  = np.percentile(y_10, 35)   # bottom 35%

    return {
        "tier_1_elite":         y_10 >= p10,
        "tier_2_above_average": (y_10 >= p35) & (y_10 < p10),
        "tier_3_average":       (y_10 >= p65) & (y_10 < p35),
        "tier_4_below_average": y_10 < p65,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Anchor Faces from dataset")
    parser.add_argument("--dataset",   default="data/SCUT-FBP5500/dataset.npz")
    parser.add_argument("--out_dir",   default="weights")
    parser.add_argument("--n_anchors", type=int, default=5,
                        help="Number of anchor faces per tier")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}\n"
              f"Please run src/scripts/extract_features.py first.")
        return

    # ── load ────────────────────────────────────────────────────────────────
    print("Loading dataset…")
    data = np.load(args.dataset)
    X    = data["X"].astype(np.float32)    # (N, 427)
    y    = data["y"].astype(np.float32)    # (N,) in [1, 5]
    y_10 = ((y - 1.0) / 4.0) * 10.0       # rescale to [0, 10]
    N    = len(X)
    print(f"  {N} samples | score range: [{y_10.min():.2f}, {y_10.max():.2f}] "
          f"mean={y_10.mean():.2f}")

    # ── standardise BEFORE clustering ───────────────────────────────────────
    scaler_path = os.path.join(args.out_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"  Loaded existing scaler from {scaler_path}")
    else:
        print("  Fitting new StandardScaler (no existing scaler found)…")
        scaler = StandardScaler()
        scaler.fit(X)

    X_scaled = scaler.transform(X)

    # ── tier masks ───────────────────────────────────────────────────────────
    tiers = get_tier_masks(y_10)
    for name, mask in tiers.items():
        print(f"  {name}: {mask.sum()} samples "
              f"| score range [{y_10[mask].min():.2f}, {y_10[mask].max():.2f}]")

    # ── cluster and find medoids ─────────────────────────────────────────────
    anchors_out = {}
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nClustering ({args.n_anchors} anchors per tier)…")
    for tier_name, mask in tiers.items():
        group_X_sc = X_scaled[mask]   # scaled subset for KMeans
        group_X    = X[mask]          # original for saving
        group_y    = y_10[mask]
        group_idx  = np.where(mask)[0]  # global indices

        if len(group_X_sc) < args.n_anchors:
            print(f"  [{tier_name}] Only {len(group_X_sc)} samples "
                  f"— using all as anchors.")
            medoid_local = np.arange(len(group_X_sc))
        else:
            kmeans = KMeans(
                n_clusters=args.n_anchors,
                random_state=42,
                n_init=20,     # more initialisations → more stable centroids
                max_iter=500,
            )
            kmeans.fit(group_X_sc)
            medoid_local = find_medoids(group_X_sc, kmeans.cluster_centers_)

        anchor_records = []
        for local_i in medoid_local:
            anchor_records.append({
                "global_index":   int(group_idx[local_i]),
                "score_0_10":     float(round(group_y[local_i], 3)),
                "feature_vector": group_X[local_i].tolist(),  # unscaled, for inference
            })

        anchors_out[tier_name] = anchor_records
        scores = [r["score_0_10"] for r in anchor_records]
        print(f"  [{tier_name}] {len(anchor_records)} medoids selected "
              f"| scores: {[round(s,2) for s in scores]}")

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, "anchors.json")
    with open(out_path, "w") as f:
        json.dump(anchors_out, f, indent=2)

    print(f"\nAnchors saved to: {out_path}")
    print("Each anchor is a real face from the dataset (medoid), not a synthetic average.")
    print("Use AnchorScorer in scoring.py with these anchors for metric-learning inference.")


if __name__ == "__main__":
    main()
