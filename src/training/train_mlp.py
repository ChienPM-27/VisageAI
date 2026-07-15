"""
train_mlp.py — MLP Regressor for Face Aesthetic Scoring.

Improvements over initial version:
  1. Proper 3-way split (70% train / 15% val / 15% test).
     Validation set drives model selection; test set is touched only once.
  2. Learning-rate scheduling: ReduceLROnPlateau halves LR when val loss
     plateaus for 10 epochs, enabling better final convergence.
  3. Loss function: combined MSE + Pearson correlation loss.
     Pearson correlation is the standard benchmark metric for SCUT-FBP5500,
     so optimising for it directly improves leaderboard performance.
  4. Architecture: replaced BatchNorm with LayerNorm. BatchNorm statistics
     are unstable with small batches and when combined with Dropout.
     LayerNorm normalises over feature dim per sample — safe with any batch.
  5. Activation: GELU instead of ReLU (smoother gradient, better for regression).
  6. Training curve logging: saves per-epoch metrics to CSV for analysis.

Usage:
    python src/training/train_mlp.py \\
        --dataset data/SCUT-FBP5500/dataset.npz \\
        --epochs  150 \\
        --out_dir weights
"""

import argparse
import os
import csv
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


# ─── Loss ─────────────────────────────────────────────────────────────────────

def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson r, differentiable. Minimising this maximises correlation."""
    vx = pred   - pred.mean()
    vy = target - target.mean()
    r  = (vx * vy).sum() / (
        torch.sqrt((vx ** 2).sum() + 1e-8) *
        torch.sqrt((vy ** 2).sum() + 1e-8)
    )
    return 1.0 - r


def pairwise_ranking_loss(pred: torch.Tensor,
                          target: torch.Tensor,
                          margin: float = 0.5) -> torch.Tensor:
    """
    Margin Ranking Loss on randomly sampled pairs within a batch.

    For each pair (i, j) where target[i] > target[j] + margin,
    the model is penalised if it predicts pred[i] <= pred[j].

    This directly optimises the ranking capability of the model —
    matching how humans perceive attractiveness comparatively, not absolutely.

    margin: minimum score difference to form a valid (hard) pair.
            Pairs where |target_i - target_j| < margin are skipped.
    """
    pred   = pred.squeeze()
    target = target.squeeze()
    B = pred.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=pred.device)

    # Sample all pairs; filter by margin
    idx_i = torch.arange(B, device=pred.device).unsqueeze(1).expand(B, B).reshape(-1)
    idx_j = torch.arange(B, device=pred.device).unsqueeze(0).expand(B, B).reshape(-1)

    diff_target = target[idx_i] - target[idx_j]
    valid = diff_target > margin     # target_i is strictly better than target_j

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred_i  = pred[idx_i[valid]]
    pred_j  = pred[idx_j[valid]]
    # We want pred_i > pred_j; MarginRankingLoss penalises when it's not
    y_sign  = torch.ones_like(pred_i)   # +1 means first arg should be larger
    return nn.functional.margin_ranking_loss(pred_i, pred_j, y_sign, margin=0.0)


def combined_loss(pred: torch.Tensor,
                  target: torch.Tensor,
                  alpha: float = 0.5,
                  beta: float  = 0.2) -> torch.Tensor:
    """
    alpha * MSE + (1-alpha-beta) * Pearson-loss + beta * Pairwise-Ranking-loss.

    Default weights:
        alpha = 0.50  → MSE     (absolute accuracy)
        beta  = 0.20  → Ranking (pairwise ordering)
        rest  = 0.30  → Pearson (correlation quality)

    Pearson and Ranking together make the model optimise *how well it orders*
    faces relative to each other, not just minimise absolute number errors.
    """
    mse   = nn.functional.mse_loss(pred, target)
    p_l   = pearson_loss(pred.squeeze(), target.squeeze())
    rank_l = pairwise_ranking_loss(pred, target)
    gamma = max(0.0, 1.0 - alpha - beta)
    return alpha * mse + gamma * p_l + beta * rank_l


# ─── Model ────────────────────────────────────────────────────────────────────

class FaceRatingMLP(nn.Module):
    """
    4-layer MLP with LayerNorm + GELU + Dropout.

    Architecture (input_dim -> 512 -> 256 -> 64 -> 1):
    - LayerNorm: stable normalisation per-sample (no batch-size dependency)
    - GELU: smooth gradient flow, better than ReLU for regression
    - Dropout: regularisation (0.35 on first layer, 0.20 on second)
    - Final linear: no activation — free to predict any real value
    """

    def __init__(self, input_dim: int = 427):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(256, 64),
            nn.GELU(),

            nn.Linear(64, 1),
        )
        # Initialise last layer near zero to avoid large initial predictions
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Metrics helper ───────────────────────────────────────────────────────────

def evaluate(model, X_t, y_t, device, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(X_t.to(device)).cpu().numpy().squeeze()
        trues = y_t.cpu().numpy().squeeze()
    mse  = float(np.mean((preds - trues) ** 2))
    mae  = float(mean_absolute_error(trues, preds))
    r, _ = pearsonr(trues, preds)
    return mse, mae, float(r), preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Face Rating MLP")
    parser.add_argument("--dataset",    default="data/SCUT-FBP5500/dataset.npz")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="MSE weight (0-1). Remaining split between Pearson and Ranking.")
    parser.add_argument("--beta",       type=float, default=0.2,
                        help="Pairwise ranking loss weight.")
    parser.add_argument("--out_dir",    default="weights")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        print("Please run src/scripts/extract_features.py first.")
        return

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading {args.dataset}…")
    data = np.load(args.dataset)
    X    = data["X"].astype(np.float32)
    y    = data["y"].astype(np.float32)

    # SCUT scores are 1–5 → rescale to 0–10
    y = ((y - 1.0) / 4.0) * 10.0
    print(f"  X: {X.shape} | y: [{y.min():.2f}, {y.max():.2f}]  mean={y.mean():.2f}")

    # ── 3-way stratified split (70 / 15 / 15) ───────────────────────────────
    # Stratify by quantile bins to preserve score distribution in each split
    bins = np.digitize(y, np.percentile(y, [20, 40, 60, 80]))
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=bins, random_state=42
    )
    bins_tv = np.digitize(y_tv, np.percentile(y_tv, [20, 40, 60, 80]))
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15 of total
        stratify=bins_tv, random_state=42
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Feature standardisation ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    os.makedirs(args.out_dir, exist_ok=True)
    scaler_path = os.path.join(args.out_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {scaler_path}")

    # ── Tensors ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    X_tr_t  = to_tensor(X_train)
    y_tr_t  = to_tensor(y_train).unsqueeze(1)
    X_val_t = to_tensor(X_val)
    y_val_t = to_tensor(y_val).unsqueeze(1)
    X_te_t  = to_tensor(X_test)
    y_te_t  = to_tensor(y_test).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model     = FaceRatingMLP(input_dim=X.shape[1]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10, min_lr=1e-5, verbose=True
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_pearson = -1.0
    best_model_path  = os.path.join(args.out_dir, "face_rater_v1.pt")
    csv_path         = os.path.join(args.out_dir, "training_curves.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_mse", "val_mae", "val_pearson", "lr"])

    print(f"\nTraining for {args.epochs} epochs…")
    for epoch in range(1, args.epochs + 1):
        # ── train step ───────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            optimiser.zero_grad()
            pred = model(bX)
            loss = combined_loss(pred, by, alpha=args.alpha, beta=args.beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            running_loss += loss.item() * bX.size(0)
        train_loss = running_loss / len(loader.dataset)

        # ── validation step ───────────────────────────────────────────────
        val_mse, val_mae, val_r, _ = evaluate(model, X_val_t, y_val_t, device, combined_loss)
        scheduler.step(val_mse)

        # ── checkpoint on best Pearson r ──────────────────────────────────
        if val_r > best_val_pearson:
            best_val_pearson = val_r
            torch.save(model.state_dict(), best_model_path)

        # ── log ──────────────────────────────────────────────────────────
        lr_now = optimiser.param_groups[0]["lr"]
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}",
                             f"{val_mse:.4f}", f"{val_mae:.4f}",
                             f"{val_r:.4f}", f"{lr_now:.2e}"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_mse={val_mse:.4f} | val_mae={val_mae:.4f} | "
                  f"val_r={val_r:.4f} | lr={lr_now:.2e}"
                  f"{'  ← best' if val_r == best_val_pearson else ''}")

    # ── final test set evaluation (touched only once) ────────────────────────
    print("\nLoading best model for final test evaluation…")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_mse, test_mae, test_r, _ = evaluate(model, X_te_t, y_te_t, device, combined_loss)

    print("\n" + "=" * 55)
    print("Training complete!")
    print(f"  Best model : {best_model_path}")
    print(f"  Val Pearson: {best_val_pearson:.4f}  (best during training)")
    print(f"  Test MSE   : {test_mse:.4f}")
    print(f"  Test MAE   : {test_mae:.4f}")
    print(f"  Test Pearson r: {test_r:.4f}")
    print(f"  Curves CSV : {csv_path}")
    print("=" * 55)

    # Save metadata for inference
    meta = {
        "input_dim":      int(X.shape[1]),
        "score_range":    [0.0, 10.0],
        "scut_raw_range": [1.0, 5.0],
        "val_pearson":    round(best_val_pearson, 4),
        "test_pearson":   round(test_r, 4),
        "test_mae":       round(test_mae, 4),
    }
    import json
    meta_path = os.path.join(args.out_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata   : {meta_path}")


if __name__ == "__main__":
    main()
