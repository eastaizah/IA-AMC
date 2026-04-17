#!/usr/bin/env python3
"""
train.py — Train and evaluate DL models for Automatic Modulation Classification.

Supports all architectures in models.py.  Produces per-SNR accuracy tables,
confusion matrices, and training curves consumed by the plotting scripts.

Usage:
    python train.py --model MS-SENet --epochs 60 --batch-size 256
    python train.py --model CNN-4Layer --epochs 40
    python train.py --self-test          # quick 2-epoch smoke test
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

from generate_dataset import (
    generate_dataset, MOD_CLASSES, NUM_CLASSES,
    apply_training_augmentation,
)
from models import get_model, count_parameters, MODEL_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════

def make_loaders(X, y, snrs, batch_size=256, val_split=0.15, test_split=0.15):
    """Split data and return train/val/test DataLoaders + test SNR array."""
    n = len(y)
    idx = np.arange(n)
    np.random.seed(0)
    torch.manual_seed(0)
    np.random.shuffle(idx)

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    def _loader(indices, shuffle=False):
        Xt = torch.tensor(X[indices])
        yt = torch.tensor(y[indices])
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    return (
        _loader(train_idx, shuffle=True),
        _loader(val_idx),
        _loader(test_idx),
        snrs[test_idx],
        y[test_idx],
    )


def _augment_batch_tensor(xb):
    """Apply training augmentation to a batch tensor (B, 2, 128)."""
    B, C, L = xb.shape
    device = xb.device
    # Random circular time shift (±8 samples)
    shifts = torch.randint(-8, 9, (B,), device=device)
    xb_shifted = torch.zeros_like(xb)
    for i in range(B):
        xb_shifted[i] = torch.roll(xb[i], shifts[i].item(), dims=-1)
    # Gaussian noise injection (sigma=0.01)
    xb_shifted = xb_shifted + torch.randn_like(xb_shifted) * 0.01
    return xb_shifted


def train_one_epoch(model, loader, criterion, optimizer, device, augment=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if augment:
            xb = _augment_batch_tensor(xb)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_per_snr(model, X_test, y_test, snrs_test, device, batch_size=256):
    """Return dict: snr -> accuracy.  Uses batched inference to limit VRAM."""
    model.eval()
    results = {}
    unique_snrs = sorted(set(snrs_test.tolist()))
    for s in unique_snrs:
        mask = snrs_test == s
        X_sub = X_test[mask]
        y_sub = y_test[mask]
        correct, total = 0, 0
        for start in range(0, len(y_sub), batch_size):
            end = min(start + batch_size, len(y_sub))
            Xb = torch.tensor(X_sub[start:end]).to(device)
            yb = torch.tensor(y_sub[start:end]).to(device)
            correct += (model(Xb).argmax(1) == yb).sum().item()
            total += len(yb)
            del Xb, yb
        results[int(s) if s == int(s) else s] = round(100.0 * correct / total, 1)
    return results


@torch.no_grad()
def get_confusion_matrix(model, X_test, y_test, snrs_test, device,
                         snr_min=10, batch_size=256):
    """Confusion matrix for SNR >= snr_min.  Uses batched inference."""
    model.eval()
    mask = snrs_test >= snr_min
    X_sub = X_test[mask]
    yt = y_test[mask]
    all_preds = []
    for start in range(0, len(yt), batch_size):
        end = min(start + batch_size, len(yt))
        Xb = torch.tensor(X_sub[start:end]).to(device)
        all_preds.append(model(Xb).argmax(1).cpu().numpy())
        del Xb
    preds = np.concatenate(all_preds)
    return confusion_matrix(yt, preds, labels=list(range(NUM_CLASSES)))


@torch.no_grad()
def per_modulation_accuracy(model, X_test, y_test, snrs_test, device,
                            snr_min=10, batch_size=256):
    """Per-class accuracy for SNR >= snr_min.  Uses batched inference."""
    model.eval()
    mask = snrs_test >= snr_min
    X_sub = X_test[mask]
    yt = y_test[mask]
    all_preds = []
    for start in range(0, len(yt), batch_size):
        end = min(start + batch_size, len(yt))
        Xb = torch.tensor(X_sub[start:end]).to(device)
        all_preds.append(model(Xb).argmax(1).cpu().numpy())
        del Xb
    preds = np.concatenate(all_preds)
    accs = {}
    for c in range(NUM_CLASSES):
        c_mask = yt == c
        if c_mask.sum() > 0:
            accs[MOD_CLASSES[c]] = round(100.0 * (preds[c_mask] == c).mean(), 1)
    return accs


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    model_name,
    epochs=60,
    batch_size=256,
    lr=1e-3,
    num_per_class=500,
    device=None,
    output_dir="results",
    verbose=True,
):
    """Train a model and return results dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    if verbose:
        print(f"Generating dataset ({num_per_class} per class per SNR) …")
    X, y, snrs = generate_dataset(num_per_class=num_per_class, seed=42)
    train_ld, val_ld, test_ld, snrs_test, y_test = make_loaders(
        X, y, snrs, batch_size)

    # Keep raw test arrays for per-SNR evaluation
    n = len(y)
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    n_test = int(n * 0.15)
    X_test = X[idx[:n_test]]

    # ── Model ───────────────────────────────────────────────────────────
    model = get_model(model_name).to(device)
    total_p, _ = count_parameters(model)
    if verbose:
        print(f"Model: {model_name}  |  Parameters: {total_p:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=1e-6)

    # ── Training ────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, criterion,
                                          optimizer, device, augment=True)
        va_loss, va_acc = evaluate(model, val_ld, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc * 100, 2))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc * 100, 2))

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if patience_counter >= patience and epoch > 20:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  "
                  f"val_acc={va_acc*100:.1f}%")

    train_time = time.time() - t0
    if verbose:
        print(f"  Training time: {train_time:.1f}s  "
              f"Best val acc: {best_val_acc*100:.1f}%")

    # ── Evaluate best model ─────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)
    _, test_acc = evaluate(model, test_ld, criterion, device)

    per_snr = evaluate_per_snr(model, X_test, y_test, snrs_test, device)
    cm = get_confusion_matrix(model, X_test, y_test, snrs_test, device).tolist()
    per_mod = per_modulation_accuracy(model, X_test, y_test, snrs_test, device)

    # Inference latency (average over 200 iterations with warmup)
    model.eval()
    dummy = torch.randn(1, 2, 128).to(device)
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        times.append(time.perf_counter() - t0)
    latency_ms = round(np.median(times) * 1000, 2)

    # ── Save results ────────────────────────────────────────────────────
    results = {
        "model": model_name,
        "parameters": total_p,
        "test_accuracy": round(test_acc * 100, 2),
        "per_snr_accuracy": per_snr,
        "per_modulation_accuracy": per_mod,
        "confusion_matrix": cm,
        "latency_ms": latency_ms,
        "training_time_s": round(train_time, 1),
        "history": history,
    }

    out_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"  Results saved to {out_path}")

    # Save model weights
    wt_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.pt")
    torch.save(best_state, wt_path)

    # ── GPU cleanup ─────────────────────────────────────────────────────
    model.cpu()
    del model, best_state, X, y, snrs, X_test, dummy
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Self-test (smoke test with 2 epochs)
# ═══════════════════════════════════════════════════════════════════════════

def self_test():
    """Train MS-SENet for 2 epochs on a tiny dataset to verify the pipeline."""
    print("Running training self-test (2 epochs, tiny dataset) …")
    results = train_model(
        "MS-SENet", epochs=2, batch_size=64, num_per_class=20,
        output_dir="/tmp/deepamc_selftest", verbose=True,
    )
    assert results["test_accuracy"] > 0, "Zero accuracy — something is broken"
    # Also quick-test one baseline
    results2 = train_model(
        "CNN-2Layer", epochs=2, batch_size=64, num_per_class=20,
        output_dir="/tmp/deepamc_selftest", verbose=True,
    )
    assert results2["test_accuracy"] > 0
    print("Training self-test PASSED ✓")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train AMC model")
    parser.add_argument("--model", type=str, default="MS-SENet",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-per-class", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    train_model(
        args.model, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_per_class=args.num_per_class,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
