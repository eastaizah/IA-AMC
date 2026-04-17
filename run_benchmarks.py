#!/usr/bin/env python3
"""
run_benchmarks.py — Train all architectures and generate comparative tables.

Produces JSON result files consumed by the plotting scripts to recreate
every table and figure in the article.

Usage:
    python run_benchmarks.py                  # full benchmark (all models)
    python run_benchmarks.py --models MS-SENet CNN-4Layer
    python run_benchmarks.py --quick          # reduced dataset for testing
    python run_benchmarks.py --self-test      # minimal smoke test
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from train import train_model
from models import MODEL_REGISTRY, get_model, count_parameters
from generate_dataset import generate_dataset, MOD_CLASSES, NUM_CLASSES


# ── Channel robustness evaluation ──────────────────────────────────────────

def _batched_predict(model, X_np, device, batch_size=256):
    """Run inference in batches to avoid GPU OOM on large datasets."""
    import torch
    all_preds = []
    n = len(X_np)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = torch.tensor(X_np[start:end]).to(device)
        with torch.no_grad():
            preds_b = model(Xb).argmax(1).cpu().numpy()
        all_preds.append(preds_b)
        del Xb
    return np.concatenate(all_preds)


def evaluate_channel_robustness(model_name, output_dir="results", device=None,
                                num_per_class=500, eval_batch_size=256):
    """Evaluate pre-trained model on Rayleigh channel and various CFO levels.

    Uses batched inference and explicit GPU memory cleanup to stay within
    8 GB VRAM budgets.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wt_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.pt")
    if not os.path.exists(wt_path):
        print(f"  [SKIP] No weights found for {model_name}")
        return None

    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(wt_path, map_location=device,
                                     weights_only=True))
    model.eval()

    results = {"model": model_name}

    # ── Rayleigh channel (SNR=10 dB) ────────────────────────────────────
    print(f"  Evaluating {model_name} on Rayleigh channel …")
    X_ray, y_ray, snrs_ray = generate_dataset(
        num_per_class=num_per_class, channel="rayleigh", seed=99)
    mask_10 = snrs_ray == 10.0
    X_sub, y_sub = X_ray[mask_10], y_ray[mask_10]
    preds = _batched_predict(model, X_sub, device, eval_batch_size)
    ray_acc = round(100.0 * (preds == y_sub).mean(), 1)
    results["rayleigh_snr10_acc"] = ray_acc
    del X_ray, y_ray, snrs_ray, X_sub, y_sub, preds
    if device != "cpu":
        torch.cuda.empty_cache()

    # ── CFO sweep ───────────────────────────────────────────────────────
    cfo_accs = {}
    for cfo in [0.0, 0.05, 0.10, 0.15, 0.20]:
        print(f"  Evaluating {model_name} with CFO={cfo:.2f} …")
        X_cfo, y_cfo, snrs_cfo = generate_dataset(
            num_per_class=num_per_class, cfo=cfo, seed=77)
        mask = snrs_cfo >= 10.0
        X_sub, y_sub = X_cfo[mask], y_cfo[mask]
        preds = _batched_predict(model, X_sub, device, eval_batch_size)
        cfo_accs[str(cfo)] = round(100.0 * (preds == y_sub).mean(), 1)
        del X_cfo, y_cfo, snrs_cfo, X_sub, y_sub, preds
        if device != "cpu":
            torch.cuda.empty_cache()
    results["cfo_accuracy"] = cfo_accs

    # Clean up model from GPU
    model.cpu()
    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return results


# ── Memory profiling ────────────────────────────────────────────────────────

def profile_memory(model_name, device="cpu", profiling_batch=64):
    """Estimate GPU memory and peak activations for a training batch."""
    import torch
    model = get_model(model_name).to(device)
    total_p, _ = count_parameters(model)

    # Parameter memory (float32 = 4 bytes)
    param_mb = total_p * 4 / 1e6

    # Forward pass activation estimate via hooks
    activation_sizes = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            activation_sizes.append(out.nelement() * out.element_size())
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    activation_sizes.append(o.nelement() * o.element_size())
                elif isinstance(o, (tuple, list)):
                    for oo in o:
                        if isinstance(oo, torch.Tensor):
                            activation_sizes.append(oo.nelement() * oo.element_size())

    hooks = []
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook_fn))

    x = torch.randn(profiling_batch, 2, 128).to(device)
    model.eval()
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    # Peak activations in MB (for the profiling batch)
    peak_act_mb = sum(activation_sizes) / 1e6

    # Training memory overhead:
    # gradients + Adam optimizer (2x params) + framework overhead
    gradient_mb = param_mb
    optimizer_mb = param_mb * 2
    overhead_mb = 5.0

    total_mb = round(peak_act_mb + gradient_mb + optimizer_mb + overhead_mb, 2)

    return {
        "model": model_name,
        "parameters_K": round(total_p / 1000, 1),
        "disk_size_MB": round(param_mb, 2),
        "gpu_memory_MB": total_mb,
        "peak_activations_MB": round(peak_act_mb, 2),
    }


# ── Main benchmark pipeline ────────────────────────────────────────────────

def _gpu_cleanup():
    """Release GPU memory between phases / model evaluations."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_full_benchmark(
    models=None,
    epochs=60,
    num_per_class=500,
    output_dir="results",
    quick=False,
):
    """Run complete benchmark: train all models, evaluate robustness, profile."""
    if models is None:
        models = list(MODEL_REGISTRY.keys())

    if quick:
        epochs = 8
        num_per_class = 50
        print("⚡ Quick mode: 8 epochs, 50 samples/class/SNR")

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # Phase 1: Train all models
    print("=" * 60)
    print("Phase 1: Training all models")
    print("=" * 60)
    for name in models:
        print(f"\n{'─' * 40}")
        print(f"Training: {name}")
        print(f"{'─' * 40}")
        res = train_model(
            name, epochs=epochs, num_per_class=num_per_class,
            output_dir=output_dir,
        )
        all_results[name] = res
        _gpu_cleanup()

    # Phase 2: Channel robustness
    print("\n" + "=" * 60)
    print("Phase 2: Channel robustness evaluation")
    print("=" * 60)
    _gpu_cleanup()
    robustness = {}
    key_models = [m for m in ["CNN-4Layer", "LSTM-2Layer", "CNN-LSTM",
                               "ResNet-LSTM", "MS-SENet"] if m in models]
    for name in key_models:
        rob = evaluate_channel_robustness(
            name, output_dir=output_dir,
            num_per_class=num_per_class)
        if rob:
            robustness[name] = rob
        _gpu_cleanup()

    # Phase 3: Memory profiling (always on CPU -- safe)
    print("\n" + "=" * 60)
    print("Phase 3: Memory profiling")
    print("=" * 60)
    memory = {}
    for name in models:
        mem = profile_memory(name)
        memory[name] = mem
        print(f"  {name:15s} | {mem['parameters_K']:>7.1f}K params | "
              f"{mem['disk_size_MB']:.2f} MB")

    # ── Aggregate and save ──────────────────────────────────────────────
    summary = {
        "models": list(all_results.keys()),
        "accuracy": {k: v["test_accuracy"] for k, v in all_results.items()},
        "per_snr": {k: v["per_snr_accuracy"] for k, v in all_results.items()},
        "per_mod": {k: v["per_modulation_accuracy"] for k, v in all_results.items()},
        "latency": {k: v["latency_ms"] for k, v in all_results.items()},
        "memory": memory,
        "robustness": robustness,
    }

    with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {output_dir}/benchmark_summary.json")

    # ── Print comparative table ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARATIVE RESULTS")
    print("=" * 60)
    print(f"{'Model':15s} | {'SNR<0':>7s} | {'0≤SNR<10':>8s} | "
          f"{'SNR≥10':>7s} | {'Avg':>6s} | {'Lat(ms)':>7s}")
    print("-" * 60)

    for name in models:
        if name not in all_results:
            continue
        ps = all_results[name]["per_snr_accuracy"]
        low = np.mean([v for k, v in ps.items() if float(k) < 0])
        mid = np.mean([v for k, v in ps.items() if 0 <= float(k) < 10])
        high = np.mean([v for k, v in ps.items() if float(k) >= 10])
        avg = np.mean(list(ps.values()))
        lat = all_results[name]["latency_ms"]
        print(f"{name:15s} | {low:7.1f} | {mid:8.1f} | {high:7.1f} | "
              f"{avg:6.1f} | {lat:7.2f}")

    return summary


# ── Self-test ───────────────────────────────────────────────────────────────

def self_test():
    """Minimal smoke test: 2 models, 2 epochs, tiny dataset."""
    print("Running benchmark self-test …")
    summary = run_full_benchmark(
        models=["CNN-2Layer", "MS-SENet"],
        epochs=2,
        num_per_class=20,
        output_dir="/tmp/deepamc_bench_test",
    )
    assert len(summary["models"]) == 2
    for m in summary["models"]:
        assert summary["accuracy"][m] > 0
    print("\nBenchmark self-test PASSED ✓")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run AMC benchmarks")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to benchmark (default: all)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num-per-class", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer epochs and samples")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_full_benchmark(
        models=args.models, epochs=args.epochs,
        num_per_class=args.num_per_class,
        output_dir=args.output_dir, quick=args.quick,
    )


if __name__ == "__main__":
    main()
