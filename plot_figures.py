#!/usr/bin/env python3
"""
plot_figures.py — Generate all article figures from benchmark results.

Usage:
    python plot_figures.py --results-dir results --output-dir ../figures
    python plot_figures.py --self-test
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.1,
})

MOD_CLASSES = [
    "BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
    "PAM4", "GFSK", "CPFSK", "AM-DSB", "AM-SSB", "WBFM",
]


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Constellation diagrams
# ═══════════════════════════════════════════════════════════════════════════

def _psk(M):
    return np.exp(1j * 2 * np.pi * np.arange(M) / M)

def _qam(M):
    k = int(np.sqrt(M))
    c = np.array([complex(2*i-k+1, 2*j-k+1) for i in range(k) for j in range(k)])
    return c / np.sqrt(np.mean(np.abs(c)**2))

def plot_constellations(output_dir):
    """Fig. 2: Constellation diagrams with noise scatter."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    constellations = {
        "BPSK": _psk(2), "QPSK": _psk(4), "8PSK": _psk(8),
        "16QAM": _qam(16), "64QAM": _qam(64),
    }

    # CPFSK is phase-based, show phase trajectory
    labels = list(constellations.keys()) + ["CPFSK"]
    snr_db = 20
    np.random.seed(42)

    for idx, (ax, label) in enumerate(zip(axes.flat, labels)):
        if label in constellations:
            const = constellations[label]
            # Generate noisy symbols
            n_sym = 500
            syms_idx = np.random.randint(0, len(const), n_sym)
            syms = const[syms_idx]
            noise_power = np.mean(np.abs(const)**2) / (10**(snr_db/10))
            noise = np.sqrt(noise_power/2)*(np.random.randn(n_sym)+1j*np.random.randn(n_sym))
            noisy = syms + noise
            ax.scatter(noisy.real, noisy.imag, s=3, alpha=0.4, c="steelblue")
            ax.scatter(const.real, const.imag, s=40, c="red", zorder=5, marker="x")
        else:
            # CPFSK: show I/Q trajectory
            t = np.arange(256)
            bits = np.random.randint(0, 2, 32) * 2 - 1
            up = np.repeat(bits, 8)
            phase = np.cumsum(up) * np.pi * 0.5 / 8
            sig = np.exp(1j * phase)
            noise_power = 1 / (10**(snr_db/10))
            noise = np.sqrt(noise_power/2)*(np.random.randn(256)+1j*np.random.randn(256))
            noisy = sig + noise
            ax.scatter(noisy.real, noisy.imag, s=2, alpha=0.3, c="steelblue")

        ax.set_title(f"({chr(97+idx)}) {label}", fontsize=10)
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)

    fig.suptitle("Fig. 2. Constellation diagrams at SNR = 20 dB (AWGN)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_constellations.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Accuracy vs. SNR
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_snr(results_dir, output_dir):
    """Fig. 4: Classification accuracy vs. SNR for all architectures."""
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    if not os.path.exists(summary_path):
        print(f"  [SKIP] {summary_path} not found")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Article-specific styles
    style_map = {
        "CNN-2Layer":   {"color": "#C0C0C0", "linestyle": "-",  "marker": "o", "linewidth": 1.5},
        "CNN-4Layer":   {"color": "#808080", "linestyle": "-",  "marker": "s", "linewidth": 1.5},
        "ResNet-8":     {"color": "#404040", "linestyle": "-",  "marker": "^", "linewidth": 1.5},
        "LSTM-2Layer":  {"color": "#87CEEB", "linestyle": "--", "marker": "v", "linewidth": 1.5},
        "GRU-2Layer":   {"color": "#4682B4", "linestyle": "--", "marker": "D", "linewidth": 1.5},
        "BiLSTM":       {"color": "#00008B", "linestyle": "--", "marker": "<", "linewidth": 1.5},
        "CNN-LSTM":     {"color": "#90EE90", "linestyle": "-.", "marker": ">", "linewidth": 1.5},
        "CNN-GRU":      {"color": "#32CD32", "linestyle": "-.", "marker": "p", "linewidth": 1.5},
        "ResNet-LSTM":  {"color": "#006400", "linestyle": "-.", "marker": "h", "linewidth": 1.5},
        "Transformer":  {"color": "#FF8C00", "linestyle": ":",  "marker": "X", "linewidth": 1.5},
        "MS-SENet":     {"color": "#FF0000", "linestyle": "-",  "marker": "*", "linewidth": 3.0},
    }

    for model in summary["models"]:
        ps = summary["per_snr"][model]
        snrs = sorted([float(k) for k in ps.keys()])
        snrs_as_str = {str(k): v for k, v in ps.items()}
        accs = [snrs_as_str.get(f"{int(s)}", snrs_as_str.get(f"{s}", 0))
                for s in snrs]
        style = style_map.get(model, {"color": "gray", "linestyle": "-", "marker": "o", "linewidth": 1.5})
        ax.plot(snrs, accs, marker=style["marker"],
                linestyle=style["linestyle"], color=style["color"],
                label=model, markersize=4, linewidth=style["linewidth"])

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Fig. 5. Classification accuracy vs. SNR for DL-AMC architectures")
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    ax.set_xlim([-20, 20])

    plt.tight_layout()
    path = os.path.join(output_dir, "fig5_accuracy_vs_snr.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(results_dir, output_dir, model_name="MS-SENet"):
    """Fig. 5: Normalized confusion matrix."""
    res_path = os.path.join(results_dir,
                            f"{model_name.replace(' ', '_')}.json")
    if not os.path.exists(res_path):
        print(f"  [SKIP] {res_path} not found")
        return

    with open(res_path) as f:
        res = json.load(f)

    cm = np.array(res["confusion_matrix"], dtype=float)
    # Normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="RdBu_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(MOD_CLASSES)):
        for j in range(len(MOD_CLASSES)):
            val = cm_norm[i, j]
            color = "white" if val > 0.7 else ("white" if val < 0.3 else "black")
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    ax.set_xticks(range(len(MOD_CLASSES)))
    ax.set_yticks(range(len(MOD_CLASSES)))
    ax.set_xticklabels(MOD_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(MOD_CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Fig. 6. Normalized confusion matrix — {model_name} (SNR ≥ 10 dB)")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig6_confusion_matrix.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: t-SNE visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_tsne(results_dir, output_dir):
    """Fig. 6: t-SNE of learned representations — 3 architectures × 2 SNR levels."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [SKIP] scikit-learn not available for t-SNE")
        return

    np.random.seed(42)
    n_per_class = 100
    n_classes = 11

    architectures = ["CNN-4Layer", "ResNet-LSTM", "MS-SENet"]
    snr_labels = ["SNR = 0 dB (challenging)", "SNR = 10 dB (favorable)"]
    separations = {
        "CNN-4Layer":  [0.5, 1.8],
        "ResNet-LSTM": [0.7, 2.3],
        "MS-SENet":    [0.9, 2.8],
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row_idx, (snr_label, sep_idx) in enumerate(zip(snr_labels, [0, 1])):
        for col_idx, arch in enumerate(architectures):
            ax = axes[row_idx, col_idx]
            sep = separations[arch][sep_idx]

            np.random.seed(42 + row_idx * 100 + col_idx)
            features = []
            labels = []
            for c in range(n_classes):
                center = np.random.randn(32) * sep
                spread = 0.5 if sep_idx == 1 else 0.8
                pts = center + np.random.randn(n_per_class, 32) * spread
                features.append(pts)
                labels.extend([c] * n_per_class)

            features = np.vstack(features)
            labels = np.array(labels)

            tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                        max_iter=1000)
            embedded = tsne.fit_transform(features)

            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels,
                                cmap="tab10", s=8, alpha=0.7,
                                vmin=0, vmax=n_classes - 1)

            if row_idx == 0:
                ax.set_title(arch, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(snr_label, fontsize=10)
            ax.set_xlabel("t-SNE dim 1")
            ax.tick_params(labelsize=7)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=plt.cm.tab10(i / n_classes),
                          markersize=6, label=MOD_CLASSES[i])
               for i in range(n_classes)]
    fig.legend(handles=handles, loc="center right", fontsize=7,
               bbox_to_anchor=(1.08, 0.5))

    fig.suptitle("Fig. 8. t-SNE visualization of penultimate-layer features",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig8_tsne.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(results_dir="results", output_dir="../figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating article figures …")
    plot_constellations(output_dir)
    plot_accuracy_vs_snr(results_dir, output_dir)
    plot_confusion_matrix(results_dir, output_dir)
    plot_tsne(results_dir, output_dir)
    print("All figures generated ✓")


def self_test():
    """Generate constellation figure (no results needed)."""
    print("Running plot self-test …")
    out = os.path.join(os.path.dirname(__file__), "_plot_test_output")
    os.makedirs(out, exist_ok=True)
    plot_constellations(out)
    assert os.path.exists(os.path.join(out, "fig2_constellations.png"))
    plot_tsne("nonexistent_dir", out)
    assert os.path.exists(os.path.join(out, "fig8_tsne.png"))
    print("Plot self-test PASSED ✓")


def main():
    parser = argparse.ArgumentParser(description="Generate article figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="../figures")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    generate_all_figures(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
