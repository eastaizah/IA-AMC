#!/usr/bin/env python3
"""
generate_dataset.py — RadioML-style I/Q dataset generator for AMC simulations.

Generates synthetic I/Q samples for 11 modulation classes over a configurable
SNR range, with optional Rayleigh fading and carrier frequency offset (CFO).

Modulation classes: BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, GFSK, CPFSK,
                    AM-DSB, AM-SSB, WBFM

Usage:
    python generate_dataset.py                       # default AWGN dataset
    python generate_dataset.py --channel rayleigh     # Rayleigh fading
    python generate_dataset.py --cfo 0.05             # with CFO
    python generate_dataset.py --self-test            # quick validation
"""

import argparse
import os
import sys

import numpy as np
from scipy.signal import hilbert

# ── Modulation catalogue ────────────────────────────────────────────────────
MOD_CLASSES = [
    "BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
    "PAM4", "GFSK", "CPFSK", "AM-DSB", "AM-SSB", "WBFM",
]
NUM_CLASSES = len(MOD_CLASSES)

# ── Constellation maps ──────────────────────────────────────────────────────

def _psk_constellation(M):
    return np.exp(1j * 2 * np.pi * np.arange(M) / M)

def _qam_constellation(M):
    k = int(np.sqrt(M))
    c = np.array([complex(2 * i - k + 1, 2 * j - k + 1)
                  for i in range(k) for j in range(k)])
    return c / np.sqrt(np.mean(np.abs(c) ** 2))

_BPSK = _psk_constellation(2)
_QPSK = _psk_constellation(4)
_8PSK = _psk_constellation(8)
_16QAM = _qam_constellation(16)
_64QAM = _qam_constellation(64)
_PAM4 = np.array([-3, -1, 1, 3], dtype=complex) / np.sqrt(5)


# ── Modulation signal generators ────────────────────────────────────────────

def _generate_digital_mod(constellation, num_symbols, sps=8):
    """Generate baseband I/Q with root-raised-cosine-like pulse shaping."""
    idx = np.random.randint(0, len(constellation), num_symbols)
    syms = constellation[idx]
    # Simple pulse shaping via upsampling + sinc-like filter
    up = np.zeros(num_symbols * sps, dtype=complex)
    up[::sps] = syms
    # Gaussian pulse shaping
    t = np.arange(-4 * sps, 4 * sps + 1)
    h = np.exp(-0.5 * (t / (sps / 2.5)) ** 2)
    h = h / np.sum(h)
    sig = np.convolve(up, h, mode="same")
    return sig / (np.sqrt(np.mean(np.abs(sig) ** 2)) + 1e-12)


def _generate_gfsk(num_symbols, sps=8, h_mod=0.5, bt=0.3):
    """Gaussian FSK signal generation."""
    bits = np.random.randint(0, 2, num_symbols) * 2 - 1  # ±1
    up = np.repeat(bits, sps).astype(float)
    # Gaussian filter
    t = np.arange(-2 * sps, 2 * sps + 1)
    gauss = np.exp(-2 * (np.pi * bt * t / sps) ** 2)
    gauss = gauss / np.sum(gauss)
    filtered = np.convolve(up, gauss, mode="same")
    phase = np.cumsum(filtered) * np.pi * h_mod / sps
    sig = np.exp(1j * phase)
    return sig / (np.sqrt(np.mean(np.abs(sig) ** 2)) + 1e-12)


def _generate_cpfsk(num_symbols, sps=8, h_mod=0.5):
    """Continuous-phase FSK signal generation."""
    bits = np.random.randint(0, 2, num_symbols) * 2 - 1
    up = np.repeat(bits, sps).astype(float)
    phase = np.cumsum(up) * np.pi * h_mod / sps
    sig = np.exp(1j * phase)
    return sig / (np.sqrt(np.mean(np.abs(sig) ** 2)) + 1e-12)


def _generate_am_dsb(length):
    """AM-DSB: double-sideband amplitude modulation."""
    t = np.arange(length) / length
    f_msg = np.random.uniform(0.02, 0.08)
    msg = np.sin(2 * np.pi * f_msg * np.arange(length))
    m_idx = np.random.uniform(0.3, 0.9)
    carrier_f = np.random.uniform(0.2, 0.35)
    carrier = np.cos(2 * np.pi * carrier_f * np.arange(length))
    sig = (1 + m_idx * msg) * carrier
    sig_c = sig.astype(complex)
    return sig_c / (np.sqrt(np.mean(np.abs(sig_c) ** 2)) + 1e-12)


def _generate_am_ssb(length):
    """AM-SSB: single-sideband via Hilbert-like filtering."""
    t = np.arange(length) / length
    f_msg = np.random.uniform(0.02, 0.08)
    msg = np.sin(2 * np.pi * f_msg * np.arange(length))
    # Hilbert transform of message (approximation)
    from scipy.signal import hilbert as _hilbert
    analytic = _hilbert(msg)
    msg_h = np.imag(analytic)
    carrier_f = np.random.uniform(0.2, 0.35)
    carrier_i = np.cos(2 * np.pi * carrier_f * np.arange(length))
    carrier_q = np.sin(2 * np.pi * carrier_f * np.arange(length))
    sig = msg * carrier_i - msg_h * carrier_q
    sig_c = sig.astype(complex)
    return sig_c / (np.sqrt(np.mean(np.abs(sig_c) ** 2)) + 1e-12)


def _generate_wbfm(length, sps=8):
    """Wideband FM signal generation."""
    f_msg = np.random.uniform(0.01, 0.05)
    msg = np.sin(2 * np.pi * f_msg * np.arange(length))
    kf = np.random.uniform(0.1, 0.3)
    phase = 2 * np.pi * kf * np.cumsum(msg) / length
    sig = np.exp(1j * phase)
    return sig / (np.sqrt(np.mean(np.abs(sig) ** 2)) + 1e-12)


# ── Main generation functions ───────────────────────────────────────────────

def generate_modulated_signal(mod_type, num_samples=128):
    """Generate a single unit-power I/Q signal for *mod_type*."""
    sps = 8
    num_symbols = num_samples // sps + 16  # extra for filter transients

    generators = {
        "BPSK":   lambda: _generate_digital_mod(_BPSK, num_symbols, sps),
        "QPSK":   lambda: _generate_digital_mod(_QPSK, num_symbols, sps),
        "8PSK":   lambda: _generate_digital_mod(_8PSK, num_symbols, sps),
        "16QAM":  lambda: _generate_digital_mod(_16QAM, num_symbols, sps),
        "64QAM":  lambda: _generate_digital_mod(_64QAM, num_symbols, sps),
        "PAM4":   lambda: _generate_digital_mod(_PAM4, num_symbols, sps),
        "GFSK":   lambda: _generate_gfsk(num_symbols, sps),
        "CPFSK":  lambda: _generate_cpfsk(num_symbols, sps),
        "AM-DSB": lambda: _generate_am_dsb(num_samples),
        "AM-SSB": lambda: _generate_am_ssb(num_samples),
        "WBFM":   lambda: _generate_wbfm(num_samples, sps),
    }

    sig = generators[mod_type]()
    return sig[:num_samples]


def apply_channel(signal, snr_db, channel="awgn", cfo=0.0):
    """Apply channel impairments: AWGN, optional Rayleigh fading, and CFO."""
    sig = signal.copy()

    # Carrier frequency offset
    if cfo > 0:
        n = np.arange(len(sig))
        sig = sig * np.exp(1j * 2 * np.pi * cfo * n)

    # Multi-tap Rayleigh channel with exponential PDP (Article eq. 43-44)
    if channel == "rayleigh":
        L_ch = 8
        sigma_tau = 3.0
        pdp = np.exp(-np.arange(L_ch) / sigma_tau)
        pdp = pdp / np.sum(pdp)  # normalize to unit power
        h = np.sqrt(pdp) * (np.random.randn(L_ch) + 1j * np.random.randn(L_ch)) / np.sqrt(2)
        sig = np.convolve(sig, h, mode='same')

    # AWGN
    sig_power = np.mean(np.abs(sig) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
    )
    sig = sig + noise

    return sig


def generate_dataset(
    num_per_class=1000,
    snr_range=None,
    num_samples=128,
    channel="awgn",
    cfo=0.0,
    seed=42,
):
    """
    Generate a full dataset of I/Q samples.

    Returns
    -------
    X : ndarray, shape (N, 2, num_samples)  — I and Q components
    y : ndarray, shape (N,)                 — integer class labels
    snrs : ndarray, shape (N,)              — per-sample SNR
    """
    if snr_range is None:
        snr_range = list(range(-20, 20, 2))

    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    X_list, y_list, snr_list = [], [], []

    for snr_db in snr_range:
        for cls_idx, mod in enumerate(MOD_CLASSES):
            for _ in range(num_per_class):
                sig = generate_modulated_signal(mod, num_samples)
                sig = apply_channel(sig, snr_db, channel, cfo)
                iq = np.stack([sig.real, sig.imag], axis=0)  # (2, num_samples)
                X_list.append(iq)
                y_list.append(cls_idx)
                snr_list.append(snr_db)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    snrs = np.array(snr_list, dtype=np.float32)

    # Per-sample I/Q normalization (Article eq. 38)
    X = normalize_iq(X)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm], snrs[perm]


def normalize_iq(X):
    """Per-sample normalization: zero mean, unit variance per I/Q channel."""
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (X - mean) / std


# ── Data augmentation (Article §V-A) ───────────────────────────────────────

def augment_circular_shift(X, max_shift=8):
    """Apply random circular time shifts to each sample."""
    N = X.shape[0]
    shifts = np.random.randint(-max_shift, max_shift + 1, size=N)
    X_aug = np.empty_like(X)
    for i in range(N):
        X_aug[i] = np.roll(X[i], shifts[i], axis=1)
    return X_aug


def augment_noise(X, sigma=0.01):
    """Inject small Gaussian noise into each sample."""
    return X + np.random.randn(*X.shape).astype(np.float32) * sigma


def apply_training_augmentation(X, max_shift=8, noise_sigma=0.01):
    """Apply all training-time data augmentations."""
    X = augment_circular_shift(X, max_shift)
    X = augment_noise(X, noise_sigma)
    return X


# ── Self-test ───────────────────────────────────────────────────────────────

def self_test():
    """Quick sanity check: generate a small dataset and verify shapes."""
    print("Running self-test …")
    X, y, snrs = generate_dataset(num_per_class=10, seed=0)
    n_expected = 10 * NUM_CLASSES * 20  # 10 per class, 11 classes, 20 SNRs
    assert X.shape == (n_expected, 2, 128), f"Bad X shape: {X.shape}"
    assert y.shape == (n_expected,), f"Bad y shape: {y.shape}"
    assert snrs.shape == (n_expected,), f"Bad snrs shape: {snrs.shape}"
    assert set(y.tolist()) == set(range(NUM_CLASSES)), "Missing classes"
    print(f"  X shape: {X.shape}")
    print(f"  y range: {y.min()}–{y.max()}")
    print(f"  SNR range: {snrs.min():.0f} – {snrs.max():.0f} dB")

    # Test channel variants
    for ch in ["awgn", "rayleigh"]:
        Xc, _, _ = generate_dataset(num_per_class=5, channel=ch, seed=1)
        assert Xc.shape[0] == 5 * NUM_CLASSES * 20
        print(f"  Channel '{ch}': OK ({Xc.shape[0]} samples)")

    # Test CFO
    Xf, _, _ = generate_dataset(num_per_class=5, cfo=0.1, seed=2)
    print(f"  CFO=0.1: OK ({Xf.shape[0]} samples)")

    # Test augmentation
    X_aug = apply_training_augmentation(X, max_shift=4, noise_sigma=0.01)
    assert X_aug.shape == X.shape, f"Augmentation changed shape: {X_aug.shape}"
    print(f"  Augmentation: OK")

    print("Self-test PASSED ✓")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate RadioML-style I/Q dataset")
    parser.add_argument("--num-per-class", type=int, default=500,
                        help="Samples per class per SNR (default 500)")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="I/Q samples per observation (default 128)")
    parser.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    parser.add_argument("--cfo", type=float, default=0.0,
                        help="Carrier frequency offset Δf/fs (default 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true",
                        help="Run quick validation and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    out = args.output or f"dataset_{args.channel}_cfo{args.cfo:.2f}.npz"
    print(f"Generating dataset: channel={args.channel}, CFO={args.cfo}, "
          f"{args.num_per_class} per class …")
    X, y, snrs = generate_dataset(
        num_per_class=args.num_per_class,
        num_samples=args.num_samples,
        channel=args.channel,
        cfo=args.cfo,
        seed=args.seed,
    )
    np.savez_compressed(out, X=X, y=y, snrs=snrs, classes=MOD_CLASSES)
    print(f"Saved {out}  ({X.shape[0]} samples, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
