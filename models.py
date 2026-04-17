#!/usr/bin/env python3
"""
models.py — DL model zoo for Automatic Modulation Classification.

Implements baseline and proposed architectures:
  - CNN-2Layer, CNN-4Layer, ResNet-8
  - LSTM-2Layer, GRU-2Layer, BiLSTM
  - CNN-LSTM, CNN-GRU, ResNet-LSTM
  - Transformer
  - **MS-SENet** (proposed Multi-Scale Squeeze-Excitation Network)

All models accept input shape (batch, 2, 128) — I/Q samples.
Output: (batch, num_classes) logits.

Usage:
    python models.py --self-test   # quick validation of all models
"""

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 11
INPUT_LEN = 128
INPUT_CHANNELS = 2


# ═══════════════════════════════════════════════════════════════════════════
# Helper modules
# ═══════════════════════════════════════════════════════════════════════════

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for 1-D feature maps."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L)
        w = self.fc(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


class ResBlock1D(nn.Module):
    """1-D residual block with optional SE attention."""

    def __init__(self, channels, kernel_size=3, use_se=False):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + x)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline models
# ═══════════════════════════════════════════════════════════════════════════

class CNN2Layer(nn.Module):
    """Simple 2-layer CNN baseline."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, 8, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


class CNN4Layer(nn.Module):
    """Deeper 4-layer CNN baseline."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, 8, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


class ResNet8(nn.Module):
    """8-layer ResNet for 1-D signals."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.layer1 = ResBlock1D(64, 3)
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.layer3 = ResBlock1D(128, 3)
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.layer5 = ResBlock1D(256, 3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.classifier(self.pool(x).squeeze(-1))


class LSTM2Layer(nn.Module):
    """2-layer LSTM baseline."""

    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_CHANNELS, hidden, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, 2, 128) -> (B, 128, 2)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class GRU2Layer(nn.Module):
    """2-layer GRU baseline."""

    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.gru = nn.GRU(INPUT_CHANNELS, hidden, num_layers=2,
                          batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


class BiLSTM(nn.Module):
    """Bidirectional LSTM baseline."""

    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_CHANNELS, hidden, num_layers=2,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM architecture."""

    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, 8, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # (B, 128, L')
        x = x.permute(0, 2, 1)   # (B, L', 128)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class CNNGRU(nn.Module):
    """Hybrid CNN-GRU architecture."""

    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, 8, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gru = nn.GRU(128, hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


class ResNetLSTM(nn.Module):
    """Hybrid ResNet-LSTM architecture."""

    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res1 = ResBlock1D(64, 3)
        self.down = nn.Sequential(
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.res2 = ResBlock1D(128, 3)
        self.lstm = nn.LSTM(128, hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class TransformerAMC(nn.Module):
    """Transformer-based AMC with positional encoding."""

    def __init__(self, num_classes=NUM_CLASSES, d_model=64, nhead=4,
                 num_layers=2, dim_ff=256):
        super().__init__()
        self.proj = nn.Linear(INPUT_CHANNELS, d_model)
        self.pos = nn.Parameter(torch.randn(1, INPUT_LEN, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)            # (B, 128, 2)
        x = self.proj(x) + self.pos        # (B, 128, d_model)
        x = self.encoder(x)
        x = x.mean(dim=1)                  # global average pooling
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════
# ★ PROPOSED MODEL: Multi-Scale Squeeze-Excitation Network (MS-SENet)
# ═══════════════════════════════════════════════════════════════════════════

class MultiScaleConv(nn.Module):
    """Multi-scale convolution block: parallel branches with different kernel
    sizes capture both fine-grained and coarse temporal patterns."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 3 == 0, f"out_ch must be divisible by 3, got {out_ch}"
        br = out_ch // 3
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, br, 3, padding=1), nn.BatchNorm1d(br), nn.ReLU())
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, br, 5, padding=2), nn.BatchNorm1d(br), nn.ReLU())
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_ch, br, 7, padding=3), nn.BatchNorm1d(br), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)


class MSSENet(nn.Module):
    """
    Multi-Scale Squeeze-Excitation Network (MS-SENet) for AMC.

    Architecture:
      1. Multi-scale convolution (kernels 3, 5, 7) → captures both fine and
         coarse I/Q patterns simultaneously.
      2. Two SE-enhanced residual blocks → learn channel-wise feature
         importance with skip connections for stable deep training.
      3. Bidirectional LSTM → models long-range temporal dependencies in
         both forward and backward directions.
      4. Global attention pooling → weighted aggregation of LSTM outputs.
      5. Fully-connected classifier with dropout regularization.

    This design specifically addresses the AMC challenges of:
      - Discriminating between high-order modulations (multi-scale features)
      - Robustness under low SNR (SE attention + residual connections)
      - Temporal structure exploitation (BiLSTM)
      - Computational efficiency (moderate model size ~450K params)
    """

    def __init__(self, num_classes=NUM_CLASSES, lstm_hidden=96):
        super().__init__()
        # Stage 1: Multi-scale feature extraction
        self.ms_conv = MultiScaleConv(INPUT_CHANNELS, 96)  # 3 branches × 32
        self.pool1 = nn.MaxPool1d(2)

        # Stage 2: SE-Residual refinement
        self.se_res1 = ResBlock1D(96, 3, use_se=True)
        self.down1 = nn.Sequential(
            nn.Conv1d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.se_res2 = ResBlock1D(128, 3, use_se=True)

        # Stage 3: Bidirectional LSTM
        self.lstm = nn.LSTM(128, lstm_hidden, num_layers=1,
                            batch_first=True, bidirectional=True)

        # Stage 4: Attention pooling
        self.attn_fc = nn.Linear(2 * lstm_hidden, 1)

        # Stage 5: Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 2, 128)
        x = self.ms_conv(x)       # (B, 96, 128)
        x = self.pool1(x)         # (B, 96, 64)
        x = self.se_res1(x)       # (B, 96, 64)
        x = self.down1(x)         # (B, 128, 32)
        x = self.se_res2(x)       # (B, 128, 32)

        # LSTM
        x = x.permute(0, 2, 1)    # (B, 32, 128)
        lstm_out, _ = self.lstm(x)  # (B, 32, 2*hidden)

        # Attention pooling
        attn_w = torch.softmax(self.attn_fc(lstm_out), dim=1)  # (B, 32, 1)
        ctx = (attn_w * lstm_out).sum(dim=1)                   # (B, 2*hidden)

        return self.classifier(ctx)


# ═══════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "CNN-2Layer":   CNN2Layer,
    "CNN-4Layer":   CNN4Layer,
    "ResNet-8":     ResNet8,
    "LSTM-2Layer":  LSTM2Layer,
    "GRU-2Layer":   GRU2Layer,
    "BiLSTM":       BiLSTM,
    "CNN-LSTM":     CNNLSTM,
    "CNN-GRU":      CNNGRU,
    "ResNet-LSTM":  ResNetLSTM,
    "Transformer":  TransformerAMC,
    "MS-SENet":     MSSENet,
}


def get_model(name, num_classes=NUM_CLASSES):
    """Instantiate a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


def count_parameters(model):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Self-test ───────────────────────────────────────────────────────────────

def self_test():
    print("Running models self-test …")
    x = torch.randn(4, INPUT_CHANNELS, INPUT_LEN)

    for name, cls in MODEL_REGISTRY.items():
        model = cls(num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            y = model(x)
        total, trainable = count_parameters(model)
        assert y.shape == (4, NUM_CLASSES), f"{name}: bad output shape {y.shape}"
        print(f"  {name:15s} | params: {total:>8,} | output: {y.shape}")

    print("Models self-test PASSED ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        self_test()
