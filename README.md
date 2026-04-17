# DeepAMC Simulations

Simulation scripts for the article:
**"MS-SENet: A Multi-Scale Squeeze-Excitation Network for Deep Learning-Based
Automatic Modulation Classification in Cognitive Radio Systems"**

## Quick Start

```bash
pip install -r requirements.txt

# Run self-tests for each script
python generate_dataset.py --self-test
python models.py --self-test
python train.py --self-test
python run_benchmarks.py --self-test
python plot_figures.py --self-test
```

## Scripts

| Script | Description |
|---|---|
| `generate_dataset.py` | RadioML-style I/Q dataset generator (11 modulations, configurable SNR/channel/CFO) |
| `models.py` | All DL model implementations (10 baselines + proposed MS-SENet) |
| `train.py` | Training and evaluation pipeline for individual models |
| `run_benchmarks.py` | Full comparative benchmark: trains all models, evaluates robustness |
| `plot_figures.py` | Generates all article figures from benchmark results |

## Full Benchmark

```bash
# Quick benchmark (8 epochs, small dataset — ~10 minutes)
python run_benchmarks.py --quick

# Full benchmark (60 epochs, 500 samples/class/SNR — several hours)
python run_benchmarks.py --output-dir results

# Generate figures from benchmark results
python plot_figures.py --results-dir results --output-dir ../figures
```

## Individual Model Training

```bash
python train.py --model MS-SENet --epochs 60 --batch-size 256
python train.py --model CNN-4Layer --epochs 40
python train.py --model ResNet-LSTM --epochs 60
```

## Model Architectures

| Model | Parameters | Description |
|---|---|---|
| CNN-2Layer | 44K | Simple 2-layer CNN baseline |
| CNN-4Layer | 373K | Deeper 4-layer CNN |
| ResNet-8 | 647K | 8-layer ResNet with residual blocks |
| LSTM-2Layer | 201K | 2-layer LSTM |
| GRU-2Layer | 151K | 2-layer GRU |
| BiLSTM | 533K | Bidirectional LSTM |
| CNN-LSTM | 176K | Hybrid CNN + LSTM |
| CNN-GRU | 143K | Hybrid CNN + GRU |
| ResNet-LSTM | 284K | Hybrid ResNet + LSTM |
| Transformer | 109K | Transformer encoder with positional encoding |
| **MS-SENet** | **406K** | **Proposed: Multi-Scale SE + BiLSTM + Attention Pooling** |

## Dataset

Synthetic I/Q samples for 11 modulation classes:
BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, GFSK, CPFSK, AM-DSB, AM-SSB, WBFM

- SNR range: -20 to 18 dB (2 dB steps, 20 levels)
- Default: 500 samples per class per SNR = 110,000 total
- Channels: AWGN (default), Rayleigh fading
- Configurable carrier frequency offset (CFO)
