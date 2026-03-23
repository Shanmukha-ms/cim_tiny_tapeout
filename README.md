# tt_um_bnn — Binary Neural Network Breast Cancer Classifier

[![TinyTapeout](https://img.shields.io/badge/TinyTapeout-TTIHP-blue)](https://tinytapeout.com)

An 8-input Binary Neural Network perceptron trained on the Breast Cancer Wisconsin
dataset, implemented in ~140 gate equivalents for TinyTapeout.

**89.5% test accuracy** · **~140 GE** · **3 logic levels** combinational latency

## Design summary

| Property | Value |
|----------|-------|
| Dataset | Breast Cancer Wisconsin (sklearn) |
| Task | Malignant (0) vs Benign (1) |
| Accuracy | 89.5% test set |
| Architecture | XNOR + popcount + threshold |
| Gate count | ~140 GE total |
| Top module | `tt_um_bnn` |

## Pinout

| Pin | Direction | Description |
|-----|-----------|-------------|
| `ui_in[7:0]` | in | Binarized feature vector (see `docs/info.md`) |
| `uo_out[0]` | out | Combinational result (1=benign, 0=malignant) |
| `uo_out[1]` | out | Sequential result (latched, valid after 8 clocks) |
| `uo_out[2]` | out | Sequential valid strobe |
| `uio_in[0]` | in | Start pulse for sequential engine |

## Project structure

```
├── src/
│   ├── project.v         # tt_um_bnn top-level TT wrapper
│   ├── tiny_bnn_comb.v   # combinational XNOR+popcount engine (~55 GE)
│   ├── tiny_bnn_seq.v    # sequential 1-bit/cycle engine    (~84 GE)
│   ├── tiny_bnn_2out.v   # 2-neuron extension (reference)
│   └── config.json       # OpenLane synthesis config
├── test/
│   ├── tb.v              # cocotb Verilog wrapper
│   ├── test.py           # cocotb Python test suite
│   ├── Makefile          # make → runs all tests
│   └── requirements.txt
├── docs/
│   └── info.md           # full design documentation
├── scripts/
│   ├── train_bnn.py      # training pipeline (sklearn → Verilog weights)
│   ├── infer_bnn.py      # software inference (mirrors hardware gate-for-gate)
│   ├── gen_weights.py    # weight conversion utilities
│   └── trained_weights.json
└── info.yaml             # TinyTapeout project metadata
```

## Quick start

```bash
# Train (regenerate weights from scratch)
python3 scripts/train_bnn.py

# Software inference on test set
python3 scripts/infer_bnn.py

# Simulate (requires iverilog + cocotb)
cd test && make
```

## How it works

See [docs/info.md](docs/info.md) for full design documentation including
the feature-to-pin mapping, inference protocol, and training procedure.
