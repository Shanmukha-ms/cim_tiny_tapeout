#!/usr/bin/env python3
"""
infer_bnn.py — Software BNN inference matching the Verilog exactly.

Usage:
    python3 infer_bnn.py                    # run on full test set
    python3 infer_bnn.py --demo             # interactive single-sample demo

This script loads the trained weights from trained_weights.json and
simulates the XNOR + popcount + threshold logic gate-for-gate.
"""

import json, argparse, pathlib
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Load trained artefacts
# ─────────────────────────────────────────────────────────────────────────────
HERE    = pathlib.Path(__file__).parent
weights = json.loads((HERE / "trained_weights.json").read_text())

W_BIN        = np.array(weights["W_bin"],        dtype=np.uint8)   # shape (8,)
THRESH       = weights["THRESH"]
FEAT_NAMES   = weights["feat_names"]
FEAT_MEDIANS = np.array(weights["feat_medians"])
X_TEST       = np.array(weights["X_test_bin"],   dtype=np.uint8)
Y_TEST       = np.array(weights["y_test"],        dtype=np.uint8)

W_VERILOG    = weights["W_verilog"]
T_VERILOG    = weights["THRESH_verilog"]

LABELS = ["malignant", "benign"]


# ─────────────────────────────────────────────────────────────────────────────
# Core BNN inference — mirrors Verilog exactly
# ─────────────────────────────────────────────────────────────────────────────
def bnn_infer_single(x_bin: np.ndarray) -> dict:
    """
    Run one inference step.  Returns dict with full internals for inspection.

    x_bin : uint8 array of shape (8,), each element 0 or 1
    """
    assert x_bin.shape == (8,) and x_bin.dtype == np.uint8

    # Stage 1: XNOR  (match[i] = ~(x[i] XOR w[i]))
    match   = (~(x_bin ^ W_BIN)) & 1

    # Stage 2: popcount
    popcount = int(match.sum())

    # Stage 3: threshold comparator
    y = int(popcount >= THRESH)

    return {
        "x_bin"   : x_bin,
        "match"   : match,
        "popcount": popcount,
        "y"       : y,
        "label"   : LABELS[y],
    }


def binarize_features(raw_values: list) -> np.ndarray:
    """
    Convert 8 raw float measurements to binary features.
    raw_values must be in the order shown in FEAT_NAMES.
    """
    raw = np.array(raw_values, dtype=float)
    assert len(raw) == 8, f"Need 8 values, got {len(raw)}"
    return (raw > FEAT_MEDIANS).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation on test set
# ─────────────────────────────────────────────────────────────────────────────
def run_test_set():
    print("=" * 64)
    print("  BNN Inference — Breast Cancer Wisconsin test set")
    print(f"  W = {W_VERILOG}   THRESH = {T_VERILOG}")
    print("=" * 64)

    y_preds = []
    for x, y_true in zip(X_TEST, Y_TEST):
        res = bnn_infer_single(x)
        y_preds.append(res["y"])

    y_preds = np.array(y_preds)
    correct  = (y_preds == Y_TEST).sum()
    total    = len(Y_TEST)
    acc      = correct / total

    # Confusion matrix
    tp = int(((y_preds == 1) & (Y_TEST == 1)).sum())
    tn = int(((y_preds == 0) & (Y_TEST == 0)).sum())
    fp = int(((y_preds == 1) & (Y_TEST == 0)).sum())
    fn = int(((y_preds == 0) & (Y_TEST == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    ppv         = tp / (tp + fp) if (tp + fp) else 0   # precision

    print(f"\n  Test set size : {total} samples")
    print(f"  Accuracy      : {correct}/{total} = {acc*100:.1f}%")
    print(f"\n  Confusion matrix:")
    print(f"               Predicted")
    print(f"               malignant  benign")
    print(f"  Actual mal.  {tn:>9d}  {fp:>6d}   (TN / FP)")
    print(f"  Actual ben.  {fn:>9d}  {tp:>6d}   (FN / TP)")
    print(f"\n  Sensitivity (benign recall)    : {sensitivity*100:.1f}%")
    print(f"  Specificity (malignant recall) : {specificity*100:.1f}%")
    print(f"  Precision   (when y=1)         : {ppv*100:.1f}%")

    print(f"\n  First 12 test samples:")
    print(f"  {'x (hex)':>8}  {'popcount':>9}  {'pred':>6}  {'actual':>12}  {'ok':>4}")
    print("  " + "-" * 48)
    for x, y_true in zip(X_TEST[:12], Y_TEST[:12]):
        res = bnn_infer_single(x)
        x_int = int(''.join(str(b) for b in reversed(x)), 2)
        ok = "✓" if res["y"] == y_true else "✗"
        print(f"  0x{x_int:02X}       {res['popcount']:>9}  "
              f"{res['label']:>10}  {LABELS[y_true]:>12}  {ok:>4}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive single-sample demo
# ─────────────────────────────────────────────────────────────────────────────
def run_demo():
    print("=" * 64)
    print("  BNN Interactive Inference Demo")
    print("  Enter 8 raw patient measurements (press Enter for defaults)")
    print("=" * 64)
    print()
    print("  Feature thresholds (enter values in these units):")
    for i, (name, med) in enumerate(zip(FEAT_NAMES, FEAT_MEDIANS)):
        print(f"    [{i}] {name:<36s}  (median = {med:.4f})")
    print()

    # Default: use first test sample for demonstration
    defaults = [float(v) * (m * 1.1 if v else m * 0.9)
                for v, m in zip(X_TEST[0], FEAT_MEDIANS)]
    # Use a clear benign-like example (all features below median)
    benign_example  = FEAT_MEDIANS * 0.7   # 30% below median on all features
    malig_example   = FEAT_MEDIANS * 1.4   # 40% above median on all features

    print("  --- Example 1: All features 30% below median (expect benign) ---")
    x_bin = binarize_features(benign_example)
    _show_inference(benign_example, x_bin)

    print()
    print("  --- Example 2: All features 40% above median (expect malignant) ---")
    x_bin = binarize_features(malig_example)
    _show_inference(malig_example, x_bin)

    print()
    print("  --- Example 3: Mixed (high perimeter+area, low rest) ---")
    mixed = FEAT_MEDIANS.copy()
    mixed[1] = FEAT_MEDIANS[1] * 1.2   # x[1] = mean perimeter HIGH
    mixed[2] = FEAT_MEDIANS[2] * 1.2   # x[2] = mean area HIGH
    mixed[0] = FEAT_MEDIANS[0] * 0.8   # x[0] = mean radius LOW
    x_bin = binarize_features(mixed)
    _show_inference(mixed, x_bin)


def _show_inference(raw_values, x_bin):
    print(f"  {'Feature':<36}  {'Raw':>10}  {'Median':>10}  {'x_bit':>6}  {'w':>4}  {'match':>6}")
    print("  " + "-" * 78)
    for i, (name, raw, med, xb, wb) in enumerate(
            zip(FEAT_NAMES, raw_values, FEAT_MEDIANS, x_bin, W_BIN)):
        match = int((~(int(xb) ^ int(wb))) & 1)
        print(f"  [{i}] {name:<33}  {raw:>10.4f}  {med:>10.4f}  "
              f"{xb:>6}  {wb:>4}  {match:>6}")

    res = bnn_infer_single(x_bin)
    x_int = int(''.join(str(b) for b in reversed(x_bin)), 2)
    print()
    print(f"  x (binary) = {format(x_int,'08b')}  (0x{x_int:02X})")
    print(f"  popcount   = {res['popcount']} / 8")
    print(f"  threshold  = {THRESH}")
    print(f"  RESULT     → y = {res['y']}  ({res['label'].upper()})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Show step-by-step inference on example patients")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_test_set()
        print()
        run_demo()
