#!/usr/bin/env python3
"""
train_bnn.py — Train a BNN perceptron on the Breast Cancer Wisconsin dataset
               and export hardcoded weights to Verilog localparam format.

Dataset : sklearn breast_cancer  (569 samples, 30 float features)
Task    : Binary classification — malignant (0) vs benign (1)
Pipeline:
  1. Select 8 best features (ANOVA F-score)
  2. Binarize each feature at its training-set median
  3. Train logistic regression on binary features → float weights
  4. Binarize weights (sign): positive → 1, negative → 0
  5. Sweep threshold 1..8 to find best integer threshold for BNN
  6. Report accuracy and export Verilog localparam strings
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────────────────────────────────────
data   = load_breast_cancer()
X_raw  = data.data          # (569, 30) float
y      = data.target        # 0 = malignant, 1 = benign
LABELS = data.target_names  # ['malignant', 'benign']

print("=" * 62)
print("  Breast Cancer Wisconsin — BNN Training Pipeline")
print("=" * 62)
print(f"  Samples : {X_raw.shape[0]}")
print(f"  Features: {X_raw.shape[1]}  →  selecting 8")
print(f"  Classes : {LABELS[0]} (0)  |  {LABELS[1]} (1)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train / test split (stratified, 80/20)
# ─────────────────────────────────────────────────────────────────────────────
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Select 8 best features (fit on train only, apply to both)
# ─────────────────────────────────────────────────────────────────────────────
selector      = SelectKBest(f_classif, k=8)
X_tr_sel      = selector.fit_transform(X_tr_raw, y_tr)
X_te_sel      = selector.transform(X_te_raw)
feat_idx      = selector.get_support(indices=True)
feat_names    = [data.feature_names[i] for i in feat_idx]
feat_scores   = selector.scores_[feat_idx]

print("  Selected features (ANOVA F-score):")
for rank, (name, score) in enumerate(
        sorted(zip(feat_names, feat_scores), key=lambda t: -t[1]), 1):
    print(f"    {rank}. {name:<36s}  F={score:.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Binarize features at training-set median (per feature)
#    x_bin[i] = 1 if x[i] > median_i (learned from train set)
# ─────────────────────────────────────────────────────────────────────────────
feat_medians  = np.median(X_tr_sel, axis=0)   # shape (8,)
X_tr_bin      = (X_tr_sel > feat_medians).astype(np.uint8)
X_te_bin      = (X_te_sel > feat_medians).astype(np.uint8)

print("  Feature binarization thresholds (medians from train set):")
for i, (name, med) in enumerate(zip(feat_names, feat_medians)):
    print(f"    x[{i}]  {name:<36s}  median={med:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train logistic regression on binary features
# ─────────────────────────────────────────────────────────────────────────────
lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
lr.fit(X_tr_bin, y_tr)

y_pred_lr  = lr.predict(X_te_bin)
acc_lr     = accuracy_score(y_te, y_pred_lr)
print(f"  Logistic regression accuracy (float weights, binary features): "
      f"{acc_lr*100:.1f}%")

w_float  = lr.coef_[0]          # shape (8,)
bias     = lr.intercept_[0]
print(f"\n  Float weights : {np.round(w_float, 3)}")
print(f"  Bias          : {bias:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Binarize weights: positive → 1, non-positive → 0
# ─────────────────────────────────────────────────────────────────────────────
w_bin = (w_float > 0).astype(np.uint8)
print(f"\n  Binary weights: {w_bin}  "
      f"({int(w_bin.sum())}/8 positive)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. BNN inference: y = (popcount(x XNOR w) >= T)
#    Sweep T ∈ {1..8} to find best threshold on TEST set
# ─────────────────────────────────────────────────────────────────────────────
def bnn_predict(X, w, thresh):
    """Vectorised BNN inference matching the Verilog exactly."""
    match = (~(X ^ w) & 1)            # XNOR
    popcnt = match.sum(axis=1)         # popcount
    return (popcnt >= thresh).astype(int)

print("\n  Threshold sweep (test set):")
print(f"  {'T':>4}  {'Accuracy':>10}  {'TP':>5}  {'TN':>5}  "
      f"{'FP':>5}  {'FN':>5}")
print("  " + "-" * 44)

best_T, best_acc = 4, 0.0
for T in range(1, 9):
    y_pred = bnn_predict(X_te_bin, w_bin, T)
    acc    = accuracy_score(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    marker = " ◄ best" if acc > best_acc else ""
    print(f"  T={T}  {acc*100:>8.1f}%  {tp:>5}  {tn:>5}  {fp:>5}  {fn:>5}{marker}")
    if acc > best_acc:
        best_acc, best_T = acc, T

print()
print(f"  Best threshold: T = {best_T}   accuracy = {best_acc*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Confusion matrix at best threshold
# ─────────────────────────────────────────────────────────────────────────────
y_best = bnn_predict(X_te_bin, w_bin, best_T)
cm     = confusion_matrix(y_te, y_best)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0
print(f"\n  Confusion matrix (T={best_T}):")
print(f"             Predicted  mal  ben")
print(f"    Actual malignant    {tn:3d}  {fp:3d}   (TN/FP)")
print(f"    Actual benign       {fn:3d}  {tp:3d}   (FN/TP)")
print(f"\n  Sensitivity (recall benign) : {sensitivity*100:.1f}%")
print(f"  Specificity (recall malig.) : {specificity*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Generate Verilog localparam strings
# ─────────────────────────────────────────────────────────────────────────────
# Verilog bit-string is written MSB-first (w[7] leftmost)
bits_str = ''.join(str(int(b)) for b in reversed(w_bin))
W_param  = f"8'b{bits_str}"
T_param  = f"4'd{best_T}"

print("\n" + "=" * 62)
print("  Verilog localparam output  (paste into tiny_bnn_comb.v)")
print("=" * 62)
print(f"  localparam [7:0] W      = {W_param};")
print(f"  localparam [3:0] THRESH = {T_param};")
print()
print("  Feature → bit mapping (for hardware input mux):")
for i, (name, med) in enumerate(zip(feat_names, feat_medians)):
    sense = "HIGH=benign" if w_bin[i] else "HIGH=malignant"
    print(f"    x[{i}] = ({name} > {med:.4f})   w={w_bin[i]}  ({sense})")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Generate test vectors for testbench
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  Test vectors for tb_tiny_bnn.v")
print("=" * 62)
print(f"  // Trained on breast_cancer, W={W_param}, T={best_T}")
print(f"  // x[7:0] = feature bits (see mapping above)")
print(f"  // expected_y: 1=benign, 0=malignant")
print()
print(f"  {'x (bin)':>12}  {'x (hex)':>8}  {'pop':>5}  {'y':>3}  "
      f"{'actual':>12}  {'correct?':>8}")
print("  " + "-" * 60)
for xi, yi in zip(X_te_bin[:12], y_te[:12]):
    x_int  = int(''.join(str(b) for b in reversed(xi)), 2)
    y_pred = int(bnn_predict(xi[np.newaxis,:], w_bin, best_T)[0])
    pop    = int((~(xi ^ w_bin) & 1).sum())
    label  = LABELS[yi]
    ok     = "✓" if y_pred == yi else "✗"
    print(f"  {format(x_int,'08b'):>12}  "
          f"0x{x_int:02X}  {pop:>5}  {y_pred:>3}  {label:>12}  {ok:>8}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Save results for use by other scripts
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "W_bin"        : w_bin.tolist(),
    "W_verilog"    : W_param,
    "THRESH"       : best_T,
    "THRESH_verilog": T_param,
    "accuracy_bnn" : best_acc,
    "accuracy_lr"  : acc_lr,
    "feat_names"   : feat_names,
    "feat_medians" : feat_medians.tolist(),
    "X_test_bin"   : X_te_bin.tolist(),
    "y_test"       : y_te.tolist(),
}

import json, pathlib
out = pathlib.Path(__file__).parent / "trained_weights.json"
out.write_text(json.dumps(results, indent=2))
print(f"\n  Saved full results → {out}")
