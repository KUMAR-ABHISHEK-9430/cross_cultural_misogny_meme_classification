"""
utils/metrics.py — Per-culture + macro evaluation metrics.
"""
from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
)

CULTURES = ["indian", "irish", "chinese"]

def compute_metrics(
    all_logits: Dict[str, List[float]],
    all_labels: Dict[str, List[float]],
) -> Dict[str, float]:
    results = {}
    f1s = []
    for culture in CULTURES:
        logits = np.array(all_logits.get(culture, []))
        labels = np.array(all_labels.get(culture, []))
        if len(labels) == 0:
            continue
        preds = (logits > 0.0).astype(int)
        f1  = f1_score(labels, preds, zero_division=0)
        pre = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, logits)
        except ValueError:
            auc = float("nan")
        results[f"{culture}_f1"]        = f1
        results[f"{culture}_precision"] = pre
        results[f"{culture}_recall"]    = rec
        results[f"{culture}_accuracy"]  = acc
        results[f"{culture}_auc"]       = auc
        f1s.append(f1)
    if f1s:
        results["val_f1_macro"] = float(np.mean(f1s))
    return results

def format_metrics(metrics: Dict[str, float]) -> str:
    lines = []
    for culture in CULTURES:
        f1  = metrics.get(f"{culture}_f1",  float("nan"))
        auc = metrics.get(f"{culture}_auc", float("nan"))
        lines.append(f"  {culture:<8} F1={f1:.3f}  AUC={auc:.3f}")
    macro = metrics.get("val_f1_macro", float("nan"))
    lines.append(f"  macro_F1 = {macro:.3f}")
    return "\n".join(lines)