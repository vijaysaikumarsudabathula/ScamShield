"""Shared config, logging, metric and plotting helpers for ScamShield."""
import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, confusion_matrix, brier_score_loss)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
FIG, TAB, CACHE = ROOT / "figures", ROOT / "tables", ROOT / "cache"
for _p in (FIG, TAB, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
FAST = os.environ.get("SS_FAST", "0") == "1"      # quick smoke-test mode
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC"]

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "font.family": "DejaVu Sans"})
C_SPAM, C_HAM, C_ACC = "#D1495B", "#2E86AB", "#EDAE49"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log(f"  figure -> figures/{name}")


def save_table(df, name, index=False):
    df.to_csv(TAB / f"{name}.csv", index=index)
    log(f"  table  -> tables/{name}.csv  {df.shape}")


def fetch(url, fname):
    """Download once, then reuse the cached copy (so re-runs work offline)."""
    import requests
    p = CACHE / fname
    if not p.exists():
        log(f"  downloading {url}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


def compute_metrics(y, p, thr=0.5):
    y = np.asarray(y); p = np.asarray(p)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, yh),
        "Precision": precision_score(y, yh, zero_division=0),
        "Recall": recall_score(y, yh, zero_division=0),
        "F1": f1_score(y, yh, zero_division=0),
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "MCC": matthews_corrcoef(y, yh),
        "Balanced Acc": balanced_accuracy_score(y, yh),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
        "Brier": brier_score_loss(y, p),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
