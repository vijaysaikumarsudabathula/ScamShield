"""Generic benchmarking engine: CV + held-out test + all standard plots."""
import time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from ss_common import *

SCORING = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1": "f1",
           "ROC-AUC": "roc_auc", "PR-AUC": "average_precision",
           "MCC": make_scorer(matthews_corrcoef)}


def benchmark(models, Xtr, ytr, Xte, yte, tag, cv=5):
    """Cross-validate every model on the training split, then fit on all of it and score the test split."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    cv_rows, test_rows, probas, fitted, eff = [], [], {}, {}, []
    for name, model in models.items():
        t0 = time.time()
        res = cross_validate(clone(model), Xtr, ytr, cv=skf, scoring=SCORING, n_jobs=1,
                             error_score="raise")
        for f in range(cv):
            row = {"Model": name, "Fold": f + 1}
            row.update({m: res[f"test_{m}"][f] for m in METRICS})
            cv_rows.append(row)
        m = clone(model)
        t1 = time.time(); m.fit(Xtr, ytr); fit_t = time.time() - t1
        t2 = time.time(); p = m.predict_proba(Xte)[:, 1]; pred_t = time.time() - t2
        probas[name] = p; fitted[name] = m
        r = {"Model": name}; r.update(compute_metrics(yte, p)); test_rows.append(r)
        eff.append({"Model": name, "Train time (s)": fit_t,
                    "Inference (ms / sample)": 1000 * pred_t / len(Xte)})
        log(f"  [{tag}] {name:<22} CV-F1={np.mean(res['test_F1']):.4f}  "
            f"test-F1={test_rows[-1]['F1']:.4f}  ({time.time()-t0:.0f}s)")
    cv_long = pd.DataFrame(cv_rows)
    cv_mean = cv_long.groupby("Model", sort=False)[METRICS].mean()
    cv_std = cv_long.groupby("Model", sort=False)[METRICS].std()
    test_df = pd.DataFrame(test_rows).set_index("Model")
    eff_df = pd.DataFrame(eff).set_index("Model")
    return dict(cv_long=cv_long, cv_mean=cv_mean, cv_std=cv_std, test=test_df,
                proba=probas, fitted=fitted, eff=eff_df, yte=np.asarray(yte))


def cv_pm_table(res):
    out = pd.DataFrame(index=res["cv_mean"].index)
    for m in METRICS:
        out[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(res["cv_mean"][m], res["cv_std"][m])]
    return out.reset_index()


def rank_models(res):
    d = res["cv_mean"].sort_values(["F1", "PR-AUC"], ascending=False)
    return list(d.index)


def mcnemar_vs_best(res, best):
    y = res["yte"]; pb = (res["proba"][best] >= 0.5).astype(int)
    rows = []
    for n, p in res["proba"].items():
        if n == best:
            continue
        po = (p >= 0.5).astype(int)
        b = int(((pb == y) & (po != y)).sum()); c = int(((pb != y) & (po == y)).sum())
        pv = 1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue
        rows.append({"Best model": best, "Compared with": n, "Best right / other wrong": b,
                     "Other right / best wrong": c, "McNemar exact p": pv,
                     "Significant (p<0.05)": "Yes" if pv < 0.05 else "No"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
def plot_cv_heatmap(res, tag, title):
    d = res["cv_mean"].loc[rank_models(res)]
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(d) + 2))
    sns.heatmap(d, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5, cbar_kws={"label": "mean score"},
                vmin=max(0, d.values.min() - .02), vmax=1, ax=ax)
    ax.set_title(f"{title}: {len(res['cv_long']['Fold'].unique())}-fold CV metrics (mean), models ranked by F1")
    ax.set_ylabel(""); ax.tick_params(axis="x", rotation=0)
    savefig(f"{tag}_cv_heatmap.png")


def plot_test_bars(res, tag, title):
    d = res["test"][METRICS].loc[rank_models(res)].reset_index().melt("Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(13, 5.2))
    sns.barplot(data=d, x="Model", y="Score", hue="Metric", palette="viridis", ax=ax)
    lo = max(0, d.Score.min() - .05)
    ax.set_ylim(lo, 1.005); ax.set_title(f"{title}: held-out test metrics by model")
    ax.tick_params(axis="x", rotation=30); plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend(ncol=7, loc="lower center", bbox_to_anchor=(.5, 1.0), frameon=False, fontsize=8, title=None)
    savefig(f"{tag}_test_metrics_bar.png")


def plot_test_heatmap(res, tag, title):
    d = res["test"][METRICS + ["Balanced Acc", "Specificity"]].loc[rank_models(res)]
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(d) + 2))
    sns.heatmap(d, annot=True, fmt=".3f", cmap="rocket_r", linewidths=.5, ax=ax,
                vmin=max(0, d.values.min() - .02), vmax=1)
    ax.set_title(f"{title}: held-out test metrics heatmap"); ax.set_ylabel("")
    savefig(f"{tag}_test_heatmap.png")


def plot_confusions(res, tag, title, labels=("Legit", "Scam")):
    names = rank_models(res); n = len(names); cols = 4 if n > 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, nm in zip(axes, names):
        r = res["test"].loc[nm]
        cm = np.array([[int(r.TN), int(r.FP)], [int(r.FN), int(r.TP)]])
        pct = cm / cm.sum(1, keepdims=True)
        ann = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
        sns.heatmap(pct, annot=ann, fmt="", cmap="Blues", cbar=False, vmin=0, vmax=1, ax=ax,
                    xticklabels=labels, yticklabels=labels, linewidths=1, linecolor="white")
        ax.set_title(nm, fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}: confusion matrices (test set, row-normalised)", y=1.01, fontweight="bold")
    savefig(f"{tag}_confusion_matrices.png")


def plot_roc_pr(res, tag, title):
    y = res["yte"]; names = rank_models(res)
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for i, nm in enumerate(names):
        p = res["proba"][nm]; fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        auc = res["test"].loc[nm, "ROC-AUC"]; ap = res["test"].loc[nm, "PR-AUC"]
        c = cmap[i % 10]
        axes[0].plot(fpr, tpr, color=c, lw=1.6, label=f"{nm} ({auc:.4f})")
        axes[1].plot(fpr, tpr, color=c, lw=1.6)
        axes[2].plot(rc, pr, color=c, lw=1.6, label=f"{nm} ({ap:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set_title("ROC curves (AUC)")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].set_xlim(0, .1); axes[1].set_ylim(.8, 1.005); axes[1].set_title("ROC zoom (FPR ≤ 0.10)")
    axes[1].set_xlabel("False positive rate")
    axes[2].set_title("Precision-Recall curves (AP)"); axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].set_ylim(.5, 1.005); axes[2].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{title}: discrimination on the held-out test set", fontweight="bold", y=1.02)
    savefig(f"{tag}_roc_pr_curves.png")


def plot_cv_box(res, tag, title, metric="F1"):
    order = rank_models(res)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=res["cv_long"], x="Model", y=metric, order=order, palette="Set2", ax=ax, width=.6)
    sns.stripplot(data=res["cv_long"], x="Model", y=metric, order=order, color="k", size=3.5, alpha=.6, ax=ax)
    ax.set_title(f"{title}: {metric} across CV folds (stability)")
    ax.tick_params(axis="x", rotation=30); plt.setp(ax.get_xticklabels(), ha="right")
    savefig(f"{tag}_cv_{metric.lower()}_boxplot.png")


def plot_calibration(res, tag, title, top=4):
    y = res["yte"]; names = rank_models(res)[:top]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for nm in names:
        fp, mp = calibration_curve(y, res["proba"][nm], n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", lw=1.4, label=nm)
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"{title}: calibration (top {top} models)"); ax.legend(fontsize=8)
    savefig(f"{tag}_calibration.png")


def plot_efficiency(res, tag, title):
    d = res["eff"].join(res["test"][["F1"]]).reset_index()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.scatter(d["Inference (ms / sample)"], d["F1"], s=40 + 10 * np.log1p(d["Train time (s)"]) * 8,
               c=np.log1p(d["Train time (s)"]), cmap="viridis_r", edgecolor="k", alpha=.85)
    for _, r in d.iterrows():
        ax.annotate(r["Model"], (r["Inference (ms / sample)"], r["F1"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Inference latency (ms / sample, log scale)")
    ax.set_ylabel("Test F1"); ax.set_title(f"{title}: accuracy vs. speed (bubble = training time)")
    savefig(f"{tag}_efficiency.png")


def plot_threshold(res, tag, title, name):
    y = res["yte"]; p = res["proba"][name]
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    k = int(np.argmax(f1))
    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.plot(th, pr[:-1], label="Precision", color=C_HAM); ax.plot(th, rc[:-1], label="Recall", color=C_SPAM)
    ax.plot(th, f1, label="F1", color="k", lw=2)
    ax.axvline(.5, color="gray", ls=":", label="default 0.5")
    ax.axvline(th[k], color=C_ACC, ls="--", label=f"best F1 @ {th[k]:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score"); ax.legend(loc="lower left")
    ax.set_title(f"{title}: threshold analysis for {name}")
    savefig(f"{tag}_threshold_analysis.png")
    return pd.DataFrame({"Threshold": th, "Precision": pr[:-1], "Recall": rc[:-1], "F1": f1})


def full_report_figs(res, tag, title, labels=("Legit", "Scam")):
    plot_cv_heatmap(res, tag, title); plot_test_heatmap(res, tag, title)
    plot_test_bars(res, tag, title); plot_confusions(res, tag, title, labels)
    plot_roc_pr(res, tag, title); plot_cv_box(res, tag, title)
    plot_calibration(res, tag, title); plot_efficiency(res, tag, title)
