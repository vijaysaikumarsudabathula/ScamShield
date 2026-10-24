"""Paper-sized (single-column, 3.35 in) figures for the ScamShield IEEE paper. All data read from tables/ or recomputed from cache."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path
import ss_common  # import first: it applies a seaborn theme that our rcParams must override

ROOT = Path(__file__).resolve().parent.parent; TAB = ROOT / "tables"; OUT = ROOT / "paper" / "figs"
T = lambda n: pd.read_csv(TAB / f"{n}.csv")
C_S, C_H, C_A, C_G = "#D1495B", "#2E86AB", "#EDAE49", "#2F6B50"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": .6,
                     "axes.titleweight": "bold", "xtick.major.width": .5, "ytick.major.width": .5,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": .4, "grid.color": "#DDDDDD"})
SHORT = {"Multinomial NB": "NB (multinomial)", "Complement NB": "NB (complement)", "Logistic Regression": "Logistic Reg.",
         "Linear SVM": "Linear SVM", "Random Forest": "Random Forest", "Extra Trees": "Extra Trees", "XGBoost": "XGBoost",
         "LightGBM": "LightGBM", "MLP (neural net)": "MLP", "Voting Ensemble": "Voting Ens.", "k-Nearest Neighbors": "k-NN",
         "Decision Tree": "Decision Tree", "Gradient Boosting": "Grad. Boosting", "HistGradientBoosting": "Hist. GB",
         "ScamShield-hardened (Logistic Regression)": "ScamShield-hardened (LR)"}
def save(fig, name, dpi=300):
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.03); plt.close(fig); print("  ", name)

# ------------------------------------------------------------------ Fig 1: methodology flowchart (vertical, like AgroBranch)
def flowchart():
    W, H = 3.2, 4.55
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    FILL, EDGE = "#EAF2EC", C_G
    def box(cx, ytop, w, lines, note=None, fill=FILL, edge=EDGE, fs=6.6):
        h = 0.16 + 0.135 * len(lines) + (0.13 if note else 0)
        ax.add_patch(FancyBboxPatch((cx - w / 2, ytop - h), w, h, boxstyle="round,pad=0.0,rounding_size=0.05", fc=fill, ec=edge, lw=1.0))
        y = ytop - 0.09 - 0.0
        for i, l in enumerate(lines):
            ax.text(cx, ytop - 0.1 - 0.135 * (i + .5) + 0.02, l, ha="center", va="center", fontsize=fs, color="#1F2D24")
        if note:
            ax.text(cx, ytop - h + 0.1, note, ha="center", va="center", fontsize=5.6, style="italic", color="#3B4A63")
        return ytop - h, h
    def arrow(cx, y1, y2):
        ax.add_patch(FancyArrowPatch((cx, y1), (cx, y2), arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#34495E"))
    cx, w = 1.22, 2.08
    y = H - 0.08; gap = 0.17; tops = []
    spec = [(["Public data", "SMS Spam Collection · Phishing.Database", "OpenDNS top domains"], None, FILL, EDGE),
            (["Phase 1 — Scam-text detection", "word + char TF-IDF + style → 10 models"], "best by 5-fold CV F1", FILL, EDGE),
            (["Phase 2 — Phishing-domain detection", "67 lexical / brand features → 12 models"], "ablation + hard subset", FILL, EDGE),
            (["Phase 3 — Adversarial hardening", "adaptive attacker vs. canonicaliser", "+ adversarial training (3 rounds)"], "evasion rate on caught scams", FILL, EDGE),
            (["Phase 4 — Concept-drift probe", "60 modern messages + online adaptation"], "legacy → modern gap", FILL, EDGE),
            (["Phase 5 — Multi-agent pipeline", "Extractor → Analyzer → Verifier", "→ Scorer → Advisor"], "text model + domain model as tools", FILL, EDGE),
            (["Validation", "5-fold CV · McNemar · ablation ·", "unseen attacks · hard subset"], None, "#EFE9F2", "#6B4C8C"),
            (["Explanation layer", "evidence-grounded verdict and advice"], None, FILL, EDGE),
            (["Report — figures, tables, IEEE paper"], None, FILL, EDGE)]
    prev = None; ys = {}
    for i, (lines, note, fill, edge) in enumerate(spec):
        if prev is not None:
            arrow(cx, prev, prev - gap); y = prev - gap
        bottom, h = box(cx, y, w, lines, note, fill, edge); ys[i] = (y, bottom); prev = bottom
    # side box beside Phase 2: brand look-alike book
    yt, yb = ys[2]
    sx, sw, sh = 2.78, 0.78, 0.62
    ax.add_patch(FancyBboxPatch((sx - sw / 2, (yt + yb) / 2 - sh / 2), sw, sh, boxstyle="round,pad=0.0,rounding_size=0.05", fc="#F8EFE6", ec="#A0521D", lw=1.0))
    for i, l in enumerate(["Brand", "look-alike", "book", "(top-400)"]):
        ax.text(sx, (yt + yb) / 2 + 0.2 - 0.135 * i, l, ha="center", va="center", fontsize=6.0, color="#1F2D24")
    ax.add_patch(FancyArrowPatch((sx - sw / 2, (yt + yb) / 2), (cx + w / 2, (yt + yb) / 2), arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#34495E"))
    # side box beside Phase 3: unseen attacks
    yt3, yb3 = ys[3]
    ax.add_patch(FancyBboxPatch((sx - sw / 2, (yt3 + yb3) / 2 - sh / 2), sw, sh, boxstyle="round,pad=0.0,rounding_size=0.05", fc="#F8EFE6", ec="#A0521D", lw=1.0))
    for i, l in enumerate(["7 attacks", "4 seen,", "3 unseen"]):
        ax.text(sx, (yt3 + yb3) / 2 + 0.16 - 0.135 * i, l, ha="center", va="center", fontsize=6.0, color="#1F2D24")
    ax.add_patch(FancyArrowPatch((sx - sw / 2, (yt3 + yb3) / 2), (cx + w / 2, (yt3 + yb3) / 2), arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#34495E"))
    ax.set_ylim(prev - 0.05, H); fig.set_size_inches(W, H - (prev - 0.05))
    save(fig, "fig1_flowchart.png", 400)

# ------------------------------------------------------------------ Fig 2: run-time architecture
def architecture():
    W, H = 3.35, 3.05
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    def box(x, y, w, h, lines, fill="#EAF2EC", edge=C_G, fs=6.1, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05", fc=fill, ec=edge, lw=0.9))
        n = len(lines)
        for i, l in enumerate(lines):
            ax.text(x + w / 2, y + h / 2 + (n - 1) * 0.0675 - i * 0.135, l, ha="center", va="center", fontsize=fs, color="#1F2D24", fontweight="bold" if (bold and i == 0) else "normal")
    def arr(x1, y1, x2, y2, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#34495E", linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    box(1.0, 2.62, 1.35, 0.34, ["Incoming message"], fill="#FFF3CD", edge="#B58900", bold=True)
    box(0.05, 1.68, 1.05, 0.62, ["① Extractor", "links · phones", "· amounts"], bold=True)
    box(1.22, 1.68, 1.0, 0.62, ["Text model", "canonicaliser +", "TF-IDF + LR"], fill="#EFE9F2", edge="#6B4C8C", bold=True)
    box(2.34, 1.68, 0.96, 0.62, ["② Analyzer", "7 manipulation", "tactics"], bold=True)
    box(0.05, 0.84, 1.05, 0.62, ["③ Verifier", "phishing DB ·", "look-alike ·", "domain model"], bold=True)
    box(1.32, 0.84, 1.18, 0.62, ["④ Scorer", r"$R = 1-(1-p_t)(1-p_d)$"], fill="#FADBD8", edge=C_S, bold=True)
    box(0.35, 0.08, 2.75, 0.5, ["⑤ Advisor", "verdict · evidence · next steps"], fill="#E8DAEF", edge="#6B4C8C", bold=True)
    for x in (0.575, 1.72, 2.82): arr(1.68, 2.62, x, 2.3)
    arr(0.575, 1.68, 0.575, 1.46)          # extractor -> verifier
    arr(1.1, 1.15, 1.32, 1.15)             # verifier -> scorer
    arr(1.72, 1.68, 1.9, 1.46)             # text model -> scorer
    arr(1.9, 0.84, 1.9, 0.58)              # scorer -> advisor
    arr(2.82, 1.68, 2.82, 0.58)            # analyzer -> advisor
    ax.text(2.88, 1.1, "tactics", fontsize=5.8, style="italic", color="#3B4A63")
    save(fig, "fig2_architecture.png", 400)

# ------------------------------------------------------------------ Fig 3: SMS EDA
def eda():
    from ss_data import load_sms
    df = load_sms(); df["len"] = df.text.str.len(); df["dig"] = df.text.apply(lambda t: sum(c.isdigit() for c in t))
    fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.55))
    for lab, c, nm in [("ham", C_H, "Legitimate"), ("spam", C_S, "Scam")]:
        sns.kdeplot(df[df.label == lab]["len"], ax=ax[0], color=c, fill=True, alpha=.35, label=nm, clip=(0, 400), lw=.9)
        sns.kdeplot(df[df.label == lab]["dig"], ax=ax[1], color=c, fill=True, alpha=.35, label=nm, clip=(0, 40), lw=.9)
    ax[0].set_xlabel("Message length (characters)"); ax[1].set_xlabel("Digits per message")
    ax[0].set_ylabel("Density"); ax[1].set_ylabel(""); ax[0].legend(frameon=False, loc="upper right")
    ax[0].set_title("(a) Length", fontsize=6.5); ax[1].set_title("(b) Digits", fontsize=6.5)
    fig.tight_layout(w_pad=1.0); save(fig, "fig3_sms_eda.png")

def top_terms():
    t = T("sms_top_terms"); pos = t[t.coef > 0].nlargest(8, "coef"); neg = t[t.coef < 0].nsmallest(8, "coef")
    d = pd.concat([pos, neg]).sort_values("coef")
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.barh(d.feature, d.coef, color=[C_S if c > 0 else C_H for c in d.coef], height=.7)
    ax.axvline(0, color="k", lw=.6); ax.set_xlabel("Logistic-regression coefficient (+ scam / − legitimate)")
    for l in ax.get_yticklabels(): l.set_fontsize(6)
    fig.tight_layout(); save(fig, "fig4_top_terms.png")

def confusion(df_name, model, labels, fname, title=None):
    r = T(df_name).set_index("Model").loc[model]
    cm = np.array([[r.TN, r.FP], [r.FN, r.TP]], dtype=int); pct = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.55, 2.2))
    ann = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
    sns.heatmap(pct, annot=ann, fmt="", cmap="Blues", cbar=False, vmin=0, vmax=1, ax=ax, xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="white", annot_kws={"fontsize": 7})
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); save(fig, fname)

def cv_bars():
    d = T("sms_cv_metrics_mean_std"); m = T("sms_cv_metrics_mean").set_index("Model")
    s = pd.DataFrame({"Model": m.index, "F1": m.F1.values})
    sd = d.set_index("Model").F1.str.extract(r"± ([\d.]+)")[0].astype(float)
    s["sd"] = sd.loc[s.Model].values; s = s.sort_values("F1")
    best = s.F1.idxmax()
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    cols = [C_S if f == s.F1.max() else C_H for f in s.F1]
    ax.barh([SHORT[x] for x in s.Model], s.F1, xerr=s.sd, color=cols, height=.68, error_kw=dict(lw=.6, capsize=1.5))
    ax.set_xlim(.90, 1.01); ax.set_xlabel("5-fold CV F1 (mean ± sd)"); ax.grid(axis="x")
    for i, f in enumerate(s.F1): ax.text(.9025, i, f"{f:.3f}", va="center", fontsize=5.6, color="white")
    fig.tight_layout(); save(fig, "fig6_sms_cv_f1.png")

def importance():
    import joblib
    from sklearn.model_selection import train_test_split
    import shap
    from ss_data import load_domains, BrandBook, domain_features
    us = joblib.load(ROOT / "cache" / "url_state.joblib")
    dd, ranked, phish, info = load_domains(10000)
    F = domain_features(dd.domain.tolist(), BrandBook(ranked)); y = dd.y.values
    Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=.2, stratify=y, random_state=42)
    lgbm = us["fitted"]["LightGBM"]; samp = Xte.sample(1500, random_state=42)
    sv = shap.TreeExplainer(lgbm).shap_values(samp)
    if isinstance(sv, list): sv = sv[1]
    imp = pd.Series(np.abs(sv).mean(0), index=F.columns).sort_values(ascending=False).head(10)[::-1]
    pd.DataFrame({"feature": imp.index[::-1], "mean_abs_shap": imp.values[::-1]}).to_csv(ROOT / "paper" / "shap_top10.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    ax.barh(imp.index, imp.values, color=C_A, edgecolor="k", lw=.4, height=.7)
    ax.set_xlabel("Mean |SHAP value| (LightGBM, 1,500 test domains)"); ax.grid(axis="x")
    fig.tight_layout(); save(fig, "fig8_shap.png")

def ablation():
    a = T("url_ablation").copy()
    a["lab"] = a.Configuration.str.replace("− ", "− ", regex=False).str.replace(r"\(.*?\)", "", regex=True).str.replace("Lexical + brand + keywords only", "Lexical + brand + kw. only", regex=False).str.strip()
    a = a.iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    cols = [C_S if i == len(a) - 1 else (C_A if "only" in l else C_H) for i, l in enumerate(a.lab)]
    ax.barh(a.lab, a["Test F1"], color=cols, height=.68)
    for i, v in enumerate(a["Test F1"]): ax.text(.8005, i, f"{v:.3f}", va="center", fontsize=5.8, color="white")
    ax.set_xlim(.80, .915); ax.set_xlabel("Held-out test F1 (LightGBM)"); ax.grid(axis="x")
    fig.tight_layout(); save(fig, "fig9_ablation.png")

CFG = ["Baseline", "+ Canonicalization", "+ Adversarial training", "+ Canon. & Adv. training"]
CFG_S = ["Baseline", "Canon.", "Adv. training", "Canon. + adv."]
def adv_bars():
    a = T("adv_final_summary"); a = a[a.Model == "Logistic Regression"].set_index("Config").loc[CFG]
    cols = ["Evasion (seen attacks)", "Evasion (unseen attacks)", "Evasion (all attacks)"]
    fig, ax = plt.subplots(figsize=(3.35, 2.1)); x = np.arange(4); w = .26
    for i, (c, col, nm) in enumerate(zip(cols, [C_H, C_A, C_S], ["Seen attacks", "Unseen attacks", "All attacks"])):
        b = ax.bar(x + (i - 1) * w, a[c] * 100, w, color=col, label=nm, edgecolor="k", lw=.3)
        for xi, v in zip(x + (i - 1) * w, a[c] * 100): ax.text(xi, v + .3, f"{v:.1f}", ha="center", fontsize=5.4)
    ax.set_xticks(x); ax.set_xticklabels(CFG_S); ax.set_ylabel("Evasion rate (%)"); ax.set_ylim(0, 22); ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y"); fig.tight_layout(); save(fig, "fig10_adv_defences.png")

def adv_heat():
    p = T("adv_per_attack"); p = p[p.Model == "Logistic Regression"].pivot(index="Attack", columns="Config", values="Evasion rate")[CFG] * 100
    order = ["leet", "space", "synonym", "punct", "homoglyph", "padding", "zerowidth"]; p = p.loc[order]
    names = {"leet": "leet (seen)", "space": "spaced letters (seen)", "synonym": "synonyms (seen)", "punct": "punctuation (seen)", "homoglyph": "homoglyphs (unseen)", "padding": "benign padding (unseen)", "zerowidth": "zero-width (unseen)"}
    p.index = [names[i] for i in p.index]; p.columns = ["Baseline", "Canon.", "Adv.\ntraining", "Canon.\n+ adv."]
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    sns.heatmap(p, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=10, cbar_kws={"label": "Evasion (%)", "shrink": .8}, ax=ax,
                linewidths=.5, linecolor="white", annot_kws={"fontsize": 6.2})
    ax.axhline(4, color="k", lw=1.2); ax.tick_params(length=0); ax.set_ylabel(""); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=5.8)
    fig.tight_layout(); save(fig, "fig11_adv_heatmap.png")

def adv_rounds():
    r = T("adv_rounds"); r = r[r.Model == "Logistic Regression"]
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    for cfg, ls, nm in [("+ Adversarial training", "-", "Adv. training"), ("+ Canon. & Adv. training", "--", "Canon. + adv.")]:
        d = r[r.Config == cfg].sort_values("Round")
        ax.plot(d.Round, d["Evasion (all attacks)"] * 100, ls, marker="o", ms=3, color=C_S, lw=1.1, label=f"{nm}: all")
        ax.plot(d.Round, d["Evasion (unseen attacks)"] * 100, ls, marker="s", ms=3, color=C_A, lw=1.1, label=f"{nm}: unseen")
        ax.plot(d.Round, d["Evasion (seen attacks)"] * 100, ls, marker="^", ms=3, color=C_H, lw=1.1, label=f"{nm}: seen")
    ax.set_xticks([0, 1, 2, 3]); ax.set_xlabel("Adversarial-training round"); ax.set_ylabel("Evasion rate (%)"); ax.grid(True)
    ax.legend(frameon=False, ncol=3, fontsize=5.0, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.6); ax.set_ylim(0, 20)
    fig.tight_layout(); save(fig, "fig12_adv_rounds.png")

def drift():
    d = T("probe_model_comparison").sort_values("Modern-probe F1")
    fig, ax = plt.subplots(figsize=(3.35, 2.15)); y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()): ax.plot([r["Modern-probe F1"], r["SMS test F1"]], [i, i], color="#999", lw=1.2, zorder=1)
    ax.scatter(d["SMS test F1"], y, s=22, color=C_H, edgecolor="k", lw=.4, zorder=2, label="Legacy SMS test F1")
    ax.scatter(d["Modern-probe F1"], y, s=22, color=C_S, edgecolor="k", lw=.4, zorder=2, label="Modern-probe F1")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in d.Model]); ax.set_xlabel("F1"); ax.set_xlim(.25, 1.0); ax.grid(axis="x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2); fig.tight_layout(); save(fig, "fig13_drift.png")

def adapt():
    s = T("probe_online_update_summary")
    fig, ax = plt.subplots(figsize=(3.35, 1.7)); k = s["Reported messages (k)"]
    for col, c, m, nm in [("Held-out ROC-AUC", C_H, "o", "ROC-AUC"), ("Held-out Specificity", C_A, "s", "Specificity"), ("Held-out Recall", C_S, "^", "Recall")]:
        ax.errorbar(k, s[f"{col} (mean)"], yerr=s[f"{col} (std)"], marker=m, ms=3.2, lw=1.1, color=c, capsize=2, elinewidth=.6, label=nm)
    ax.set_xlabel("Newly reported messages used for the update (k)"); ax.set_ylabel("Held-out score"); ax.set_xticks(list(k)); ax.set_ylim(.7, 1.02)
    ax.grid(True); ax.legend(frameon=False, ncol=3, loc="lower center"); fig.tight_layout(); save(fig, "fig14_adaptation.png")

def agent_bars():
    demo = json.load(open(TAB / "agent_demo_reports.json"))
    pick = [0, 1, 3, 4, 5, 7, 9, 11]
    d = [demo[i] for i in pick]
    lab = []
    for r in d:
        t = r["text"]; lab.append((t[:30] + "…") if len(t) > 31 else t)
    fig, ax = plt.subplots(figsize=(3.35, 2.15)); y = np.arange(len(d))[::-1]
    ax.barh(y + .2, [r["p_text"] for r in d], .36, color=C_H, label="Text model P(scam)")
    ax.barh(y - .2, [r["p_url"] for r in d], .36, color=C_A, label="Domain risk")
    ax.scatter([r["risk"] for r in d], y, marker="D", s=14, color=C_S, edgecolor="k", lw=.4, zorder=5, label="Fused risk R")
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=5.4); ax.axvline(.5, color="gray", ls=":", lw=.7); ax.axvline(.85, color="gray", ls="--", lw=.7)
    ax.set_xlim(0, 1.03); ax.set_xlabel("Risk score (dotted 0.5, dashed 0.85)"); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, fontsize=5.2, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(); save(fig, "fig15_agent.png")
    return [(i, d_["text"]) for i, d_ in zip(pick, d)]


def confusion_pair():
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55))
    for ax, (nm, model, labels, ttl) in zip(axes, [("sms_test_metrics", "Logistic Regression", ["Legit", "Scam"], "(a) Scam text"),
                                                   ("url_test_metrics", "Voting Ensemble", ["Legit", "Phish"], "(b) Phishing domain")]):
        r = T(nm).set_index("Model").loc[model]
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]], dtype=int); pct = cm / cm.sum(1, keepdims=True)
        ann = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
        sns.heatmap(pct, annot=ann, fmt="", cmap="Blues", cbar=False, vmin=0, vmax=1, ax=ax, xticklabels=labels, yticklabels=labels,
                    linewidths=1, linecolor="white", annot_kws={"fontsize": 5.8})
        ax.set_title(ttl, fontsize=6.5); ax.set_xlabel("Predicted", fontsize=6); ax.set_ylabel("Actual", fontsize=6); ax.tick_params(length=0, labelsize=5.6)
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout(w_pad=0.8); save(fig, "fig_cm_pair.png")

if __name__ == "__main__":
    flowchart(); architecture(); eda(); top_terms()
    confusion("sms_test_metrics", "Logistic Regression", ["Legit", "Scam"], "fig5_sms_cm.png")
    cv_bars(); confusion("url_test_metrics", "Voting Ensemble", ["Legit", "Phishing"], "fig7_url_cm.png")
    importance(); ablation(); adv_bars(); adv_heat(); adv_rounds(); drift(); adapt(); print(agent_bars())
