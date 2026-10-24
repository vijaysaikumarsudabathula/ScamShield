"""Builds ScamShield_Report.html: one self-contained page with flow charts, every figure, tables and full source code."""
import base64, io, json, html, re
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent
FIG, TAB = ROOT / "figures", ROOT / "tables"
T = lambda n: pd.read_csv(TAB / f"{n}.csv")
esc = html.escape


# ------------------------------------------------------------------ extra flow chart
def experiment_flow():
    fig, ax = plt.subplots(figsize=(15, 6.4)); ax.set_xlim(0, 15); ax.set_ylim(0, 6.4); ax.axis("off")
    def box(x, y, w, h, t, fc, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="k", lw=1.1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
    def arr(x1, y1, x2, y2, c="#333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=1.5, color=c))
    cA, cB, cC, cD, cE = "#D6EAF8", "#D5F5E3", "#FDEBD0", "#FADBD8", "#E8DAEF"
    ax.text(0.1, 6.05, "Stages 1-2: build and benchmark the detectors", fontsize=10, fontweight="bold")
    xs = [0.1, 3.1, 6.1, 9.1, 12.1]
    tx = ["Download data\n(3 GitHub-raw URLs,\nno manual steps)", "Clean + de-duplicate\n(403 duplicate SMS removed;\n10 label-conflict domains)", "Stratified 80/20 split\n(seed 42)",
          "Tune hyper-parameters\n(3-fold CV on train only)", "Benchmark: 5-fold CV\n10 text models,\n12 domain models"]
    for x, t in zip(xs, tx): box(x, 4.3, 2.7, 1.4, t, cA)
    for i in range(4): arr(xs[i] + 2.7, 5.0, xs[i + 1], 5.0)
    arr(13.45, 4.3, 13.45, 3.95)
    ax.text(0.1, 3.65, "Selection, then stress-testing", fontsize=10, fontweight="bold")
    ys = 2.2; xs2 = [12.1, 9.1, 6.1, 3.1, 0.1]
    tx2 = ["Fit on full train,\nscore once on held-out test", "Pick best by CV F1\n+ McNemar significance,\nSHAP, ablations", "Stage 3: adaptive attacker\nvs. 3 defences\n(3 rounds)", "Stage 4: modern-scam drift\nprobe + adaptation curve", "Stage 5: agent demo,\nfigures, Excel workbook,\nHTML report"]
    cs = [cB, cB, cD, cD, cE]
    for x, t, c in zip(xs2, tx2, cs): box(x, ys, 2.7, 1.4, t, c)
    for i in range(4): arr(xs2[i], ys + 0.7, xs2[i + 1] + 2.7, ys + 0.7)
    ax.text(0.1, 1.55, "Reading guide: blue = data and training, green = evaluation, red = robustness, purple = delivery.", fontsize=9, style="italic")
    ax.set_title("ScamShield experiment flow - from raw data to final report", fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG / "00b_experiment_flow.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()


# ------------------------------------------------------------------ helpers
_fig_n, _tab_n = [0], [0]


def img_uri(name, maxw=1700):
    im = Image.open(FIG / name).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(name, title, text="", narrow=False):
    _fig_n[0] += 1
    cls = "plate narrow" if narrow else "plate"
    return (f'<figure id="fig{_fig_n[0]}"><div class="{cls}"><img src="{img_uri(name)}" alt="{esc(title)}" loading="lazy" class="zoom"></div>'
            f'<figcaption><b>Figure {_fig_n[0]}.</b> {esc(title)}. {text}</figcaption></figure>')


def table(df, title, text="", best_max=(), best_min=(), fmt=None, open_=True, details=None, pct=()):
    _tab_n[0] += 1
    fmt = fmt or {}
    best = {}
    for c in best_max:
        if c in df: best[c] = df[c].max()
    for c in best_min:
        if c in df: best[c] = df[c].min()
    head = "".join(f"<th>{esc(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                s = f"{v:.1%}" if c in pct else (fmt[c](v) if c in fmt else f"{v:.4f}")
            elif isinstance(v, (int, np.integer)):
                s = f"{v:,}"
            else:
                s = esc(str(v))
            cls = ' class="best"' if (c in best and isinstance(v, (float, int, np.floating, np.integer)) and abs(v - best[c]) < 1e-12) else ""
            num = " num" if (isinstance(v, (int, float, np.integer, np.floating)) or (isinstance(v, str) and "±" in v)) else ""
            tds.append(f'<td class="{cls[8:-1] if cls else ""}{num}">{s}</td>'.replace('class=""', ""))
        rows.append("<tr>" + "".join(tds) + "</tr>")
    cap = f'<div class="tcap"><b>Table {_tab_n[0]}.</b> {esc(title)}. {text}</div>'
    body = f'<div class="tscroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    if details:
        return f'<details class="tdet"{" open" if open_ else ""}><summary>{cap}</summary>{body}</details>'
    return cap + body


def prose(*ps):
    return "".join(f"<p>{p}</p>" for p in ps)


def section(i, id_, title, lead, body):
    return f'<section id="{id_}"><h2><span class="sn">{i}</span>{esc(title)}</h2><p class="lead">{lead}</p>{body}</section>'


M7 = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC"]
MAXC = M7 + ["Balanced Acc", "Specificity"]
MINC = ["FPR", "FNR", "Brier"]

# ------------------------------------------------------------------ gather numbers for the hero
sms_t, url_t = T("sms_test_metrics"), T("url_test_metrics")
sms_cv, url_cv = T("sms_cv_metrics_mean"), T("url_cv_metrics_mean")
sb = sms_cv.sort_values(["F1", "PR-AUC"], ascending=False).Model.iloc[0]
ub = url_cv.sort_values(["F1", "PR-AUC"], ascending=False).Model.iloc[0]
sr, ur = sms_t[sms_t.Model == sb].iloc[0], url_t[url_t.Model == ub].iloc[0]
adv = T("adv_final_summary"); lr = adv[adv.Model == "Logistic Regression"].set_index("Config")
pm = T("probe_model_comparison")
glance = pd.DataFrame([
    ["SMS scam text", sb, sr.Accuracy, sr.Precision, sr.Recall, sr.F1, sr["ROC-AUC"]],
    ["Phishing domain", ub, ur.Accuracy, ur.Precision, ur.Recall, ur.F1, ur["ROC-AUC"]]],
    columns=["Task", "Best model (5-fold CV F1)", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])

# ------------------------------------------------------------------ BUILD BODY
experiment_flow()
S = []

hero = f"""<div class="hero"><h1>ScamShield</h1>
<p>Everyone gets scam texts and fake delivery links. Typical filters say "spam" without a reason, go stale as scammers reword their messages, and stop at the label. This project builds a scam checker that explains itself, is attacked on purpose to find its weak points, is measured against modern scam styles, and tells the person what to do next.</p>
<div class="strip" aria-label="Pipeline"><span class="good">Collect public data</span><span class="good">Detect scam text</span><span class="good">Detect phishing domains</span><span class="bad">Attack it</span><span class="bad">Harden it</span><span class="bad">Test on modern scams</span><span class="good">Explain and advise</span></div></div>
{table(glance, "Results at a glance (held-out test, best model chosen by 5-fold cross-validation)", "Details, caveats and every other model are in sections 3 and 4.", best_max=[])}
{fig("01_executive_dashboard.png", "Executive dashboard", "Top row: test F1 for every model on both tasks (red = best by cross-validation). Bottom left: how much obfuscated scam text evades each defence. Bottom right: the drop from the old test set to the modern probe.")}
{prose(f"Under a deliberate attacker, about <b>{lr.loc['Baseline','Evasion (all attacks)']:.0%}</b> of caught scams slip past the baseline text model; after hardening that falls to about <b>{lr.loc['+ Adversarial training','Evasion (all attacks)']:.0%}</b>. On 60 hand-written modern messages, the legacy-trained models score F1 between <b>{pm['Modern-probe F1'].min():.2f}</b> (Random Forest) and <b>{pm['Modern-probe F1'].max():.2f}</b>, against 0.94 to 0.97 on the old SMS test set.")}"""


# 1 ---- flow
files = pd.DataFrame([
    ["ss_common.py", "Config, logging, metric computation, plotting defaults", "all"],
    ["ss_data.py", "Data download + cleaning, SMS features, 67 domain features, brand look-alike book", "1, 2"],
    ["ss_attacks.py", "Seven evasion attacks (attacker) and the input canonicaliser (defender)", "3"],
    ["ss_eval.py", "Benchmark engine: CV, test, McNemar, 8 standard plots", "1, 2"],
    ["ss_agent.py", "Five-step agent: Extractor, Analyzer, Verifier, Scorer, Advisor", "4"],
    ["s1_sms.py", "SMS text models: EDA, tuning, 10-model benchmark", "1"],
    ["s2_url.py", "Domain models: 12-model benchmark, SHAP, ablation, hard subset", "2"],
    ["s3_adv.py", "Attacker-vs-defender loop (3 rounds, 4 configurations, 3 models)", "3"],
    ["s4_probe_agent.py", "60-message drift probe, adaptation curve, agent demo", "4"],
    ["s5_report.py", "Architecture figure, dashboard, Excel workbook", "5"],
    ["run_all.py", "Runs every stage in order", "all"]], columns=["File", "What it does", "Stage"])
S.append(section(1, "flow", "How the system and the experiment fit together",
    "The first chart shows what happens to a single message at run time. The second shows what the project did, in order, to build and test it.",
    fig("00_architecture.png", "Run-time architecture", "A message goes to an Extractor (links, phone numbers, amounts) and an Analyzer (manipulation tactics). A Verifier checks any domain against a phishing database and the domain model. Text and domain risks are fused by a noisy-OR rule and an Advisor explains the verdict and next steps. The orange arrow marks the offline loop in which an attacker rewrites scams and the text model is retrained.")
    + fig("00b_experiment_flow.png", "Experiment flow", "Every number in this report comes from one run of <code>run_all.py</code> (about 17 minutes on one CPU).")
    + table(files, "Where each part lives in the code")))

# 2 ---- data
S.append(section(2, "data", "The data",
    "Everything downloads from public URLs inside the code, so there is nothing to fetch by hand.",
    table(T("sms_dataset_summary"), "SMS dataset") + table(T("url_dataset_summary"), "Domain dataset",
        "Phishing domains are a random sample of the feed; legitimate domains are the OpenDNS popularity list. Ten domains appearing in both were removed as label noise.")
    + table(T("sms_eda_summary"), "How scam and legitimate texts differ",
        "Scams are about twice as long, carry roughly 15 digits against 0.3, and use an exclamation mark in 47% of messages against 12%.", fmt={"Legit": lambda v: f"{v:.2f}", "Scam": lambda v: f"{v:.2f}"})
    + fig("sms_01_eda.png", "SMS exploratory analysis", "Scam messages are 12.6% of the data after removing 403 duplicates. Length and digit count separate the classes clearly.")
    + fig("url_01_eda.png", "Domain exploratory analysis", "Phishing domains skew longer, higher in character entropy, and deeper in sub-domains.")
    + fig("url_02_correlation_heatmap.png", "Feature correlation heatmap", "The last row and column are the phishing label. Entropy (0.60), length (0.56) and sub-domain count (0.46) correlate most with the label, and several structure features are strongly collinear with each other.")))

# 3 ---- SMS
sms_test_cols = ["Model"] + MAXC + MINC[:2] + ["TN", "FP", "FN", "TP"]
bodys = prose("Ten models were tuned and then compared with 5-fold cross-validation on the training split. The best model is chosen by cross-validated F1, never by the test set.")
bodys += table(T("sms_tuning").drop(columns=["Search space"]), "Hyper-parameter search (3-fold CV, F1)", best_max=["Best CV F1"])
bodys += table(T("sms_cv_metrics_mean_std"), "5-fold cross-validation, mean ± standard deviation",
               "Logistic Regression leads on F1 (0.968) with the Voting Ensemble essentially tied (0.967).")
bodys += fig("sms_cv_heatmap.png", "Cross-validated metrics, models ranked by F1", "Darker is better.")
bodys += table(T("sms_test_metrics")[sms_test_cols], "Held-out test metrics (1,035 messages, 131 scams)",
               "Green marks the best value per column. Multinomial NB has the highest test F1 (0.965), but see the McNemar table below.",
               best_max=MAXC, best_min=MINC[:2])
bodys += fig("sms_test_heatmap.png", "Held-out test metrics heatmap")
bodys += fig("sms_test_metrics_bar.png", "Held-out test metrics by model")
bodys += fig("sms_confusion_matrices.png", "Confusion matrices for all ten models", "For Logistic Regression: 901 of 904 legitimate messages are correct (3 false alarms) and 122 of 131 scams are caught (9 missed).")
bodys += fig("sms_roc_pr_curves.png", "ROC and precision-recall curves", "All ten models reach ROC-AUC between 0.994 and 0.999, so the left panel saturates; the zoomed middle panel is where they differ.")
bodys += table(T("sms_best_classification_report"), "Per-class report for the best model (Logistic Regression)")
bodys += table(T("sms_mcnemar_vs_best"), "Is the best model really better? Exact McNemar test",
               "No comparison is significant (all p &gt; 0.25). With only 131 scams in the test set, the top models cannot be separated.", fmt={"McNemar exact p": lambda v: f"{v:.3f}"})
bodys += fig("sms_cv_f1_boxplot.png", "F1 across the five CV folds", "Logistic Regression's F1 varies by about ±0.023 between folds, which is larger than the gaps between top models.")
bodys += fig("sms_calibration.png", "Calibration of the top four models", "The closer a curve is to the diagonal, the more its probabilities can be read as real likelihoods.")
bodys += table(T("sms_efficiency"), "Speed", "Every text model predicts in under 0.25 ms per message. Training takes 0.6 s for Logistic Regression and 18 s for XGBoost.",
               fmt={"Train time (s)": lambda v: f"{v:.2f}", "Inference (ms / sample)": lambda v: f"{v:.3f}"}, best_min=["Train time (s)", "Inference (ms / sample)"])
bodys += fig("sms_efficiency.png", "Accuracy versus speed")
bodys += fig("sms_threshold_analysis.png", "Choosing the decision threshold", "On the test set the best F1 occurs at a threshold near 0.15 (F1 0.958, precision 0.948, recall 0.970) against F1 0.953 at 0.5. That threshold was picked on the test set, so treat the gain as optimistic; choose thresholds on validation data.")
bodys += fig("sms_learning_curve.png", "Learning curve", "Training F1 is 1.0 (the model memorises its training data) while cross-validated F1 rises from 0.927 to 0.961 and is flattening, so more of the same data would help only a little.")
tt = T("sms_top_terms"); bodys += fig("sms_02_top_terms.png", "Words that drive the scam decision", "'text', 'free', 'txt', 'won' and 'reply' push a message toward scam; casual words such as 'hey' and 'ur' push it toward legitimate.")
bodys += table(T("sms_learning_curve"), "Learning-curve values", fmt={"Train size": lambda v: f"{int(v):,}"}, open_=False, details=True)
S.append(section(3, "sms", "Detecting scam text", "Task: given an SMS, is it a scam? Data: the UCI SMS Spam Collection (5,171 unique messages).", bodys))

# 4 ---- URL
ub_ = prose("Twelve models were compared on 67 hand-built features of the domain string (length, entropy, digits, hyphens, sub-domains, TLD, keywords, free-hosting provider, look-alike similarity to popular brands).")
ub_ += table(T("url_cv_metrics_mean_std"), "5-fold cross-validation, mean ± standard deviation", "Every model lands within about 0.887 to 0.906 F1.")
ub_ += fig("url_cv_heatmap.png", "Cross-validated metrics, models ranked by F1")
ub_ += table(T("url_test_metrics")[sms_test_cols], "Held-out test metrics (3,996 domains)", "Green marks the best value per column.", best_max=MAXC, best_min=MINC[:2])
ub_ += fig("url_test_heatmap.png", "Held-out test metrics heatmap")
ub_ += fig("url_test_metrics_bar.png", "Held-out test metrics by model")
ub_ += fig("url_confusion_matrices.png", "Confusion matrices for all twelve models", "For the Voting Ensemble: 1,913 of 1,998 legitimate domains are correct (85 false alarms) and 1,726 of 1,998 phishing domains are caught (272 missed).")
ub_ += fig("url_roc_pr_curves.png", "ROC and precision-recall curves", "ROC-AUC ranges from 0.931 (Decision Tree) to 0.961.")
ub_ += table(T("url_best_classification_report"), "Per-class report for the best model (Voting Ensemble)")
ub_ += table(T("url_mcnemar_vs_best"), "Exact McNemar test against the best model", fmt={"McNemar exact p": lambda v: f"{v:.3f}"}, open_=False, details=True)
ub_ += fig("url_cv_f1_boxplot.png", "F1 across the five CV folds")
ub_ += fig("url_calibration.png", "Calibration of the top four models")
ub_ += table(T("url_efficiency"), "Speed", fmt={"Train time (s)": lambda v: f"{v:.2f}", "Inference (ms / sample)": lambda v: f"{v:.3f}"}, best_min=["Train time (s)", "Inference (ms / sample)"])
ub_ += fig("url_efficiency.png", "Accuracy versus speed")
ub_ += fig("url_threshold_analysis.png", "Choosing the decision threshold")
ub_ += prose("<b>What is the model actually using?</b> The next three figures show that sub-domain count dominates. That is partly a property of the data (the legitimate list holds bare domains) and not a pure phishing signal, which is why the ablation and hard-subset tests follow.")
ub_ += fig("url_feature_importance.png", "Feature importance by LightGBM gain", "Sub-domain count carries about half of the total gain.")
ub_ += fig("url_shap_summary.png", "SHAP summary", "Each dot is a domain. Many sub-domains (red) push strongly toward phishing.")
ub_ += fig("url_shap_bar.png", "Mean absolute SHAP value", "Sub-domain count is first by a wide margin, followed by length and entropy.", narrow=True)
ub_ += table(T("url_char_ngram_baseline")[["Model"] + M7], "A feature-free baseline", "A character n-gram model on the raw domain string scores F1 0.899, nearly the same as all the engineered-feature models.", best_max=M7)
ub_ += table(T("url_ablation"), "Feature-group ablation (LightGBM)", "Dropping the TLD features hurts most (test F1 0.905 to 0.879). Lexical, brand and keyword features alone reach 0.867.", best_max=["CV F1", "Test F1", "Test ROC-AUC", "Test Recall"])
ub_ += fig("url_ablation.png", "Ablation chart")
ub_ += table(T("url_hard_subset"), "The hard subset: bare domains with no sub-domain and no free host", "On these 2,576 test domains (595 phishing) F1 falls to about 0.65 and recall to about 0.55. This is the more honest estimate of how well the string alone identifies a phishing domain.")
ub_ += table(T("url_feature_importance").head(15), "Top 15 features by gain share", fmt={"Gain share": lambda v: f"{v:.3%}"})
ub_ += table(T("url_label_correlations").head(12), "Features most correlated with the label", open_=False, details=True)
S.append(section(4, "url", "Detecting phishing domains", "Task: given only the domain name, is it phishing? Data: 9,990 phishing domains from a public feed and 9,990 popular domains.", ub_))

# 5 ---- adversarial
cfg_order = ["Baseline", "+ Canonicalization", "+ Adversarial training", "+ Canon. & Adv. training"]
lr_t = adv[adv.Model == "Logistic Regression"].set_index("Config").loc[cfg_order].reset_index()
ad_cols = ["Config", "Round", "Accuracy", "Precision", "Recall", "F1", "FPR", "Evasion (seen attacks)", "Evasion (unseen attacks)", "Evasion (all attacks)", "FPR on obfuscated legit"]
pa = T("adv_per_attack"); pa_lr = pa[pa.Model == "Logistic Regression"].pivot(index="Attack", columns="Config", values="Evasion rate")[cfg_order].reset_index()
pa_lr["Known to defender"] = pa_lr.Attack.map(lambda a: "seen" if a in ("leet", "space", "synonym", "punct") else "unseen")
ab = prose("An attacker rewrites scams that the detector currently catches, using seven tricks (letter-for-symbol swaps, spaced letters, synonyms, inserted punctuation, look-alike Unicode letters, zero-width characters and appended harmless text). For each message it tries 30 combinations and keeps the one with the lowest scam score. <b>Evasion</b> is the share of caught scams that slip through.",
           "Four defences are compared: none, a canonicaliser that undoes cheap obfuscation, adversarial training (three rounds of retraining on the attacker's best rewrites), and both. Four of the seven tricks are visible to defenders in training; look-alike letters, zero-width characters and appended text are held back as <i>unseen</i>.")
ab += table(lr_t[ad_cols], "Final robustness summary, Logistic Regression (the best text model)",
            "Adversarial training cuts evasion of known tricks from 9.0% to 0.8% without hurting clean F1 (0.953 to 0.957). Unseen tricks only fall from 18.0% to 12.3%. With 122 caught scams, one message is 0.8 points, so 1 to 2 point gaps between defences are noise.",
            best_min=["Evasion (seen attacks)", "Evasion (unseen attacks)", "Evasion (all attacks)"], pct=("Evasion (seen attacks)", "Evasion (unseen attacks)", "Evasion (all attacks)", "FPR", "FPR on obfuscated legit"))
ab += fig("adv_02_attack_defence_heatmap.png", "Evasion by attack type and defence", "Appended harmless text is the one trick that no defence fixed: its single-attack evasion rose from 5% to 8% after hardening.")
ab += table(pa_lr, "Evasion per single attack type (Logistic Regression)", pct=tuple(cfg_order), best_min=cfg_order)
ab += fig("adv_01_evasion_by_round.png", "Evasion round by round for the three top models", "Dashed lines are defences without rounds; solid lines are adversarial training over three rounds.")
ab += fig("adv_03_robustness_tradeoff.png", "Robustness versus accuracy", "Top-left is best: low evasion, high clean F1.")
ab += fig("adv_04_cost_of_hardening.png", "Cost of hardening", "Clean F1 does not fall. False alarms on obfuscated legitimate messages rise slightly, from 0.1% to 0.4%, which is still tiny.")
ab += table(adv[["Model"] + ad_cols].sort_values(["Model", "Config"]), "Final summary for all three attacked models", open_=False, details=True,
            pct=("Evasion (seen attacks)", "Evasion (unseen attacks)", "Evasion (all attacks)", "FPR", "FPR on obfuscated legit"))
rr = T("adv_rounds"); ab += table(rr, "Every round of every configuration", "Full detail behind the figures above.", open_=False, details=True,
            pct=("Evasion (seen attacks)", "Evasion (unseen attacks)", "Evasion (all attacks)", "FPR", "FPR on obfuscated legit"))
S.append(section(5, "adv", "Attacker versus defender", "Scammers adapt, so a detector has to be tested against an adaptive attacker and not only against a fixed test set.", ab))

# 6 ---- drift
pd_cols = ["Model", "SMS test F1", "Modern-probe Accuracy", "Modern-probe Precision", "Modern-probe Recall", "Modern-probe F1", "Modern-probe Specificity", "Modern-probe ROC-AUC", "F1 drop (test → probe)"]
up = T("probe_online_update_summary"); up = up[["Reported messages (k)"] + [c for c in up.columns if "(mean)" in c]]
up.columns = [c.replace(" (mean)", "") for c in up.columns]
dr = prose("The SMS data is old. To measure how much that matters, 60 modern messages (30 scams, 30 legitimate) were written by hand: delivery-fee, toll, KYC, job, crypto, family-emergency and romance scams, plus tricky legitimate messages such as one-time passcodes and payment receipts. This is a small stress test, not a benchmark, and one message is worth 1.7 points.")
dr += table(T("probe_model_comparison")[pd_cols], "Legacy-trained models on the modern probe",
            "Linear and Naive Bayes models keep F1 around 0.80 to 0.86. Tree models collapse: Random Forest 0.33, Extra Trees 0.38, XGBoost 0.55. The hardened model is highest at 0.86, but that edge is within the noise of 60 messages.",
            best_max=["Modern-probe Accuracy", "Modern-probe Precision", "Modern-probe Recall", "Modern-probe F1", "Modern-probe Specificity", "Modern-probe ROC-AUC"])
dr += fig("probe_01_model_comparison.png", "Modern-probe metrics by model")
dr += fig("probe_02_drift_gap.png", "The drift gap", "Each line connects a model's legacy test F1 (blue) to its modern-probe F1 (red).")
dr += prose("<b>Can it learn from newly reported scams?</b> The hardened model was retrained with k reported messages and tested on 30 held-out messages, repeated over five random splits.")
dr += table(up, "Adaptation curve (mean over five splits)", "Ranking quality improves (ROC-AUC 0.924 to 0.936) and specificity rises (0.88 to 0.92), but recall does not (0.87 to 0.84). Performance on the original SMS data is kept (F1 0.957).", best_max=["Held-out ROC-AUC", "Held-out PR-AUC", "Held-out F1", "Held-out Specificity"], fmt={"Reported messages (k)": lambda v: f"{int(v)}"})
dr += fig("probe_03_online_update.png", "Adaptation curve")
dr += table(T("modern_probe_set"), "The 60 probe messages (y = 1 is a scam)", open_=False, details=True)
S.append(section(6, "drift", "When scams change style", "A detector trained on yesterday's scams meets tomorrow's. This section measures the drop and how quickly new examples repair it.", dr))

# 7 ---- agent
demo = json.load(open(TAB / "agent_demo_reports.json"))
def report_card(r):
    risk = r["risk"]; col = "var(--scam)" if risk >= .85 else ("var(--gold)" if risk >= .5 else "var(--legit)")
    ev = "".join(f"<li>{esc(w)}</li>" for w in r["why"]) or "<li>No strong signals found.</li>"
    terms = ", ".join(esc(a) for a, _ in r["top_terms"]) or "none"
    return (f'<article class="rc"><div class="rc-top"><span class="verdict" style="color:{col}">{esc(r["verdict"])}</span>'
            f'<span class="rbar"><i style="width:{risk*100:.0f}%;background:{col}"></i></span><span class="rnum">risk {risk:.2f}</span></div>'
            f'<blockquote>{esc(r["text"])}</blockquote>'
            f'<div class="rmeta">Text model P(scam) <b>{r["p_text"]:.3f}</b> &nbsp;|&nbsp; domain risk <b>{r["p_url"]:.3f}</b> &nbsp;|&nbsp; top terms: {terms}</div>'
            f'<ul>{ev}</ul><div class="radv"><b>Advice:</b> {esc(" ".join(r["advice"][:3]))}</div></article>')
ag = prose("Five specialists run in order. The <b>Extractor</b> pulls out links, phone numbers and amounts. The <b>Analyzer</b> tags manipulation tactics. The <b>Verifier</b> checks each domain against a phishing database, a look-alike test and the domain model. The <b>Scorer</b> fuses text and domain risk, and the <b>Advisor</b> writes the verdict and next steps. The agents are code-orchestrated tools calling the trained models; no language model is involved in the reported run.")
ag += table(T("agent_vs_text_only_probe"), "Agent versus text-only model on the modern probe", "The agent makes exactly the same decisions as the text model here (3 false alarms, 5 missed scams); only AUC moves, from 0.906 to 0.909. Illustrative, because the probe author chose the domain styles.", best_max=M7 + ["Specificity"])
ag += fig("probe_04_text_vs_agent_confusion.png", "Text model versus agent, confusion matrices")
ag += fig("agent_01_demo_risk.png", "Risk evidence for twelve example messages", "Blue bars are text-model scores, gold bars are domain-verifier scores, red diamonds the fused risk.")
ag += "<h3>Example reports</h3><p>Two failures are worth noticing: the family-emergency and romance scams score close to zero because they contain no link and no scammy vocabulary, and a legitimate payment receipt is flagged high risk.</p>" + "".join(report_card(r) for r in demo)
ag += table(T("agent_demo_table"), "The same examples as a table", open_=False, details=True)
S.append(section(7, "agent", "The ScamShield agent", "Beyond a yes/no label: show the evidence and tell the person what to do.", ag))

# 8 ---- code
fm = HtmlFormatter(nowrap=True, classprefix="t-")
order = ["run_all.py", "ss_common.py", "ss_data.py", "ss_attacks.py", "ss_eval.py", "ss_agent.py", "s1_sms.py", "s2_url.py", "s3_adv.py", "s4_probe_agent.py", "s5_report.py"]
code = prose("The complete source, in the order it runs. Use the buttons to fold everything away.") + '<div class="btns"><button type="button" id="expand">Expand all</button><button type="button" id="collapse">Collapse all</button></div>'
for fn in order:
    src = (ROOT / fn).read_text(); n = src.count("\n") + 1
    code += f'<details class="code" open><summary><code>{fn}</code> <span class="muted">{n} lines</span></summary><pre><code>{highlight(src, PythonLexer(), fm)}</code></pre></details>'
S.append(section(8, "code", "Source code", "Everything needed to reproduce the results.", code))

# 9 ---- limits
lim = "<ol class='lim'>" + "".join(f"<li>{x}</li>" for x in [
    "The SMS corpus is old, UK-style spam. Scores near 0.95 reflect an easy, dated benchmark, which the modern probe makes visible.",
    "The 60-message probe was written by the project author. It is a stress test, not a benchmark, and the agent-versus-text comparison on it is illustrative.",
    "Legitimate domains are popular bare domains and phishing domains come from a feed, so sub-domain structure is a shortcut. The hard-subset F1 of about 0.65 is the more cautious estimate.",
    "The attacker is rule-based. A language-model attacker would probably find more evasions. A Claude-based advisor hook exists in <code>ss_agent.py</code> but was never run.",
    "Combining text and domain risk was not validated on a jointly labelled dataset.",
    "Top models are statistically indistinguishable on the 131-scam test set, so 'best' means 'best by cross-validation' and not 'proven better'."]) + "</ol>"
S.append(section(9, "limits", "What these results do not show", "Read these before quoting a number.", lim + prose("The complete set of tables (including every cross-validation fold) is in <code>ScamShield_Results.xlsx</code>; all 39 figures are in the <code>figures/</code> folder.")))

# ------------------------------------------------------------------ CSS / JS / assemble
CSS = """
:root{--bg:#F4F6F8;--surface:#FFFFFF;--ink:#16212C;--muted:#566677;--rule:#D3DBE3;--scam:#B8334A;--legit:#1B6F96;--gold:#A8730A;--codebg:#EDF1F5;--tint:#E3F1E8;--tinkInk:#14532D;
--k:#8A2BE2;--s:#1B7F4C;--c:#6B7B8C;--n:#B05A00;--f:#1B6F96}
@media (prefers-color-scheme:dark){:root{--bg:#0D141B;--surface:#141D26;--ink:#E3EAF0;--muted:#92A2B2;--rule:#27333F;--scam:#F0788B;--legit:#5EB6DC;--gold:#E8B64C;--codebg:#0F1820;--tint:#173826;--tinkInk:#B8F0CC;
--k:#C792EA;--s:#7FD6A0;--c:#6E8196;--n:#F2A25C;--f:#7CC4E8}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.62 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1,h2,h3{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-weight:700;letter-spacing:-.01em;line-height:1.15}
.wrap{display:grid;grid-template-columns:230px minmax(0,1fr);gap:48px;max-width:1360px;margin:0 auto;padding:0 28px}
nav.toc{position:sticky;top:0;align-self:start;padding:36px 0;max-height:100vh;overflow:auto}
nav.toc a{display:block;color:var(--muted);text-decoration:none;padding:6px 0 6px 14px;border-left:2px solid var(--rule);font-size:14.5px}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--ink);border-left-color:var(--scam);outline:none}
nav.toc .t{font-family:Georgia,serif;font-weight:700;font-size:18px;margin-bottom:12px;display:block}
main{padding:40px 0 100px;min-width:0}
.hero h1{font-size:clamp(40px,6vw,64px);margin:0 0 14px}
.hero p{max-width:68ch;font-size:18px;color:var(--ink)}
.strip{display:flex;flex-wrap:wrap;align-items:center;gap:0;margin:26px 0 8px;font-size:14.5px}
.strip span{padding:7px 14px;background:var(--surface);border:1px solid var(--rule)}
.strip span:not(:last-child){border-right:none}
.strip span:first-child{border-radius:8px 0 0 8px}.strip span:last-child{border-radius:0 8px 8px 0}
.strip .bad{border-bottom:3px solid var(--scam)}.strip .good{border-bottom:3px solid var(--legit)}
section{margin-top:72px;scroll-margin-top:20px}
h2{font-size:32px;margin:0 0 8px;display:flex;align-items:baseline;gap:14px}
h2 .sn{font-size:15px;font-family:system-ui,sans-serif;color:var(--muted);border:1px solid var(--rule);border-radius:50%;width:30px;height:30px;display:inline-flex;align-items:center;justify-content:center;flex:none;transform:translateY(-4px)}
h3{font-size:22px;margin:30px 0 6px}
.lead{font-size:18px;color:var(--muted);max-width:70ch;margin:0 0 22px}
p{max-width:75ch}
figure{margin:26px 0 34px}
.plate{background:#fff;border:1px solid var(--rule);border-radius:6px;padding:10px;overflow:hidden}
.plate.narrow{max-width:760px}
.plate img{display:block;width:100%;height:auto;cursor:zoom-in}
figcaption{font-size:14.5px;color:var(--muted);max-width:90ch;margin-top:10px}
figcaption b{color:var(--ink)}
.tcap{font-size:14.5px;color:var(--muted);margin:26px 0 8px;max-width:95ch}.tcap b{color:var(--ink)}
.tscroll{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface);margin-bottom:30px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 12px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
td.num{text-align:right}
td:not(.num){white-space:normal;min-width:90px}
thead th{background:var(--codebg);position:sticky;top:0;font-weight:600;border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:none}tbody tr:hover{background:var(--codebg)}
td.best{background:var(--tint);color:var(--tinkInk);font-weight:700}
details.tdet>summary{cursor:pointer;list-style:none}details.tdet>summary .tcap{margin-bottom:8px}
details.tdet>summary::before{content:"+ ";font-weight:700;color:var(--muted)}details.tdet[open]>summary::before{content:"− "}
details.code{border:1px solid var(--rule);border-radius:6px;margin:14px 0;background:var(--surface)}
details.code>summary{cursor:pointer;padding:11px 16px;font-size:15px;position:sticky;top:0;background:var(--surface);border-radius:6px;z-index:2}
details.code pre{margin:0;padding:16px;overflow:auto;background:var(--codebg);border-top:1px solid var(--rule);font:12.8px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;max-height:70vh}
.muted{color:var(--muted)}
.t-k,.t-kn,.t-kc,.t-kd,.t-kr,.t-ow{color:var(--k)}.t-s,.t-s1,.t-s2,.t-sa,.t-sb,.t-se,.t-si,.t-sd{color:var(--s)}
.t-c,.t-c1,.t-cm,.t-cs{color:var(--c);font-style:italic}.t-mi,.t-mf,.t-mh,.t-m{color:var(--n)}.t-nf,.t-nc,.t-nd,.t-fm{color:var(--f)}.t-nb,.t-bp{color:var(--f)}
.btns{margin:8px 0 14px;display:flex;gap:10px}
button{font:inherit;font-size:14px;color:var(--ink);background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:6px 14px;cursor:pointer}
button:hover{border-color:var(--scam)}button:focus-visible,summary:focus-visible,.plate img:focus-visible{outline:2px solid var(--legit);outline-offset:2px}
.rc{border:1px solid var(--rule);border-radius:6px;background:var(--surface);padding:14px 18px;margin:14px 0;max-width:920px}
.rc-top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.verdict{font-weight:700}
.rbar{flex:1;min-width:120px;height:8px;background:var(--codebg);border-radius:4px;overflow:hidden}.rbar i{display:block;height:100%}
.rnum{font-variant-numeric:tabular-nums;color:var(--muted);font-size:14px}
.rc blockquote{margin:10px 0 8px;padding-left:12px;border-left:3px solid var(--rule);color:var(--ink)}
.rmeta{font-size:13.5px;color:var(--muted)}.rc ul{margin:8px 0 6px 18px;padding:0;font-size:14.5px}.radv{font-size:14.5px}
ol.lim{padding-left:22px;max-width:78ch}ol.lim li{margin:8px 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;z-index:50;padding:20px;cursor:zoom-out}
#lb img{max-width:100%;max-height:100%;background:#fff;border-radius:4px}#lb.on{display:flex}
@media (max-width:980px){.wrap{display:block;padding:0 16px}nav.toc{position:static;max-height:none;padding:16px 0 0;display:flex;gap:6px;overflow-x:auto}
nav.toc a{border-left:none;border-bottom:2px solid var(--rule);white-space:nowrap;padding:6px 8px}nav.toc .t{display:none}h2{font-size:26px}}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""
JS = """
document.querySelectorAll('img.zoom').forEach(function(i){i.tabIndex=0;function o(){var l=document.getElementById('lb');l.querySelector('img').src=i.src;l.classList.add('on')}
i.addEventListener('click',o);i.addEventListener('keydown',function(e){if(e.key==='Enter')o()})});
var lb=document.getElementById('lb');lb.addEventListener('click',function(){lb.classList.remove('on')});document.addEventListener('keydown',function(e){if(e.key==='Escape')lb.classList.remove('on')});
document.getElementById('expand').onclick=function(){document.querySelectorAll('details.code').forEach(function(d){d.open=true})};
document.getElementById('collapse').onclick=function(){document.querySelectorAll('details.code').forEach(function(d){d.open=false})};
"""
toc_items = [("flow", "Flow charts"), ("data", "Data"), ("sms", "Scam text"), ("url", "Phishing domains"), ("adv", "Attacker vs defender"), ("drift", "Style drift"), ("agent", "The agent"), ("code", "Source code"), ("limits", "Limitations")]
toc = '<nav class="toc" aria-label="Contents"><span class="t">ScamShield</span>' + "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in toc_items) + "</nav>"
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScamShield - full results report</title><style>{CSS}</style></head><body><div class="wrap">{toc}<main>{hero}{''.join(S)}</main></div>
<div id="lb" role="dialog" aria-label="Enlarged figure"><img alt=""></div><script>{JS}</script></body></html>"""
out = ROOT / "ScamShield_Report.html"; out.write_text(page, encoding="utf-8")
print("report", out.name, f"{out.stat().st_size/1e6:.1f} MB", "figures", _fig_n[0], "tables", _tab_n[0])
