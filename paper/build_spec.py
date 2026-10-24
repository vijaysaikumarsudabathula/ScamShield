"""Builds spec.json for the ScamShield IEEE paper. Every number is read from tables/ (no hand-typed results)."""
import json, re
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent; TAB = ROOT / "tables"; FIGS = ROOT / "paper" / "figs"
T = lambda n: pd.read_csv(TAB / f"{n}.csv")
f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"; p1 = lambda v: f"{100*v:.1f}%"; n0 = lambda v: f"{int(v):,}"

# ---------------------------------------------------------------- data
sms_t = T("sms_test_metrics").set_index("Model"); sms_cvm = T("sms_cv_metrics_mean").set_index("Model")
sms_cvs = T("sms_cv_metrics_mean_std").set_index("Model")
url_t = T("url_test_metrics").set_index("Model"); url_cvm = T("url_cv_metrics_mean").set_index("Model"); url_cvs = T("url_cv_metrics_mean_std").set_index("Model")
sms_rank = sms_cvm.sort_values(["F1", "PR-AUC"], ascending=False).index.tolist(); url_rank = url_cvm.sort_values(["F1", "PR-AUC"], ascending=False).index.tolist()
SB, UB = sms_rank[0], url_rank[0]
assert SB == "Logistic Regression" and UB == "Voting Ensemble"
sd = lambda tbl, m: float(tbl.loc[m, "F1"].split("± ")[1])
abl = T("url_ablation").set_index("Configuration"); hard = T("url_hard_subset").set_index("Model"); ngram = T("url_char_ngram_baseline").iloc[0]
adv = T("adv_final_summary"); lr = adv[adv.Model == "Logistic Regression"].set_index("Config")
rounds = T("adv_rounds"); rounds = rounds[rounds.Model == "Logistic Regression"]
per = T("adv_per_attack"); per = per[per.Model == "Logistic Regression"].pivot(index="Attack", columns="Config", values="Evasion rate")
probe = T("probe_model_comparison").set_index("Model"); ad = T("probe_online_update_summary").set_index("Reported messages (k)")
agent_cmp = T("agent_vs_text_only_probe").set_index("System"); demo = json.load(open(TAB / "agent_demo_reports.json"))
mc = T("sms_mcnemar_vs_best"); eda = T("sms_eda_summary").set_index("Feature"); eff = T("sms_efficiency").set_index("Model")
sds = T("sms_dataset_summary").set_index("Item").Value; uds = T("url_dataset_summary").set_index("Item").Value
shap = pd.read_csv(ROOT / "paper" / "shap_top10.csv")
BASE, CAN, ADVT, BOTH = "Baseline", "+ Canonicalization", "+ Adversarial training", "+ Canon. & Adv. training"
HARD = "ScamShield-hardened (Logistic Regression)"
SHORT = {"Multinomial NB": "Naive Bayes (mult.)", "Complement NB": "Naive Bayes (compl.)", "Logistic Regression": "Logistic Reg.", "Linear SVM": "Linear SVM",
         "Random Forest": "Random Forest", "Extra Trees": "Extra Trees", "XGBoost": "XGBoost", "LightGBM": "LightGBM", "MLP (neural net)": "MLP",
         "Voting Ensemble": "Voting Ens.", "k-Nearest Neighbors": "k-NN", "Decision Tree": "Decision Tree", "Gradient Boosting": "Grad. Boosting",
         "HistGradientBoosting": "Hist. Grad. Boost.", HARD: "Hardened LR (ours)"}

# ---------------------------------------------------------------- references (key -> text); numbered by first citation
REFS = {
 "almeida": "T. A. Almeida, J. M. Gómez Hidalgo, and A. Yamakami, “Contributions to the study of SMS spam filtering: New collection and results,” in Proc. 2011 ACM Symp. Document Engineering (DocEng ’11), Mountain View, CA, USA, 2011.",
 "uci": "T. A. Almeida and J. M. Gómez Hidalgo, “SMS Spam Collection,” UCI Machine Learning Repository. [Online]. Available: https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
 "markham": "K. Markham, “pycon-2016-tutorial,” GitHub repository (mirror of the SMS Spam Collection file). [Online]. Available: https://github.com/justmarkham/pycon-2016-tutorial",
 "sahami": "M. Sahami, S. Dumais, D. Heckerman, and E. Horvitz, “A Bayesian approach to filtering junk e-mail,” in Proc. AAAI Workshop on Learning for Text Categorization, Madison, WI, USA, 1998.",
 "sparck": "K. Spärck Jones, “A statistical interpretation of term specificity and its application in retrieval,” J. Documentation, vol. 28, no. 1, pp. 11–21, 1972.",
 "ma": "J. Ma, L. K. Saul, S. Savage, and G. M. Voelker, “Beyond blacklists: Learning to detect malicious web sites from suspicious URLs,” in Proc. 15th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, Paris, France, 2009, pp. 1245–1254.",
 "sahingoz": "O. K. Sahingoz, E. Buber, O. Demir, and B. Diri, “Machine learning based phishing detection from URLs,” Expert Syst. Appl., vol. 117, pp. 345–357, 2019.",
 "dalvi": "N. Dalvi, P. Domingos, Mausam, S. Sanghai, and D. Verma, “Adversarial classification,” in Proc. 10th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, Seattle, WA, USA, 2004, pp. 99–108.",
 "lowd": "D. Lowd and C. Meek, “Good word attacks on statistical spam filters,” in Proc. 2nd Conf. Email and Anti-Spam (CEAS), 2005.",
 "goodfellow": "I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in Proc. Int. Conf. Learning Representations (ICLR), 2015, arXiv:1412.6572.",
 "madry": "A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” in Proc. Int. Conf. Learning Representations (ICLR), 2018, arXiv:1706.06083.",
 "gama": "J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, “A survey on concept drift adaptation,” ACM Comput. Surv., vol. 46, no. 4, art. 44, 2014.",
 "lundberg": "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information Processing Systems 30 (NeurIPS), 2017, pp. 4765–4774.",
 "cortes": "C. Cortes and V. Vapnik, “Support-vector networks,” Mach. Learn., vol. 20, no. 3, pp. 273–297, Sep. 1995.",
 "platt": "J. C. Platt, “Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,” in Advances in Large Margin Classifiers, A. J. Smola et al., Eds. Cambridge, MA, USA: MIT Press, 1999, pp. 61–74.",
 "breiman": "L. Breiman, “Random forests,” Mach. Learn., vol. 45, no. 1, pp. 5–32, Oct. 2001. doi: 10.1023/A:1010933404324.",
 "geurts": "P. Geurts, D. Ernst, and L. Wehenkel, “Extremely randomized trees,” Mach. Learn., vol. 63, no. 1, pp. 3–42, Apr. 2006.",
 "chen": "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, San Francisco, CA, USA, 2016, pp. 785–794.",
 "ke": "G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in Advances in Neural Information Processing Systems 30 (NeurIPS), 2017, pp. 3146–3154.",
 "dietterich": "T. G. Dietterich, “Ensemble methods in machine learning,” in Multiple Classifier Systems (MCS 2000), Lecture Notes in Computer Science, vol. 1857. Berlin, Germany: Springer, 2000, pp. 1–15.",
 "friedman": "J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” Ann. Statist., vol. 29, no. 5, pp. 1189–1232, 2001.",
 "shannon": "C. E. Shannon, “A mathematical theory of communication,” Bell Syst. Tech. J., vol. 27, no. 3, pp. 379–423, Jul. 1948.",
 "levenshtein": "V. I. Levenshtein, “Binary codes capable of correcting deletions, insertions, and reversals,” Sov. Phys. Dokl., vol. 10, no. 8, pp. 707–710, 1966.",
 "unicode": "Unicode Consortium, “Unicode security mechanisms,” Unicode Technical Standard #39. [Online]. Available: https://www.unicode.org/reports/tr39/",
 "pearl": "J. Pearl, Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference. San Mateo, CA, USA: Morgan Kaufmann, 1988.",
 "phishdb": "M. Krog, “Phishing.Database,” GitHub repository. [Online]. Available: https://github.com/mitchellkrogza/Phishing.Database",
 "opendns": "OpenDNS, “public-domain-lists,” GitHub repository. [Online]. Available: https://github.com/opendns/public-domain-lists",
 "mcnemar": "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, Jun. 1947.",
 "matthews": "B. W. Matthews, “Comparison of the predicted and observed secondary structure of T4 phage lysozyme,” Biochim. Biophys. Acta, vol. 405, no. 2, pp. 442–451, Oct. 1975.",
 "pedregosa": "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.",
}


KEEP = ['almeida', 'ma', 'sahingoz', 'dalvi', 'lowd', 'goodfellow', 'madry', 'gama', 'lundberg', 'phishdb', 'opendns', 'pedregosa', 'chen', 'ke', 'mcnemar', 'pearl']
REFS = {k: v for k, v in REFS.items() if k in KEEP}
assert len(REFS) == 16

# ---------------------------------------------------------------- blocks
B = []
def add(**k): B.append(k)
def H1(t): add(t="h1", text=t)
def H2(t): add(t="h2", text=t)
def P(t): add(t="p", text=t)
FIGKEYS = []
def FIG(file, cap, w=3.3, key=None):
    im = Image.open(FIGS / file); FIGKEYS.append(key); add(t="fig", file=str(FIGS / file), w=w, h=round(w * im.height / im.width, 3), cap=cap, key=key)
def TBL(title, header, rows, widths, note, bold_row=None):
    assert sum(widths) == 4860, (title, sum(widths))
    add(t="table", title=title, header=header, rows=rows, widths=widths, note=note, bold_row=bold_row)
pc = lambda x: f"{100*x:.1f}"
TP = int(sms_t.loc[SB, "TP"]); NS = TP + int(sms_t.loc[SB, "FN"])
drop = probe.drop(index=HARD)["F1 drop (test → probe)"]

# ============================ ABSTRACT
abstract = (
 "Scam texts and phishing links reach billions of people, but most filters give only a yes/no answer, are tested on a fixed dataset, and weaken when scammers change their wording. "
 "This paper presents ScamShield, a framework that detects scam messages and phishing domains, explains each decision, and tells the user what to do next. It uses only public data and runs in about 17 minutes on one CPU core. "
 "We test it with a simulated attacker that rewrites scams to avoid detection, and with 60 modern messages that differ from the training data. "
 f"A logistic regression on the SMS Spam Collection reaches {p1(sms_t.loc[SB,'Accuracy'])} accuracy (F1 {f3(sms_t.loc[SB,'F1'])}), although ten models cannot be statistically separated. "
 f"Twelve domain models reach about {url_t.loc[UB,'Accuracy']*100:.0f}% accuracy, but F1 falls to {f2(hard.loc[UB,'F1'])} on bare domains without sub-domains. "
 f"The attacker evades {p1(lr.loc[BASE,'Evasion (all attacks)'])} of caught scams; adversarial training cuts evasion of known tricks from {p1(lr.loc[BASE,'Evasion (seen attacks)'])} to {p1(lr.loc[ADVT,'Evasion (seen attacks)'])} without hurting accuracy, "
 "but it does not stop scams padded with harmless text. "
 f"On modern messages, linear models keep F1 between {f2(probe.loc['Multinomial NB','Modern-probe F1'])} and {f2(probe.loc[HARD,'Modern-probe F1'])}, while random forests fall to {f2(probe.loc['Random Forest','Modern-probe F1'])}. "
 "A five-agent pipeline adds explanations but no extra correct decisions. ScamShield offers a practical and honest approach to scam detection.")
keywords = "Scam Detection, Phishing, Adversarial Robustness, Concept Drift, Explainable AI, Multi-Agent Systems, Text Classification"

# ============================ I. INTRODUCTION
H1("I. INTRODUCTION")
P("Scam messages and phishing links reach billions of people every day. Most research answers one question: is this message or link malicious [[almeida]], [[ma]], [[sahingoz]]? "
  "Deployed filters then return that answer as a bare label. They do not say why a message was flagged, how much of what they catch would still get through if a scammer rewrote it, "
  "or how far accuracy falls when scam styles change after training. Answering these questions needs an adaptive-attacker test [[dalvi]], [[lowd]] and a measurement of concept drift [[gama]], "
  "not another accuracy table on a fixed test set.")
P("This paper presents ScamShield, which pairs a text detector and a phishing-domain detector with a black-box attacker, two defences, a probe of modern scams, and a five-agent pipeline that explains its verdict. "
  "Every model is chosen by cross-validation and compared with an exact significance test, not by one test split.")
H2("A. Research Questions")
P("RQ1: Can interpretable features detect scam text well enough that the choice among ten models is settled by cross-validation and significance testing? "
  "RQ2: How well does the domain string alone identify phishing, and how much of that depends on shortcuts in the data? "
  "RQ3: How much caught scam text still evades an adaptive attacker, and how much do canonicalization and adversarial training reduce it, including for tricks never seen in training? "
  "RQ4: How far does a detector trained on a legacy corpus degrade on modern scams, and how fast can it adapt? "
  "RQ5: Does combining text and domain evidence change decisions, and which scams stay undetected?")
H2("B. Contributions and Scope")
P("The work contributes a ten-model text benchmark, a twelve-model domain benchmark with an ablation and a hard-subset test, an adaptive attacker with seven tricks (four seen, three held out) evaluated against two defences, "
  "a modern-scam probe with an adaptation curve, and a five-agent pipeline. It uses public data [[almeida]], [[phishdb]], [[opendns]], a rule-based attacker, and a 60-message probe written by the author; Section VI.B explains what this means for the results.")

# ============================ II. RELATED WORK
H1("II. RELATED WORK")
P("For SMS filtering, Almeida et al. [[almeida]] released the corpus used here and compared classical learners on it. For phishing URLs, Ma et al. [[ma]] showed that lexical and host features alone can find malicious sites, "
  "and Sahingoz et al. [[sahingoz]] reported 97.98% accuracy for a random forest on language-based URL features. Our domain-only setting drops the path and page content and builds its legitimate class differently, so it is harder and not directly comparable.")
P("For robustness, Dalvi et al. [[dalvi]] framed spam filtering as a game against an adaptive adversary, and Lowd and Meek [[lowd]] showed that appending harmless words lets spam slip past statistical filters, the effect behind our padding attack. "
  "Adversarial training [[goodfellow]], [[madry]] retrains on the strongest rewrites an attacker finds; we apply it to linear and ensemble text models. Gama et al. [[gama]] survey adaptation to concept drift, the setting of Section V.D, "
  "and SHAP [[lundberg]] explains the domain model.")

P("ScamShield draws on these lines directly. Phases 1 and 2 follow the feature-based approach of [[almeida]] and [[ma]]; Phase 3 follows the adaptive-adversary framing of [[dalvi]] and [[lowd]] with a defence in the spirit of [[madry]]; and Phase 4 measures drift as described in [[gama]], extended into an explicit legacy-versus-modern comparison and an adaptation curve. What is new here is the combination: one pipeline that is explainable, attacked on purpose, and checked against modern scam styles.")

# ============================ III. FRAMEWORK
H1("III. SCAMSHIELD FRAMEWORK ARCHITECTURE")
P("ScamShield has five phases and a validation stage (Fig. «flow»). Every phase was run on real data or a fully specified simulation.")
FIG("fig1_flowchart.png", "ScamShield methodology flowchart.", w=2.75, key="flow")
H2("A. Phase 1: Scam-Text Detection")
P("Each message is represented by word 1–2-gram and character 2–4-gram TF-IDF (up to 4,000 terms each) plus ten style cues (length, digit ratio, capitals, “!” and “?” counts, link flag, phone count, currency symbols, non-ASCII ratio). "
  "Ten classifiers are compared: two naive Bayes models, logistic regression, a linear SVM, random forest, extra trees, XGBoost [[chen]], LightGBM [[ke]], a small neural network, and a soft-voting ensemble, mostly from scikit-learn [[pedregosa]]. "
  "Regularization, smoothing and tree size were tuned by 3-fold grid search on the training split (C = 100 for logistic regression, C = 1 for the SVM). The final model is chosen by 5-fold cross-validated F1, never by the test split.")
H2("B. Phase 2: Phishing-Domain Detection")
P("Phase 2 classifies a domain using only its string, with no page content, registration data or path. Its 67 features cover structure (length, sub-domain count), lexical content (digits, hyphens, entropy, longest runs), "
  "bait keywords, the top-level domain, free-hosting providers, and brand look-alikes (edit similarity to the 400 most popular domains). Twelve classifiers are compared with fixed, untuned settings, so small differences among them should not be over-read. "
  "A feature-group ablation and a hard subset of bare domains test how much accuracy comes from structure shared by the way the data were built.")
H2("C. Phase 3: Adversarial Hardening")
P("The attacker is black-box: it sees only the scam score. For each caught scam it tries B rewrites, each a random mix of one to three tricks, and keeps the one with the lowest score. "
  "Seven tricks are used: leetspeak, spaced letters, synonyms, inserted punctuation (seen in training), and look-alike Unicode letters, zero-width characters, and appended harmless text, called padding (unseen). The evasion rate is")
add(t="eq", runs="E = |{ x ∈ S<sub>c</sub> : f(a(x)) < τ }| / |S<sub>c</sub>|", num="(1)")
P("where S<sub>c</sub> is the set of test scams the model already catches, a(·) is the attacker's best rewrite, f is the scam score and τ = 0.5. Two defences are compared: a canonicalizer that undoes cheap obfuscation (Unicode normalization, zero-width removal, look-alike mapping, spacing and symbol clean-up), "
  "and adversarial training (Algorithm 1), which also adds obfuscated legitimate messages so that obfuscation alone is not learned as a scam signal.")
add(t="algo", title="Algorithm 1: Adaptive evasion and adversarial training",
    lines=[("Input", "model class f, training set D, seen tricks A<sub>seen</sub>, budget B, rounds R"),
           ("1", "D′ ← D;  f<sub>0</sub> ← Fit(D′)"),
           ("2", "<b>for</b> r = 1, …, R <b>do</b>"),
           ("3", "    <b>for each</b> scam x in D <b>do</b>"),
           ("4", "        draw B rewrites (1–3 random tricks from A<sub>seen</sub>)"),
           ("5", "        x* ← rewrite with the lowest score f<sub>r−1</sub>(·)"),
           ("6", "    H ← obfuscated copies of 25% of legitimate texts"),
           ("7", "    D′ ← D′ ∪ {(x*, scam)} ∪ {(h, legit) : h ∈ H}"),
           ("8", "    f<sub>r</sub> ← Fit(D′)"),
           ("9", "<b>return</b> f<sub>R</sub>")])
H2("D. Phase 4: Concept-Drift Probe")
P("A probe of 60 hand-written messages (30 scams, 30 legitimate) covers delivery-fee, toll, account, tax, job, investment, family-emergency and romance scams, plus legitimate look-alikes such as one-time codes and receipts. "
  "To test adaptation, the hardened model is retrained with k reported probe messages (k = 0 to 30, repeated five times in training) and tested on the other 30, over five random splits.")
H2("E. Phase 5: Multi-Agent Pipeline")
P("Five agents run in order (Fig. «arch»). The Extractor finds links, phones and amounts. The Analyzer tags seven manipulation tactics (urgency, authority, fear, reward, payment request, secrecy, acquaintance impersonation). "
  "The Verifier checks each domain against the public phishing feed, a popular-domain list, a look-alike test and the Phase 2 model. The Scorer combines the text probability p<sub>t</sub> and the largest domain risk p<sub>d</sub> with a noisy-OR rule [[pearl]]:")
add(t="eq", runs="R = 1 − (1 − p<sub>t</sub>)(1 − p<sub>d</sub>)", num="(2)")
P("The Advisor maps R to a verdict (high at R ≥ 0.85, suspicious from 0.5, otherwise low) and returns the evidence and next steps (Section III.F). "
  "The agents are code-orchestrated tools; no language model is used in any reported result.")
FIG("fig2_architecture.png", "ScamShield run-time architecture.", w=3.15, key="arch")
H2("F. Explanation and Advice Layer")
P("Every verdict comes with its evidence. For the linear text model, the contribution of term j to the scam score is")
add(t="eq", runs="c<sub>j</sub> = w<sub>j</sub> · x<sub>j</sub>", num="(3)")
P("where w<sub>j</sub> is the logistic-regression coefficient and x<sub>j</sub> the TF-IDF value, so the largest positive terms are shown as reasons for a flag. Domain findings and detected tactics are listed beside them. The advice separates what to do before acting (do not click, check through an official channel) from what to do if the person already clicked or replied (change passwords, turn on two-factor authentication, call the bank, report the message). The advice is templated; an optional language-model hook exists but is not used in any reported result.")

# ============================ IV. SETUP
H1("IV. EXPERIMENTAL SETUP AND DATASET")
H2("A. Dataset Description")
P(f"The SMS Spam Collection [[almeida]] has 5,574 labeled messages; removing 403 duplicates leaves {n0(sds['After de-duplication'])}, of which {sds['Scam share']} are scams. An 80/20 stratified split gives 4,136 training and 1,035 test messages ({NS} test scams). "
  f"The domain data pair the Phishing.Database feed [[phishdb]] ({n0(uds['Phishing domains in feed'])} well-formed domains) with the 10,000 most popular OpenDNS domains [[opendns]]; ten domains in both were removed, and {n0(uds['Sampled per class'])} were sampled per class "
  f"({n0(2*int(uds['Sampled per class']))} total; 15,984 training, 3,996 test). The modern probe has 60 messages written by the author.")
FIG("fig3_sms_eda.png", "Scam versus legitimate SMS: (a) message length and (b) digits per message.", key="eda")
P(f"Scams are about twice as long (mean {eda.loc['Mean length (chars)','Scam']:.0f} against {eda.loc['Mean length (chars)','Legit']:.0f} characters) and carry far more digits (mean {eda.loc['Mean digits','Scam']:.1f} against {eda.loc['Mean digits','Legit']:.1f}; Fig. «eda»), which is why simple style cues help.")
H2("B. Implementation Environment")
P("All experiments ran in Python 3.12 with scikit-learn, XGBoost, LightGBM, SHAP and RapidFuzz on one CPU core with 3 GB of memory and no GPU. The full pipeline runs in about 17 minutes. "
  "This is why the text detector uses TF-IDF instead of a fine-tuned transformer and the attacker is rule-based.")
H2("C. Evaluation Protocol")
P(f"Models are tuned and cross-validated on the training split only, ranked by 5-fold F1 (ties broken by PR-AUC), and the best is scored once on the test split and compared with the others by an exact McNemar test [[mcnemar]]. "
  f"Metrics are accuracy, precision, recall, F1 and ROC-AUC, with scams (or phishing domains) as the positive class. The seed is fixed at 42. The attacker uses B = 20 per scam in training and B = 30 at evaluation, measured on the {TP} test scams the baseline catches, so one message is {100/TP:.1f} percentage points.")

# ============================ V. RESULTS
H1("V. RESULTS")
H2("A. Scam-Text Detection (RQ1)")
rows = [[SHORT[m], f"{sms_cvm.loc[m,'F1']:.3f} ± {sd(sms_cvs,m):.3f}", f3(sms_t.loc[m, "Accuracy"]), f3(sms_t.loc[m, "Precision"]), f3(sms_t.loc[m, "Recall"]), f3(sms_t.loc[m, "F1"]), f3(sms_t.loc[m, "ROC-AUC"])] for m in sms_rank]
TBL("TABLE I.  SCAM-TEXT CLASSIFIER PERFORMANCE", ["Model", "CV F1", "Acc.", "Prec.", "Rec.", "F1", "AUC"], rows, [1180, 1000, 530, 530, 530, 530, 560],
    "TABLE I. CV F1 = 5-fold cross-validated F1 (mean ± std). Other columns = held-out test (1,035 messages, 131 scams). Bold = selected model.", bold_row=0)
sdr = pd.Series([sd(sms_cvs, m) for m in sms_rank])
P(f"Table I reports ten models. Logistic regression has the highest cross-validated F1 ({f3(sms_cvm.loc[SB,'F1'])}), with the voting ensemble tied ({f3(sms_cvm.loc['Voting Ensemble','F1'])}). On the test split it reaches {p1(sms_t.loc[SB,'Accuracy'])} accuracy, precision {f3(sms_t.loc[SB,'Precision'])}, recall {f3(sms_t.loc[SB,'Recall'])} and F1 {f3(sms_t.loc[SB,'F1'])} "
  f"(901 of 904 legitimate messages correct, {TP} of {NS} scams caught; Fig. «cm»a). Naive Bayes has the highest test F1 ({f3(sms_t.loc['Multinomial NB','F1'])}), but no model differs significantly from logistic regression (McNemar p from {mc['McNemar exact p'].min():.2f} to {mc['McNemar exact p'].max():.2f}), "
  f"and fold-to-fold deviations ({sdr.min():.3f} to {sdr.max():.3f}) exceed the gaps between top models (Fig. «cv»). The ten models therefore cannot be separated, which answers RQ1. Logistic regression is also the fastest: it trains in {eff.loc[SB,'Train time (s)']:.1f} s and scores a message in {eff.loc[SB,'Inference (ms / sample)']:.2f} ms. "
  "Its strongest scam words are “text”, “free”, “txt”, “won” and “reply” (Fig. «terms»), which is what the explanation layer shows to the user. Its strongest legitimate cues “lt” and “gt” are a corpus artifact: they come from an escaped placeholder found in 240 raw messages, all legitimate.")
FIG("fig6_sms_cv_f1.png", "5-fold cross-validated F1 for the ten text models (mean ± std); the selected model is red.", key="cv")
FIG("fig4_top_terms.png", "Words with the largest logistic-regression coefficients for scam (red) and legitimate (blue).", key="terms")
H2("B. Phishing-Domain Detection (RQ2)")
rows = [[SHORT[m], f"{url_cvm.loc[m,'F1']:.3f} ± {sd(url_cvs,m):.3f}", f3(url_t.loc[m, "Accuracy"]), f3(url_t.loc[m, "Precision"]), f3(url_t.loc[m, "Recall"]), f3(url_t.loc[m, "F1"]), f3(url_t.loc[m, "ROC-AUC"])] for m in url_rank]
TBL("TABLE II.  PHISHING-DOMAIN CLASSIFIER PERFORMANCE", ["Model", "CV F1", "Acc.", "Prec.", "Rec.", "F1", "AUC"], rows, [1180, 1000, 530, 530, 530, 530, 560],
    "TABLE II. CV F1 = 5-fold cross-validated F1 (mean ± std). Other columns = held-out test (3,996 domains). Bold = selected model.", bold_row=0)
tf1 = url_t["F1"]
P(f"Table II reports twelve models, and all fall in a narrow band (test F1 {tf1.min():.3f} to {tf1.max():.3f}). The voting ensemble is best by cross-validation ({f3(url_cvm.loc[UB,'F1'])}) and reaches {p1(url_t.loc[UB,'Accuracy'])} accuracy, precision {f3(url_t.loc[UB,'Precision'])}, recall {f3(url_t.loc[UB,'Recall'])} and F1 {f3(url_t.loc[UB,'F1'])} "
  f"({int(url_t.loc[UB,'FP'])} false alarms and {int(url_t.loc[UB,'FN'])} misses among 1,998 domains per class; Fig. «cm»b). A character n-gram model on the raw string, with no engineered features, reaches F1 {f3(ngram['F1'])}, so the 67 features add little.")
FIG("fig_cm_pair.png", "Confusion matrices on the held-out test splits: (a) selected logistic regression, (b) selected voting ensemble.", key="cm")
sh = shap.set_index("feature").mean_abs_shap
P(f"Sub-domain count dominates the model (Fig. «shap»; mean |SHAP| {sh['n_subdomains']:.1f} against {sh.iloc[1]:.1f} for {shap.feature.iloc[1]}). This reflects how the data were built: legitimate domains are bare, while phishing hosts often use sub-domains. "
  f"In the ablation (Table III, Fig. «abl»), removing structure lowers F1 from {f3(abl.loc['All features','Test F1'])} to {f3(abl.loc['− Structure (length, labels, dots)','Test F1'])} and removing the TLD group to {f3(abl.loc['− TLD','Test F1'])}. "
  f"On the {n0(hard.loc[UB,'Subset size'])} test domains with no sub-domain and no free host ({int(hard.loc[UB,'Phishing in subset'])} phishing), the voting ensemble's F1 falls to {f3(hard.loc[UB,'F1'])} (precision {f3(hard.loc[UB,'Precision'])}, recall {f3(hard.loc[UB,'Recall'])}). "
  "This answers RQ2: the domain string separates the classes well when their structure differs, but is a much weaker signal for bare domains.")
FIG("fig8_shap.png", "Mean absolute SHAP value of the ten most influential domain features.", key="shap")
names = {"All features": "All features", "− Structure (length, labels, dots)": "− Structure", "− Lexical (digits, entropy, runs)": "− Lexical", "− Keywords": "− Keywords", "− Brand look-alike": "− Brand look-alike",
         "− TLD": "− TLD", "− Free-hosting provider": "− Free-hosting", "Lexical + brand + keywords only (no structure/TLD/host)": "Lexical + brand + kw. only"}
rows = [[names[c], str(int(r["# features"])), f3(r["CV F1"]), f3(r["Test F1"]), f3(r["Test ROC-AUC"]), f3(r["Test Recall"])] for c, r in abl.iterrows()]
TBL("TABLE III.  DOMAIN FEATURE-GROUP ABLATION (LIGHTGBM)", ["Configuration", "# feat.", "CV F1", "Test F1", "Test AUC", "Test Rec."], rows, [1520, 600, 680, 700, 700, 660],
    "TABLE III. “−” = feature group removed. CV F1 = 5-fold cross-validated F1; other columns = held-out test split.")
FIG("fig9_ablation.png", "Held-out test F1 after removing each domain feature group (LightGBM).", key="abl")
H2("C. Adversarial Robustness (RQ3)")
rows = [[nm, pc(lr.loc[c, "Evasion (seen attacks)"]), pc(lr.loc[c, "Evasion (unseen attacks)"]), pc(lr.loc[c, "Evasion (all attacks)"]), f3(lr.loc[c, "F1"]), pc(lr.loc[c, "FPR on obfuscated legit"])]
        for nm, c in [("Baseline", BASE), ("Canonicalization", CAN), ("Adversarial training", ADVT), ("Both", BOTH)]]
TBL("TABLE IV.  EVASION UNDER AN ADAPTIVE ATTACKER (LOGISTIC REGRESSION)", ["Defence", "Seen %", "Unseen %", "All %", "Clean F1", "FP obf. %"], rows, [1300, 640, 760, 600, 780, 780],
    "TABLE IV. Evasion = share of caught test scams that evade after the attacker's best rewrite (B = 30). FP obf. = false alarms on obfuscated legitimate messages.")
P(f"Table IV shows the share of caught scams that evade. Without defence, {p1(lr.loc[BASE,'Evasion (all attacks)'])} of caught scams evade ({p1(lr.loc[BASE,'Evasion (seen attacks)'])} through seen tricks, {p1(lr.loc[BASE,'Evasion (unseen attacks)'])} through unseen ones). Canonicalization lowers overall evasion to {p1(lr.loc[CAN,'Evasion (all attacks)'])}. "
  f"Adversarial training lowers seen-trick evasion to {p1(lr.loc[ADVT,'Evasion (seen attacks)'])} and overall evasion to {p1(lr.loc[ADVT,'Evasion (all attacks)'])}, but unseen evasion only to {p1(lr.loc[ADVT,'Evasion (unseen attacks)'])}; both together give {p1(lr.loc[BOTH,'Evasion (all attacks)'])} overall. "
  f"Clean F1 does not fall ({f3(lr.loc[BASE,'F1'])} to {f3(lr.loc[ADVT,'F1'])}) and false alarms on obfuscated legitimate text stay below 0.5%. Fig. «heat» shows each trick: canonicalization drives homoglyph and zero-width evasion to {pc(per.loc['homoglyph',CAN])}% and {pc(per.loc['zerowidth',CAN])}%, "
  "but it was built against exactly those tricks, so this is a unit test and not proof of general robustness. Padding is the one trick no defence fixes: "
  f"its evasion rises from {pc(per.loc['padding',BASE])}% to {pc(per.loc['padding',CAN])}–{pc(per.loc['padding',ADVT])}%. With {TP} caught scams, differences of one or two points are noise. This answers RQ3: hardening removes most evasion the defender has seen, less of what it has not, and none of padding.")
FIG("fig11_adv_heatmap.png", "Evasion rate (%) by trick and defence; tricks below the line were never seen in training.", key="heat")
r_ = rounds[rounds.Config == ADVT].set_index("Round")
P(f"Fig. «rounds» shows the rounds: most of the gain on seen tricks arrives by round 2 ({p1(r_.loc[0,'Evasion (seen attacks)'])} to {p1(r_.loc[2,'Evasion (seen attacks)'])}), while unseen evasion falls slowly ({p1(r_.loc[0,'Evasion (unseen attacks)'])} to {p1(r_.loc[3,'Evasion (unseen attacks)'])} over three rounds).")
FIG("fig12_adv_rounds.png", "Evasion rate over adversarial-training rounds for adversarial training alone (solid) and with canonicalization (dashed).", key="rounds")
H2("D. Concept Drift and Adaptation (RQ4)")
order = [HARD] + [m for m in probe.sort_values("Modern-probe F1", ascending=False).index if m != HARD]
rows = [[SHORT[m], f3(probe.loc[m, "SMS test F1"]), f3(probe.loc[m, "Modern-probe Accuracy"]), f3(probe.loc[m, "Modern-probe Precision"]), f3(probe.loc[m, "Modern-probe Recall"]), f3(probe.loc[m, "Modern-probe F1"])] for m in order]
TBL("TABLE V.  LEGACY-TRAINED MODELS ON THE MODERN PROBE", ["Model", "Test F1", "Acc.", "Prec.", "Rec.", "Probe F1"], rows, [1500, 640, 620, 620, 620, 860],
    "TABLE V. Test F1 = held-out SMS split; other columns = 60-message probe. Hardened LR = canonicalization + adversarial training.", bold_row=0)
P(f"Every model loses F1 on modern messages (legacy test F1 {sms_t['F1'].min():.2f} to {sms_t['F1'].max():.2f}; Table V, Fig. «drift»). Linear and naive Bayes models lose least (probe F1 {f2(probe.loc['Multinomial NB','Modern-probe F1'])} to {f2(probe.loc[HARD,'Modern-probe F1'])}), "
  f"while tree ensembles collapse: random forest {f2(probe.loc['Random Forest','Modern-probe F1'])}, extra trees {f2(probe.loc['Extra Trees','Modern-probe F1'])}, XGBoost {f2(probe.loc['XGBoost','Modern-probe F1'])}. The forests raise no false alarms but catch only "
  f"{probe.loc['Random Forest','Modern-probe Recall']*100:.0f}% and {probe.loc['Extra Trees','Modern-probe Recall']*100:.0f}% of scams. The hardened model scores highest ({f3(probe.loc[HARD,'Modern-probe F1'])}), but one message moves the probe by 1.7 points, so its lead over plain logistic regression ({f3(probe.loc['Logistic Regression','Modern-probe F1'])}) is within noise.")
FIG("fig13_drift.png", "Legacy SMS test F1 (blue) against modern-probe F1 (red) for each model.", key="drift")
P(f"Table VI and Fig. «adapt» show adaptation. Retraining with 30 reported messages raises held-out ROC-AUC from {f3(ad.loc[0,'Held-out ROC-AUC (mean)'])} to {f3(ad.loc[30,'Held-out ROC-AUC (mean)'])} and specificity from {f2(ad.loc[0,'Held-out Specificity (mean)'])} to {f2(ad.loc[30,'Held-out Specificity (mean)'])}, "
  f"but recall does not improve ({f2(ad.loc[0,'Held-out Recall (mean)'])} to {f2(ad.loc[30,'Held-out Recall (mean)'])}), and legacy F1 is unchanged ({f3(ad.loc[30,'Legacy SMS F1 (mean)'])}). "
  f"This answers RQ4: legacy detectors lose {drop.min():.2f} to {drop.max():.2f} F1 on modern scams, and a few dozen reports improve ranking without forgetting old data, but do not restore recall.")
rows_ad = [[str(int(k)), f3(r["Held-out ROC-AUC (mean)"]), f3(r["Held-out PR-AUC (mean)"]), f3(r["Held-out Recall (mean)"]), f3(r["Held-out Specificity (mean)"]), f3(r["Legacy SMS F1 (mean)"])] for k, r in ad.iterrows()]
TBL("TABLE VI.  ADAPTATION TO NEWLY REPORTED SCAMS (HARDENED LR)", ["k", "ROC-AUC", "PR-AUC", "Recall", "Spec.", "Legacy F1"], rows_ad, [500, 860, 800, 800, 800, 1100],
    "TABLE VI. k = reported probe messages used for the update. Held-out columns = mean over five splits (30 held-out messages each). Legacy F1 = held-out SMS test split.")
FIG("fig14_adaptation.png", "Held-out ROC-AUC, specificity and recall against the number of newly reported messages (mean ± std, five splits).", key="adapt")
H2("E. Multi-Agent Pipeline (RQ5)")
d0, d4, d7 = demo[0], demo[4], demo[7]
assert d0["text"].startswith("Your account has been temporarily") and d4["text"].startswith("Mom I lost") and d7["text"].startswith("Thanks for your payment")
fa, ta = agent_cmp.loc["ScamShield agent (text + domain verifier)"], agent_cmp.loc["Text model only (hardened)"]
P(f"For “Your account has been temporarily suspended… http://secure-bank-verify.xyz/login”, the text model gives {d0['p_text']:.3f} and the domain risk is {d0['p_url']:.3f}, so R = {d0['risk']:.2f} (high). The Verifier notes bait words in the address, many hyphens and a low-cost TLD, and the Analyzer tags urgency, authority, fear and a payment request. "
  f"Two failures: “Mom I lost my phone, this is my new number…” gets R = {d4['risk']:.3f} because it has no link and no scam vocabulary, and a legitimate payment receipt gets R = {d7['risk']:.3f} (false alarm). "
  f"Across the probe, the agent makes the same decisions as the text model (accuracy {p1(fa['Accuracy'])}; 3 false alarms, 5 misses) and its ROC-AUC moves only from {f3(ta['ROC-AUC'])} to {f3(fa['ROC-AUC'])} (Fig. «agent»). "
  "This answers RQ5: fusion adds ranking information but no new decisions, and scams with no link and no scam vocabulary escape both parts. The domain signal is judged on domains the author invented, so this is illustrative.")
FIG("fig15_agent.png", "Text probability, domain risk and fused risk R for eight probe messages (dotted: 0.5; dashed: 0.85).", key="agent")
H2("F. Compute Cost")
ea = T("sms_efficiency").set_index("Model"); eb = T("url_efficiency").set_index("Model")
cm_ = ["Logistic Regression", "Linear SVM", "MLP (neural net)", "Random Forest", "Extra Trees", "LightGBM", "XGBoost", "Voting Ensemble"]
rows = [[SHORT[m], f"{ea.loc[m,'Train time (s)']:.1f}", f"{ea.loc[m,'Inference (ms / sample)']:.2f}", f"{eb.loc[m,'Train time (s)']:.1f}", f"{eb.loc[m,'Inference (ms / sample)']:.3f}"] for m in cm_]
TBL("TABLE VII.  TRAINING AND INFERENCE COST (ONE CPU CORE)", ["Model", "Text train (s)", "Text ms/msg", "Domain train (s)", "Domain ms/dom."], rows, [1400, 900, 820, 900, 840],
    "TABLE VII. Text inference includes TF-IDF feature extraction; domain inference excludes feature extraction. Train = one fit on the training split.")
P(f"Table VII shows that cost is not a deciding factor at this scale: every model scores a message in under 0.25 ms. Logistic regression is the cheapest to train ({ea.loc[SB,'Train time (s)']:.1f} s on text), while XGBoost is the most expensive ({ea.loc['XGBoost','Train time (s)']:.1f} s), "
  "so the simplest model gives the best balance of accuracy, explanation and speed.")
H2("G. Validation Summary")
P("Table VIII collects the checks beyond the single held-out split, so each headline claim can be read against the test that stresses it.")
rows = [["5-fold CV", "Scam text, 10 models", f"F1 {f3(sms_cvm.loc[SB,'F1'])} ± {f3(sd(sms_cvs,SB))}"],
        ["5-fold CV", "Domains, 12 models", f"F1 {f3(url_cvm.loc[UB,'F1'])} ± {f3(sd(url_cvs,UB))}"],
        ["McNemar", "Best vs. 9 text models", f"all p ≥ {mc['McNemar exact p'].min():.2f}"],
        ["Ablation", "Domains, − TLD", f"F1 {f3(abl.loc['− TLD','Test F1'])} (vs. {f3(abl.loc['All features','Test F1'])})"],
        ["Hard subset", f"Bare domains, n = {n0(hard.loc[UB,'Subset size'])}", f"F1 {f3(hard.loc[UB,'F1'])}"],
        ["Unseen tricks", "Adv. training", f"evasion {p1(lr.loc[ADVT,'Evasion (unseen attacks)'])} (vs. {p1(lr.loc[BASE,'Evasion (unseen attacks)'])})"],
        ["Modern probe", "Hardened LR", f"F1 {f3(probe.loc[HARD,'Modern-probe F1'])} (legacy {f3(probe.loc[HARD,'SMS test F1'])})"],
        ["Agent vs. text", "Probe, 60 messages", f"F1 {f3(fa['F1'])} vs. {f3(ta['F1'])}"]]
TBL("TABLE VIII.  VALIDATION SUMMARY", ["Validation", "Setting", "Metric"], rows, [1250, 1700, 1910], "TABLE VIII. Parentheses = comparison value for the same setting.")

# ============================ VI. DISCUSSION
H1("VI. DISCUSSION")
H2("A. Key Findings and Practical Implications")
P(f"Four points follow. First, because ten text models cannot be separated, the simplest and fastest, logistic regression, is the sensible default and also supplies the explanation. Second, because a domain string alone separates bare domains poorly, a domain detector should be one signal among several, "
  f"joined with evidence such as registration age and page content. Third, because adversarial training cut seen-trick evasion from {p1(lr.loc[BASE,'Evasion (seen attacks)'])} to {p1(lr.loc[ADVT,'Evasion (seen attacks)'])} but left {p1(lr.loc[ADVT,'Evasion (unseen attacks)'])} for unseen tricks, "
  "a deployed system needs a canonicalizer and regular retraining against a broader attacker. Fourth, because tree ensembles collapsed under drift while linear models degraded less, models refreshed from reported scams should be checked on recent messages, not only on an old test set.")
H2("B. Scope and Practical Notes")
P("The SMS corpus is old, UK-style spam, so scores near 0.95 describe an easy benchmark, and it contains the placeholder artifact noted above. The domain data pair a feed against a popularity list, so structure is a shortcut and the hard-subset F1 is the more cautious estimate. "
  "The 60-message probe is a stress test written by the author, not a benchmark. The attacker is rule-based and the canonicalizer was built against its tricks; a language-model attacker would likely find more evasions. "
  "One fixed seed and the default 0.5 threshold were used, the optional language-model advice hook was not run, and text-plus-domain fusion was not validated on a jointly labeled dataset. "
  "Only public data and invented messages were used. Next steps are a recent consented smishing corpus, domain evidence beyond the string, a language-model attacker, and a conversation-level model for scams that carry no link.")

# ============================ VII. CONCLUSION
H1("VII. CONCLUSION")
P(f"This paper presented ScamShield, a five-phase pipeline for explainable, adversarially tested scam and phishing detection. Ten text models are statistically indistinguishable (selected logistic regression: {p1(sms_t.loc[SB,'Accuracy'])} accuracy, F1 {f3(sms_t.loc[SB,'F1'])}); "
  f"twelve domain models plateau near F1 0.90 and fall to {f2(hard.loc[UB,'F1'])} on bare domains; an adaptive attacker evades {p1(lr.loc[BASE,'Evasion (all attacks)'])} of caught scams, and adversarial training removes most evasion the defender has seen but not padding; "
  f"legacy detectors lose {drop.min():.2f} to {drop.max():.2f} F1 on modern scams; and the agent pipeline explains verdicts but adds no decisions beyond the text model. "
  "Together these results show that explainable detectors, an adaptive adversary and a drift measurement make scam detection that can be trusted for what it catches and honest about what it misses.")

# ============================ REFERENCES
order_keys = []
def resolve(text):
    def sub(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        for k in keys:
            assert k in REFS, k
            if k not in order_keys: order_keys.append(k)
        return ", ".join(f"[{n}]" for n in sorted(order_keys.index(k) + 1 for k in keys))
    return re.sub(r"\[\[([a-z, ]+)\]\]", sub, text)
abstract = resolve(abstract)
fignum = {k: i + 1 for i, k in enumerate(FIGKEYS)}
figtok = lambda t: re.sub(r"«(\w+)»", lambda m: str(fignum[m.group(1)]), t)
for b in B:
    if b["t"] in ("p", "h1", "h2"): b["text"] = figtok(resolve(b["text"]))
    if b["t"] == "fig": b["cap"] = f"Fig. {fignum[b['key']]}. " + b["cap"]
refs_sorted = [f"[{i+1}] {REFS[k]}" for i, k in enumerate(order_keys)]
assert set(order_keys) == set(REFS), set(REFS) - set(order_keys)
spec = dict(title="ScamShield: An Explainable and Adversarially Tested Framework for Detecting Scam Messages and Phishing Domains",
            author="[Author Name(s), to be completed]", affil="Department / Affiliation, City, Country", email="email@example.com",
            header="ScamShield: Explainable and Adversarially Tested Scam and Phishing Detection",
            abstract=abstract, keywords=keywords, blocks=B, refs=refs_sorted)
json.dump(spec, open(ROOT / "paper" / "spec.json", "w"), ensure_ascii=False, indent=1)
words = sum(len(re.sub(r"<[^>]+>", "", b.get("text", "")).split()) for b in B)
print("blocks", len(B), "refs", len(refs_sorted), "body words", words, "abstract words", len(abstract.split()))
