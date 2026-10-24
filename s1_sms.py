"""Stage 1: SMS scam-text detection - EDA, tuning, 10-model benchmark, all plots/tables."""
import re, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ss_common import *
from ss_data import load_sms, make_sms_pipe, URL_RE
from ss_eval import *


def get_split():
    df = load_sms()
    Xtr, Xte, ytr, yte = train_test_split(df.text.values, df.y.values, test_size=0.2, stratify=df.y.values, random_state=SEED)
    return df, Xtr, Xte, ytr, yte


def lgbm(**kw):
    p = dict(n_estimators=300, num_leaves=31, learning_rate=0.08, colsample_bytree=.5, subsample=.8,
             subsample_freq=1, min_child_samples=5, n_jobs=1, verbose=-1, random_state=SEED)
    p.update(kw); return LGBMClassifier(**p)


def build_sms_clfs(t):
    """t = dict of tuned hyper-parameters."""
    ne = 60 if FAST else 250
    lr = LogisticRegression(C=t["lr_C"], max_iter=3000, class_weight="balanced")
    svm = CalibratedClassifierCV(LinearSVC(C=t["svm_C"], class_weight="balanced"), cv=3)
    mnb = MultinomialNB(alpha=t["mnb_alpha"])
    lg = lgbm(num_leaves=t["lgb_leaves"], n_estimators=t["lgb_n"])
    return {
        "Multinomial NB": mnb,
        "Complement NB": ComplementNB(alpha=t["cnb_alpha"]),
        "Logistic Regression": lr,
        "Linear SVM": svm,
        "Random Forest": RandomForestClassifier(n_estimators=ne, class_weight="balanced_subsample", n_jobs=1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=ne, class_weight="balanced_subsample", n_jobs=1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=80 if FAST else 300, max_depth=6, learning_rate=.1, subsample=.8,
                                 colsample_bytree=.5, tree_method="hist", n_jobs=1, eval_metric="logloss", random_state=SEED),
        "LightGBM": lg,
        "MLP (neural net)": MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED),
        "Voting Ensemble": VotingClassifier([("lr", LogisticRegression(C=t["lr_C"], max_iter=3000, class_weight="balanced")),
                                             ("svm", CalibratedClassifierCV(LinearSVC(C=t["svm_C"], class_weight="balanced"), cv=3)),
                                             ("nb", MultinomialNB(alpha=t["mnb_alpha"])),
                                             ("lgb", lgbm(num_leaves=t["lgb_leaves"], n_estimators=t["lgb_n"]))], voting="soft"),
    }


def tune(Xtr, ytr):
    specs = {
        "mnb_alpha": ("Multinomial NB", MultinomialNB(), {"clf__alpha": [.03, .1, .3, 1.0]}),
        "cnb_alpha": ("Complement NB", ComplementNB(), {"clf__alpha": [.1, .3, 1.0]}),
        "lr_C": ("Logistic Regression", LogisticRegression(max_iter=3000, class_weight="balanced"), {"clf__C": [1, 3, 10, 30, 100]}),
        "svm_C": ("Linear SVM", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3), {"clf__estimator__C": [.1, .3, 1, 3]}),
        "lgb": ("LightGBM", lgbm(), {"clf__num_leaves": [15, 31], "clf__n_estimators": [200, 400]}),
    }
    if FAST:
        specs["lr_C"] = (specs["lr_C"][0], specs["lr_C"][1], {"clf__C": [3, 30]})
        specs["lgb"] = (specs["lgb"][0], specs["lgb"][1], {"clf__num_leaves": [15], "clf__n_estimators": [100]})
    t, rows = {}, []
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for key, (nm, clf, grid) in specs.items():
        gs = GridSearchCV(make_sms_pipe(clf), grid, scoring="f1", cv=cv, n_jobs=1)
        gs.fit(Xtr, ytr); bp = gs.best_params_
        rows.append({"Model": nm, "Search space": str(grid).replace("clf__", "").replace("estimator__", ""),
                     "Best parameters": str(bp).replace("clf__", "").replace("estimator__", ""),
                     "Best CV F1": gs.best_score_})
        log(f"  tuned {nm}: {bp}  F1={gs.best_score_:.4f}")
        if key == "lgb":
            t["lgb_leaves"], t["lgb_n"] = bp["clf__num_leaves"], bp["clf__n_estimators"]
        elif key == "svm_C":
            t["svm_C"] = bp["clf__estimator__C"]
        else:
            t[key] = list(bp.values())[0]
    return t, pd.DataFrame(rows)


def eda_figs(df):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    vc = df.label.value_counts()
    ax[0].bar(["Legit (ham)", "Scam (spam)"], [vc["ham"], vc["spam"]], color=[C_HAM, C_SPAM])
    for i, v in enumerate([vc["ham"], vc["spam"]]):
        ax[0].text(i, v + 40, f"{v:,}\n({v/len(df):.1%})", ha="center")
    ax[0].set_title("Class distribution (after de-duplication)"); ax[0].set_ylim(0, vc.max() * 1.18)
    df["len"] = df.text.str.len(); df["digits"] = df.text.apply(lambda t: sum(c.isdigit() for c in t))
    for lab, c in [("ham", C_HAM), ("spam", C_SPAM)]:
        sns.kdeplot(df[df.label == lab]["len"], ax=ax[1], color=c, fill=True, alpha=.35, label=lab, clip=(0, 400))
        sns.kdeplot(df[df.label == lab]["digits"], ax=ax[2], color=c, fill=True, alpha=.35, label=lab, clip=(0, 40))
    ax[1].set_title("Message length (characters)"); ax[2].set_title("Number of digits in message")
    ax[1].legend(); ax[2].legend()
    savefig("sms_01_eda.png")
    s = pd.DataFrame({"Feature": ["Mean length (chars)", "Mean digits", "Contains URL (%)", "Contains '!' (%)", "ALL-CAPS ratio"],
                      "Legit": [df[df.y == 0]["len"].mean(), df[df.y == 0].digits.mean(),
                                100 * df[df.y == 0].text.apply(lambda t: bool(URL_RE.search(t))).mean(),
                                100 * df[df.y == 0].text.str.contains("!").mean(),
                                df[df.y == 0].text.apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)).mean()],
                      "Scam": [df[df.y == 1]["len"].mean(), df[df.y == 1].digits.mean(),
                               100 * df[df.y == 1].text.apply(lambda t: bool(URL_RE.search(t))).mean(),
                               100 * df[df.y == 1].text.str.contains("!").mean(),
                               df[df.y == 1].text.apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)).mean()]})
    save_table(s.round(3), "sms_eda_summary")


def top_terms(pipe_lr, k=15):
    feats = pipe_lr.named_steps["feat"].get_feature_names_out()
    coef = pipe_lr.named_steps["clf"].coef_.ravel()
    d = pd.DataFrame({"feature": feats, "coef": coef})
    d = d[d.feature.str.startswith("word__")].assign(feature=lambda x: x.feature.str[6:])
    top = pd.concat([d.nlargest(k, "coef"), d.nsmallest(k, "coef")])
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(data=top, y="feature", x="coef", palette=[C_SPAM if c > 0 else C_HAM for c in top.coef], ax=ax)
    ax.axvline(0, color="k", lw=.8); ax.set_ylabel(""); ax.set_xlabel("Logistic-regression coefficient (+ scam / − legit)")
    ax.set_title("Most influential words for the scam decision")
    savefig("sms_02_top_terms.png")
    save_table(top.reset_index(drop=True).round(4), "sms_top_terms")
    return d.nlargest(60, "coef").feature.tolist()


def run():
    log("STAGE 1 - SMS text classification")
    df, Xtr, Xte, ytr, yte = get_split()
    log(f"  messages={len(df)} (raw {df.attrs['n_raw']}, duplicates removed {df.attrs['n_raw']-len(df)}), "
        f"train={len(Xtr)}, test={len(Xte)}, spam rate={df.y.mean():.3%}")
    ds = pd.DataFrame({"Item": ["Source", "Raw messages", "After de-duplication", "Scam (spam) messages", "Legit (ham) messages",
                                "Scam share", "Train / test split", "Cross-validation"],
                       "Value": ["UCI SMS Spam Collection (GitHub mirror, loaded via URL)", df.attrs["n_raw"], len(df),
                                 int(df.y.sum()), int((1 - df.y).sum()), f"{df.y.mean():.2%}",
                                 f"{len(Xtr)} / {len(Xte)} (stratified 80/20)", "5-fold stratified on training split"]})
    save_table(ds, "sms_dataset_summary")
    eda_figs(df)

    t, tune_df = tune(Xtr, ytr); save_table(tune_df.round(4), "sms_tuning")
    clfs = build_sms_clfs(t)
    models = {n: make_sms_pipe(c) for n, c in clfs.items()}
    res = benchmark(models, Xtr, ytr, Xte, yte, "sms", cv=3 if FAST else 5)
    ranked = rank_models(res); best = ranked[0]
    log(f"  BEST by CV-F1: {best}")

    save_table(cv_pm_table(res), "sms_cv_metrics_mean_std")
    save_table(res["cv_mean"].reset_index().round(4), "sms_cv_metrics_mean")
    save_table(res["cv_long"].round(4), "sms_cv_metrics_per_fold")
    te = res["test"].loc[ranked].reset_index(); save_table(te.round(4), "sms_test_metrics")
    save_table(res["eff"].loc[ranked].reset_index().round(4), "sms_efficiency")
    mc = mcnemar_vs_best(res, best); save_table(mc, "sms_mcnemar_vs_best")
    cr = classification_report(yte, (res["proba"][best] >= .5).astype(int), target_names=["Legit", "Scam"], output_dict=True)
    save_table(pd.DataFrame(cr).T.reset_index().rename(columns={"index": "Class"}).round(4), "sms_best_classification_report")

    full_report_figs(res, "sms", "SMS scam detection")
    plot_threshold_df = plot_threshold(res, "sms", "SMS scam detection", best)
    save_table(plot_threshold_df.iloc[::max(1, len(plot_threshold_df) // 60)].round(4), "sms_threshold_table")

    trig = top_terms(res["fitted"]["Logistic Regression"])
    # learning curve for the best model
    log("  learning curve ...")
    sizes = np.linspace(.15, 1.0, 4 if FAST else 6)
    ts, trs, vs = learning_curve(make_sms_pipe(clfs[best]), Xtr, ytr, cv=3 if FAST else 5, scoring="f1",
                                 train_sizes=sizes, n_jobs=1, random_state=SEED, shuffle=True)
    fig, ax = plt.subplots(figsize=(7.6, 5))
    for arr, c, l in [(trs, C_HAM, "Training F1"), (vs, C_SPAM, "Cross-validated F1")]:
        ax.plot(ts, arr.mean(1), "o-", color=c, label=l); ax.fill_between(ts, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), color=c, alpha=.15)
    ax.set_xlabel("Training examples"); ax.set_ylabel("F1"); ax.set_title(f"Learning curve - {best}"); ax.legend(loc="lower right")
    savefig("sms_learning_curve.png")
    save_table(pd.DataFrame({"Train size": ts, "Train F1": trs.mean(1), "CV F1": vs.mean(1), "CV F1 std": vs.std(1)}).round(4), "sms_learning_curve")

    joblib.dump(dict(tuned=t, ranked=ranked, best=best, triggers=trig, fitted=res["fitted"],
                     test_proba=res["proba"], yte=yte), CACHE / "sms_state.joblib")
    log("STAGE 1 done")


if __name__ == "__main__":
    run()
