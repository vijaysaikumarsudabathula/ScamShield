"""Stage 2: phishing-DOMAIN detection - features, 12-model benchmark, ablation, SHAP."""
import joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ss_common import *
from ss_data import load_domains, BrandBook, domain_features, FEATURE_GROUPS
from ss_eval import *


def build_url_models():
    ne = 50 if FAST else 300
    lg = lambda: LGBMClassifier(n_estimators=ne, num_leaves=31, learning_rate=.05, subsample=.8, subsample_freq=1,
                                colsample_bytree=.8, n_jobs=1, verbose=-1, random_state=SEED)
    xg = lambda: XGBClassifier(n_estimators=ne, max_depth=6, learning_rate=.08, subsample=.8, colsample_bytree=.8,
                               tree_method="hist", n_jobs=1, eval_metric="logloss", random_state=SEED)
    rf = lambda: RandomForestClassifier(n_estimators=ne, n_jobs=1, random_state=SEED)
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)),
        "Linear SVM": make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(C=.3, max_iter=5000), cv=3)),
        "k-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=SEED),
        "Random Forest": rf(),
        "Extra Trees": ExtraTreesClassifier(n_estimators=ne, n_jobs=1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=40 if FAST else 150, max_depth=4, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=ne, random_state=SEED),
        "XGBoost": xg(),
        "LightGBM": lg(),
        "MLP (neural net)": make_pipeline(StandardScaler(), MLPClassifier((64, 32), early_stopping=True, max_iter=300, random_state=SEED)),
        "Voting Ensemble": VotingClassifier([("rf", rf()), ("xgb", xg()), ("lgb", lg()),
                                             ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))], voting="soft"),
    }


def run():
    log("STAGE 2 - phishing-domain classification")
    n = 1500 if FAST else 10000
    dd, ranked_benign, phish_all, info = load_domains(n)
    brands = BrandBook(ranked_benign)
    F = domain_features(dd.domain.tolist(), brands)
    y = dd.y.values
    log(f"  domains={len(dd)}  features={F.shape[1]}  {info}")
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(F, y, dd.domain.values, test_size=.2, stratify=y, random_state=SEED)

    ds = pd.DataFrame({"Item": ["Phishing source", "Legitimate source", "Phishing domains in feed", "Legit domains in list",
                                "Overlap removed (label noise)", "Sampled per class", "Engineered features", "Train / test split", "Cross-validation"],
                       "Value": ["Phishing.Database ACTIVE domains (GitHub raw)", "OpenDNS public top-10k domains (GitHub raw)",
                                 info["phish_total"], info["benign_total"], info["overlap_removed"], info["n_per_class"], F.shape[1],
                                 f"{len(Xtr)} / {len(Xte)} (stratified 80/20)", "5-fold stratified on training split"]})
    save_table(ds, "url_dataset_summary")

    # ---------------- EDA
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    d = F.assign(Class=np.where(y == 1, "Phishing", "Legit"))
    pal = {"Phishing": C_SPAM, "Legit": C_HAM}
    for a, col, ttl in zip(ax.ravel()[:5], ["length", "entropy", "n_subdomains", "brand_sim", "n_digits"],
                           ["Domain length", "Shannon entropy", "Number of sub-domains", "Look-alike similarity to top brands", "Digits in domain"]):
        if col in ("n_subdomains", "n_digits"):
            sns.countplot(data=d[d[col] <= 6], x=col, hue="Class", palette=pal, ax=a)
        else:
            sns.kdeplot(data=d, x=col, hue="Class", palette=pal, fill=True, alpha=.35, common_norm=False, ax=a)
        a.set_title(ttl)
    tl = d.assign(tld=[c.replace("tld_", "") for c in F[[f"tld_{t}" for t in ["com", "net", "org", "xyz", "top", "info", "online", "site", "club", "icu", "tk", "ml", "ga", "cf", "gq", "app", "io", "co", "in", "ru"]]].idxmax(1)])
    tt = tl.groupby(["tld", "Class"]).size().reset_index(name="n"); order = tt.groupby("tld").n.sum().nlargest(10).index
    sns.barplot(data=tt[tt.tld.isin(order)], x="tld", y="n", hue="Class", palette=pal, order=order, ax=ax.ravel()[5])
    ax.ravel()[5].set_title("Top TLDs by class")
    savefig("url_01_eda.png")

    num = [c for c in F.columns if not c.startswith("tld_")]
    cm = F[num].assign(is_phishing=y).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Pearson r"}, xticklabels=True, yticklabels=True)
    ax.set_title("Feature correlation heatmap (last row/col = phishing label)"); ax.tick_params(labelsize=8)
    savefig("url_02_correlation_heatmap.png")
    save_table(cm["is_phishing"].drop("is_phishing").sort_values(key=abs, ascending=False).reset_index()
               .rename(columns={"index": "Feature", "is_phishing": "Correlation with label"}).round(4), "url_label_correlations")

    # ---------------- benchmark
    models = build_url_models()
    res = benchmark(models, Xtr, ytr, Xte, yte, "url", cv=3 if FAST else 5)
    rk = rank_models(res); best = rk[0]; log(f"  BEST by CV-F1: {best}")
    save_table(cv_pm_table(res), "url_cv_metrics_mean_std"); save_table(res["cv_mean"].reset_index().round(4), "url_cv_metrics_mean")
    save_table(res["cv_long"].round(4), "url_cv_metrics_per_fold")
    save_table(res["test"].loc[rk].reset_index().round(4), "url_test_metrics")
    save_table(res["eff"].loc[rk].reset_index().round(4), "url_efficiency")
    save_table(mcnemar_vs_best(res, best), "url_mcnemar_vs_best")
    cr = classification_report(yte, (res["proba"][best] >= .5).astype(int), target_names=["Legit", "Phishing"], output_dict=True)
    save_table(pd.DataFrame(cr).T.reset_index().rename(columns={"index": "Class"}).round(4), "url_best_classification_report")
    full_report_figs(res, "url", "Phishing-domain detection", labels=("Legit", "Phishing"))
    thr = plot_threshold(res, "url", "Phishing-domain detection", best)
    save_table(thr.iloc[::max(1, len(thr) // 60)].round(4), "url_threshold_table")

    # ---------------- char n-gram text model on the raw domain string (feature-free baseline)
    from sklearn.model_selection import cross_validate
    ng = Pipeline([("tf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3, max_features=30000, sublinear_tf=True)),
                   ("lr", LogisticRegression(C=10, max_iter=3000))])
    ng.fit(dtr, ytr); png = ng.predict_proba(dte)[:, 1]
    row = {"Model": "Char n-gram TF-IDF + LR (raw string)"}; row.update(compute_metrics(yte, png))
    ngdf = pd.DataFrame([row]); save_table(ngdf.round(4), "url_char_ngram_baseline")
    log(f"  char-ngram baseline F1={row['F1']:.4f}")

    # ---------------- feature importance + SHAP (LightGBM)
    lgb_m = res["fitted"]["LightGBM"]
    imp = pd.Series(lgb_m.booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    top = imp.head(20)[::-1]; ax.barh(top.index, top.values / imp.sum(), color=C_ACC, edgecolor="k")
    ax.set_xlabel("Share of total split gain"); ax.set_title("Top-20 features (LightGBM gain)")
    savefig("url_feature_importance.png")
    save_table((imp / imp.sum()).reset_index().rename(columns={"index": "Feature", 0: "Gain share"}).round(5), "url_feature_importance")
    try:
        import shap
        samp = Xte.sample(min(1500, len(Xte)), random_state=SEED)
        ex = shap.TreeExplainer(lgb_m); sv = ex.shap_values(samp)
        if isinstance(sv, list): sv = sv[1]
        plt.figure(figsize=(9, 7)); shap.summary_plot(sv, samp, max_display=15, show=False)
        plt.title("SHAP summary - LightGBM phishing-domain model"); savefig("url_shap_summary.png")
        plt.figure(figsize=(8, 6)); shap.summary_plot(sv, samp, plot_type="bar", max_display=15, show=False)
        plt.title("Mean |SHAP| - global feature importance"); savefig("url_shap_bar.png")
    except Exception as e:
        log(f"  SHAP skipped: {e}")

    # ---------------- feature-group ablation (LightGBM, same CV)
    log("  ablation ...")
    skf = StratifiedKFold(3 if FAST else 5, shuffle=True, random_state=SEED)
    base = LGBMClassifier(n_estimators=50 if FAST else 200, learning_rate=.05, num_leaves=31, subsample=.8, subsample_freq=1,
                          colsample_bytree=.8, n_jobs=1, verbose=-1, random_state=SEED)
    def score(cols):
        cvf = cross_val_score(base, Xtr[cols], ytr, cv=skf, scoring="f1").mean()
        m = base.fit(Xtr[cols], ytr); pr = m.predict_proba(Xte[cols])[:, 1]
        return cvf, compute_metrics(yte, pr)
    allc = list(F.columns); rows = []
    c, m = score(allc); rows.append({"Configuration": "All features", "# features": len(allc), "CV F1": c, "Test F1": m["F1"], "Test ROC-AUC": m["ROC-AUC"], "Test Recall": m["Recall"]})
    for g, cols in FEATURE_GROUPS.items():
        keep = [x for x in allc if x not in cols]; c, m = score(keep)
        rows.append({"Configuration": f"− {g}", "# features": len(keep), "CV F1": c, "Test F1": m["F1"], "Test ROC-AUC": m["ROC-AUC"], "Test Recall": m["Recall"]})
    hard = [x for x in allc if x not in FEATURE_GROUPS["Structure (length, labels, dots)"] + FEATURE_GROUPS["TLD"] + FEATURE_GROUPS["Free-hosting provider"]]
    c, m = score(hard); rows.append({"Configuration": "Lexical + brand + keywords only (no structure/TLD/host)", "# features": len(hard), "CV F1": c, "Test F1": m["F1"], "Test ROC-AUC": m["ROC-AUC"], "Test Recall": m["Recall"]})
    ab = pd.DataFrame(rows); save_table(ab.round(4), "url_ablation")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    a2 = ab.iloc[::-1]; ax.barh(a2.Configuration, a2["Test F1"], color=[C_ACC if i == len(a2) - 1 else C_HAM for i in range(len(a2))], edgecolor="k")
    for i, v in enumerate(a2["Test F1"]): ax.text(v + .001, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(max(0, a2["Test F1"].min() - .05), 1.01); ax.set_xlabel("Test F1"); ax.set_title("Feature-group ablation (LightGBM)")
    savefig("url_ablation.png")

    # ---------------- hard subset: bare registrable domains only (no sub-domain, no free host)
    mask = ((Xte.n_subdomains == 0) & (Xte.free_host == 0)).values
    rows = []
    for nm in rk[:5]:
        p = res["proba"][nm][mask]; yy = yte[mask]
        if len(np.unique(yy)) == 2:
            r = {"Model": nm, "Subset size": int(mask.sum()), "Phishing in subset": int(yy.sum())}; r.update({k: v for k, v in compute_metrics(yy, p).items() if k in METRICS}); rows.append(r)
    save_table(pd.DataFrame(rows).round(4), "url_hard_subset"); log("  hard subset done")

    joblib.dump(dict(best=best, ranked=rk, fitted=res["fitted"], columns=list(F.columns), brands=brands,
                     phish_all=phish_all, benign_ranked=ranked_benign, test_proba=res["proba"], yte=yte), CACHE / "url_state.joblib")
    log("STAGE 2 done")


if __name__ == "__main__":
    run()
