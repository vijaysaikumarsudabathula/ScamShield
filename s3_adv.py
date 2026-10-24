"""Stage 3: attacker-vs-defender loop on the SMS detector.
Attacker = black-box adaptive evasion (obfuscation, synonyms, padding...).  Defenders = canonicalisation,
adversarial training, and both.  Metric of interest: evasion rate on spam the model originally caught."""
import joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.base import clone
from ss_common import *
from ss_data import make_sms_pipe
from ss_attacks import Attacker, SEEN_ATTACKS, UNSEEN_ATTACKS, ALL_ATTACKS
from s1_sms import get_split, build_sms_clfs

BUDGET = 8 if FAST else 30        # candidate variants tried per message (black-box queries)
ROUNDS = 2 if FAST else 3
CONFIGS = {"Baseline": (False, False), "+ Canonicalization": (True, False),
           "+ Adversarial training": (False, True), "+ Canon. & Adv. training": (True, True)}


def adaptive_attack(pipe, texts, att, kinds, budget, kmax=3):
    """For each message, try `budget` random obfuscations and keep the one with the lowest spam score."""
    texts = list(texts)
    cands = [att.random_variant(t, kinds, kmax) for t in texts for _ in range(budget)]
    p = pipe.predict_proba(cands)[:, 1].reshape(len(texts), budget)
    j = p.argmin(1)
    best = [cands[i * budget + j[i]] for i in range(len(texts))]
    return best, p.min(1)


def evaluate(pipe, Xte, yte, triggers):
    p = pipe.predict_proba(Xte)[:, 1]; m = compute_metrics(yte, p)
    spam_txt = Xte[(yte == 1) & (p >= .5)]; ham_txt = Xte[yte == 0]
    ev = lambda kinds, b, k: float((adaptive_attack(pipe, spam_txt, Attacker(triggers, 999), kinds, b, k)[1] < .5).mean())
    out = {"Accuracy": m["Accuracy"], "Precision": m["Precision"], "Recall": m["Recall"], "F1": m["F1"],
           "ROC-AUC": m["ROC-AUC"], "FPR": m["FPR"],
           "Evasion (seen attacks)": ev(SEEN_ATTACKS, BUDGET, 3),
           "Evasion (unseen attacks)": ev(UNSEEN_ATTACKS, BUDGET, 3),
           "Evasion (all attacks)": ev(ALL_ATTACKS, BUDGET, 3)}
    for k in ALL_ATTACKS:
        out[f"single:{k}"] = ev([k], 10, 1)
    a = Attacker(triggers, 777)
    hv = [a.random_variant(t, SEEN_ATTACKS, 2) for t in ham_txt]
    out["FPR on obfuscated legit"] = float((pipe.predict_proba(hv)[:, 1] >= .5).mean())
    return out


def run():
    log("STAGE 3 - adversarial evasion study")
    st = joblib.load(CACHE / "sms_state.joblib")
    df, Xtr, Xte, ytr, yte = get_split()
    triggers = st["triggers"]; clfs = build_sms_clfs(st["tuned"])
    models = st["ranked"][:3]
    log(f"  models under attack: {models}   budget={BUDGET} rounds={ROUNDS}")
    rows, final_pipes, final_aug = [], {}, {}
    for mn in models:
        for cname, (canon, adv) in CONFIGS.items():
            mk = lambda: make_sms_pipe(clone(clfs[mn]), canon=canon)
            pipe = mk(); pipe.fit(Xtr, ytr)
            r = evaluate(pipe, Xte, yte, triggers); r.update(Model=mn, Config=cname, Round=0); rows.append(r)
            log(f"  {mn:<20} {cname:<26} r0  F1={r['F1']:.4f} evade(all)={r['Evasion (all attacks)']:.1%} unseen={r['Evasion (unseen attacks)']:.1%}")
            aug_X, aug_y = [], []
            if adv:
                rng = np.random.RandomState(SEED)
                for rd in range(1, ROUNDS + 1):
                    att = Attacker(triggers, 100 + rd)
                    best_txt, _ = adaptive_attack(pipe, Xtr[ytr == 1], att, SEEN_ATTACKS, 10 if FAST else 20, 3)
                    ham_tr = Xtr[ytr == 0]; sel = rng.choice(len(ham_tr), len(ham_tr) // 4, replace=False)
                    hv = [att.random_variant(ham_tr[i], SEEN_ATTACKS, 2) for i in sel]
                    aug_X += best_txt + hv; aug_y += [1] * len(best_txt) + [0] * len(hv)
                    pipe = mk(); pipe.fit(np.concatenate([Xtr, np.array(aug_X, dtype=object)]), np.concatenate([ytr, aug_y]))
                    r = evaluate(pipe, Xte, yte, triggers); r.update(Model=mn, Config=cname, Round=rd); rows.append(r)
                    log(f"  {mn:<20} {cname:<26} r{rd}  F1={r['F1']:.4f} evade(all)={r['Evasion (all attacks)']:.1%} unseen={r['Evasion (unseen attacks)']:.1%}")
            final_pipes[(mn, cname)] = pipe; final_aug[(mn, cname)] = (aug_X, aug_y)
    R = pd.DataFrame(rows)
    lead = ["Model", "Config", "Round"]
    R = R[lead + [c for c in R.columns if c not in lead]]
    single = [c for c in R.columns if c.startswith("single:")]
    save_table(R.drop(columns=single).round(4), "adv_rounds")
    last = R.sort_values("Round").groupby(["Model", "Config"], sort=False).tail(1)
    save_table(last.drop(columns=single).round(4), "adv_final_summary")
    pa = last.melt(id_vars=["Model", "Config"], value_vars=single, var_name="Attack", value_name="Evasion rate")
    pa["Attack"] = pa.Attack.str.replace("single:", ""); pa["Known to defender"] = pa.Attack.isin(SEEN_ATTACKS).map({True: "Seen", False: "Unseen"})
    save_table(pa.round(4), "adv_per_attack")

    # ---- FIGURES
    cfg_col = {"Baseline": "#6c757d", "+ Canonicalization": C_ACC, "+ Adversarial training": C_HAM, "+ Canon. & Adv. training": C_SPAM}
    ev_cols = ["Evasion (seen attacks)", "Evasion (unseen attacks)", "Evasion (all attacks)"]
    fig, axes = plt.subplots(3, len(models), figsize=(5.2 * len(models), 11), sharey=True)
    for j, mn in enumerate(models):
        for i, ec in enumerate(ev_cols):
            ax = axes[i, j]; d = R[R.Model == mn]
            for cname in CONFIGS:
                dd = d[d.Config == cname].sort_values("Round")
                if len(dd) == 1:
                    ax.axhline(dd[ec].iloc[0], color=cfg_col[cname], ls="--", lw=2, label=cname)
                else:
                    ax.plot(dd.Round, dd[ec], "o-", color=cfg_col[cname], lw=2.2, label=cname)
            ax.set_ylim(-.02, 1.02); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_title(f"{mn}\n{ec}", fontsize=10); ax.set_xticks(range(ROUNDS + 1))
            if i == 2: ax.set_xlabel("Adversarial-training round")
            if j == 0: ax.set_ylabel("Evasion rate")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Attacker vs. defender: share of caught scams that evade after obfuscation", fontweight="bold", y=1.0)
    savefig("adv_01_evasion_by_round.png")

    bm = models[0]
    hm = pa[pa.Model == bm].pivot(index="Attack", columns="Config", values="Evasion rate")[list(CONFIGS)].loc[ALL_ATTACKS]
    ad = last[last.Model == bm].set_index("Config")[ev_cols].T.rename(index=lambda s: "ADAPTIVE: " + s.replace("Evasion ", ""))[list(CONFIGS)]
    hm = pd.concat([hm, ad])
    fig, ax = plt.subplots(figsize=(10, 6.4))
    sns.heatmap(hm, annot=True, fmt=".0%", cmap="Reds", vmin=0, vmax=1, linewidths=.5, ax=ax, cbar_kws={"label": "evasion rate"})
    ax.axhline(len(SEEN_ATTACKS), color="k", lw=2); ax.axhline(len(ALL_ATTACKS), color="k", lw=2)
    ax.set_title(f"Evasion rate by attack type and defence - {bm}\n(first 4 rows = attacks defenders trained on, next 3 = unseen)")
    ax.set_ylabel(""); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=15)
    savefig("adv_02_attack_defence_heatmap.png")

    fig, ax = plt.subplots(figsize=(8.8, 6)); mk = dict(zip(models, ["o", "s", "^", "D"]))
    for _, r in last.iterrows():
        ax.scatter(r["Evasion (all attacks)"], r["F1"], s=170, marker=mk[r.Model], color=cfg_col[r.Config], edgecolor="k", alpha=.9)
    for cname, c in cfg_col.items(): ax.scatter([], [], color=c, label=cname, s=90)
    for mn_, m_ in mk.items(): ax.scatter([], [], marker=m_, color="w", edgecolor="k", label=mn_, s=90)
    ax.set_xlabel("Evasion rate (all attacks, lower is better)"); ax.set_ylabel("Clean test F1 (higher is better)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Robustness vs. accuracy trade-off (top-left is best)")
    savefig("adv_03_robustness_tradeoff.png")

    # drop in clean performance when hardening
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    d = last[last.Model == bm].set_index("Config").loc[list(CONFIGS)]
    x = np.arange(len(d)); w = .25
    ax.bar(x - w, d["F1"], w, label="Clean F1", color=C_HAM); ax.bar(x, d["Recall"], w, label="Clean recall", color=C_SPAM)
    ax.bar(x + w, 1 - d["FPR on obfuscated legit"], w, label="Specificity on obfuscated legit", color=C_ACC)
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=12); ax.set_ylim(.8, 1.01); ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Cost of hardening: clean-data quality - {bm}")
    savefig("adv_04_cost_of_hardening.png")

    # choose hardened model: best model, canon + adv
    hk = (models[0], "+ Canon. & Adv. training")
    joblib.dump(dict(hardened=final_pipes[hk], baseline=st["fitted"][models[0]], aug=final_aug[hk], best_name=models[0],
                     models=models), CACHE / "adv_state.joblib")
    log("STAGE 3 done")


if __name__ == "__main__":
    run()
