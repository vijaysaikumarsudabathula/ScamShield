"""Stage 4: (a) modern-scam drift probe, (b) online update experiment, (c) multi-agent demo."""
import json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from ss_common import *
from ss_data import make_sms_pipe
from s1_sms import get_split, build_sms_clfs
from ss_agent import ScamShieldAgent

SCAM = [
 "Your account has been temporarily suspended due to unusual activity. Verify your identity within 24 hours: http://secure-bank-verify.xyz/login",
 "Dear customer, your KYC is expiring today. Update now to avoid account block: http://kyc-update-online.top/form",
 "Your package could not be delivered due to an incorrect address. Pay a $1.99 redelivery fee: http://postal-redeliver.info/pay",
 "Final notice: unpaid toll balance of $12.40. Pay now to avoid a late fee of $50 http://toll-pay-now.xyz",
 "Congratulations! You've been shortlisted for a work-from-home job paying Rs 5000 per day. Message us on WhatsApp to start today.",
 "Hi, I'm Emma. I think I texted the wrong number, but you seem nice. Can we chat?",
 "Your electricity will be disconnected tonight at 9:30 pm because last month's bill was not updated. Call our officer immediately.",
 "Tax refund of $842 is pending. Confirm your details to receive the deposit: http://tax-refund-claim.top",
 "Your Netflix payment failed. Update your billing information to keep watching: http://netflix-billing-help.xyz/update",
 "Security alert: someone tried to sign in to your Apple ID from another country. Tap to secure your account http://appleid-locked.info",
 "We have detected a virus on your device. Call Microsoft Support now at 1-800-555-0199 to remove it.",
 "Earn 30% weekly returns with our guaranteed crypto investment group. Limited slots. Reply YES to join.",
 "Dear user, you've won Rs 25,00,000 in the lucky draw. Send your bank details to claim the prize.",
 "Your PAN card will be blocked today. Update immediately: http://pan-verify-india.xyz",
 "Amazon: Your order of iPhone 15 for $999 has been placed. If this wasn't you, call 1-888-555-0123 to cancel immediately.",
 "Your bank account is on hold. Click here to unlock: http://bit.ly/3xKLm92",
 "URGENT: your parcel is held at customs. Pay the clearance fee of Rs 1,450 to release it: http://customs-clearance-pay.top",
 "Mom I lost my phone, this is my new number. Can you send me money for an urgent bill? I'll explain later.",
 "You have a new voicemail. Listen now: http://voice-mail-secure.xyz/play",
 "Coinbase alert: a withdrawal of 0.8 BTC was requested. Cancel here if this wasn't you: http://coinbase-secure-help.top",
 "Get a pre-approved personal loan of Rs 5,00,000 instantly with no documents. Click: http://instant-loan-approve.online",
 "Your SIM will be deactivated in 2 hours. Complete the re-verification now: http://sim-reverify.xyz",
 "Hello, I am a recruiter. Like videos on YouTube and earn Rs 200 per task. Join our Telegram channel.",
 "Dear customer your credit card reward points worth Rs 9,850 expire today. Redeem now: http://reward-redeem-bank.top",
 "FedEx: we attempted delivery but nobody was home. Reschedule here: http://fedex-reschedule.info",
 "Your Instagram account violated copyright rules and will be deleted in 24 hours. Appeal here: http://insta-appeal-center.xyz",
 "Hi, I saw your profile and I am a US soldier looking for a serious relationship. Please add me on Telegram.",
 "DHL: Your shipment is on hold. A customs fee of $3.50 is required: http://dhl-customs-fee.top",
 "Walmart gift card $500 is waiting for you. Complete a short survey to claim: http://walmart-gift-survey.xyz",
 "Your linked bank account will be frozen. Share the OTP you just received to continue."]
LEGIT = [
 "Your OTP for login is 482913. It is valid for 10 minutes. Do not share it with anyone.",
 "Hi, your appointment with Dr. Rao is confirmed for tomorrow at 4:30 PM. Reply C to confirm or R to reschedule.",
 "Your order #48213 has been shipped and will arrive on Thursday. Track it in the app.",
 "Hey, are we still on for dinner tonight? I'm thinking Italian.",
 "Reminder: your electricity bill of Rs 1,240 is due on the 28th. Pay via the official app or website.",
 "Meeting moved to 3 pm in conference room B. Please bring the quarterly numbers.",
 "Thanks for your payment of $45.00 to City Water. Your receipt is available in your account.",
 "Your Uber is arriving in 3 minutes. Look for a white Toyota with plate ending 4821.",
 "Can you pick up some milk and bread on your way home?",
 "Your prescription is ready for pickup at the pharmacy. Store closes at 9 pm.",
 "Happy birthday! Hope you have an amazing day. Let's catch up this weekend.",
 "Your flight to Delhi is on time. Boarding starts at 6:15 PM at gate 12.",
 "Your verification code is 771204. If you did not request this, you can ignore this message.",
 "Class is cancelled tomorrow due to the holiday. Assignment deadline extended by one day.",
 "Your bank statement for September is now available. Log in to the official app to view it.",
 "Reminder: parent-teacher meeting on Saturday at 10 am in the school hall.",
 "I'll be about 15 minutes late, start without me.",
 "Your food order is on the way. Delivery partner Ravi will reach you shortly.",
 "Your package was delivered to the front desk at 2:14 PM.",
 "Thanks for visiting! Your 10% discount code for next time is THANKS10.",
 "Don't forget to submit the expense report by Friday.",
 "Your electricity meter reading has been recorded. Thank you for submitting.",
 "Debit alert: Rs 2,300 spent at the grocery store on card ending 4417. If not you, call the number on the back of your card.",
 "Your library book is due in 2 days. Renew it online or at the front desk.",
 "Mom: Dinner at 8, come hungry!",
 "Your cab to the airport is booked for 5:30 AM tomorrow.",
 "The doctor's office called, please call back during working hours.",
 "Your subscription renews on 3 November for Rs 199. Manage it in your account settings.",
 "Congrats on the new job! Drinks on me this weekend.",
 "Your Wi-Fi password was changed successfully. If this wasn't you, contact support through the official app."]


def run():
    log("STAGE 4 - drift probe, online update, agent demo")
    st = joblib.load(CACHE / "sms_state.joblib"); ad = joblib.load(CACHE / "adv_state.joblib"); us = joblib.load(CACHE / "url_state.joblib")
    df, Xtr, Xte, ytr, yte = get_split()
    probe = pd.DataFrame({"text": SCAM + LEGIT, "y": [1] * len(SCAM) + [0] * len(LEGIT)})
    save_table(probe, "modern_probe_set")
    P, yp = probe.text.values, probe.y.values

    # ---------- (a) drift probe
    models = dict(st["fitted"]); models["ScamShield-hardened (" + ad["best_name"] + ")"] = ad["hardened"]
    rows = []
    for n, m in models.items():
        pt = m.predict_proba(Xte)[:, 1]; pp = m.predict_proba(P)[:, 1]
        a = compute_metrics(yte, pt); b = compute_metrics(yp, pp)
        rows.append({"Model": n, "SMS test F1": a["F1"], "Modern-probe Accuracy": b["Accuracy"], "Modern-probe Precision": b["Precision"],
                     "Modern-probe Recall": b["Recall"], "Modern-probe F1": b["F1"], "Modern-probe Specificity": b["Specificity"],
                     "Modern-probe ROC-AUC": b["ROC-AUC"], "F1 drop (test → probe)": a["F1"] - b["F1"]})
    D = pd.DataFrame(rows).sort_values("Modern-probe F1", ascending=False); save_table(D.round(4), "probe_model_comparison")
    d = D.set_index("Model")[["Modern-probe Accuracy", "Modern-probe Precision", "Modern-probe Recall", "Modern-probe F1", "Modern-probe Specificity"]]
    fig, ax = plt.subplots(figsize=(11.5, 5.4)); d.plot.bar(ax=ax, colormap="viridis", width=.82, edgecolor="k", lw=.4)
    ax.set_title("Concept drift: legacy-trained models on 60 hand-written modern messages"); ax.set_ylim(0, 1.05)
    ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=30, ha="right"); ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.35))
    savefig("probe_01_model_comparison.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.6)); dd = D.sort_values("Modern-probe F1")
    for i, (_, r) in enumerate(dd.iterrows()):
        ax.plot([r["Modern-probe F1"], r["SMS test F1"]], [i, i], color="gray", lw=2, zorder=1)
    ax.scatter(dd["SMS test F1"], range(len(dd)), s=90, color=C_HAM, label="Legacy SMS test F1", zorder=2, edgecolor="k")
    ax.scatter(dd["Modern-probe F1"], range(len(dd)), s=90, color=C_SPAM, label="Modern-probe F1", zorder=2, edgecolor="k")
    ax.set_yticks(range(len(dd))); ax.set_yticklabels(dd.Model); ax.set_xlabel("F1"); ax.legend(loc="lower left"); ax.set_title("The drift gap: same models, new scam styles")
    savefig("probe_02_drift_gap.png")

    # ---------- (b) online update: how fast does the detector adapt when users report new scams?
    log("  online-update (adaptation-curve) experiment")
    clfs = build_sms_clfs(st["tuned"]); bn = ad["best_name"]; aug_X, aug_y = ad["aug"]
    KS = [0, 6, 12, 18, 30] if not FAST else [0, 12, 30]
    sss = StratifiedShuffleSplit(n_splits=3 if FAST else 5, test_size=.5, random_state=SEED)
    rows = []
    for sp, (pool, held) in enumerate(sss.split(P, yp)):
        # order the pool so any prefix is class-balanced
        pos = [i for i in pool if yp[i] == 1]; neg = [i for i in pool if yp[i] == 0]
        order = [i for pair in zip(pos, neg) for i in pair]
        for k in KS:
            sel = np.array(order[:k], dtype=int)
            if k == 0:
                mdl = ad["hardened"]
            else:
                X2 = np.concatenate([Xtr, np.array(aug_X, dtype=object), np.repeat(P[sel], 5)])
                y2 = np.concatenate([ytr, aug_y, np.repeat(yp[sel], 5)])
                mdl = make_sms_pipe(clone(clfs[bn]), canon=True); mdl.fit(X2, y2)
            ph = mdl.predict_proba(P[held])[:, 1]; m = compute_metrics(yp[held], ph)
            leg = compute_metrics(yte, mdl.predict_proba(Xte)[:, 1])
            rows.append({"Split": sp + 1, "Reported messages (k)": k, "Held-out ROC-AUC": m["ROC-AUC"], "Held-out PR-AUC": m["PR-AUC"],
                         "Held-out Recall": m["Recall"], "Held-out F1": m["F1"], "Held-out Specificity": m["Specificity"],
                         "Mean P(scam) on held-out scams": float(ph[yp[held] == 1].mean()), "Legacy SMS F1": leg["F1"]})
    O = pd.DataFrame(rows); save_table(O.round(4), "probe_online_update_splits")
    S = O.drop(columns="Split").groupby("Reported messages (k)").agg(["mean", "std"]); S.columns = [f"{a} ({b})" for a, b in S.columns]
    save_table(S.reset_index().round(4), "probe_online_update_summary")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, cols, ttl in zip(axes, [["Held-out ROC-AUC", "Held-out PR-AUC"], ["Held-out Recall", "Held-out F1", "Held-out Specificity"],
                                    ["Mean P(scam) on held-out scams", "Legacy SMS F1"]],
                             ["Ranking quality on unseen modern messages", "Decisions at threshold 0.5", "Confidence on new scams vs. legacy retention"]):
        for c, col in zip(cols, sns.color_palette("deep", 3)):
            g = O.groupby("Reported messages (k)")[c]; ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o", lw=2, capsize=3, label=c.replace("Held-out ", ""), color=col)
        ax.set_xlabel("Number of newly reported messages used for the update (k)"); ax.set_title(ttl); ax.legend(fontsize=8); ax.set_ylim(.3, 1.02)
    fig.suptitle(f"Adaptation curve - {bn} updated with k reported messages (mean ± sd over {O.Split.nunique()} splits, 30 held-out messages each)", fontweight="bold", y=1.03)
    savefig("probe_03_online_update.png")

    # ---------- (c) agent
    lr = st["fitted"]["Logistic Regression"]
    agent = ScamShieldAgent(ad["hardened"], us["fitted"][us["best"]], us["columns"], us["brands"],
                            us["phish_all"], set(us["benign_ranked"]), lr)
    # text-only vs fused on the probe set (ILLUSTRATIVE - probe author chose the domain styles)
    reps = [agent.run(t) for t in P]
    pt = np.array([r["p_text"] for r in reps]); rk = np.array([r["risk"] for r in reps])
    mt, mf = compute_metrics(yp, pt), compute_metrics(yp, rk)
    cmp_ = pd.DataFrame([{"System": "Text model only (hardened)", **{k: mt[k] for k in METRICS + ["Specificity"]}},
                         {"System": "ScamShield agent (text + domain verifier)", **{k: mf[k] for k in METRICS + ["Specificity"]}}])
    save_table(cmp_.round(4), "agent_vs_text_only_probe")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (ttl, m) in zip(axes, [("Text model only", mt), ("ScamShield agent", mf)]):
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]]); pct = cm / cm.sum(1, keepdims=True)
        sns.heatmap(pct, annot=np.array([[f"{cm[i, j]}\n({pct[i, j]:.0%})" for j in range(2)] for i in range(2)]), fmt="", cmap="Blues", cbar=False,
                    vmin=0, vmax=1, ax=ax, xticklabels=["Legit", "Scam"], yticklabels=["Legit", "Scam"], linewidths=1, linecolor="w")
        ax.set_title(f"{ttl}\nF1={m['F1']:.3f}  recall={m['Recall']:.3f}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    savefig("probe_04_text_vs_agent_confusion.png")

    demo_idx = [0, 2, 7, 15, 17, 30, 31, 38, 45, 58]
    demo_idx = [i for i in [0, 2, 7, 15, 17, 5] + [len(SCAM) + 0, len(SCAM) + 6, len(SCAM) + 12]]
    demo = [agent.run(P[i], region="GLOBAL") for i in demo_idx]
    demo.append(agent.run("FREE entry in 2 a weekly comp to win FA Cup tickets! Text FA to 87121 now. T&C apply 08452810075", "GLOBAL"))
    demo.append(agent.run("U r a WINNER! Claim your cash prize now - call 09061701461 urgently", "GLOBAL"))
    demo.append(agent.run("F.R.E.E entry! W1n a pr1ze - cl@1m your ca$h today", "GLOBAL"))
    json.dump(demo, open(TAB / "agent_demo_reports.json", "w"), indent=2, default=str)
    T = pd.DataFrame([{"Message": r["text"][:110] + ("…" if len(r["text"]) > 110 else ""), "Text P(scam)": r["p_text"], "Domain risk": r["p_url"],
                       "Final risk": r["risk"], "Verdict": r["verdict"], "Tactics": "; ".join(r["tactics"]),
                       "Domain evidence": "; ".join(w for w in r["why"] if "tactics" not in w), "Top terms": ", ".join(a for a, _ in r["top_terms"][:4])} for r in demo])
    save_table(T, "agent_demo_table")
    fig, ax = plt.subplots(figsize=(11, 5.8)); y = np.arange(len(demo))[::-1]
    ax.barh(y, [r["p_text"] for r in demo], color=C_HAM, label="Text model P(scam)", height=.38)
    ax.barh(y - .4, [r["p_url"] for r in demo], color=C_ACC, label="Domain verifier risk", height=.38)
    ax.scatter([r["risk"] for r in demo], y - .2, marker="D", color=C_SPAM, s=70, zorder=5, label="Fused final risk", edgecolor="k")
    ax.set_yticks(y - .2); ax.set_yticklabels([r["text"][:58] + "…" for r in demo], fontsize=8)
    ax.axvline(.5, color="gray", ls=":"); ax.axvline(.85, color="gray", ls="--"); ax.set_xlim(0, 1.03); ax.set_xlabel("risk score")
    ax.legend(loc="lower right", fontsize=8); ax.set_title("Multi-agent ScamShield - evidence per message (dotted = suspicious, dashed = high risk)")
    savefig("agent_01_demo_risk.png")

    md = ["# ScamShield agent - demo reports\n"]
    for r in demo:
        md += [f"### {r['verdict']}  (risk {r['risk']:.2f})", f"> {r['text']}\n",
               f"- Text model P(scam): **{r['p_text']:.3f}**; domain risk: **{r['p_url']:.3f}**",
               f"- Tactics: {', '.join(r['tactics']) or 'none detected'}",
               f"- Evidence: {'; '.join(r['why']) or 'none'}",
               f"- Top contributing terms: {', '.join(a for a, _ in r['top_terms']) or 'n/a'}",
               "- Advice: " + " ".join(r["advice"][:3]) + "\n"]
    (TAB / "agent_demo_reports.md").write_text("\n".join(md))
    joblib.dump(agent, CACHE / "agent.joblib")
    log("STAGE 4 done")


if __name__ == "__main__":
    run()
