"""ScamShield multi-agent pipeline: Extractor -> Analyzer -> Verifier -> Scorer -> Advisor.
Agents are tool-using specialists orchestrated in code; each calls the trained ML models as tools.
An OPTIONAL LLM advisor hook is included at the bottom (needs ANTHROPIC_API_KEY; not exercised in the reported run)."""
import re, os
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from ss_data import URL_RE, PHONE_RE, host_of, split_domain, domain_features, SHORTENERS, FREE_HOSTS

TACTICS = {
    "Urgency / time pressure": r"\b(urgent|immediately|within \d+ ?(hours?|hrs?|minutes?|mins?)|today only|final notice|expires?|expiring|last chance|act now|asap|tonight|right away|before midnight)\b",
    "Authority impersonation": r"\b(bank|irs|tax|customs|police|court|government|microsoft|apple|amazon|netflix|fedex|dhl|usps|paypal|coinbase|officer|kyc|pan card|sim|telecom|electricity|walmart|instagram)\b",
    "Fear / threat": r"\b(suspended|blocked|frozen|freeze|disconnect(ed)?|deleted|deactivated|arrest|legal action|penalty|virus|unauthori[sz]ed|unusual activity|locked|on hold|late fee|violated)\b",
    "Reward / greed": r"\b(won|winner|prize|lottery|lucky draw|cash ?back|reward|free|gift|bonus|refund|congratulations|earn|returns?|per day|weekly|shortlisted|pre-approved)\b",
    "Payment / credential request": r"\b(pay|fee|deposit|transfer|send money|gift card|bitcoin|crypto|bank details|otp|pin|cvv|password|verify your identity|billing information|share the)\b",
    "Secrecy / off-platform move": r"\b(don't tell|do not tell|keep (this )?secret|confidential|whatsapp|telegram|add me)\b",
    "Impersonating an acquaintance / romance": r"\b(wrong number|serious relationship|new number|lost my phone|you seem nice|soldier)\b",
}
REPORT = {
    "GLOBAL": "Report it to your bank (if money or cards are involved) and to your national cyber-crime or consumer-fraud authority.",
    "US": "Forward the text to 7726 (SPAM) and report it at reportfraud.ftc.gov.",
    "UK": "Forward the text to 7726 and report it to Action Fraud (actionfraud.police.uk).",
    "IN": "Report it at cybercrime.gov.in, or call the national cyber-crime helpline 1930 if money was lost.",
}


RISKY_TLD = {"xyz", "top", "info", "online", "site", "club", "icu", "tk", "ml", "ga", "cf", "gq", "buzz", "click", "vip", "link", "shop", "live"}


class ScamShieldAgent:
    def __init__(self, text_model, url_model, url_cols, brands, phish_set, top_set, lr_explainer):
        self.tm, self.um, self.cols, self.brands = text_model, url_model, url_cols, brands
        self.phish, self.top, self.lr = phish_set, top_set, lr_explainer

    # ---- Agent 1: Extractor
    def extractor(self, text):
        urls = [m.group(0).rstrip(".,;)") for m in URL_RE.finditer(text)]
        return {"urls": urls, "hosts": list(dict.fromkeys(host_of(u) for u in urls)),
                "phones": PHONE_RE.findall(text),
                "amounts": re.findall(r"(?:[$£€₹]|Rs\.?)\s?[\d,]+(?:\.\d+)?", text)}

    # ---- Agent 2: Analyzer (manipulation tactics)
    def analyzer(self, text):
        return [k for k, rx in TACTICS.items() if re.search(rx, text, re.I)]

    # ---- Agent 3: Verifier (domain intelligence)
    def verifier(self, host):
        reg, sub, tld = split_domain(host)
        look = max(((fuzz.ratio(reg.split(".")[0], b), b) for b, r in zip(self.brands.labels, self.brands.regs) if r != reg),
                   default=(0, ""))
        feat = domain_features([host], self.brands)[self.cols]
        p = float(self.um.predict_proba(feat)[0, 1])
        in_db, in_top = host in self.phish or reg in self.phish, (host in self.top or reg in self.top)
        if in_db:
            risk = 0.99
        elif host in SHORTENERS or reg in SHORTENERS:
            risk = max(p, 0.6)
        elif in_top and not sub:
            risk = min(p, 0.05)
        else:
            risk = p
        return {"host": host, "in_phishing_db": in_db, "in_top_domains": in_top, "url_shortener": reg in SHORTENERS,
                "free_hosting": any(host.endswith("." + h) for h in FREE_HOSTS),
                "lookalike_of": look[1] if look[0] >= 80 else "", "lookalike_score": look[0],
                "domain_model_p": round(p, 4), "domain_risk": round(risk, 4)}

    # ---- explanation from the linear model (coefficient x tf-idf value = additive contribution)
    META_LABEL = {"has_url": "contains a link", "n_currency": "mentions money", "n_phone": "contains a phone number",
                  "digit_ratio": "many digits", "upper_ratio": "ALL-CAPS text", "n_exclaim": "exclamation marks"}

    def explain(self, text, k=5):
        feat = self.lr.named_steps["feat"]; clf = self.lr.named_steps["clf"]
        x = feat.transform([text]); names = feat.get_feature_names_out()
        contrib = np.asarray(x.multiply(clf.coef_).todense()).ravel()
        out = []
        for i in np.argsort(-contrib):
            if contrib[i] <= 0 or len(out) >= k:
                break
            n = names[i]
            if n.startswith("word__"):
                out.append((f"word '{n[6:]}'", round(float(contrib[i]), 3)))
            elif n.startswith("meta__") and n[6:] in self.META_LABEL:
                out.append((self.META_LABEL[n[6:]], round(float(contrib[i]), 3)))
        return out

    # ---- Agent 4/5: scorer + advisor
    def run(self, text, region="GLOBAL"):
        ent = self.extractor(text); tac = self.analyzer(text)
        ver = [self.verifier(h) for h in ent["hosts"] if "." in h]
        p_text = float(self.tm.predict_proba([text])[0, 1])
        p_url = max([v["domain_risk"] for v in ver], default=0.0)
        risk = 1 - (1 - p_text) * (1 - p_url)
        level = "HIGH - likely scam" if risk >= .85 else ("SUSPICIOUS - verify first" if risk >= .5 else "LOW - looks legitimate")
        why = []
        if ver:
            for v in ver:
                if v["in_phishing_db"]: why.append(f"{v['host']} is listed in a public phishing database")
                if v["lookalike_of"]: why.append(f"{v['host']} imitates '{v['lookalike_of']}' (similarity {v['lookalike_score']:.0f}/100)")
                if v["url_shortener"]: why.append(f"{v['host']} is a link shortener that hides the real destination")
                if v["free_hosting"]: why.append(f"{v['host']} is on free web hosting")
                reg, sub, tld = split_domain(v["host"]); h = v["host"]
                kws = [k for k in ("login", "verify", "secure", "update", "account", "bank", "pay", "claim", "refund", "reward", "kyc", "unlock", "support") if k in h]
                if kws and not v["in_top_domains"]: why.append(f"{h} stuffs security/finance words into the address ({', '.join(kws[:3])})")
                if h.count("-") >= 2: why.append(f"{h} uses many hyphens, a common phishing pattern")
                if tld.split('.')[-1] in RISKY_TLD: why.append(f"'.{tld.split('.')[-1]}' is a low-cost TLD heavily used by phishing sites")
                if v["domain_model_p"] >= .8 and not v["in_phishing_db"]: why.append(f"domain model rates {h} as phishing-like (P={v['domain_model_p']:.2f})")
        if tac: why.append("manipulation tactics: " + ", ".join(tac))
        adv = self.advise(risk, region)
        return {"text": text, "p_text": round(p_text, 4), "p_url": round(p_url, 4), "risk": round(risk, 4), "verdict": level,
                "entities": ent, "tactics": tac, "domain_checks": ver, "top_terms": self.explain(text), "why": why, "advice": adv}

    @staticmethod
    def advise(risk, region):
        if risk >= .85:
            a = ["Do not click any link, call any number or reply.", "Never share OTPs, PINs, passwords or card details.",
                 "Contact the organisation only via its official app or the number printed on your card/bill.",
                 REPORT.get(region, REPORT["GLOBAL"]), "Then delete the message and block the sender."]
            a.append("If you ALREADY clicked or replied: change the affected passwords, turn on two-factor authentication, "
                     "call your bank to freeze cards/transfers if you shared financial details, and run a malware scan.")
        elif risk >= .5:
            a = ["Do not click the link yet.", "Check the claim through the official app/website (type the address yourself).",
                 "If anything feels pressured or unusual, treat it as a scam and report it."]
        else:
            a = ["No strong scam signals found. Still avoid sharing OTPs or passwords with anyone who contacts you first."]
        return a


# ---------------------------------------------------------------------------------------------
# OPTIONAL: LLM-backed advisor / attacker (NOT used in the reported experiments; untested here).
def llm_advise(report: dict, model="claude-sonnet-5-5"):
    """Rewrite the structured report into friendly plain-language guidance. Requires `pip install anthropic`."""
    import anthropic
    client = anthropic.Anthropic()                       # reads ANTHROPIC_API_KEY
    msg = client.messages.create(model=model, max_tokens=400, messages=[{"role": "user", "content":
        f"Explain in 4 short sentences, for a non-technical person, why this message was judged '{report['verdict']}' "
        f"and what to do. Evidence: {report['why']}. Message: {report['text']}"}])
    return msg.content[0].text
