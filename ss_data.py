"""Data loading + feature engineering for the SMS (text) and domain (URL) models."""
import re, math, csv
import numpy as np, pandas as pd
from collections import Counter
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, FunctionTransformer
from rapidfuzz import process, fuzz
from ss_common import *
from ss_attacks import canon_list

SMS_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
PHISH_URL = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt"
BENIGN_URL = "https://raw.githubusercontent.com/opendns/public-domain-lists/master/opendns-top-domains.txt"

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.(?:com|net|org|xyz|top|info|online|site|club|icu|tk|ml|ga|cf|gq|app|io|co|in|ly|me|cc|link|live|shop|buzz|click|vip)(?:/\S*)?\b")
PHONE_RE = re.compile(r"\d[\d\s\-]{7,}\d")


# ----------------------------------------------------------------------------- SMS
def load_sms():
    p = fetch(SMS_URL, "sms.tsv")
    df = pd.read_csv(p, sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE, encoding="utf-8")
    n_raw = len(df)
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
    df["y"] = (df.label == "spam").astype(int)
    df.attrs["n_raw"] = n_raw
    return df


class SMSMeta(BaseEstimator, TransformerMixin):
    """Hand-crafted style features: length, digits, caps, punctuation, links, phones, currency, non-ASCII."""
    names = ["log_len", "log_words", "digit_ratio", "upper_ratio", "n_exclaim", "n_question",
             "has_url", "n_phone", "n_currency", "non_ascii_ratio"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            t = str(t); n = max(len(t), 1)
            rows.append([np.log1p(len(t)), np.log1p(len(t.split())), sum(c.isdigit() for c in t) / n,
                         sum(c.isupper() for c in t) / n, t.count("!"), t.count("?"),
                         int(bool(URL_RE.search(t))), len(PHONE_RE.findall(t)),
                         sum(t.count(c) for c in "£$€₹"), sum(ord(c) > 127 for c in t) / n])
        return np.array(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.names)


def sms_features():
    mf = 3000 if FAST else 4000
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=mf, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=mf, sublinear_tf=True)),
        ("meta", Pipeline([("m", SMSMeta()), ("s", MaxAbsScaler())])),
    ])


def make_sms_pipe(clf, canon=False):
    from sklearn.pipeline import Pipeline as P
    steps = []
    if canon:
        steps.append(("canon", FunctionTransformer(canon_list, validate=False)))
    steps += [("feat", sms_features()), ("clf", clf)]
    return P(steps)


# ----------------------------------------------------------------------------- Domain / URL
KEYWORDS = ["login", "signin", "secure", "account", "verify", "update", "bank", "pay", "support", "wallet",
            "confirm", "free", "bonus", "gift", "claim", "auth", "billing", "invoice", "recover", "unlock",
            "suspend", "alert", "kyc", "refund", "prize", "reward", "service", "online", "web", "mail"]
FREE_HOSTS = ["000webhostapp.com", "weebly.com", "weeblysite.com", "wixsite.com", "web.app", "firebaseapp.com",
              "blogspot.com", "github.io", "netlify.app", "herokuapp.com", "glitch.me", "godaddysites.com",
              "pages.dev", "workers.dev", "myshopify.com", "wordpress.com", "sites.google.com", "ukit.me",
              "jimdofree.com", "square.site", "webflow.io", "vercel.app", "onrender.com", "repl.co",
              "surge.sh", "ngrok.io", "duckdns.org", "no-ip.org", "ddns.net", "hopto.org",
              "infinityfreeapp.com", "rf.gd", "epizy.com", "yolasite.com", "tripod.com", "sharepoint.com",
              "r2.dev", "weebly.net", "wixsite.net", "000webhost.com", "page.link", "typedream.app"]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rb.gy",
              "shorturl.at", "tiny.cc", "rebrand.ly", "t.ly", "lnkd.in", "wa.me"}
TLDS = ["com", "net", "org", "xyz", "top", "info", "online", "site", "club", "icu", "tk", "ml", "ga", "cf",
        "gq", "app", "dev", "io", "co", "us", "uk", "de", "ru", "cn", "in", "br", "shop", "live", "buzz",
        "cc", "link", "page", "me", "ly", "cloud", "vip", "click", "store", "tech", "website"]
SLD2 = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.in", "net.in", "org.in", "com.br",
        "co.jp", "com.cn", "co.za", "com.mx", "com.tr", "co.kr", "com.sg", "co.nz", "com.ar", "com.hk",
        "com.tw", "com.ua", "co.id", "com.pk", "com.eg", "com.ng", "co.ke", "com.my", "com.ph", "com.vn", "com.sa"}
_DOMAIN_OK = re.compile(r"^[a-z0-9][a-z0-9.\-]*\.[a-z0-9\-]{2,}$")


def split_domain(d):
    labels = d.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in SLD2:
        return ".".join(labels[-3:]), labels[:-3], ".".join(labels[-2:])
    return ".".join(labels[-2:]), labels[:-2], labels[-1]


def host_of(url):
    u = re.sub(r"(?i)^[a-z]+://", "", str(url).strip())
    u = u.split("/")[0].split("?")[0].split("#")[0].split("@")[-1].split(":")[0]
    return u.lower().strip(".")


def load_domains(n_per_class=10000, seed=SEED):
    pf = fetch(PHISH_URL, "phishing_domains.txt"); bf = fetch(BENIGN_URL, "top_domains.txt")
    clean = lambda lines: [l.strip().lower() for l in lines if _DOMAIN_OK.match(l.strip().lower())]
    benign_ranked = list(dict.fromkeys(clean(bf.read_text().splitlines())))      # popularity order kept
    phish_all = set(clean(pf.read_text(errors="ignore").splitlines()))
    overlap = phish_all & set(benign_ranked)
    phish_pool = sorted(phish_all - overlap)
    benign = [d for d in benign_ranked if d not in overlap]
    n = min(n_per_class, len(benign), len(phish_pool))
    rng = np.random.RandomState(seed)
    phish = list(rng.choice(phish_pool, n, replace=False))
    benign = benign[:n]
    df = pd.DataFrame({"domain": phish + benign, "y": [1] * n + [0] * n})
    info = dict(phish_total=len(phish_all), benign_total=len(benign_ranked), overlap_removed=len(overlap), n_per_class=n)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True), benign_ranked, phish_all, info


def _entropy(s):
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def _longest(pattern, s):
    m = re.findall(pattern, s)
    return max((len(x) for x in m), default=0)


class BrandBook:
    """Top-N popular domains used to detect look-alike / brand-impersonating domains."""

    def __init__(self, ranked_domains, top=400):
        regs, labels = [], []
        for d in ranked_domains[:top * 2]:
            reg, _, _ = split_domain(d); lab = reg.split(".")[0]
            if len(lab) >= 4 and lab not in labels:
                regs.append(reg); labels.append(lab)
            if len(labels) >= top:
                break
        self.regs, self.labels = np.array(regs), labels


def domain_features(domains, brands: BrandBook):
    rows, mains, subs, regs_ = [], [], [], []
    for d in domains:
        reg, sub, tld = split_domain(d); labels = d.split(".")
        main = reg.split(".")[0]
        letters = sum(c.isalpha() for c in d); digits = sum(c.isdigit() for c in d)
        vowels = sum(c in "aeiou" for c in d)
        sub_str = ".".join(sub)
        b_sub = sum(1 for b in brands.labels if b in sub_str and reg != brands.regs[brands.labels.index(b)]) if sub_str else 0
        b_main = sum(1 for i, b in enumerate(brands.labels) if b in main and main != b and reg != brands.regs[i])
        row = {
            "length": len(d), "n_labels": len(labels), "n_subdomains": len(sub), "n_dots": d.count("."),
            "n_hyphens": d.count("-"), "n_digits": digits, "digit_ratio": digits / len(d),
            "letter_ratio": letters / len(d), "vowel_ratio": vowels / max(letters, 1),
            "entropy": _entropy(d), "main_entropy": _entropy(main), "main_len": len(main),
            "max_label_len": max(len(x) for x in labels), "avg_label_len": np.mean([len(x) for x in labels]),
            "longest_digit_run": _longest(r"\d+", d), "longest_consonant_run": _longest(r"[bcdfghjklmnpqrstvwxyz]+", d),
            "has_punycode": int("xn--" in d), "is_ip": int(bool(re.fullmatch(r"[\d.]+", d))),
            "kw_count": sum(d.count(k) for k in KEYWORDS), "has_kw": int(any(k in d for k in KEYWORDS)),
            "free_host": int(any(d.endswith("." + h) for h in FREE_HOSTS)),
            "is_shortener": int(reg in SHORTENERS), "tld_len": len(tld.split(".")[-1]),
            "brand_in_sub": b_sub, "brand_in_main": b_main,
        }
        top_tld = tld.split(".")[-1]
        for t in TLDS:
            row[f"tld_{t}"] = int(top_tld == t)
        row["tld_other"] = int(top_tld not in TLDS)
        rows.append(row); mains.append(main); regs_.append(reg)
        subs.append(labels)
    F = pd.DataFrame(rows)
    # fuzzy similarity of every token of the registrable label to every popular brand (self excluded)
    toks, owner = [], []
    for i, m in enumerate(mains):
        parts = {m} | {t for t in re.split(r"[-_\d]+", m) if len(t) >= 4}
        for t in parts:
            toks.append(t); owner.append(i)
    sim = process.cdist(toks, brands.labels, scorer=fuzz.ratio, dtype=np.uint8, workers=1).astype(float)
    reg_arr = np.array(regs_)[np.array(owner)]
    sim[reg_arr[:, None] == brands.regs[None, :]] = 0
    best = pd.Series(sim.max(1)).groupby(np.array(owner)).max()
    F["brand_sim"] = best.reindex(range(len(domains))).fillna(0).values
    return F


FEATURE_GROUPS = {
    "Structure (length, labels, dots)": ["length", "n_labels", "n_subdomains", "n_dots", "max_label_len", "avg_label_len", "main_len"],
    "Lexical (digits, entropy, runs)": ["n_hyphens", "n_digits", "digit_ratio", "letter_ratio", "vowel_ratio", "entropy",
                                        "main_entropy", "longest_digit_run", "longest_consonant_run", "has_punycode", "is_ip"],
    "Keywords": ["kw_count", "has_kw"],
    "Brand look-alike": ["brand_in_sub", "brand_in_main", "brand_sim"],
    "TLD": ["tld_len", "is_shortener"] + [f"tld_{t}" for t in TLDS] + ["tld_other"],
    "Free-hosting provider": ["free_host"],
}
