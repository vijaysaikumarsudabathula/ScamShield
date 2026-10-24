"""Scam-text evasion attacks (the 'attacker') and the input canonicaliser (a 'defender')."""
import random, re, unicodedata

SEEN_ATTACKS = ["leet", "space", "synonym", "punct"]            # known to defenders at training time
UNSEEN_ATTACKS = ["homoglyph", "padding", "zerowidth"]          # only used at test time
ALL_ATTACKS = SEEN_ATTACKS + UNSEEN_ATTACKS

LEET = {"a": ["@", "4"], "e": ["3"], "i": ["1", "!"], "o": ["0"], "s": ["$", "5"], "l": ["1"]}
HOMO = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441",
        "x": "\u0445", "y": "\u0443", "i": "\u0456"}             # Latin -> Cyrillic look-alikes
HOMO_REV = {v: k for k, v in HOMO.items()}
HOMO_REV.update({"\u03bf": "o", "\u03b1": "a", "\u03b5": "e", "\u0455": "s", "\u0458": "j"})  # a few Greek/Cyrillic extras
ZW = "\u200b"
SYN = {"free": "complimentary", "win": "get", "won": "earned", "winner": "selected", "prize": "gift",
       "cash": "funds", "claim": "collect", "urgent": "important", "call": "contact", "text": "message",
       "txt": "msg", "reply": "respond", "offer": "deal", "guaranteed": "assured",
       "congratulations": "well done", "mobile": "handset", "service": "facility", "stop": "end",
       "click": "tap", "account": "profile", "award": "grant", "bonus": "extra", "customer": "client",
       "selected": "chosen", "latest": "newest", "private": "personal", "valid": "active",
       "draw": "lottery", "ringtone": "tune", "chat": "talk", "contact": "reach", "send": "forward",
       "now": "today", "prizes": "gifts", "per": "each", "week": "7 days"}
PADDING = ["ok see you tomorrow at lunch", "thanks for the help yesterday", "I will be home late tonight",
           "can you pick up some milk on the way", "hope you have a great day", "the meeting moved to monday",
           "let me know when you reach home", "love you, take care", "sorry I missed your call earlier",
           "what time is dinner", "are we still on for the weekend", "just saw your message, will reply soon"]
FALLBACK_TRIGGERS = ["free", "win", "winner", "prize", "claim", "cash", "urgent", "call", "text", "reply",
                     "offer", "guaranteed", "congratulations", "account", "bonus", "click", "txt", "mobile"]
_WORD = re.compile(r"[A-Za-z]+")


class Attacker:
    """Black-box text attacker. Obfuscates 'trigger' words that the detector relies on."""

    def __init__(self, triggers=None, seed=0):
        self.rng = random.Random(seed)
        self.triggers = set(t.lower() for t in (triggers or FALLBACK_TRIGGERS))

    def _targets(self, text):
        toks = {m.group(0).lower() for m in _WORD.finditer(text)}
        hit = toks & self.triggers
        if hit:
            return hit
        long_ = [t for t in toks if len(t) >= 4]
        return set(self.rng.sample(long_, min(2, len(long_))))

    def _word_op(self, text, fn, min_len=1):
        tg = self._targets(text)
        return _WORD.sub(lambda m: fn(m.group(0)) if (m.group(0).lower() in tg and len(m.group(0)) >= min_len)
                         else m.group(0), text)

    def apply(self, text, kind):
        r = self.rng
        if kind == "leet":
            def f(w):
                out = []; changed = False
                for ch in w:
                    if ch.lower() in LEET and r.random() < .6:
                        out.append(r.choice(LEET[ch.lower()])); changed = True
                    else:
                        out.append(ch)
                if not changed:
                    for i, ch in enumerate(w):
                        if ch.lower() in LEET:
                            out[i] = LEET[ch.lower()][0]; break
                return "".join(out)
            return self._word_op(text, f)
        if kind == "space":
            return self._word_op(text, lambda w: r.choice([" ", ".", "-"]).join(w), min_len=4)
        if kind == "synonym":
            return self._word_op(text, lambda w: SYN.get(w.lower(), w))
        if kind == "punct":
            def f(w):
                if len(w) < 3: return w
                i = r.randint(1, len(w) - 1)
                return w[:i] + r.choice("*_~^|") + w[i:]
            return self._word_op(text, f)
        if kind == "homoglyph":
            return self._word_op(text, lambda w: "".join(HOMO[c] if (c in HOMO and r.random() < .8) else c for c in w))
        if kind == "zerowidth":
            return self._word_op(text, lambda w: ZW.join(w), min_len=3)
        if kind == "padding":
            k = r.randint(1, 3); extra = ". ".join(r.sample(PADDING, k))
            return f"{text} {extra}" if r.random() < .6 else f"{extra}. {text}"
        raise ValueError(kind)

    def random_variant(self, text, kinds, kmax=3):
        k = self.rng.randint(1, min(kmax, len(kinds)))
        for kind in self.rng.sample(kinds, k):
            text = self.apply(text, kind)
        return text


# ----------------------------------------------------------------------------- canonicaliser
_SPACED = re.compile(r"(?<![A-Za-z])[A-Za-z]([ .\-_])(?:[A-Za-z]\1){2,}[A-Za-z](?![A-Za-z])")
_INNER = re.compile(r"(?<=[A-Za-z])[\*_~\^\|](?=[A-Za-z])")
_LEETTOK = re.compile(r"[A-Za-z][@$0134!5]+[A-Za-z]")
_DELEET = str.maketrans({"@": "a", "4": "a", "3": "e", "1": "i", "!": "i", "0": "o", "$": "s", "5": "s"})
_ZW = re.compile("[\u200b\u200c\u200d\ufeff\u2060]")


def canonicalize(text):
    """Undo cheap obfuscations: unicode tricks, zero-width chars, spaced/punctuated letters, leetspeak."""
    t = unicodedata.normalize("NFKC", str(text))
    t = _ZW.sub("", t)
    t = "".join(HOMO_REV.get(c, c) for c in t)
    t = _SPACED.sub(lambda m: re.sub(r"[\s\.\-_]", "", m.group(0)), t)
    t = _INNER.sub("", t)
    toks = []
    for tok in t.split(" "):
        core = tok.rstrip(".,;:?!")
        if sum(c.isalpha() for c in core) >= 2 and _LEETTOK.search(core):
            tok = core.translate(_DELEET) + tok[len(core):]
        toks.append(tok)
    return " ".join(toks)


def canon_list(X):
    return [canonicalize(x) for x in X]
