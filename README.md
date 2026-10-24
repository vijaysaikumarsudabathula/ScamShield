# ScamShield — explainable, adversarially-hardened scam detection

**Problem.** Everyone receives scam texts, fake delivery links and phishing messages. Filters are black boxes, go stale as scammers reword messages, and stop at a label. ScamShield (1) detects scam text and phishing domains, (2) explains *why*, (3) is stress-tested and hardened against an adaptive attacker, (4) measures how badly it degrades on modern scam styles, and (5) tells the user what to do next.

## Run it
```bash
pip install -r requirements.txt
python run_all.py            # full run, ~17 min on 1 CPU (all data downloads itself from GitHub raw URLs)
SS_FAST=1 python run_all.py  # ~8 min smoke test
python run_all.py 3 4 5      # only some stages
```
Data (auto-downloaded, no manual steps): UCI SMS Spam Collection · Phishing.Database ACTIVE domains · OpenDNS top-10k domains.

## Headline results (held-out test, selection by 5-fold CV F1)
| Task | Best model (CV) | Acc | Prec | Recall | F1 | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| SMS scam text (n=1,035; 131 scams) | Logistic Regression | 0.988 | 0.976 | 0.931 | 0.953 | 0.997 | 0.947 |
| Phishing domain (n=3,996) | Voting Ensemble | 0.911 | 0.953 | 0.864 | 0.906 | 0.960 | 0.825 |

* SMS: 10 models; Multinomial NB has the highest *test* F1 (0.965) but **no pairwise difference is statistically significant** (exact McNemar, all p>0.25) — the test set has only 131 scams.
* Domain: 12 models all plateau at F1≈0.90; a feature-free character n-gram baseline gets 0.899. `n_subdomains` carries ~50% of LightGBM gain (dataset shortcut). On the hard subset (bare domains, no sub-domain/free host) F1 drops to ≈0.65 (recall ≈0.55).

## Adversarial study (Logistic Regression, adaptive black-box attacker, 30 queries/message)
| Defence | Evade: seen attacks | Evade: unseen attacks | Evade: all | Clean F1 |
|---|---|---|---|---|
| Baseline | 9.0% | 18.0% | 13.9% | 0.953 |
| + Canonicalization | 4.9% | 5.7% | 9.0% | 0.953 |
| + Adversarial training (3 rounds) | 0.8% | 12.3% | 8.2% | 0.957 |
| + Both | 0.8% | 9.8% | 8.9% | 0.957 |

Hardening cuts evasion on known attacks to ~1% with no loss of clean accuracy, but **does not solve the padding attack** (benign text appended to a scam: 5% → 8%). With 122 detected scams, one message = 0.8 points, so differences of 1–2 points between defences are noise.

## Concept drift (60 hand-written modern messages)
Legacy-trained linear models keep F1≈0.82–0.86; tree models collapse (Random Forest 0.33, Extra Trees 0.38, XGBoost 0.55). Updating with 30 newly reported messages raises held-out ROC-AUC 0.924→0.936 and specificity 0.88→0.92 but **not recall** (0.87→0.84), and legacy F1 is retained (0.957).

## Agent
Extractor → Analyzer → Verifier → Scorer → Advisor (see `figures/00_architecture.png`, `tables/agent_demo_reports.md`). On the probe, text-only and the full agent give the same F1 (0.862; AUC 0.906→0.909). Known failures: family-emergency and romance scams score ≈0 (no links, no scammy vocabulary), and a legitimate payment receipt is flagged HIGH (0.96).

## Limitations (state these in any write-up)
1. SMS corpus is old, UK-style spam — easy and dated; the modern probe exists to show this.
2. The 60-message probe was written by the project author: a stress test, not a benchmark. Agent-vs-text results on it are illustrative.
3. Legit domains are popular bare domains; phishing domains come from a feed — structural shortcuts inflate scores.
4. The attacker is rule-based. An LLM attacker would likely be stronger (`ss_agent.llm_advise` is an untested optional hook).
5. Text+domain fusion is not validated on a jointly labelled dataset.

## Repository layout
| Path | What it is |
|---|---|
| `run_all.py`, `s1_sms.py` … `s5_report.py`, `ss_*.py` | The full pipeline: data download, 10 text + 12 domain models, adversarial study, drift probe, agent, report assets |
| `figures/`, `tables/` | 39 result figures and 38 CSV tables produced by the run |
| `ScamShield_Results.xlsx` | Every table in one workbook (38 sheets) |
| `ScamShield_Report.html`, `build_report.py` | Self-contained results report (flow charts, all figures, tables, source code) |
| `paper/` | The 8-page IEEE-format paper (`.docx` and `.pdf`) and the scripts that build it |
| `run_full.log` | Console log of the full run used for every reported number |

## Rebuild everything
```bash
pip install -r requirements.txt
python run_all.py                    # ~17 min on one CPU; downloads the public datasets itself, writes cache/, figures/, tables/
python build_report.py               # HTML report
cd paper
python make_paper_figs.py            # paper-sized figures (needs the cache/ created by run_all.py for the SHAP chart)
python build_spec.py                 # paper content; every number is read from ../tables
node render.js                       # needs: npm install docx   -> ScamShield_IEEE_Research_Paper.docx
```

## Notes
* `cache/` (downloaded data and fitted models, ~570 MB) is not committed; `run_all.py` recreates it.
* Author, affiliation and e-mail in the paper are placeholders.
* Reference details in the paper should be checked against the originals before submission.
