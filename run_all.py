#!/usr/bin/env python3
"""Run the complete ScamShield pipeline.   python run_all.py          (full run, ~30-45 min on 1 CPU)
                                           SS_FAST=1 python run_all.py (smoke test, ~8 min)
                                           python run_all.py 3 4 5      (only some stages)"""
import sys, time, traceback
sys.path.insert(0, ".")
import s1_sms, s2_url, s3_adv, s4_probe_agent, s5_report
STAGES = {"1": s1_sms, "2": s2_url, "3": s3_adv, "4": s4_probe_agent, "5": s5_report}
if __name__ == "__main__":
    want = sys.argv[1:] or list(STAGES)
    t0 = time.time()
    for k in want:
        try:
            STAGES[k].run()
        except Exception:
            traceback.print_exc(); print(f"!! stage {k} failed"); break
    print(f"TOTAL {time.time()-t0:.0f}s")
