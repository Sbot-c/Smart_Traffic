"""Run the full experiment pipeline and write every CSV used by the dashboard.

    python run_experiments.py           # full run (~6-8 minutes)
    python run_experiments.py --quick   # quick check (~2 minutes)
"""
import sys
import time

import experiments as X

if __name__ == "__main__":
    cfg = X.QUICK if "--quick" in sys.argv else X.FULL
    t0 = time.time()
    res = X.run_all(cfg)
    X.export_all(res, "data")
    print(f"done in {time.time() - t0:.0f} s")
