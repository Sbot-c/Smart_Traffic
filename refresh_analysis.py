"""Recompute the component studies and the decision-network samples from the saved
Q-tables (no retraining) and rewrite those CSV files."""
import time

import experiments as X

if __name__ == "__main__":
    t0 = time.time()
    models = X.load_models("data/q_tables.npz")
    abl = X.ablations(models, X.FULL, verbose=True)
    abl.round(4).to_csv("data/ablation_results.csv", index=False)
    dn = X.dn_samples(models, X.FULL, verbose=True)
    dn.round(4).to_csv("data/decision_network_samples.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")
