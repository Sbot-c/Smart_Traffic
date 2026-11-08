"""Training-seed robustness: re-adapt the general tables with different random seeds and
evaluate on the same held-out episodes. Writes data/training_robustness.csv."""
import sys
import numpy as np
import pandas as pd
import agents as A
import experiments as X

ALGO = {"Q-learning": "qlearning", "SARSA": "sarsa"}


def run(scenarios=("normal", "rush_hour", "event_surge"), seeds=(1, 2, 3), episodes=80,
        eval_seeds=4, path="data/training_robustness.csv"):
    models = X.load_models("data/q_tables.npz")
    rows = []
    for sc in scenarios:
        for pol, algo in ALGO.items():
            for sd in seeds:
                s2 = A.MultiAgentSystem(algo, algo, algo, coop=True, seed=500 + sd)
                s2.load(models[pol]["general"])
                rng = np.random.default_rng(9000 + sd)
                for ep in range(episodes):
                    eps = X.eps_schedule(ep, episodes, 0.10, 0.02)
                    s2.run_episode(sc, seed=600_000 + 1000 * sd + ep, train=True, eps=eps,
                                   overrides=X.training_overrides(rng, sc, "adapt"))
                ks = [s2.run_episode(sc, seed=X.EVAL_SEED0 + j).kpis() for j in range(eval_seeds)]
                rows.append(dict(scenario=sc, policy=pol, training_seed=sd,
                                 avg_delay_s=np.mean([k["avg_delay_s"] for k in ks]),
                                 conflicts_per_1000=np.mean([k["conflicts_per_1000"] for k in ks]),
                                 co2_kg=np.mean([k["co2_kg"] for k in ks]),
                                 jaywalks=np.mean([k["jaywalks"] for k in ks])))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(path, index=False)
    return df


if __name__ == "__main__":
    run()
