"""
experiments.py - training, scenario evaluation, decision-network data and CSV export.

Pipeline (run_all):
  1. Pre-train each algorithm (Q-learning, SARSA) on a round-robin of all scenarios with
     ONE shared signal table (parameter sharing: nine junctions pool their experience).
  2. Adapt: for every scenario, copy the general tables into nine independent junction
     agents (plus the AV and bus fleets) and keep learning on that scenario only.
  3. Evaluate the four integrated policies on held-out seeds with common random numbers
     (the same arrivals for every policy) -> kpi_results.csv.
  4. Record one representative episode per scenario x policy -> simulation log CSV and
     the animation files used by the dashboard.
  5. Generate decision-network training data: every combination of weather x density x
     pedestrian flow x signal policy x routing policy -> junction-window samples.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import agents as A
import config as C
import traffic_env as TE

FULL = dict(pre_episodes=280, adapt_episodes=80, eval_seeds=8, dn_seeds=2)
QUICK = dict(pre_episodes=70, adapt_episodes=15, eval_seeds=3, dn_seeds=1)
EVAL_SEED0 = 900_000
ALGOS = {"Q-learning": "qlearning", "SARSA": "sarsa"}


def eps_schedule(ep: int, n: int, start: float, end: float) -> float:
    frac = min(1.0, ep / max(1.0, 0.8 * n))
    return start + (end - start) * frac


TRAIN_PED_LEVELS = (0.5, 1.0, 2.0, 4.0)


def training_overrides(rng, scenario: str, stage: str) -> dict:
    """Domain randomisation: training episodes sample the uncertain conditions that the
    decision network models (pedestrian flow, weather, demand level), so the learned
    policies are robust to them rather than tuned to one value."""
    ov = {"ped_mult": float(rng.choice(TRAIN_PED_LEVELS))}
    if scenario in ("normal", "rain", "sandstorm"):
        ov["demand_mult"] = float(rng.choice([0.75, 1.0]))
    if scenario == "rush_hour" or (stage == "pre-train" and scenario not in ("rain",
                                                                          "sandstorm")):
        u = rng.random()
        ov["weather"] = "Clear" if u < 0.7 else ("Rain" if u < 0.85 else "Sandstorm")
    return ov


def _curve_row(policy, stage, ep, sc, eps, env):
    k = env.kpis()
    return dict(policy=policy, stage=stage, episode=ep, scenario=sc, epsilon=round(eps, 4),
                weather=env.weather, ped_mult=env.sc["ped_mult"],
                demand_mult=env.sc["demand_mult"],
                signal_return=env.signal_return, avg_delay_s=k["avg_delay_s"],
                conflicts_per_1000=k["conflicts_per_1000"], co2_kg=k["co2_kg"],
                bus_on_time_pct=k["bus_on_time_pct"], av_avg_delay_s=k["av_avg_delay_s"],
                ped_avg_wait_s=k["ped_avg_wait_s"])


# ======================================================================================
# 1-2. Training
# ======================================================================================
def train_algorithm(policy: str, cfg: dict, scenarios=None, seed: int = 0,
                    verbose: bool = True):
    algo = ALGOS[policy]
    scenarios = scenarios or C.SCENARIO_ORDER
    sysm = A.MultiAgentSystem(algo, algo, algo, coop=True, seed=seed)
    sysm.signals = A.RLSignals(algo, seed, shared=True)
    curve = []
    t0 = time.time()
    rng = np.random.default_rng(4242 + seed)
    n_pre = cfg["pre_episodes"]
    for ep in range(n_pre):
        sc = scenarios[ep % len(scenarios)]
        eps = eps_schedule(ep, n_pre, 0.30, 0.05)
        env = sysm.run_episode(sc, seed=100_000 + ep, train=True, eps=eps,
                               overrides=training_overrides(rng, sc, "pre-train"))
        curve.append(_curve_row(policy, "pre-train", ep, sc, eps, env))
        if verbose and (ep % 40 == 0 or ep == n_pre - 1):
            print(f"  [{policy}] pre-train {ep + 1:4d}/{n_pre}  {sc:12s} "
                  f"delay {curve[-1]['avg_delay_s']:6.1f} s  ({time.time() - t0:.0f} s)")
    general = sysm.snapshot()
    adapted = {}
    for k, sc in enumerate(scenarios):
        s2 = A.MultiAgentSystem(algo, algo, algo, coop=True, seed=seed + 1 + k)
        s2.load(general)
        n_ad = cfg["adapt_episodes"]
        for ep in range(n_ad):
            eps = eps_schedule(ep, n_ad, 0.10, 0.02)
            env = s2.run_episode(sc, seed=200_000 + 1000 * k + ep, train=True, eps=eps,
                                 overrides=training_overrides(rng, sc, "adapt"))
            curve.append(_curve_row(policy, "adapt", ep, sc, eps, env))
        adapted[sc] = s2.snapshot()
        if verbose:
            print(f"  [{policy}] adapted to {sc:12s} last delay "
                  f"{curve[-1]['avg_delay_s']:6.1f} s  ({time.time() - t0:.0f} s)")
    return dict(general=general, adapted=adapted, curve=pd.DataFrame(curve))


# ======================================================================================
# 3. Evaluation
# ======================================================================================
def policy_tables(models, policy, scenario, adapted=True):
    if policy not in ALGOS:
        return None
    m = models[policy]
    return m["adapted"][scenario] if adapted else m["general"]


def evaluate_policies(models, cfg, scenarios=None, verbose=True):
    scenarios = scenarios or C.SCENARIO_ORDER
    rows = []
    plan = A.webster_plan("normal")
    for sc in scenarios:
        for pol in A.POLICY_ORDER:
            sysm = A.make_system(pol, policy_tables(models, pol, sc), plan=plan)
            for j in range(cfg["eval_seeds"]):
                env = sysm.run_episode(sc, seed=EVAL_SEED0 + j)
                k = env.kpis()
                k.update(scenario=sc, scenario_label=C.SCENARIOS[sc]["label"],
                         family=C.SCENARIO_FAMILY[sc], policy=pol, seed=j,
                         weather=env.weather)
                rows.append(k)
        if verbose:
            df = pd.DataFrame([r for r in rows if r["scenario"] == sc])
            msg = "  ".join(f"{p}: {df[df.policy == p].avg_delay_s.mean():5.1f}"
                            for p in A.POLICY_ORDER)
            print(f"  eval {sc:12s} delay  {msg}")
    df = pd.DataFrame(rows)
    front = ["scenario", "scenario_label", "family", "weather", "policy", "seed"]
    return df[front + [c for c in df.columns if c not in front]]


def adapt_selfish(models, cfg, scenario, seed=0):
    """Re-adapt the general Q-learning tables with a selfish AV reward (no externality)."""
    s2 = A.MultiAgentSystem("qlearning", "qlearning", "qlearning", coop=False, seed=seed + 77)
    s2.load(models["Q-learning"]["general"])
    rng = np.random.default_rng(7777 + seed)
    for ep in range(cfg["adapt_episodes"]):
        eps = eps_schedule(ep, cfg["adapt_episodes"], 0.10, 0.02)
        s2.run_episode(scenario, seed=300_000 + ep, train=True, eps=eps,
                       overrides=training_overrides(rng, scenario, "adapt"))
    return s2.snapshot()


def ablations(models, cfg, verbose=True):
    """Component studies: AV routing mode and bus priority strategy (Q-learning signals)."""
    rows = []
    plan = A.webster_plan("normal")
    for sc in ["incident", "rush_hour", "normal"]:
        coop_tables = policy_tables(models, "Q-learning", sc)
        selfish_tables = adapt_selfish(models, cfg, sc)
        for label, routing, tables in [("Static routes", "static", coop_tables),
                                       ("Selfish RL routing", "qlearning", selfish_tables),
                                       ("Cooperative RL routing", "qlearning", coop_tables)]:
            sysm = A.MultiAgentSystem("qlearning", routing, "qlearning", plan=plan)
            sysm.load(tables)
            for j in range(cfg["eval_seeds"]):
                k = sysm.run_episode(sc, seed=EVAL_SEED0 + j).kpis()
                k.update(study="AV routing", variant=label, scenario=sc, seed=j)
                rows.append(k)
    for sc in ["rush_hour", "event_surge", "incident", "normal"]:
        for sig in DN_SIGNAL:
            for rout in DN_ROUTING:
                sysm = dn_system(models, sig, rout, sc, plan)
                for j in range(cfg["eval_seeds"]):
                    k = sysm.run_episode(sc, seed=EVAL_SEED0 + j).kpis()
                    k.update(study="Signal x routing", variant=f"{sig} + {rout}", scenario=sc,
                             seed=j)
                    rows.append(k)
    for sc in ["bus_corridor", "rush_hour", "normal"]:
        tables = policy_tables(models, "Q-learning", sc)
        for label, bus in [("No priority", "none"), ("Always request TSP", "always"),
                           ("Learned (Q-learning)", "qlearning")]:
            sysm = A.MultiAgentSystem("qlearning", "qlearning", bus, coop=True, plan=plan)
            sysm.load(tables)
            for j in range(cfg["eval_seeds"]):
                k = sysm.run_episode(sc, seed=EVAL_SEED0 + j).kpis()
                k.update(study="Bus priority", variant=label, scenario=sc, seed=j)
                rows.append(k)
    df = pd.DataFrame(rows)
    if verbose:
        print(df.groupby(["study", "scenario", "variant"])[
            ["avg_delay_s", "av_avg_delay_s", "bus_avg_delay_s", "bus_on_time_pct"]]
            .mean().round(1).to_string())
    front = ["study", "scenario", "variant", "seed"]
    return df[front + [c for c in df.columns if c not in front]]


# ======================================================================================
# 4. Simulation log (30-second windows) + animation recordings
# ======================================================================================
WINDOW_S = 30.0


def episode_log_frame(env, scenario, policy) -> pd.DataFrame:
    """Aggregate the per-step junction log of one episode into 30 s windows."""
    log = env.log                      # (steps, junctions, fields)
    per = int(WINDOW_S / C.DT)
    n_win = log.shape[0] // per
    F = TE.F
    rows = []
    for i in range(C.N_INT):
        r, c = TE.rc(i)
        for w in range(n_win):
            blk = log[w * per:(w + 1) * per, i, :]
            rows.append(dict(
                scenario=scenario, scenario_label=C.SCENARIOS[scenario]["label"],
                weather=env.weather, policy=policy, junction=C.JUNCTION_NAMES[i],
                street=C.STREET_NAMES[r], avenue=C.AVENUE_NAMES[c],
                t_start_s=int(w * WINDOW_S),
                ns_green_share=round(float(np.mean(blk[:, F["phase"]] == 0)), 3),
                phase_changes=int(blk[:, F["executed"]].sum()),
                queue_n=round(float(blk[:, F["q_n"]].mean()), 2),
                queue_e=round(float(blk[:, F["q_e"]].mean()), 2),
                queue_s=round(float(blk[:, F["q_s"]].mean()), 2),
                queue_w=round(float(blk[:, F["q_w"]].mean()), 2),
                arrivals=int(blk[:, F["arrivals"]].sum()),
                departures=int(blk[:, F["departures"]].sum()),
                vehicle_delay_s=round(float(blk[:, F["delay"]].sum()), 1),
                ped_waiting=round(float((blk[:, F["ped_wait_ns"]] +
                                         blk[:, F["ped_wait_ew"]]).mean()), 2),
                ped_delay_s=round(float(blk[:, F["ped_delay"]].sum()), 1),
                jaywalks=int(blk[:, F["jaywalks"]].sum()),
                expected_conflicts=round(float(blk[:, F["conflicts"]].sum()), 4),
                co2_g=round(float(blk[:, F["co2"]].sum()), 1),
                nox_g=round(float(blk[:, F["nox"]].sum()), 3),
                stops=int(blk[:, F["stops"]].sum()),
                spillback_blocks=int(blk[:, F["blocked"]].sum()),
                ev_present=int(blk[:, F["ev"]].max()),
                tsp_active=int(blk[:, F["tsp"]].max()),
                overrides=int((blk[:, F["override"]] > 0).sum()),
                reward=round(float(blk[:, F["reward"]].sum()), 3),
            ))
    return pd.DataFrame(rows)


def animation_payload(env, scenario, policy, t_max=None) -> dict:
    """Compact JSON for the dashboard canvas: vehicle link segments + signal timeline."""
    t_max = t_max or C.EPISODE_S
    segs = [s for s in env.rec_segments if s[3] < t_max]
    sig = [[int(p) for p in phases] for (_, phases, _) in env.rec_signal]
    sw = [[1 if x else 0 for x in flags] for (_, _, flags) in env.rec_signal]
    ped = [[a + b for (a, b) in step] for step in env.rec_ped]
    F = TE.F
    return dict(
        scenario=scenario, policy=policy, dt=C.DT, lost=env.lost_time,
        t_max=t_max, block=C.BLOCK_M, entry=C.ENTRY_M,
        lanes=list(TE.LINK_LANES), up=list(TE.LINK_UP),
        segments=[list(s[1:]) for s in segs],        # vtype, link, t_enter, t_arr, t_dep
        phase=sig, switched=sw, ped=ped,
        incident=C.SCENARIOS[scenario].get("incident"),
        queues=np.round(env.log[:, :, [F["q_n"], F["q_e"], F["q_s"], F["q_w"]]], 0)
        .astype(int).tolist(),
        override=env.log[:, :, F["override"]].astype(int).tolist(),
        chosen=env.log[:, :, F["action"]].astype(int).tolist(),
        weather=env.weather,
    )


def record_runs(models, scenarios=None, seed=0, out_dir=None, anim_minutes=15):
    scenarios = scenarios or C.SCENARIO_ORDER
    plan = A.webster_plan("normal")
    frames, anims, decisions = [], {}, []
    for sc in scenarios:
        for pol in A.POLICY_ORDER:
            sysm = A.make_system(pol, policy_tables(models, pol, sc), plan=plan)
            env = sysm.run_episode(sc, seed=EVAL_SEED0 + seed, record=True)
            frames.append(episode_log_frame(env, sc, pol))
            anims[f"{sc}|{pol}"] = animation_payload(env, sc, pol, anim_minutes * 60.0)
            for (t, vid, route, i, s, act, dev, load) in (env.bus_log or []):
                decisions.append(dict(scenario=sc, policy=pol, agent="bus", t_s=round(t, 1),
                                      agent_id=f"bus-{vid}", junction=C.JUNCTION_NAMES[i],
                                      state=json.dumps(A.decode_bus_state(s)),
                                      action=["go", "request priority", "hold 20 s"][act],
                                      detail=f"route {route}, deviation {dev:+.0f} s, "
                                             f"load {load}"))
            for (t, vid, i, cx, cy, a, h) in (env.av_log or [])[:400]:
                lv = ["free", "moderate", "heavy", "incident"]
                decisions.append(dict(scenario=sc, policy=pol, agent="AV", t_s=round(t, 1),
                                      agent_id=f"av-{vid}", junction=C.JUNCTION_NAMES[i],
                                      state=json.dumps(dict(x_link=lv[cx], y_link=lv[cy])),
                                      action=f"head {C.HEADING_NAMES[h]}",
                                      detail="x-move" if a == 0 else "y-move"))
    sim_log = pd.concat(frames, ignore_index=True)
    return sim_log, anims, pd.DataFrame(decisions)


# ======================================================================================
# 5. Decision-network training data
# ======================================================================================
# Density levels reuse the scenario demand patterns: off-peak, normal, rush hour
DN_DENSITY = {"Low": dict(demand_mult=0.75, profile="flat", corridor_mult=1.0),
              "Medium": dict(demand_mult=1.0, profile="flat", corridor_mult=1.0),
              "High": dict(demand_mult=1.30, profile="peak", corridor_mult=1.15)}
DN_PED = {"Low": 0.5, "Medium": 1.0, "High": 4.0}
DN_SIGNAL = ["Fixed-time", "Actuated", "Q-learning", "SARSA"]
DN_ROUTING = ["Static", "Cooperative RL"]
DN_WINDOWS = [(300, 800), (800, 1300), (1300, 1800)]


def condition_scenario(weather: str, density: str, ped: str) -> str:
    """Which scenario-adapted RL tables an operator would deploy under these conditions
    (demand level first, then weather)."""
    if density == "High":
        return "rush_hour"
    if weather == "Rain":
        return "rain"
    if weather == "Sandstorm":
        return "sandstorm"
    return "normal"


def dn_system(models, sig: str, rout: str, tables_scenario: str, plan=None):
    """The system behind one decision pair (signal timing x routing). Baseline signals keep
    their own bus rule; cooperative RL routing uses the Q-learning AV table unless the
    signals are SARSA (then the SARSA AV table)."""
    spec = dict(A.POLICIES[sig])
    routing = "static" if rout == "Static" else (
        spec["routing"] if spec["routing"] != "static" else "qlearning")
    sysm = A.MultiAgentSystem(spec["signal"], routing, spec["bus"], coop=True, plan=plan)
    if sig in ALGOS:
        sysm.load(models[sig]["adapted"][tables_scenario])
    if routing != "static" and sysm.routing is not None:
        src = sig if sig in ALGOS else "Q-learning"
        sysm.routing.learner.load(models[src]["adapted"][tables_scenario]["routing"])
    return sysm


def dn_samples(models, cfg, verbose=True):
    rows = []
    plan = A.webster_plan("normal")
    t0 = time.time()
    combos = [(w, d, p, s, r) for w in C.WEATHER_NAMES for d in DN_DENSITY for p in DN_PED
              for s in DN_SIGNAL for r in DN_ROUTING]
    F = TE.F
    conditions = [(w, d, p) for w in C.WEATHER_NAMES for d in DN_DENSITY for p in DN_PED]
    for n, (w, d, p, sig, rout) in enumerate(combos):
        cond = conditions.index((w, d, p))      # same seeds for every decision: paired data
        sc_tables = condition_scenario(w, d, p)
        sysm = dn_system(models, sig, rout, sc_tables, plan)
        routing = "static" if sysm.routing is None else "rl"
        ov = dict(weather=w, ped_mult=DN_PED[p], **DN_DENSITY[d])
        for j in range(cfg["dn_seeds"]):
            env = sysm.run_episode("normal", seed=700_000 + 97 * cond + j, overrides=ov)
            log = env.log
            for i in range(C.N_INT):
                for (a, b) in DN_WINDOWS:
                    s0, s1 = int(a / C.DT), int(b / C.DT)
                    blk = log[s0:s1, i, :]
                    dep = max(1.0, float(blk[:, F["departures"]].sum()))
                    rows.append(dict(
                        weather=w, density=d, ped_flow=p, signal=sig, routing=rout,
                        rl_tables=sc_tables if (sig in ALGOS or routing != "static") else "",
                        seed=j, junction=C.JUNCTION_NAMES[i], window=f"{a}-{b}",
                        delay_per_veh_s=float(blk[:, F["delay"]].sum()) / dep,
                        conflicts_per_1000=1000.0 * float(blk[:, F["conflicts"]].sum()) / dep,
                        co2_g_per_veh=float(blk[:, F["co2"]].sum()) / dep,
                        nox_g_per_veh=float(blk[:, F["nox"]].sum()) / dep,
                        jaywalks=int(blk[:, F["jaywalks"]].sum()),
                        departures=int(dep)))
        if verbose and (n + 1) % 54 == 0:
            print(f"  decision-network data {n + 1}/{len(combos)} combos "
                  f"({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows)


# ======================================================================================
# Persistence
# ======================================================================================
def save_models(models, path):
    flat = {}
    for pol, m in models.items():
        for tag, snap in [("general", m["general"])] + [(f"adapted-{k}", v)
                                                        for k, v in m["adapted"].items()]:
            for i, s in enumerate(snap["signals"]):
                flat[f"{pol}__{tag}__signal{i}__Q"] = s["Q"]
                flat[f"{pol}__{tag}__signal{i}__N"] = s["visits"]
            for part in ("routing", "bus"):
                flat[f"{pol}__{tag}__{part}__Q"] = snap[part]["Q"]
                flat[f"{pol}__{tag}__{part}__N"] = snap[part]["visits"]
    np.savez_compressed(path, **flat)


def load_models(path):
    z = np.load(path)
    models = {}
    for key in z.files:
        pol, tag, part, kind = key.split("__")
        m = models.setdefault(pol, dict(general=None, adapted={}))
        if tag == "general":
            snap = m["general"] = m["general"] or dict(signals=[None] * C.N_INT)
        else:
            sc = tag[len("adapted-"):]
            snap = m["adapted"].setdefault(sc, dict(signals=[None] * C.N_INT))
        field = "Q" if kind == "Q" else "visits"
        if part.startswith("signal"):
            i = int(part[6:])
            snap["signals"][i] = snap["signals"][i] or {}
            snap["signals"][i][field] = z[key]
        else:
            snap.setdefault(part, {})[field] = z[key]
    return models


def run_all(cfg=None, out_dir="data", verbose=True):
    cfg = cfg or FULL
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    models = {}
    for pol in ALGOS:
        if verbose:
            print(f"Training {pol} ...")
        models[pol] = train_algorithm(pol, cfg, verbose=verbose)
    curves = pd.concat([m["curve"] for m in models.values()], ignore_index=True)
    if verbose:
        print("Evaluating policies ...")
    kpi = evaluate_policies(models, cfg, verbose=verbose)
    if verbose:
        print("Ablations ...")
    abl = ablations(models, cfg, verbose=verbose)
    if verbose:
        print("Recording representative runs ...")
    sim_log, anims, decisions = record_runs(models)
    if verbose:
        print("Decision-network data ...")
    dn = dn_samples(models, cfg, verbose=verbose)
    return dict(models=models, curves=curves, kpi=kpi, ablations=abl, sim_log=sim_log,
                anims=anims, decisions=decisions, dn=dn, seconds=time.time() - t0)


def scenario_table() -> pd.DataFrame:
    rows = []
    for key in C.SCENARIO_ORDER:
        sc = C.SCENARIOS[key]
        w = C.WEATHER[sc["weather"]]
        rows.append(dict(
            scenario=key, label=sc["label"], family=C.SCENARIO_FAMILY[key],
            weather=sc["weather"], demand_multiplier=sc["demand_mult"],
            demand_profile=sc["profile"], corridor_eastbound_multiplier=sc["corridor_mult"],
            pedestrian_multiplier=sc["ped_mult"], emergency_calls_per_30min=sc["ev_per_episode"],
            bus_headway_s=sc["bus_headway_s"], bus_lane=sc["bus_lane"],
            passenger_multiplier=sc["pax_mult"],
            saturation_flow_factor=w[0], speed_factor=w[1], crash_risk_multiplier=w[2],
            incident=json.dumps(sc["incident"]) if sc.get("incident") else "",
            event=json.dumps(sc["event"]) if sc.get("event") else "",
            description=sc["description"]))
    return pd.DataFrame(rows)


def export_all(res: dict, out_dir: str = "data"):
    import gzip
    os.makedirs(out_dir, exist_ok=True)
    res["sim_log"].to_csv(os.path.join(out_dir, "smart_traffic_simulation_log.csv"),
                          index=False)
    res["kpi"].round(4).to_csv(os.path.join(out_dir, "kpi_results.csv"), index=False)
    res["curves"].round(4).to_csv(os.path.join(out_dir, "learning_curves.csv"), index=False)
    res["ablations"].round(4).to_csv(os.path.join(out_dir, "ablation_results.csv"),
                                     index=False)
    res["decisions"].to_csv(os.path.join(out_dir, "agent_decisions.csv"), index=False)
    res["dn"].round(4).to_csv(os.path.join(out_dir, "decision_network_samples.csv"),
                              index=False)
    scenario_table().to_csv(os.path.join(out_dir, "scenario_parameters.csv"), index=False)
    save_models(res["models"], os.path.join(out_dir, "q_tables.npz"))
    with gzip.open(os.path.join(out_dir, "animations.json.gz"), "wt") as f:
        json.dump(res["anims"], f, separators=(",", ":"))
