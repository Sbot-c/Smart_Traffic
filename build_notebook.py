"""Assemble the Colab notebook from the project's source files (single source of truth)."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def src(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def writefile(name):
    return code(f"%%writefile {name}\n" + src(name))


cells = []
cells.append(md(r"""
# Smart City Traffic Management
### Multi-agent reinforcement learning (Q-learning & SARSA) with decision-network evaluation

**Group 2** · Reasoning and Decision Making under Uncertainty (MAIB DSC 103) · SP Jain School of Global Management, Dubai · Term 2, 2026

This notebook implements the complete project pipeline end to end:

| Step | What it does | Section |
|---|---|---|
| 1. System design | Multi-agent ecosystem: 9 traffic-light controllers, an autonomous-vehicle (AV) fleet and bus agents on a 3 × 3 Downtown-Dubai-style grid | §1 |
| 2. Simulation | Vehicle-level traffic simulator with peak-hour congestion, pedestrian crossings, emergency-vehicle preemption, rain and sandstorm effects, an incident and an event surge | §2 |
| 3. Reinforcement learning | Tabular **Q-learning** and **SARSA** for all three agent types, with a safety shield | §3 |
| 4. Scenario experiments | Six scenario families × four control policies × 8 held-out seeds | §4 |
| 5. Decision network | Influence diagram whose probability tables are learned from the simulator; expected utility, value of information, weight sensitivity | §5 |
| 6. Data export | The CSV files used by the Streamlit dashboard | §6 |

**How to run:** *Runtime → Run all.* `QUICK = True` (below) runs a shortened version in about 3 minutes; `QUICK = False` reproduces the reported results (about 10 minutes on a Colab CPU). No GPU is needed and only standard libraries are used.

> Reference article: Wairagade, A., Ugale, K. & Waghmare, R. (2025). *AI-Powered Smart Traffic Management System for Urban Congestion Reduction.* International Journal of Scientific Research & Engineering Trends, 11(2), 1997–2001. The article proposes RL-based signal timing with emergency override and a monitoring dashboard but reports no quantitative evaluation; this project builds and evaluates such a system under uncertainty.
"""))
cells.append(code(r"""
QUICK = False   # True = ~3 minute demo run, False = full experiment (~10 minutes)

import sys, os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e6df", "font.size": 10})
POLICY_COLORS = {"Fixed-time": "#eda100", "Actuated": "#1baf7a", "Q-learning": "#2a78d6", "SARSA": "#eb6834"}
print("numpy", np.__version__, "| pandas", pd.__version__)
"""))

cells.append(md(r"""
## 1. System design: the multi-agent traffic ecosystem

The city is modelled as a **3 × 3 grid of signalised junctions** (J1–J9, 400 m blocks, 1.2 km × 1.2 km) inspired by the Downtown Dubai / Business Bay street grid. Street 2 is the main arterial (3 lanes per direction) and the bus-priority corridor; all other approaches have 2 lanes. Traffic enters at 12 boundary entries with a gravity-style origin–destination matrix.

| Agent | How many | State $s$ (discretised) | Actions $a$ | Reward $r$ |
|---|---|---|---|---|
| **Traffic-light controller** | 9 (one Q-table per junction) | phase · green time (5 bands) · queue on green · queue on red (5 bands each) · arrivals on green in next 6 s · flag (pedestrians waiting / spillback) → 2,250 states | every 5 s: **keep** or **change** phase | $-\big(\text{vehicle delay} + 0.5\,\text{pedestrian delay} + 300\,\text{expected conflicts} + 0.2\,\text{g CO}_2\big)/100$ |
| **Autonomous vehicle** (25 % of cars) | fleet, one shared table | remaining moves in x and y · congestion of the two candidate next links (free / moderate / heavy / incident) → 64 states | next link: move along **x** or **y** | $-\text{travel time to next decision}/10 - \beta\cdot\text{occupancy of chosen link}$ (cooperative term, $\beta=1$) |
| **Bus** (2 routes × 2 directions) | fleet, one shared table | schedule deviation · passenger load · headway · signal ahead · cross-street queue → 144 states | **go**, **request priority** (TSP), or **hold 20 s** | $-\big(\text{passenger-minutes}/40 + |\text{schedule deviation}|\text{ (min)} + \text{priority externality}\big)$ |

**Safety shield** (applied to every controller, learned or not): minimum green 10 s; pedestrian clearance (a green cannot end within 19 s of pedestrians starting to cross); maximum green 90 s; **emergency-vehicle preemption** (green for the emergency vehicle's approach when it is 15 s away); **transit signal priority** when a bus requests it (green extension up to 15 s, or early green).

**Baselines:** a **Webster fixed-time** plan computed from design flows (the status quo), and a **vehicle-actuated** controller (gap-out / queue-ratio / pedestrian call, max green 60 s) with unconditional bus priority.

**Environment features** (all in `config.py`, with sources): peak-hour demand profile; Poisson pedestrian arrivals with a 7 s WALK interval and HCM-calibrated impatience (jaywalking risk after 30 s, high after 60 s); emergency vehicles that overtake queues; rain (saturation flow −6 %, speed −10 %, crash risk ×1.71) and sandstorm (capacity −12 %, speed −20 %, crash risk ×2.0); an incident that removes 75 % of capacity on one corridor link; an arena event surge; scheduled buses with near-side stops and dwell times.
"""))

cells.append(md(r"""
## 2. Simulation environment

### 2.1 Parameters and scenarios — `config.py`
Every number the simulator uses lives in one file, with its source. Values marked **ASSUMPTION** are modelling choices.
"""))
cells.append(writefile("config.py"))
cells.append(md(r"""
### 2.2 The traffic simulator — `traffic_env.py`
A mesoscopic **point-queue model with storage**: each vehicle is an object; each link has a free-flow travel time and a storage limit (a full downstream link blocks departures, so spillback propagates). At each stop line vehicles wait in a FIFO queue and are discharged during green at the saturation flow rate, with costs for turning, heavy vehicles, AV platooning and crossing pedestrians. Delay, stops, idling, CO₂ and NOx are tracked per vehicle; expected conflicts (dilemma-zone exposure at phase changes, rear-end exposure at stops, jaywalking) per junction.

Demand is generated with **common random numbers**: for a given seed every controller sees exactly the same arrivals, so policy comparisons are paired.
"""))
cells.append(writefile("traffic_env.py"))
cells.append(code(r"""
import config as C, traffic_env as TE
# The environment API: every 5 s each junction gets an action (0 keep, 1 change phase),
# the safety shield vets it, and step() returns one reward per junction.
# Here a naive controller changes phase every 30 s, for one 30-minute episode.
env = TE.TrafficEnv("normal", seed=1)
t0 = time.time()
while not env.done:
    chosen = [1 if env.elapsed[i] >= 30 else 0 for i in range(C.N_INT)]
    executed, codes = env.shield(chosen)
    rewards = env.step(executed, chosen, codes)
k = env.kpis()
print(f"simulated 30 minutes in {time.time() - t0:.2f} s")
print({key: round(v, 2) if isinstance(v, float) else v for key, v in k.items()})
"""))

cells.append(md(r"""
## 3. Reinforcement learning agents — `agents.py`

All three agent types use the same tabular learner with $\varepsilon$-greedy exploration (Kochenderfer 2015, ch. 5; Sutton & Barto 2018, ch. 6):

**Q-learning (off-policy)**
$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]$$

**SARSA (on-policy)**
$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma\, Q(s',a') - Q(s,a)\big]$$

where $a'$ is the action actually taken next. Q-learning learns the value of the greedy policy while exploring; SARSA learns the value of the policy it is following, exploration included, so it tends to be more conservative.

Implementation details that matter:
* $Q$ starts at 0, which is **optimistic** because every reward is a cost, so untried actions get tried.
* Agents choose only among actions the safety shield will allow (`legal_actions_for`), and they learn from the action that was **executed**.
* Decisions are semi-Markov for AVs and buses (one transition per junction or stop), every 5 s for signals; signal episodes are time-truncated, so the last step bootstraps.
* At deployment an agent never takes an action it never tried in that state, and in a state never seen in training the signal agent falls back to the actuated rule.
* **Domain randomisation:** every training episode samples the uncertain conditions the decision network models — pedestrian demand (0.5×, 1×, 2× or 4×), weather (for rush hour and pre-training) and demand level (0.75× or 1×) — so the learned policies are robust rather than tuned to one value.
* Hyper-parameters: $\alpha = 0.1$; $\gamma = 0.90$ (signals), $0.98$ (AVs), $0.90$ (buses); $\varepsilon$ decays 0.30 → 0.05 in pre-training and 0.10 → 0.02 in adaptation.
"""))
cells.append(writefile("agents.py"))
cells.append(code(r"""
import agents as A
# the two non-learning baselines on the same episode (identical arrivals)
print("Webster plan (green s for NS, EW) per junction:", A.webster_plan("normal"))
for pol in ["Fixed-time", "Actuated"]:
    k = A.make_system(pol).run_episode("normal", seed=1).kpis()
    print(f"{pol:11s} delay {k['avg_delay_s']:.1f} s/veh | conflicts/1000 {k['conflicts_per_1000']:.2f} | CO2 {k['co2_kg']:.0f} kg")
"""))

cells.append(md(r"""
## 4. Training and scenario-based evaluation — `experiments.py`

1. **Pre-training** (parameter sharing): the nine junction agents update one shared table while cycling through all scenarios, so every junction's experience trains the same general policy.
2. **Adaptation** (decentralised): for each scenario the general table is copied into nine independent junction agents (plus the AV and bus fleets) that keep learning on that scenario only — *"RL agents adapt policies"*.
3. **Evaluation**: greedy policies on 8 held-out seeds per scenario, identical arrivals for every policy.
4. **Component studies**: AV routing (static vs selfish vs cooperative) and bus priority (none vs always vs learned).
5. **Decision-network data**: every combination of weather × density × pedestrian flow × signal policy × routing policy.
"""))
cells.append(writefile("experiments.py"))
cells.append(code(r"""
import experiments as X
CFG = X.QUICK if QUICK else X.FULL
print("configuration:", CFG)
t0 = time.time()
models = {}
for pol in X.ALGOS:
    print(f"Training {pol} ...")
    models[pol] = X.train_algorithm(pol, CFG)
curves = pd.concat([m["curve"] for m in models.values()], ignore_index=True)
print(f"training finished in {time.time() - t0:.0f} s")
"""))
cells.append(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for pol in X.ALGOS:
    sub = curves[(curves.policy == pol) & (curves.stage == "pre-train")]
    axes[0].plot(sub.episode, sub.avg_delay_s.rolling(15, min_periods=1).mean(), color=POLICY_COLORS[pol], lw=2, label=pol)
    axes[1].plot(sub.episode, sub.signal_return.rolling(15, min_periods=1).mean(), color=POLICY_COLORS[pol], lw=2, label=pol)
    ad = curves[(curves.policy == pol) & (curves.stage == "adapt") & (curves.scenario == "rush_hour")]
    axes[2].plot(ad.episode, ad.avg_delay_s.rolling(8, min_periods=1).mean(), color=POLICY_COLORS[pol], lw=2, label=pol)
axes[0].set(title="Pre-training: average delay (s/veh)", xlabel="episode")
axes[1].set(title="Pre-training: episode return (signal agents)", xlabel="episode")
axes[2].set(title="Adaptation to rush hour: delay (s/veh)", xlabel="episode")
for ax in axes: ax.legend(frameon=False)
plt.tight_layout(); plt.show()
"""))
cells.append(code(r"""
kpi = X.evaluate_policies(models, CFG)
summary = kpi.groupby(["scenario", "policy"])[["avg_delay_s", "throughput_vph", "co2_kg", "conflicts_per_1000",
                                                "safety_score", "ped_avg_wait_s", "jaywalks", "bus_on_time_pct"]].mean()
summary.round(2)
"""))
cells.append(code(r"""
# % change against the fixed-time plan (negative delay / CO2 / conflicts = better)
rows = []
for sc in C.SCENARIO_ORDER:
    base = kpi[(kpi.scenario == sc) & (kpi.policy == "Fixed-time")].mean(numeric_only=True)
    for pol in ["Actuated", "Q-learning", "SARSA"]:
        m = kpi[(kpi.scenario == sc) & (kpi.policy == pol)].mean(numeric_only=True)
        rows.append(dict(scenario=C.SCENARIOS[sc]["label"], policy=pol,
                         delay_pct=100 * (m.avg_delay_s - base.avg_delay_s) / base.avg_delay_s,
                         co2_pct=100 * (m.co2_kg - base.co2_kg) / base.co2_kg,
                         conflicts_pct=100 * (m.conflicts_per_1000 - base.conflicts_per_1000) / base.conflicts_per_1000,
                         ped_wait_pct=100 * (m.ped_avg_wait_s - base.ped_avg_wait_s) / base.ped_avg_wait_s))
improve = pd.DataFrame(rows)
improve.pivot(index="scenario", columns="policy", values="delay_pct").round(1)
"""))
cells.append(code(r"""
fig, ax = plt.subplots(figsize=(13, 4))
g = kpi.groupby(["scenario", "policy"]).avg_delay_s.agg(["mean", "std"]).reset_index()
w = 0.2
for k, pol in enumerate(A.POLICY_ORDER):
    sub = g[g.policy == pol].set_index("scenario").reindex(C.SCENARIO_ORDER)
    x = np.arange(len(C.SCENARIO_ORDER)) + (k - 1.5) * w
    ax.bar(x, sub["mean"], w * 0.9, yerr=sub["std"], color=POLICY_COLORS[pol], label=pol, capsize=2, error_kw=dict(lw=0.8, ecolor="#999"))
ax.set_xticks(np.arange(len(C.SCENARIO_ORDER)), [C.SCENARIOS[s]["label"] for s in C.SCENARIO_ORDER])
ax.set_ylabel("average delay (s/veh)"); ax.legend(frameon=False, ncol=4); ax.set_title("Average delay by scenario and policy (mean ± sd over 8 seeds)")
plt.tight_layout(); plt.show()
"""))
cells.append(code(r"""
# Q-learning vs SARSA, paired by seed (common random numbers)
from scipy import stats
rows = []
for sc in C.SCENARIO_ORDER:
    a = kpi[(kpi.scenario == sc) & (kpi.policy == "Q-learning")].sort_values("seed")
    b = kpi[(kpi.scenario == sc) & (kpi.policy == "SARSA")].sort_values("seed")
    rows.append(dict(scenario=sc, q_learning=a.avg_delay_s.mean(), sarsa=b.avg_delay_s.mean(),
                     sarsa_minus_q=(b.avg_delay_s.values - a.avg_delay_s.values).mean(),
                     p_value=stats.ttest_rel(b.avg_delay_s, a.avg_delay_s).pvalue if len(a) > 1 else np.nan))
pd.DataFrame(rows).round(3)
"""))
cells.append(code(r"""
abl = X.ablations(models, CFG, verbose=False)
abl.groupby(["study", "scenario", "variant"])[["avg_delay_s", "av_avg_delay_s", "bus_avg_delay_s", "bus_on_time_pct"]].mean().round(1)
"""))

cells.append(md(r"""
## 5. Criteria evaluation with a decision network — `decision_network.py`

The decision network (influence diagram) turns the simulation results into a **probabilistic, multi-criteria decision analysis**:

* **Chance nodes:** *Weather* {Clear, Rain, Sandstorm}, *Traffic density* {Low, Medium, High}, *Pedestrian flow* {Low, Medium, High} (operating conditions); *Congestion*, *Collision risk*, *Emissions* {Low, Medium, High}; *Health impact* {Low, High}.
* **Decision nodes:** *Signal timing* {Fixed-time, Actuated, Q-learning, SARSA}; *Routing* {Static, Cooperative RL}.
* **Utility nodes:** $U_{\text{efficiency}}(\text{Congestion})$, $U_{\text{safety}}(\text{Collision risk})$, $U_{\text{environment}}(\text{Emissions}, \text{Health})$, combined by multi-attribute utility $U = w_{e} U_{\text{eff}} + w_{s} U_{\text{safe}} + w_{v} U_{\text{env}}$ with default weights 0.40 / 0.35 / 0.25.
* **Dependencies:** Density, Weather, Signal, Routing → Congestion; Weather, Pedestrian flow, Congestion, Signal → Collision risk; Congestion, Signal → Emissions → Health impact ← Pedestrian flow (the *congestion → emissions → health* chain).

**Learning the CPTs from the simulator.** Every junction × 10-minute window of every run is one sample. Congestion is discretised by HCM level of service (LOS A / B / C-or-worse), collision risk and emissions at the tertiles of the pooled data, and each CPT is the Dirichlet(1) posterior mean $P(x\mid \text{pa}) = (n_{x,\text{pa}}+1)/(n_{\text{pa}}+|X|)$ — maximum likelihood with Laplace smoothing, as in the Probability Models lab. Priors on weather, density, pedestrian flow and the health CPT are expert assumptions. RL policies enter with the tables adapted to the matching conditions.

**Inference** is exact enumeration: $EU(d\mid e) = \sum_x P(x\mid e, d)\,U(x, d)$, the same factor-product-and-sum computation as the food-delivery example in the Utility Theory lecture.
"""))
cells.append(writefile("decision_network.py"))
cells.append(code(r"""
import decision_network as D
dn_data = X.dn_samples(models, CFG, verbose=False)
dn = D.build_traffic_network(dn_data)
print("samples:", len(dn_data), "| discretisation edges:", {k: [round(x, 2) for x in v] for k, v in dn.edges.items()})
D.cpt_frame(dn).query("node == 'Congestion'").head(12)
"""))
cells.append(code(r"""
# draw the influence diagram
import matplotlib.patches as mpatches
LAYOUT = {"Traffic density": (0.4, 2.6), "Weather": (1.9, 2.6), "Pedestrian flow": (3.4, 2.6), "Routing": (-0.5, 1.3),
          "Congestion": (0.9, 1.3), "Collision risk": (2.6, 1.3), "Signal timing": (-0.5, 0.0), "Emissions": (1.4, 0.0),
          "Health impact": (3.2, 0.0), "U_efficiency": (0.0, -1.4), "U_safety": (2.0, -1.4), "U_environment": (3.6, -1.4)}
fig, ax = plt.subplots(figsize=(11, 6.5)); ax.set_axis_off()
edges = [(p, n) for n, nd in dn.chance.items() for p in nd["parents"]] + [(p, u) for u, ud in dn.utilities.items() for p in ud["parents"]]
for a, b in edges:
    ax.annotate("", xy=LAYOUT[b], xytext=LAYOUT[a], arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.1, shrinkA=26, shrinkB=26))
for n, (x, y) in LAYOUT.items():
    if n in dn.chance:   patch = mpatches.Ellipse((x, y), 1.1, 0.6, fc="#cde2fb", ec="#256abf")
    elif n in dn.decisions: patch = mpatches.Rectangle((x - 0.55, y - 0.27), 1.1, 0.54, fc="#fde3c8", ec="#c2591f")
    else: patch = mpatches.RegularPolygon((x, y), 4, radius=0.5, fc="#d7f0e3", ec="#16865f")
    ax.add_patch(patch); ax.text(x, y, n.replace("U_", "U: "), ha="center", va="center", fontsize=9)
ax.set_xlim(-1.15, 4.3); ax.set_ylim(-1.9, 3.05); ax.set_title("Decision network: chance (ovals), decisions (rectangles), utilities (diamonds)")
plt.show()
"""))
cells.append(code(r"""
# expected utility of every decision pair, overall and per scenario
best, table_all = dn.best({}, D.DEFAULT_WEIGHTS)
print("Recommended with no evidence (priors only):", best["Signal timing"], "+", best["Routing"], f"EU = {best['EU']:.2f}")
table_all[["Signal timing", "Routing", "EU", "E[U_efficiency]", "E[U_safety]", "E[U_environment]",
           "P(Congestion=High)", "P(Collision risk=High)"]].round(3)
"""))
cells.append(code(r"""
sd = D.scenario_decisions(dn, D.DEFAULT_WEIGHTS)
sd[sd["rank"] <= 2][["scenario", "evidence", "rank", "Signal timing", "Routing", "EU", "E[U_efficiency]", "E[U_safety]", "E[U_environment]"]].round(2)
"""))
cells.append(code(r"""
# value of (perfect) information: is it worth observing a condition before choosing?
pd.DataFrame([{k: v for k, v in dn.value_of_information(v_, {}, D.DEFAULT_WEIGHTS).items() if k != "detail"}
              for v_ in ["Weather", "Traffic density", "Pedestrian flow"]]).round(3)
"""))
cells.append(code(r"""
# sensitivity of the best signal policy to the stakeholder weights (efficiency, safety, environment)
for ev in [{}, {"Traffic density": "High"}, {"Weather": "Sandstorm"}]:
    ws = D.weight_sensitivity(dn, ev, step=0.1)
    print(ev or "no evidence", "->", ws.best_signal.value_counts().to_dict())
"""))
cells.append(code(r"""
# cross-check our enumeration against pgmpy (the library used in the Probability Models lab)
try:
    import pgmpy  # noqa: F401
except ImportError:
    !pip -q install pgmpy
decision = {"Signal timing": "Q-learning", "Routing": "Cooperative RL"}
evidence = {"Weather": "Sandstorm", "Traffic density": "High"}
ours = dn.expected_utility(decision, evidence)["components"]
theirs = D.pgmpy_crosscheck(dn, decision, evidence)
pd.DataFrame({"this notebook": ours, "pgmpy VariableElimination": theirs}).round(6)
"""))

cells.append(md(r"""
## Findings (full run, 8 held-out seeds per scenario)

1. **Against the fixed-time plan**, the multi-agent RL policies cut average delay in every scenario: Q-learning by 4–49 % (rush hour −37 %, event surge −49 %) and SARSA by 4–47 %, with 1–4 % less CO₂ and up to 8 % more throughput.
2. **Against vehicle-actuated control**, RL wins where the network is stressed (rush hour −21 % for Q-learning and −10 % for SARSA; event surge −29 % and −27 %) and is within ±5 % elsewhere.
3. **The gain has a safety cost under congestion.** In rush hour and the event surge the RL policies hold longer greens, so pedestrians wait 22–28 % longer and expected conflicts rise 11–28 % against fixed-time, and jaywalking rises 45–105 % against actuated control. In light traffic they *reduce* conflicts by 9–27 % against fixed-time.
4. **Signal × routing decomposition.** Cooperative AV routing is the largest single lever in rush hour (actuated signals 87.5 → 65.0 s/veh; Q-learning signals 88.6 → 69.1 s). With that routing in place, the actuated rule is the best signal controller in rush hour, while the RL signal agents are clearly best in the event surge (46.6 vs 62.9 s).
5. **Decision network.** With weights 0.40 / 0.35 / 0.25, *actuated signals + cooperative RL routing* has the highest expected utility in 22 of the 27 weather × density × pedestrian conditions (Q-learning signals win in light, clear-weather traffic; static routing in medium-density rain), because the RL signals' pedestrian and conflict risk lowers the safety utility. With safety weighted at 0.6, fixed-time becomes the best signal choice in clear weather with high density and high pedestrian flow. The event surge's localised crowd is outside the network's generic conditions, so there the direct simulation, which favours RL signals, is the better guide.
6. **Q-learning vs SARSA.** In the main run Q-learning beat SARSA in rush hour (69.1 vs 79.0 s, paired t-test over evaluation seeds p = 0.04) and narrowly in the event surge (46.6 vs 47.7 s, p = 0.02); elsewhere the difference is not significant. That test covers evaluation noise only: re-adapting with three other seeds gave SARSA 68–70 s and Q-learning 71–75 s in rush hour, and 48–55 s against 48–79 s in the event surge. The gap between the algorithms is therefore mostly training variance, and SARSA's more conservative on-policy updates made it the more stable learner.
"""))
cells.append(md(r"""
## 6. Export the data for the dashboard

The main data file, **`smart_traffic_simulation_log.csv`**, has one row per junction per 30-second window of a representative evaluation episode for every scenario × policy (queues per approach, arrivals, departures, delay, pedestrians waiting, jaywalks, expected conflicts, CO₂, NOx, stops, spillback, emergency-vehicle and bus-priority activity, shield overrides, reward). The other files hold KPIs per episode, learning curves, component studies, agent decisions, decision-network samples, scenario parameters and the trained Q-tables.
"""))
cells.append(code(r"""
sim_log, anims, decisions = X.record_runs(models)
res = dict(models=models, curves=curves, kpi=kpi, ablations=abl, sim_log=sim_log, anims=anims, decisions=decisions, dn=dn_data)
X.export_all(res, "data")
for f in sorted(os.listdir("data")):
    print(f"{f:40s} {os.path.getsize(os.path.join('data', f)) / 1e6:6.2f} MB")
sim_log.head()
"""))
cells.append(code(r"""
# download everything as one zip (Colab only)
import shutil
shutil.make_archive("smart_traffic_data", "zip", "data")
try:
    from google.colab import files
    files.download("smart_traffic_data.zip")
except Exception:
    print("Not running in Colab - the zip is at ./smart_traffic_data.zip")
"""))
cells.append(md(r"""
## 7. Dashboard

The six-tab Streamlit prototype (City Overview · RL Simulation · Decision Network Visualizer · Scenario Explorer · KPI & Trade-off Analysis · Policy Comparison) reads the files exported above. Its source (`app.py`, `viz.py`, `sim_view.py`) is in the project repository; run it locally with `streamlit run app.py`, or deploy the folder to Streamlit Community Cloud.

## References
* Kochenderfer, M. J. (2015). *Decision Making Under Uncertainty: Theory and Application.* MIT Press — ch. 3 (decision networks, value of information) and ch. 5 (Q-learning, SARSA).
* Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press — ch. 6.
* Wairagade, A., Ugale, K. & Waghmare, R. (2025). AI-Powered Smart Traffic Management System for Urban Congestion Reduction. *IJSRET*, 11(2), 1997–2001.
* Boyan, J. A. & Littman, M. L. (1994). Packet routing in dynamically changing networks: a reinforcement learning approach. *NIPS 6*.
* Webster, F. V. (1958). *Traffic Signal Settings.* Road Research Technical Paper 39, HMSO.
* Qiu, L. & Nixon, W. A. (2008). Effects of adverse weather on traffic crashes: systematic review and meta-analysis. *Transportation Research Record*, 2055, 139–146.
* Sullivan, A., Sisiopiku, V. & Reddy-Surhabi, S. (2010). *Impacts of Signal System Timings on Rain-Related Congestion.* UTCA Report 07211 (reports Perrin et al. 2002 saturation-flow and speed reductions).
* FHWA (2013). *Guide for Highway Capacity and Operations Analysis of ATDM Strategies* (FHWA-HOP-13-042), Appendix A — HCM weather capacity adjustment factors.
* US EPA (2023). *Greenhouse Gas Emissions from a Typical Passenger Vehicle* (400 g CO₂/mile); Argonne National Laboratory, *Idling* fact sheet (0.3 gal/h car, 1 gal/h heavy vehicle).
* Transportation Research Board (2016). *Highway Capacity Manual, 6th ed.* — saturation flow, signalised-intersection LOS, pedestrian delay and non-compliance.
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python"},
                  "colab": {"provenance": [], "toc_visible": True}}
out = os.path.join(HERE, "Smart_Traffic_MARL_DecisionNetwork.ipynb")
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
