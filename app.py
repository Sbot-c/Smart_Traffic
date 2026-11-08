"""
Smart Traffic Management Dashboard - Streamlit prototype.

Multi-agent reinforcement learning (Q-learning, SARSA) for traffic lights, autonomous
vehicles and buses on a 3 x 3 Downtown-Dubai-style grid, evaluated across six scenario
families and scored with a decision network (influence diagram).

Run locally:   streamlit run app.py
All results are precomputed by run_experiments.py into ./data, so the app starts at once.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

import agents as A
import config as C
import decision_network as D
import experiments as X
import sim_view
import viz

st.set_page_config(page_title="Smart Traffic Management Dashboard", page_icon="🚦",
                   layout="wide", initial_sidebar_state="expanded")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

CSS = """
<style>
  .block-container {padding-top: 2.0rem; padding-bottom: 2rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  [data-testid="stMetricValue"] {font-size: 1.45rem;}
  [data-testid="stMetricLabel"] {color: #52514e;}
  .muted {color:#52514e; font-size:0.92rem;}
  .pill {display:inline-block; padding:2px 10px; border-radius:999px; background:#f2f1ec;
         color:#33312c; font-size:0.82rem; margin-right:6px;}
  .card {border:1px solid #e8e6df; border-radius:10px; padding:12px 14px; background:#ffffff;}
  .card h4 {margin:0 0 4px 0; font-size:1rem;}
  .card p {margin:0; color:#52514e; font-size:0.88rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ======================================================================================
# Data
# ======================================================================================
@st.cache_data(show_spinner=False)
def load_tables():
    r = lambda n: pd.read_csv(os.path.join(DATA, n))
    out = dict(kpi=r("kpi_results.csv"), curves=r("learning_curves.csv"),
               abl=r("ablation_results.csv"), sim=r("smart_traffic_simulation_log.csv"),
               dec=r("agent_decisions.csv"), dn=r("decision_network_samples.csv"),
               scen=r("scenario_parameters.csv"))
    rob = os.path.join(DATA, "training_robustness.csv")
    out["robust"] = pd.read_csv(rob) if os.path.exists(rob) else None
    sim = out["sim"]
    sim["queue_total"] = sim[["queue_n", "queue_e", "queue_s", "queue_w"]].sum(axis=1)
    return out


@st.cache_data(show_spinner=False)
def load_anims():
    with gzip.open(os.path.join(DATA, "animations.json.gz"), "rt") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_models():
    return X.load_models(os.path.join(DATA, "q_tables.npz"))


@st.cache_resource(show_spinner=False)
def build_dn(_samples: pd.DataFrame, n_rows: int):
    return D.build_traffic_network(_samples)


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove":
                            ["lasso2d", "select2d", "autoScale2d"]})


def table(df, height=None, key=None):
    kw = dict(width="stretch", hide_index=True)
    if height:
        kw["height"] = height
    st.dataframe(df, key=key, **kw)


def embed(html: str, height: int):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                            # Streamlit < 1.5x
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


with st.spinner("Loading simulation results ..."):
    T = load_tables()
    ANIMS = load_anims()
    MODELS = load_models()
    DN = build_dn(T["dn"], len(T["dn"]))

KPI = T["kpi"]
LABEL = {k: v["label"] for k, v in C.SCENARIOS.items()}
MEASURE = {"avg_delay_s": "Average delay (s/veh)", "throughput_vph": "Throughput (veh/h)",
           "co2_kg": "CO₂ emitted (kg)", "conflicts_per_1000": "Expected conflicts per 1,000 veh",
           "safety_score": "Safety score (0-100)", "ped_avg_wait_s": "Pedestrian wait (s)",
           "jaywalks": "Jaywalking pedestrians", "bus_on_time_pct": "Bus on time (%)",
           "bus_avg_delay_s": "Bus delay (s)", "switches": "Phase changes per episode",
           "vehicle_delay_s": "Vehicle delay (veh-s)", "queue_total": "Vehicles queued",
           "ped_waiting": "Pedestrians waiting", "expected_conflicts": "Expected conflicts",
           "co2_g": "CO₂ (g)", "departures": "Vehicles discharged", "jaywalks_w": "Jaywalks"}
POL = A.POLICY_ORDER

# ======================================================================================
# Sidebar
# ======================================================================================
with st.sidebar:
    st.markdown("### 🚦 Smart Traffic Management")
    st.markdown('<span class="muted">Multi-agent Q-learning & SARSA with decision-network '
                'evaluation · 3 × 3 Downtown-Dubai-style grid</span>', unsafe_allow_html=True)
    scenario = st.selectbox("Scenario", C.SCENARIO_ORDER, format_func=lambda s: LABEL[s],
                            index=1)
    policy = st.selectbox("Control policy", POL, index=2)
    st.markdown("**Stakeholder weights** (multi-attribute utility)")
    # defaults = decision_network.DEFAULT_WEIGHTS (0.40 / 0.35 / 0.25)
    w_e = st.slider("Efficiency", 0, 100, 40, step=5)
    w_s = st.slider("Safety", 0, 100, 35, step=5)
    w_v = st.slider("Environment", 0, 100, 25, step=5)
    tot = max(1, w_e + w_s + w_v)
    WEIGHTS = {"U_efficiency": w_e / tot, "U_safety": w_s / tot, "U_environment": w_v / tot}
    st.caption(f"Normalised: efficiency {WEIGHTS['U_efficiency']:.2f} · safety "
               f"{WEIGHTS['U_safety']:.2f} · environment {WEIGHTS['U_environment']:.2f}")
    with st.expander("Download the data"):
        for name in ["smart_traffic_simulation_log.csv", "kpi_results.csv",
                     "decision_network_samples.csv", "learning_curves.csv",
                     "ablation_results.csv", "agent_decisions.csv", "training_robustness.csv",
                     "scenario_parameters.csv"]:
            if not os.path.exists(os.path.join(DATA, name)):
                continue
            with open(os.path.join(DATA, name), "rb") as f:
                st.download_button(name, f.read(), file_name=name, mime="text/csv",
                                   key=f"dl-{name}")
    st.caption("Each episode = 30 simulated minutes; KPIs exclude the first 5 minutes "
               "(warm-up). Evaluation uses 8 held-out seeds with identical arrivals for "
               "every policy.")

SC = C.SCENARIOS[scenario]
k_sel = KPI[(KPI.scenario == scenario) & (KPI.policy == policy)]
k_base = KPI[(KPI.scenario == scenario) & (KPI.policy == "Fixed-time")]

st.markdown(f"## Smart Traffic Management Dashboard")
st.markdown(f'<span class="pill">{SC["label"]}</span><span class="pill">{policy}</span>'
            f'<span class="pill">Weather: {SC["weather"]}</span>'
            f'<span class="muted">{SC["description"]}</span>', unsafe_allow_html=True)

tabs = st.tabs(["City Overview", "RL Simulation", "Decision Network Visualizer",
                "Scenario Explorer", "KPI & Trade-off Analysis", "Policy Comparison"])


def anim_payload(sc, pol):
    p = dict(ANIMS[f"{sc}|{pol}"])
    p["bus_lane"] = bool(C.SCENARIOS[sc].get("bus_lane"))
    p["event"] = bool(C.SCENARIOS[sc].get("event"))
    return p


def mean(df, col):
    return float(df[col].mean()) if len(df) else float("nan")


def pct(v, b):
    return 100.0 * (v - b) / b if b else 0.0


# ======================================================================================
# 1. City overview
# ======================================================================================
with tabs[0]:
    cols = st.columns(6)
    tiles = [("Average delay", "avg_delay_s", "s/veh", True, "{:.1f}"),
             ("Throughput", "throughput_vph", "veh/h", False, "{:,.0f}"),
             ("CO₂ emitted", "co2_kg", "kg", True, "{:,.0f}"),
             ("Safety score", "safety_score", "/100", False, "{:.1f}"),
             ("Bus on time", "bus_on_time_pct", "%", False, "{:.0f}"),
             ("Pedestrian wait", "ped_avg_wait_s", "s", True, "{:.1f}")]
    for c, (lab, col, unit, lower_better, fmt) in zip(cols, tiles):
        v, b = mean(k_sel, col), mean(k_base, col)
        delta = None if policy == "Fixed-time" else f"{pct(v, b):+.1f}% vs fixed-time"
        c.metric(f"{lab} ({unit})", fmt.format(v), delta,
                 delta_color="inverse" if lower_better else "normal")
    st.caption("Means over 8 evaluation episodes; deltas compare against the Webster "
               "fixed-time plan with the same arrivals.")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Real-time traffic flow")
        st.caption("Replay of evaluation episode 1 (first 15 minutes). Vehicles are "
                   "coloured by type, signal bars show the current phase, purple dots are "
                   "pedestrians waiting.")
        embed(sim_view.render_html([anim_payload(scenario, policy)], [policy], start=120),
              height=820)
    with right:
        sim = T["sim"]
        sim_sc = sim[(sim.scenario == scenario) & (sim.policy == policy)]
        minute = st.slider("Network status at minute", 0, 29, 12, key="map-min")
        snap = sim_sc[sim_sc.t_start_s == minute * 60]
        q = [[0, 0, 0, 0] for _ in range(C.N_INT)]
        ph = [0] * C.N_INT
        for _, r in snap.iterrows():
            i = C.JUNCTION_NAMES.index(r.junction)
            q[i] = [r.queue_n, r.queue_e, r.queue_s, r.queue_w]
            ph[i] = 0 if r.ns_green_share >= 0.5 else 1
        show(viz.network_map(q, ph, incident=SC.get("incident") if SC.get("incident") and
                             SC["incident"]["t_start"] <= minute * 60 < SC["incident"]["t_end"]
                             else None, height=430), key="map")
        st.caption("Bubble size and colour = mean queue on each approach in that 30-second "
                   "window; badge = phase that held green most of the window.")
        jt = sim_sc.groupby("junction").agg(
            delay_veh_h=("vehicle_delay_s", lambda x: x.sum() / 3600),
            expected_conflicts=("expected_conflicts", "sum"),
            co2_kg=("co2_g", lambda x: x.sum() / 1000), jaywalks=("jaywalks", "sum"),
            phase_changes=("phase_changes", "sum")).reset_index()
        table(jt.round(2), height=360)

    st.markdown("#### Network queue over the episode")
    show(viz.timeseries(T["sim"], scenario, "queue_total",
                        "Vehicles queued (all junctions)", height=300, agg="sum"),
         key="ts-queue")
    st.caption("Mean queue in each 30-second window of evaluation episode 1, summed over "
               "the nine junctions.")

# ======================================================================================
# 2. RL simulation
# ======================================================================================
with tabs[1]:
    st.markdown("#### Side-by-side replay")
    c1, c2 = st.columns(2)
    left_pol = c1.selectbox("Left", POL, index=0, key="cmp-left")
    right_pol = c2.selectbox("Right", POL, index=POL.index(policy) if policy != "Fixed-time"
                             else 2, key="cmp-right")
    st.caption("Same arrivals, same seed, one clock: the only difference is the control "
               "policy.")
    embed(sim_view.render_html([anim_payload(scenario, left_pol),
                                anim_payload(scenario, right_pol)],
                               [left_pol, right_pol], start=120), height=740)

    st.markdown("#### Agent decisions")
    d1, d2, d3 = st.tabs(["Traffic-light agents", "Bus agents", "Autonomous-vehicle routing"])
    with d1:
        jn = st.selectbox("Junction", C.JUNCTION_NAMES, index=4, key="trace-j")
        i = C.JUNCTION_NAMES.index(jn)
        P = ANIMS[f"{scenario}|{policy}"]
        rows = []
        for s in range(len(P["phase"])):
            ov = P["override"][s][i]
            ch = P["chosen"][s][i]
            sw = P["switched"][s][i]
            if sw or ov or ch:
                rows.append(dict(time=f"{int(s * 5 // 60):02d}:{int(s * 5 % 60):02d}",
                                 phase_after=C.PHASE_NAMES[P["phase"][s][i]],
                                 agent_chose="change" if ch else "keep",
                                 executed="change" if sw else "keep",
                                 shield=X.TE.OVERRIDE_NAMES[ov] if ov else "",
                                 queue_N=P["queues"][s][i][0], queue_E=P["queues"][s][i][1],
                                 queue_S=P["queues"][s][i][2], queue_W=P["queues"][s][i][3]))
        st.caption(f"{policy} at {jn}: every 5-second step where the agent asked to change "
                   "phase, the phase changed, or the safety shield intervened (first 15 min).")
        table(pd.DataFrame(rows), height=320)
    with d2:
        dec = T["dec"]
        bd = dec[(dec.scenario == scenario) & (dec.policy == policy) & (dec.agent == "bus")]
        if len(bd):
            counts = bd.action.value_counts().rename_axis("action").reset_index(name="decisions")
            cc1, cc2 = st.columns([1, 2])
            with cc1:
                table(counts)
            with cc2:
                table(bd[["t_s", "agent_id", "junction", "action", "detail", "state"]]
                      .rename(columns={"t_s": "time_s"}), height=300)
        else:
            st.info("No bus decisions recorded for this run.")
    with d3:
        dec = T["dec"]
        ad = dec[(dec.scenario == scenario) & (dec.policy == policy) & (dec.agent == "AV")]
        if len(ad):
            st_ = ad.state.apply(json.loads)
            ad = ad.assign(x_link=st_.apply(lambda d: d["x_link"]),
                           y_link=st_.apply(lambda d: d["y_link"]))
            piv = ad.assign(chose_x=(ad.detail == "x-move").astype(int)).groupby(
                ["x_link", "y_link"]).agg(decisions=("chose_x", "size"),
                                          share_choosing_x=("chose_x", "mean")).reset_index()
            st.caption("Each AV with two shortest-path options picks the next link from the "
                       "congestion on both candidates. Share choosing the x-direction link:")
            table(piv.round(2))
        else:
            st.info("Static routing: autonomous vehicles follow fixed shortest paths under "
                    "this policy.")

    if policy in X.ALGOS:
        st.markdown("#### Learned signal policy")
        m = MODELS[policy]["adapted"][scenario]["signals"]
        c = st.columns(5)
        jn2 = c[0].selectbox("Junction", C.JUNCTION_NAMES, index=4, key="q-j")
        phs = c[1].selectbox("Phase", ["NS green", "EW green"], key="q-ph")
        gb = c[2].selectbox("Green so far", A.GREEN_BINS[1:], index=1, key="q-g")
        ab = c[3].selectbox("Arrivals on green (next 6 s)", ["none", "1-2", "3+"], key="q-a")
        fl = c[4].selectbox("Flag", ["none", "pedestrians waiting", "spillback"], key="q-f")
        j = C.JUNCTION_NAMES.index(jn2)
        Q, N = m[j]["Q"], m[j]["visits"]
        ph_i = 0 if phs.startswith("NS") else 1
        g_i = A.GREEN_BINS.index(gb)
        a_i = ["none", "1-2", "3+"].index(ab)
        f_i = ["none", "pedestrians waiting", "spillback"].index(fl)
        grid = np.full((A.N_Q, A.N_Q), np.nan)
        for qg in range(A.N_Q):
            for qr in range(A.N_Q):
                s = A.encode_signal_state(ph_i, g_i, qg, qr, a_i, f_i)
                tried = [a for a in (0, 1) if N[s, a] > 0]
                if tried:
                    grid[qg, qr] = float(max(tried, key=lambda a: Q[s, a]))
        show(viz.policy_heatmap(grid, A.QUEUE_BINS, A.QUEUE_BINS, height=360,
                                title=f"{policy} greedy action at {jn2} ({LABEL[scenario]})"),
             key="qheat")
        st.caption("Cells show the greedy action learned for each queue combination; '–' = "
                   "state never visited in training (the agent falls back to the actuated "
                   "rule there).")
        cb1, cb2 = st.columns(2)
        with cb1:
            st.markdown("**Bus agent policy (most visited states)**")
            bq = MODELS[policy]["adapted"][scenario]["bus"]
            vis = bq["visits"].sum(1)
            rows = []
            for s in np.argsort(-vis)[:12]:
                if vis[s] == 0:
                    continue
                dd = A.decode_bus_state(int(s))
                tried = [a for a in range(3) if bq["visits"][s, a] > 0]
                best = max(tried, key=lambda a: bq["Q"][s, a])
                rows.append(dict(**dd, visits=int(vis[s]),
                                 best=["go", "request priority", "hold 20 s"][best]))
            table(pd.DataFrame(rows), height=330)
        with cb2:
            st.markdown("**AV routing policy (shared fleet table)**")
            aq = MODELS[policy]["adapted"][scenario]["routing"]
            lv = ["free", "moderate", "heavy", "incident"]
            rows = []
            for cx in range(4):
                for cy in range(4):
                    qx = qy = 0.0
                    n = 0
                    for mx in (1, 2):
                        for my in (1, 2):
                            s = A.av_state(mx, my, cx, cy)
                            if aq["visits"][s].sum() > 0:
                                qx += aq["Q"][s, 0]
                                qy += aq["Q"][s, 1]
                                n += int(aq["visits"][s].sum())
                    if n:
                        rows.append(dict(x_link=lv[cx], y_link=lv[cy], visits=n,
                                         prefers="x-move" if qx >= qy else "y-move",
                                         q_gap_s=round(abs(qx - qy) * 10 / 4, 1)))
            table(pd.DataFrame(rows), height=330)

    st.markdown("#### Learning curves")
    st.caption("Every training episode samples its own pedestrian demand, weather and demand "
               "level (domain randomisation), so the per-episode delay moves with the sampled "
               "conditions as well as with learning. Held-out evaluation is in Policy "
               "Comparison.")
    cl1, cl2 = st.columns(2)
    with cl1:
        show(viz.learning_curves(T["curves"], "avg_delay_s", "pre-train", height=320,
                                 title="Pre-training on all scenarios (shared table)"),
             key="lc1")
    with cl2:
        show(viz.learning_curves(T["curves"], "avg_delay_s", "adapt", scenario=scenario,
                                 window=8, height=320,
                                 title=f"Adaptation to {LABEL[scenario]}"), key="lc2")

# ======================================================================================
# 3. Decision network
# ======================================================================================
with tabs[2]:
    st.markdown("#### Influence diagram: which control strategy maximises expected utility?")
    c = st.columns([1, 1, 1, 1.3])
    use_sc = c[3].toggle("Use the selected scenario's conditions", value=True)
    default = D.SCENARIO_EVIDENCE.get(scenario, {}) if use_sc else {}
    opts = lambda st_: ["unknown"] + st_
    ev = {}
    for col, node, states in [(c[0], "Weather", D.WEATHER),
                              (c[1], "Traffic density", D.DENSITY),
                              (c[2], "Pedestrian flow", D.PED)]:
        val = col.selectbox(node, opts(states),
                            index=opts(states).index(default.get(node, "unknown")),
                            key=f"ev-{node}-{use_sc}-{scenario}")
        if val != "unknown":
            ev[node] = val
    best, tab_eu = DN.best(ev, WEIGHTS)
    c2 = st.columns([1, 1, 2])
    dsig = c2[0].selectbox("Inspect signal timing", ["best"] + D.SIGNAL, key="dn-sig")
    drou = c2[1].selectbox("Inspect routing", ["best"] + D.ROUTING, key="dn-rou")
    decision = {"Signal timing": best["Signal timing"] if dsig == "best" else dsig,
                "Routing": best["Routing"] if drou == "best" else drou}
    res = DN.expected_utility(decision, ev, WEIGHTS)
    c2[2].markdown(f"**Recommended:** {best['Signal timing']} signals + {best['Routing']} "
                   f"routing &nbsp;·&nbsp; expected utility **{best['EU']:.1f}**/100  \n"
                   f'<span class="muted">Evidence: '
                   f'{", ".join(f"{k} = {v}" for k, v in ev.items()) or "none (priors)"}</span>',
                   unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        show(viz.influence_diagram(DN, res["posteriors"], decision, res["components"],
                                   height=560), key="dn-diag")
        st.caption("Ovals = chance nodes (blue), rectangles = decisions (orange), diamonds = "
                   "utilities (green). Each chance node shows its most likely state given the "
                   "evidence and the inspected decision; hover for the full distribution.")
    with right:
        show(viz.eu_bars(tab_eu, height=330, title="Expected utility of every decision pair"),
             key="dn-eu")
        show(viz.posterior_bars(DN, res["posteriors"], ["Congestion", "Collision risk",
                                                        "Emissions"], height=230,
                                title="Outcome probabilities for the inspected decision"),
             key="dn-post")
    st.markdown("#### Expected-utility table")
    show_cols = ["Signal timing", "Routing", "EU", "E[U_efficiency]", "E[U_safety]",
                 "E[U_environment]", "P(Congestion=High)", "P(Collision risk=High)",
                 "P(Emissions=High)", "P(Health impact=High)"]
    table(tab_eu[show_cols].round(3))
    cv1, cv2 = st.columns([1, 1.2])
    with cv1:
        st.markdown("#### Value of information")
        rows = []
        voi_ev = ev if len(ev) < 3 else {}
        if len(ev) == 3:
            st.caption("All three conditions are set above, so this table values observing "
                       "each one starting from the priors (no evidence).")
        for v in ["Weather", "Traffic density", "Pedestrian flow"]:
            if v in voi_ev:
                continue
            r = DN.value_of_information(v, voi_ev, WEIGHTS)
            rows.append(dict(observe=v, EVPI=round(r["evpi"], 3),
                             best_if_known="; ".join(f"{d['state']}: {d['best_signal']} + "
                                                     f"{d['best_routing']}"
                                                     for d in r["detail"])))
        table(pd.DataFrame(rows))
        st.caption("EVPI = expected gain in utility from observing the variable before "
                   "choosing. Zero means the same decision is best whatever its value.")
        st.markdown("#### Discretisation")
        e = DN.edges
        st.markdown(f"- **Congestion** (control delay per junction): Low ≤ {e['congestion'][0]:.0f} s "
                    f"(LOS A), Medium ≤ {e['congestion'][1]:.0f} s (LOS B), High > "
                    f"{e['congestion'][1]:.0f} s (LOS C or worse)\n"
                    f"- **Collision risk** (expected conflicts per 1,000 vehicles): tertiles at "
                    f"{e['risk'][0]:.2f} and {e['risk'][1]:.2f}\n"
                    f"- **Emissions** (g CO₂ from idling and stops per vehicle): tertiles at "
                    f"{e['emissions'][0]:.1f} and {e['emissions'][1]:.1f}\n"
                    f"- Learned from **{len(DN.data):,}** junction × 10-minute samples "
                    f"(216 condition × decision combinations × 2 seeds).")
    with cv2:
        st.markdown("#### Sensitivity to stakeholder weights")
        ws = D.weight_sensitivity(DN, ev, step=0.05)
        show(viz.ternary_map(ws, height=440), key="dn-tern")
        st.caption("Each dot is one combination of efficiency, safety and environment "
                   "weights (steps of 0.05); its colour is the signal policy with the highest "
                   "expected utility under the evidence above.")
    st.markdown("#### Conditional probability tables")
    node = st.selectbox("Node", DN.order, index=DN.order.index("Congestion"), key="cpt-node")
    cpt = D.cpt_frame(DN)
    table(cpt[cpt.node == node].drop(columns=["node"]).dropna(axis=1, how="all"), height=320)

# ======================================================================================
# 4. Scenario explorer
# ======================================================================================
with tabs[3]:
    st.markdown("#### Scenarios")
    scen = T["scen"]
    cols = st.columns(4)
    for k, key in enumerate(C.SCENARIO_ORDER):
        r = scen[scen.scenario == key].iloc[0]
        cols[k % 4].markdown(
            f'<div class="card"><h4>{r.label}</h4><p>{r.description}</p>'
            f'<p style="margin-top:6px"><span class="pill">{r.weather}</span>'
            f'<span class="pill">demand ×{r.demand_multiplier:g}</span>'
            f'<span class="pill">bus every {int(r.bus_headway_s // 60)} min</span></p></div>',
            unsafe_allow_html=True)
        if k % 4 == 3:
            cols = st.columns(4)
    st.markdown(f"#### {LABEL[scenario]}: outcome distributions across 8 evaluation episodes")
    ks = KPI[KPI.scenario == scenario]
    b1, b2, b3 = st.columns(3)
    with b1:
        show(viz.box_by_policy(ks, "avg_delay_s", "s/veh", height=300,
                               title="Average delay"), key="bx1")
    with b2:
        show(viz.box_by_policy(ks, "conflicts_per_1000", "per 1,000 veh", height=300,
                               title="Expected conflicts"), key="bx2")
    with b3:
        show(viz.box_by_policy(ks, "co2_kg", "kg", height=300, title="CO₂ emitted"),
             key="bx3")
    c1, c2 = st.columns([1, 1])
    jsel = c1.selectbox("Junction", ["All junctions"] + C.JUNCTION_NAMES, key="ts-j")
    msel = c2.selectbox("Measure", ["vehicle_delay_s", "queue_total", "ped_waiting",
                                    "expected_conflicts", "co2_g", "departures", "jaywalks"],
                        format_func=lambda m: MEASURE.get(m, m), key="ts-m")
    show(viz.timeseries(T["sim"], scenario, msel, MEASURE.get(msel, msel), junction=jsel,
                        height=320, title=f"{MEASURE.get(msel, msel)} per 30-second window"),
         key="ts2")
    st.markdown("#### What the decision network recommends here")
    evs = D.SCENARIO_EVIDENCE[scenario]
    _, t = DN.best(evs, WEIGHTS)
    sim_rank = ks.groupby("policy")[["avg_delay_s", "conflicts_per_1000", "co2_kg",
                                     "ped_avg_wait_s"]].mean().round(2).sort_values(
        "avg_delay_s")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("Decision network (evidence: " + ", ".join(f"{k} = {v}" for k, v in
                                                              evs.items()) + ")")
        table(t[["Signal timing", "Routing", "EU", "E[U_efficiency]", "E[U_safety]",
                 "E[U_environment]"]].head(4).round(2))
    with cc2:
        sx = T["abl"][(T["abl"].study == "Signal x routing") & (T["abl"].scenario == scenario)]
        if len(sx):
            st.caption("Direct simulation of the same signal × routing pairs (8 episodes)")
            table(sx.groupby("variant")[["avg_delay_s", "conflicts_per_1000", "co2_kg"]]
                  .mean().round(2).sort_values("avg_delay_s").reset_index().head(4))
        else:
            st.caption("Direct simulation of the four integrated policies (8 episodes)")
            table(sim_rank.reset_index())

# ======================================================================================
# 5. KPI & trade-off analysis
# ======================================================================================
with tabs[4]:
    st.markdown("#### Improvement over the fixed-time plan")
    h1, h2 = st.columns(2)
    with h1:
        show(viz.improvement_heatmap(KPI, "avg_delay_s", title="Average delay", height=330),
             key="hm1")
    with h2:
        show(viz.improvement_heatmap(KPI, "conflicts_per_1000", title="Expected conflicts",
                                     height=330), key="hm3")
    h3, h4 = st.columns(2)
    with h3:
        show(viz.improvement_heatmap(KPI, "co2_kg", title="CO₂ emissions", height=330),
             key="hm2")
    with h4:
        show(viz.improvement_heatmap(KPI, "ped_avg_wait_s", title="Pedestrian waiting time",
                                     height=330), key="hm4")
    h5, h6 = st.columns(2)
    with h5:
        show(viz.improvement_heatmap(KPI, "bus_on_time_pct", higher_is_better=True,
                                     title="Bus on-time performance", height=330), key="hm5")
    st.caption("Blue = better than fixed-time, red = worse; labels give the raw % change.")
    st.markdown("#### Trade-offs")
    t1, t2 = st.columns([1.2, 1])
    with t1:
        fig, tt = viz.tradeoff_scatter(KPI, height=430,
                                       title="Delay vs safety, all scenarios (bubble = CO₂ saved)")
        show(fig, key="tradeoff")
    with t2:
        st.markdown("**Weighted-sum score** (criteria min-max normalised within each "
                    "scenario; weights from the sidebar)")
        g = KPI.groupby(["scenario", "policy"])[["avg_delay_s", "safety_score", "co2_kg"]] \
            .mean().reset_index()
        out = []
        for s_, sub in g.groupby("scenario"):
            sub = sub.copy()
            for col, better_low in [("avg_delay_s", True), ("safety_score", False),
                                    ("co2_kg", True)]:
                lo, hi = sub[col].min(), sub[col].max()
                rng = (hi - lo) or 1.0
                sub[f"u_{col}"] = 100 * ((hi - sub[col]) / rng if better_low
                                         else (sub[col] - lo) / rng)
            sub["score"] = (WEIGHTS["U_efficiency"] * sub.u_avg_delay_s + WEIGHTS["U_safety"]
                            * sub.u_safety_score + WEIGHTS["U_environment"] * sub.u_co2_kg)
            out.append(sub)
        ws_ = pd.concat(out)
        piv = ws_.pivot(index="scenario", columns="policy", values="score").reindex(
            C.SCENARIO_ORDER)[POL]
        piv.index = [LABEL[s] for s in piv.index]
        piv["best"] = piv[POL].idxmax(axis=1)
        st.dataframe(piv.round(1), width="stretch")
        sel = ws_[ws_.scenario == scenario].set_index("policy")
        rad = pd.DataFrame({"Efficiency": sel.u_avg_delay_s, "Safety": sel.u_safety_score,
                            "Environment": sel.u_co2_kg})
        show(viz.radar(rad, height=330, title=f"Normalised criteria - {LABEL[scenario]}"),
             key="radar")
    with st.expander("Table view of the trade-off chart"):
        table(tt.assign(scenario=tt.scenario.map(LABEL)).round(2))

# ======================================================================================
# 6. Policy comparison
# ======================================================================================
with tabs[5]:
    st.markdown("#### All policies, all scenarios")
    metric = st.selectbox("Measure", ["avg_delay_s", "throughput_vph", "co2_kg",
                                      "conflicts_per_1000", "safety_score", "ped_avg_wait_s",
                                      "jaywalks", "bus_on_time_pct", "bus_avg_delay_s",
                                      "switches"], format_func=lambda m: MEASURE.get(m, m),
                          key="pc-m")
    show(viz.policy_bars(KPI, metric, MEASURE.get(metric, metric), height=380,
                         title=f"{MEASURE.get(metric, metric)}: mean ± sd over 8 seeds"),
         key="pc-bars")
    st.markdown("#### Q-learning vs SARSA (paired by evaluation seed)")
    rows = []
    try:
        from scipy import stats
    except Exception:
        stats = None
    for s_ in C.SCENARIO_ORDER:
        a = KPI[(KPI.scenario == s_) & (KPI.policy == "Q-learning")].sort_values("seed")
        b = KPI[(KPI.scenario == s_) & (KPI.policy == "SARSA")].sort_values("seed")
        d = b.avg_delay_s.values - a.avg_delay_s.values
        p = float(stats.ttest_rel(b.avg_delay_s, a.avg_delay_s).pvalue) if stats else np.nan
        rows.append(dict(scenario=LABEL[s_], q_learning_delay=a.avg_delay_s.mean(),
                         sarsa_delay=b.avg_delay_s.mean(), sarsa_minus_q=d.mean(),
                         paired_t_p=p, q_learning_conflicts=a.conflicts_per_1000.mean(),
                         sarsa_conflicts=b.conflicts_per_1000.mean(),
                         q_learning_jaywalks=a.jaywalks.mean(), sarsa_jaywalks=b.jaywalks.mean()))
    table(pd.DataFrame(rows).round(3))
    st.caption("Negative sarsa_minus_q = SARSA has less delay. Common random numbers make "
               "the comparison paired: each seed has identical arrivals for both policies.")
    if T["robust"] is not None:
        st.markdown("#### Training stability (re-adapting with different random seeds)")
        r = T["robust"].groupby(["scenario", "policy"]).avg_delay_s.agg(
            ["mean", "std", "min", "max"]).reset_index()
        r["scenario"] = r.scenario.map(LABEL)
        table(r.round(1))
    st.markdown("#### Which part of the system delivers the gain? Signal timing × routing")
    ab = T["abl"]
    sxr = ab[ab.study == "Signal x routing"]
    if len(sxr):
        scs = [x for x in C.SCENARIO_ORDER if x in sxr.scenario.unique()]
        pick = st.selectbox("Scenario", scs, index=scs.index(scenario) if scenario in scs else 0,
                            format_func=lambda x: LABEL[x], key="sxr-sc")
        cx1, cx2 = st.columns([1.3, 1])
        with cx1:
            show(viz.combo_bars(sxr[sxr.scenario == pick], height=340,
                                title=f"Every signal × routing pair, {LABEL[pick]} (8 seeds)"),
                 key="sxr")
        with cx2:
            tb = sxr[sxr.scenario == pick].groupby("variant")[
                ["avg_delay_s", "conflicts_per_1000", "co2_kg", "jaywalks"]].mean() \
                .sort_values("avg_delay_s").round(2).reset_index()
            table(tb, height=340)
        st.caption("Cooperative RL routing = the Q-learning AV fleet (SARSA fleet with SARSA "
                   "signals). These are the decision network's eight options, simulated "
                   "directly on the evaluation seeds.")
    st.markdown("#### Component studies (Q-learning signals)")
    a1, a2 = st.columns(2)
    for col, study, metrics in [(a1, "AV routing", ["avg_delay_s", "av_avg_delay_s"]),
                                (a2, "Bus priority", ["bus_avg_delay_s", "bus_on_time_pct",
                                                      "avg_delay_s"])]:
        with col:
            st.markdown(f"**{study}**")
            sub = ab[ab.study == study].groupby(["scenario", "variant"])[metrics].mean() \
                .round(1).reset_index()
            sub["scenario"] = sub.scenario.map(LABEL)
            table(sub)
    st.markdown("#### Learning curves")
    l1, l2 = st.columns(2)
    with l1:
        show(viz.learning_curves(T["curves"], "signal_return", "pre-train", height=320,
                                 label="Episode return (signal agents)",
                                 title="Return during pre-training"), key="lc3")
    with l2:
        show(viz.learning_curves(T["curves"], "conflicts_per_1000", "pre-train", height=320,
                                 label="Expected conflicts per 1,000 veh",
                                 title="Safety during pre-training (exploring)"), key="lc4")
