"""
decision_network.py - an influence diagram for evaluating traffic-control policies.

Structure (chance = oval, decision = rectangle, utility = diamond):

    Weather ----------------+---------------------> Pedestrian flow
       |                    |                            |      \\
       v                    v                            v       v
    Congestion <-- Density  Collision risk <-------------+    Health impact
    ^   ^   |               ^   ^                                ^
    |   |   +---------------+   |                                |
    |   |   +--> Emissions -----)--------------------------------+
    |   |            ^          |
  Signal timing -----+----------+         Routing --> Congestion
    (decision)                              (decision)

    U_efficiency(Congestion)   U_safety(Collision risk)   U_environment(Emissions, Health)

* Weather, Density and Pedestrian flow are the uncertain operating conditions.
* Signal timing {Fixed-time, Actuated, Q-learning, SARSA} and Routing {Static,
  Cooperative RL} are the city's decisions.
* P(Congestion | Density, Weather, Signal, Routing), P(Collision risk | Weather, Ped, Congestion,
  Signal) and P(Emissions | Congestion, Signal) are LEARNED from the simulator: every
  junction x 10-minute window of every run is one sample, discretised into Low / Medium /
  High (congestion by HCM level of service A / B / C-or-worse; collision risk and
  emissions at the tertiles of the pooled simulated distribution), and counted with a
  Dirichlet(1) prior (Laplace smoothing = Bayesian parameter learning). RL policies enter
  with the tables adapted to the matching scenario (an operator deploys the rain-adapted
  agents when it rains).
* P(Health | Emissions, Ped) and the priors are expert-specified assumptions.
* The utility is multi-attribute (MAUT): U = w_eff U_eff + w_safe U_safe + w_env U_env.

Inference is exact enumeration (the joint has < 1,000 cells per decision).
"""
from __future__ import annotations

import itertools
import string

import numpy as np
import pandas as pd


# ======================================================================================
# A small, generic decision-network engine
# ======================================================================================
class DecisionNetwork:
    def __init__(self):
        self.chance = {}       # name -> dict(states, parents, cpt)
        self.decisions = {}    # name -> states
        self.utilities = {}    # name -> dict(parents, table)
        self.order = []        # chance-node order (topological)

    def add_chance(self, name, states, parents=(), cpt=None):
        self.chance[name] = dict(states=list(states), parents=list(parents),
                                 cpt=None if cpt is None else np.asarray(cpt, float))
        self.order.append(name)

    def add_decision(self, name, states):
        self.decisions[name] = list(states)

    def add_utility(self, name, parents, table):
        self.utilities[name] = dict(parents=list(parents), table=np.asarray(table, float))

    def card(self, name):
        if name in self.chance:
            return len(self.chance[name]["states"])
        return len(self.decisions[name])

    def states(self, name):
        return self.chance[name]["states"] if name in self.chance else self.decisions[name]

    def check(self):
        for n, nd in self.chance.items():
            shape = tuple(self.card(p) for p in nd["parents"]) + (self.card(n),)
            assert nd["cpt"].shape == shape, (n, nd["cpt"].shape, shape)
            assert np.allclose(nd["cpt"].sum(-1), 1.0), f"CPT of {n} does not sum to 1"
        return True

    # ----------------------------------------------------------------------------------
    def joint(self, decision: dict, evidence: dict | None = None):
        """Posterior joint over all chance nodes for a fixed decision, given evidence."""
        evidence = evidence or {}
        letters = {n: string.ascii_letters[k] for k, n in enumerate(self.order)}
        factors, subs = [], []
        for n in self.order:
            nd = self.chance[n]
            f = nd["cpt"]
            idx, sub = [], ""
            for p in nd["parents"]:
                if p in self.decisions:
                    idx.append(self.decisions[p].index(decision[p]))
                else:
                    idx.append(slice(None))
                    sub += letters[p]
            idx.append(slice(None))
            factors.append(f[tuple(idx)])
            subs.append(sub + letters[n])
        for n, val in evidence.items():
            ind = np.zeros(self.card(n))
            ind[self.states(n).index(val)] = 1.0
            factors.append(ind)
            subs.append(letters[n])
        out = "".join(letters[n] for n in self.order)
        J = np.einsum(",".join(subs) + "->" + out, *factors)
        z = J.sum()
        return J / z if z > 0 else J, z

    def marginal(self, J, name):
        axis = tuple(k for k, n in enumerate(self.order) if n != name)
        return J.sum(axis=axis)

    def expected_utility(self, decision: dict, evidence=None, weights=None):
        J, z = self.joint(decision, evidence)
        weights = weights or {u: 1.0 for u in self.utilities}
        comps = {}
        for u, ud in self.utilities.items():
            par = ud["parents"]
            keep = [self.order.index(p) for p in par]
            axis = tuple(k for k in range(len(self.order)) if k not in keep)
            M = J.sum(axis=axis)
            # M has the parents in network order; transpose to the utility's parent order
            net_order = sorted(keep)
            perm = [net_order.index(self.order.index(p)) for p in par]
            M = np.transpose(M, perm) if len(perm) > 1 else M
            comps[u] = float(np.sum(M * ud["table"]))
        total = float(sum(weights.get(u, 0.0) * v for u, v in comps.items()))
        post = {n: self.marginal(J, n) for n in self.order}
        return dict(total=total, components=comps, posteriors=post, evidence_prob=z)

    def decision_table(self, evidence=None, weights=None):
        names = list(self.decisions)
        rows = []
        for combo in itertools.product(*[self.decisions[n] for n in names]):
            d = dict(zip(names, combo))
            r = self.expected_utility(d, evidence, weights)
            row = dict(d)
            row["EU"] = r["total"]
            row.update({f"E[{u}]": v for u, v in r["components"].items()})
            for n in ("Congestion", "Collision risk", "Emissions", "Health impact"):
                if n in r["posteriors"]:
                    for s, p in zip(self.states(n), r["posteriors"][n]):
                        row[f"P({n}={s})"] = float(p)
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("EU", ascending=False).reset_index(drop=True)
        return df

    def best(self, evidence=None, weights=None):
        t = self.decision_table(evidence, weights)
        return t.iloc[0].to_dict(), t

    def value_of_information(self, var, evidence=None, weights=None):
        """EVPI of observing chance node `var` before choosing (var must not depend on
        the decisions)."""
        evidence = dict(evidence or {})
        base_best, base_table = self.best(evidence, weights)
        dec0 = {n: base_best[n] for n in self.decisions}
        post = self.expected_utility(dec0, evidence, weights)["posteriors"][var]
        value = 0.0
        detail = []
        for s, p in zip(self.states(var), post):
            if p <= 0:
                continue
            e2 = dict(evidence)
            e2[var] = s
            b, _ = self.best(e2, weights)
            value += p * b["EU"]
            detail.append(dict(state=s, prob=float(p), best_signal=b.get("Signal timing"),
                               best_routing=b.get("Routing"), EU=b["EU"]))
        return dict(variable=var, evpi=float(value - base_best["EU"]),
                    eu_without=float(base_best["EU"]), eu_with=float(value), detail=detail)


# ======================================================================================
# The traffic decision network
# ======================================================================================
WEATHER = ["Clear", "Rain", "Sandstorm"]
DENSITY = ["Low", "Medium", "High"]
PED = ["Low", "Medium", "High"]
LEVELS3 = ["Low", "Medium", "High"]
SIGNAL = ["Fixed-time", "Actuated", "Q-learning", "SARSA"]
ROUTING = ["Static", "Cooperative RL"]

# ---- expert-specified pieces (assumptions the group must defend) ---------------------
PRIOR_WEATHER = [0.82, 0.06, 0.12]           # Dubai: mostly clear; dust events > rain days
PRIOR_DENSITY = [0.30, 0.45, 0.25]           # share of operating hours: off-peak/inter-peak/peak
P_PED_GIVEN_WEATHER = [[0.25, 0.50, 0.25],   # Clear:     Low / Medium / High
                       [0.50, 0.40, 0.10],   # Rain
                       [0.70, 0.25, 0.05]]   # Sandstorm
P_HEALTH_HIGH = {                             # P(Health impact = High | Emissions, Ped flow)
    ("Low", "Low"): 0.05, ("Low", "Medium"): 0.08, ("Low", "High"): 0.15,
    ("Medium", "Low"): 0.20, ("Medium", "Medium"): 0.28, ("Medium", "High"): 0.40,
    ("High", "Low"): 0.45, ("High", "Medium"): 0.58, ("High", "High"): 0.75,
}
U_EFF = [100.0, 60.0, 10.0]                   # Congestion Low / Medium / High
U_SAFE = [100.0, 50.0, 0.0]                   # Collision risk Low / Medium / High
U_EMIS = [100.0, 55.0, 10.0]                  # Emissions Low / Medium / High
U_HEALTH = [100.0, 0.0]                       # Health impact Low / High
DEFAULT_WEIGHTS = {"U_efficiency": 0.40, "U_safety": 0.35, "U_environment": 0.25}

# HCM signalised-intersection level of service by control delay per junction (s/veh):
# Low = LOS A (<= 10 s), Medium = LOS B (10-20 s), High = LOS C or worse (> 20 s)
LOS_EDGES = [10.0, 20.0]


def discretise(samples: pd.DataFrame, risk_edges=None, emis_edges=None, cong_edges=None,
               congestion_by_los=True):
    """Congestion by HCM level of service; risk and emissions at pooled tertiles."""
    df = samples.copy()
    risk_edges = risk_edges or list(np.quantile(df["conflicts_per_1000"], [1 / 3, 2 / 3]))
    emis_edges = emis_edges or list(np.quantile(df["co2_g_per_veh"], [1 / 3, 2 / 3]))
    if congestion_by_los:
        cong_edges = LOS_EDGES
    cong_edges = cong_edges or list(np.quantile(df["delay_per_veh_s"], [1 / 3, 2 / 3]))
    df["Congestion"] = pd.cut(df["delay_per_veh_s"], [-np.inf] + list(cong_edges) + [np.inf],
                              labels=LEVELS3).astype(str)
    df["Collision risk"] = pd.cut(df["conflicts_per_1000"],
                                  [-np.inf] + risk_edges + [np.inf], labels=LEVELS3).astype(str)
    df["Emissions"] = pd.cut(df["co2_g_per_veh"], [-np.inf] + emis_edges + [np.inf],
                             labels=LEVELS3).astype(str)
    return df, dict(congestion=[float(x) for x in cong_edges],
                    risk=[float(x) for x in risk_edges],
                    emissions=[float(x) for x in emis_edges])


def learn_cpt(df, child, child_states, parents, parent_states, alpha=1.0):
    """Dirichlet(alpha) posterior mean of P(child | parents) from counts."""
    shape = tuple(len(s) for s in parent_states) + (len(child_states),)
    counts = np.zeros(shape)
    idx_maps = [{s: k for k, s in enumerate(ps)} for ps in parent_states]
    cmap = {s: k for k, s in enumerate(child_states)}
    cols = list(parents) + [child]
    for row, n in df.groupby(cols).size().items():
        row = row if isinstance(row, tuple) else (row,)
        key = tuple(m[v] for m, v in zip(idx_maps, row[:-1])) + (cmap[row[-1]],)
        counts[key] += n
    cpt = (counts + alpha) / (counts.sum(-1, keepdims=True) + alpha * len(child_states))
    return cpt, counts


def build_traffic_network(samples: pd.DataFrame, alpha: float = 1.0, edges=None):
    edges = edges or {}
    df, edges = discretise(samples, edges.get("risk"), edges.get("emissions"),
                           edges.get("congestion"))
    dn = DecisionNetwork()
    dn.add_decision("Signal timing", SIGNAL)
    dn.add_decision("Routing", ROUTING)
    dn.add_chance("Weather", WEATHER, (), PRIOR_WEATHER)
    dn.add_chance("Traffic density", DENSITY, (), PRIOR_DENSITY)
    dn.add_chance("Pedestrian flow", PED, ("Weather",), np.array(P_PED_GIVEN_WEATHER))
    df = df.rename(columns={"weather": "Weather", "density": "Traffic density",
                            "ped_flow": "Pedestrian flow", "signal": "Signal timing",
                            "routing": "Routing"})
    cg, n_cg = learn_cpt(df, "Congestion", LEVELS3,
                         ["Traffic density", "Weather", "Signal timing", "Routing"],
                         [DENSITY, WEATHER, SIGNAL, ROUTING], alpha)
    dn.add_chance("Congestion", LEVELS3,
                  ("Traffic density", "Weather", "Signal timing", "Routing"), cg)
    cr, n_cr = learn_cpt(df, "Collision risk", LEVELS3,
                         ["Weather", "Pedestrian flow", "Congestion", "Signal timing"],
                         [WEATHER, PED, LEVELS3, SIGNAL], alpha)
    dn.add_chance("Collision risk", LEVELS3,
                  ("Weather", "Pedestrian flow", "Congestion", "Signal timing"), cr)
    em, n_em = learn_cpt(df, "Emissions", LEVELS3, ["Congestion", "Signal timing"],
                         [LEVELS3, SIGNAL], alpha)
    dn.add_chance("Emissions", LEVELS3, ("Congestion", "Signal timing"), em)
    hi = np.zeros((3, len(PED), 2))
    for a, e in enumerate(LEVELS3):
        for b, p in enumerate(PED):
            q = P_HEALTH_HIGH[(e, p)]
            hi[a, b] = [1 - q, q]
    dn.add_chance("Health impact", ["Low", "High"], ("Emissions", "Pedestrian flow"), hi)
    dn.add_utility("U_efficiency", ("Congestion",), U_EFF)
    dn.add_utility("U_safety", ("Collision risk",), U_SAFE)
    u_env = np.array([[0.7 * U_EMIS[a] + 0.3 * U_HEALTH[b] for b in range(2)]
                      for a in range(3)])
    dn.add_utility("U_environment", ("Emissions", "Health impact"), u_env)
    dn.check()
    dn.counts = {"Congestion": n_cg, "Collision risk": n_cr, "Emissions": n_em}
    dn.edges = edges
    dn.data = df
    return dn


# ---- scenario -> evidence mapping -----------------------------------------------------
SCENARIO_EVIDENCE = {
    "normal": {"Weather": "Clear", "Traffic density": "Medium", "Pedestrian flow": "Medium"},
    "rush_hour": {"Weather": "Clear", "Traffic density": "High", "Pedestrian flow": "Medium"},
    "incident": {"Weather": "Clear", "Traffic density": "High", "Pedestrian flow": "Medium"},
    "rain": {"Weather": "Rain", "Traffic density": "Medium"},
    "sandstorm": {"Weather": "Sandstorm", "Traffic density": "Medium"},
    "event_surge": {"Weather": "Clear", "Traffic density": "High", "Pedestrian flow": "High"},
    "bus_corridor": {"Weather": "Clear", "Traffic density": "Medium", "Pedestrian flow": "Medium"},
}


def cpt_frame(dn: DecisionNetwork) -> pd.DataFrame:
    rows = []
    for n in dn.order:
        nd = dn.chance[n]
        par = nd["parents"]
        cnt = getattr(dn, "counts", {}).get(n)
        for combo in itertools.product(*[range(dn.card(p)) for p in par]):
            probs = nd["cpt"][combo] if par else nd["cpt"]
            row = {"node": n, "parents": " | ".join(par) if par else "(prior)"}
            row["parent_values"] = ", ".join(f"{p}={dn.states(p)[k]}" for p, k in
                                             zip(par, combo)) if par else ""
            for s, pr in zip(nd["states"], probs):
                row[f"P={s}"] = round(float(pr), 4)
            row["samples"] = int(cnt[combo].sum()) if cnt is not None else ""
            rows.append(row)
    return pd.DataFrame(rows)


def scenario_decisions(dn, weights=None):
    rows = []
    for sc, ev in SCENARIO_EVIDENCE.items():
        t = dn.decision_table(ev, weights)
        for rank, r in t.iterrows():
            d = dict(scenario=sc, rank=rank + 1, evidence=", ".join(f"{k}={v}"
                                                                    for k, v in ev.items()))
            d.update(r.to_dict())
            rows.append(d)
    return pd.DataFrame(rows)


def weight_sensitivity(dn, evidence=None, step=0.05, routing=None):
    """Best signal policy over the weight simplex (w_eff + w_safe + w_env = 1)."""
    rows = []
    n = int(round(1 / step))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            w = {"U_efficiency": a * step, "U_safety": b * step,
                 "U_environment": 1 - (a + b) * step}
            t = dn.decision_table(evidence, w)
            if routing:
                t = t[t["Routing"] == routing]
            top = t.iloc[0]
            rows.append(dict(w_efficiency=round(w["U_efficiency"], 3),
                             w_safety=round(w["U_safety"], 3),
                             w_environment=round(w["U_environment"], 3),
                             best_signal=top["Signal timing"], best_routing=top["Routing"],
                             EU=float(top["EU"])))
    return pd.DataFrame(rows)


def pgmpy_crosscheck(dn, decision, evidence):
    """Optional: recompute the same expected utility with pgmpy (as used in the
    Probability Models lab). Decisions enter as root nodes fixed by evidence."""
    try:
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
        try:
            from pgmpy.models import DiscreteBayesianNetwork as BN
        except ImportError:                     # older pgmpy
            from pgmpy.models import BayesianNetwork as BN
    except ImportError:
        return None
    edges = []
    for n, nd in dn.chance.items():
        for p in nd["parents"]:
            edges.append((p, n))
    model = BN(edges)
    cpds = []
    for dname, st in dn.decisions.items():
        cpds.append(TabularCPD(dname, len(st), [[1 / len(st)] for _ in st],
                               state_names={dname: st}))
    for n, nd in dn.chance.items():
        par = nd["parents"]
        card = dn.card(n)
        if par:
            vals = nd["cpt"].reshape(-1, card).T
            cpds.append(TabularCPD(n, card, vals, evidence=par,
                                   evidence_card=[dn.card(p) for p in par],
                                   state_names={n: nd["states"],
                                                **{p: dn.states(p) for p in par}}))
        else:
            cpds.append(TabularCPD(n, card, [[x] for x in nd["cpt"]],
                                   state_names={n: nd["states"]}))
    model.add_cpds(*cpds)
    model.check_model()
    inf = VariableElimination(model)
    ev = dict(evidence)
    ev.update(decision)
    total = {}
    for u, ud in dn.utilities.items():
        q = inf.query(ud["parents"], evidence=ev, show_progress=False)
        vals = q.values
        order = q.variables
        perm = [order.index(p) for p in ud["parents"]]
        vals = np.transpose(vals, perm) if len(perm) > 1 else vals
        total[u] = float(np.sum(vals * ud["table"]))
    return total
