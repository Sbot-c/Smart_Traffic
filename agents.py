"""
agents.py - the three agent types and their learning rules.

Agent                 State (discretised)                          Actions
--------------------  -------------------------------------------  -------------------------
Traffic signal (x9)   phase, green time, NS queue, EW queue,       0 keep phase / 1 change
                      pedestrian pressure, downstream spillback     (512 states)
Autonomous vehicle    remaining moves (x, y), congestion on the     0 move along x / 1 along y
(fleet, shared table) two candidate next links                      (64 states)
Bus (fleet, shared)   schedule deviation, load, headway, signal     0 go / 1 request priority /
                      ahead, cross-street queue                     2 hold 20 s  (144 states)

Learning rules (Kochenderfer 2015, ch. 5; Sutton & Barto 2018, ch. 6):
  Q-learning (off-policy): Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]
  SARSA      (on-policy):  Q(s,a) <- Q(s,a) + alpha [ r + gamma Q(s',a')        - Q(s,a) ]
with epsilon-greedy exploration. Q starts at 0, which is optimistic because every reward
is a cost (negative), so untried actions get tried.
"""
from __future__ import annotations

import numpy as np

import config as C
import traffic_env as TE


# ======================================================================================
# Tabular learner shared by all agent types
# ======================================================================================
class TabularLearner:
    def __init__(self, n_states: int, n_actions: int, algo: str = "qlearning",
                 alpha: float = 0.1, gamma: float = 0.9, seed: int = 0):
        assert algo in ("qlearning", "sarsa")
        self.Q = np.zeros((n_states, n_actions))
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)
        self.algo = algo
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.td_abs = 0.0
        self.n_updates = 0

    def greedy(self, s: int, legal, deploy: bool = False) -> int:
        """Greedy action. In deployment (evaluation) mode an action that was never tried
        in this state is not taken while a tried alternative exists: its Q-value is still
        the optimistic initial 0, not a learned estimate."""
        q = self.Q[s]
        cand = legal
        if deploy:
            tried = [a for a in legal if self.visits[s, a] > 0]
            if tried:
                cand = tried
        best_v = max(q[a] for a in cand)
        best = [a for a in cand if q[a] == best_v]
        return best[0] if len(best) == 1 else best[int(self.rng.integers(len(best)))]

    def known(self, s: int) -> bool:
        return bool(self.visits[s].sum() > 0)

    def select(self, s: int, legal, eps: float, deploy: bool = False) -> int:
        if len(legal) == 1:
            return legal[0]
        if eps > 0.0 and self.rng.random() < eps:
            return legal[int(self.rng.integers(len(legal)))]
        return self.greedy(s, legal, deploy)

    def update(self, s, a, r, s2=None, a2=None, legal2=None, terminal=False) -> float:
        if terminal:
            target = r
        elif self.algo == "qlearning":
            target = r + self.gamma * max(self.Q[s2, x] for x in legal2)
        else:
            target = r + self.gamma * self.Q[s2, a2]
        td = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
        self.visits[s, a] += 1
        self.td_abs += abs(td)
        self.n_updates += 1
        return td

    def snapshot(self):
        return dict(Q=self.Q.copy(), visits=self.visits.copy())

    def load(self, snap):
        self.Q = snap["Q"].copy()
        self.visits = snap["visits"].copy()


# ======================================================================================
# State encoders
# ======================================================================================
QUEUE_EDGES = (2, 7, 15, 29)          # queue levels: 0-2 | 3-7 | 8-15 | 16-29 | 30+
GREEN_EDGES = (C.MIN_GREEN, 20, 40, 60)  # green time: <10 (locked) | 10-20 | 20-40 | 40-60 | 60+
N_Q = len(QUEUE_EDGES) + 1
N_G = len(GREEN_EDGES) + 1
N_SIGNAL_STATES = 2 * N_G * N_Q * N_Q * 3 * 3
N_AV_STATES = 4 * 4 * 4
N_BUS_STATES = 4 * 3 * 3 * 2 * 2
QUEUE_BINS = ["0-2", "3-7", "8-15", "16-29", "30+"]
GREEN_BINS = ["<10 s (locked)", "10-20 s", "20-40 s", "40-60 s", "60+ s"]


def q_level(q: int) -> int:
    for k, e in enumerate(QUEUE_EDGES):
        if q <= e:
            return k
    return len(QUEUE_EDGES)


def green_level(e: float) -> int:
    for k, edge in enumerate(GREEN_EDGES):
        if e < edge:
            return k
    return len(GREEN_EDGES)


def arrival_level(n: int) -> int:
    return 0 if n == 0 else (1 if n <= 2 else 2)


def signal_state(obs) -> int:
    """phase x green time x queue on green x queue on red x arrivals on green x flag."""
    ph, el, qg, qr, arr_g, ped, spill, _ = obs
    flag = 2 if spill else (1 if ped else 0)
    return ((((ph * N_G + green_level(el)) * N_Q + q_level(qg)) * N_Q + q_level(qr)) * 3
            + arrival_level(arr_g)) * 3 + flag


def encode_signal_state(ph, g, qg, qr, arr, flag) -> int:
    return ((((ph * N_G + g) * N_Q + qg) * N_Q + qr) * 3 + arr) * 3 + flag


def decode_signal_state(s: int):
    flag = s % 3; s //= 3
    arr = s % 3; s //= 3
    qr = s % N_Q; s //= N_Q
    qg = s % N_Q; s //= N_Q
    el = s % N_G; s //= N_G
    ph = s
    return dict(phase=ph, green=el, q_green=qg, q_red=qr, arrivals_green=arr,
                flag=["none", "pedestrians waiting", "spillback"][flag])


def av_state(mx: int, my: int, cx: int, cy: int) -> int:
    return ((min(mx, 2) - 1) * 2 + (min(my, 2) - 1)) * 16 + cx * 4 + cy


def bus_state(dev: float, load: int, hw_state: int, green: int, qc: int) -> int:
    d = 0 if dev < -60 else (1 if dev <= 60 else (2 if dev <= 180 else 3))
    ld = 0 if load < 25 else (1 if load < 55 else 2)
    c = 0 if qc < 8 else 1
    return (((d * 3 + ld) * 3 + hw_state) * 2 + green) * 2 + c


def decode_bus_state(s: int):
    c = s % 2; s //= 2
    green = s % 2; s //= 2
    hw = s % 3; s //= 3
    ld = s % 3; s //= 3
    d = s
    return dict(schedule=["early", "on time", "late", "very late"][d],
                load=["low", "medium", "high"][ld],
                headway=["bunched", "regular", "gap"][hw],
                signal=["red", "green"][green], cross_queue=["light", "heavy"][c])


# ======================================================================================
# Design flows and the Webster fixed-time plan
# ======================================================================================
def design_flows(scenario: str = "normal"):
    """Expected hourly flow on every approach link under a scenario's base demand."""
    env = TE.TrafficEnv(scenario, seed=0)
    flows = np.zeros(TE.N_LINKS)
    sc = C.SCENARIOS[scenario]
    for L in TE.ENTRY_LINKS:
        lam = env._base_rate(L) * 3600.0 * sc["demand_mult"] * env.w_demand
        exits, probs = TE.OD[L]
        i0 = L // 4
        for (di, dh), p in zip(exits, probs):
            for xf in (True, False):
                path = TE.monotone_path(i0, di, dh, xf)
                link, i = L, i0
                for h in path:
                    flows[link] += 0.5 * p * lam
                    nxt = TE.DOWN[i][h]
                    if nxt < 0:
                        break
                    link, i = nxt, nxt // 4
    return flows


def webster_plan(scenario: str = "normal"):
    """Webster (1958) optimum cycle C0 = (1.5 L + 5) / (1 - Y), greens split by y."""
    flows = design_flows(scenario)
    plan = []
    for i in range(C.N_INT):
        y = [0.0, 0.0]
        for a in range(4):
            L = i * 4 + a
            sat = TE.LINK_LANES[L] * C.SAT_FLOW * 3600.0
            ph = C.PHASE_OF_APPROACH[a]
            y[ph] = max(y[ph], flows[L] / sat)
        Y = min(sum(y), 0.9)
        lost = 2 * C.LOST_TIME
        cycle = min(max((1.5 * lost + 5.0) / (1.0 - Y), 40.0), 120.0)
        eff = cycle - lost
        g = [max(C.MIN_GREEN + 5.0, eff * y[p] / max(Y, 1e-6)) for p in (0, 1)]
        plan.append([round(g[0] / C.DT) * C.DT, round(g[1] / C.DT) * C.DT])
    return plan


# ======================================================================================
# Signal controllers
# ======================================================================================
class FixedTimeSignals:
    kind = "fixed"

    def __init__(self, plan=None):
        self.plan = plan if plan is not None else webster_plan("normal")

    def act(self, env):
        return [1 if env.elapsed[i] >= self.plan[i][env.phase[i]] else 0
                for i in range(C.N_INT)]

    def after_shield(self, env, executed):
        pass

    def store(self, env, rewards, executed):
        pass

    def end_episode(self, env):
        pass


class ActuatedSignals:
    """Vehicle-actuated control: gap-out when the green approaches empty, switch when the
    red side builds a much longer queue, serve waiting pedestrians, max green 60 s."""
    kind = "actuated"

    @staticmethod
    def rule(env, i, obs=None):
        ph, el, qg, qr, arr_g, ped, spill, q = obs if obs is not None else env.signal_obs(i)
        qg += sum(env.approaching(i, a, 4.0) for a in range(4)
                  if C.PHASE_OF_APPROACH[a] == ph)
        if el < C.MIN_GREEN:
            return 0
        if el >= 60.0:
            return 1
        if qg == 0 and qr > 0:
            return 1
        if qr >= 2 * qg + 4:
            return 1
        if ped and el >= 25.0:
            return 1
        return 0

    def act(self, env):
        return [self.rule(env, i) for i in range(C.N_INT)]

    def after_shield(self, env, executed):
        pass

    def store(self, env, rewards, executed):
        pass

    def end_episode(self, env):
        pass


class RLSignals:
    """Nine signal agents. shared=False: independent learners (one Q-table per junction);
    shared=True: parameter sharing (all junctions update one table)."""
    kind = "rl"

    def __init__(self, algo="qlearning", seed=0, alpha=None, gamma=None, shared=False):
        hp = C.RL["signal"]
        self.algo = algo
        self.shared = shared
        if shared:
            one = TabularLearner(N_SIGNAL_STATES, 2, algo, alpha or hp["alpha"],
                                 gamma or hp["gamma"], seed * 101)
            self.learners = [one] * C.N_INT
        else:
            self.learners = [TabularLearner(N_SIGNAL_STATES, 2, algo,
                                            alpha or hp["alpha"], gamma or hp["gamma"],
                                            seed * 101 + i) for i in range(C.N_INT)]
        self.train = False
        self.eps = 0.0
        self.prev = [None] * C.N_INT
        self.cur = [0] * C.N_INT
        self.ep_return = 0.0
        self.fallbacks = 0

    def begin(self, train: bool, eps: float):
        self.train, self.eps = train, eps
        self.prev = [None] * C.N_INT
        self.ep_return = 0.0

    def act(self, env):
        out = []
        for i in range(C.N_INT):
            obs = env.signal_obs(i)
            s = signal_state(obs)
            self.cur[i] = s
            legal = env.legal_actions_for(i)
            lr = self.learners[i]
            if not self.train and len(legal) > 1 and not lr.known(s):
                # state never seen in training: defer to the actuated rule (safe fallback)
                out.append(ActuatedSignals.rule(env, i, obs))
                self.fallbacks += 1
                continue
            out.append(lr.select(s, legal, self.eps if self.train else 0.0,
                                 deploy=not self.train))
        return out

    def after_shield(self, env, executed):
        if not self.train:
            return
        for i in range(C.N_INT):
            p = self.prev[i]
            if p is None:
                continue
            s0, a0, r0 = p
            legal = env.legal_actions_for(i)
            self.learners[i].update(s0, a0, r0, self.cur[i], int(executed[i]), legal)

    def store(self, env, rewards, executed):
        for i in range(C.N_INT):
            self.prev[i] = (self.cur[i], int(executed[i]), float(rewards[i]))
        self.ep_return += float(np.sum(rewards))

    def end_episode(self, env):
        if not self.train:
            return
        # time-limit truncation: bootstrap from the final state
        chosen = self.act(env)
        executed, _ = env.shield(chosen)
        self.after_shield(env, executed)

    def snapshot(self):
        return [l.snapshot() for l in self.learners]

    def load(self, snap):
        for l, s in zip(self.learners, snap):
            l.load(s)


# ======================================================================================
# Autonomous-vehicle routing
# ======================================================================================
class RLRouting:
    """Fleet-shared Q-table (connected AVs learn together). Cooperative mode adds a
    congestion externality to the reward so AVs avoid piling onto busy links
    (cf. Q-routing, Boyan & Littman 1994)."""

    def __init__(self, algo="qlearning", coop=True, seed=0):
        hp = C.RL["av"]
        self.learner = TabularLearner(N_AV_STATES, 2, algo, hp["alpha"], hp["gamma"],
                                      seed * 7 + 3)
        self.coop = coop
        self.beta = hp["coop_beta"] if coop else 0.0
        self.train = False
        self.eps = 0.0
        self.log = None

    def begin(self, train, eps, record=False):
        self.train, self.eps = train, eps
        self.log = [] if record else None

    def policy(self, env, v, i, opts, dx, dy):
        hx = 1 if dx > 0 else 3
        hy = 2 if dy > 0 else 0
        Lx, Ly = TE.DOWN[i][hx], TE.DOWN[i][hy]
        cx, cy = env.congestion_level(Lx), env.congestion_level(Ly)
        s = av_state(abs(dx), abs(dy), cx, cy)
        a = self.learner.select(s, (0, 1), self.eps if self.train else 0.0,
                                deploy=not self.train)
        if v.av_pending is not None and self.train:
            s0, a0, t0, pen0 = v.av_pending
            r = -(env.t - t0) / 10.0 - pen0
            self.learner.update(s0, a0, r, s, a, (0, 1))
        h = hx if a == 0 else hy
        L = TE.DOWN[i][h]
        pen = self.beta * min(1.0, env.occupancy_ratio(L)) * 3.0 if L >= 0 else 0.0
        v.av_pending = (s, a, env.t, pen)
        if self.log is not None:
            self.log.append((env.t, v.vid, i, cx, cy, a, h))
        return h

    def learn_terminal(self, env, v, t_exit, terminal=True):
        if v.av_pending is None:
            return
        s0, a0, t0, pen0 = v.av_pending
        if self.train:
            self.learner.update(s0, a0, -(t_exit - t0) / 10.0 - pen0, terminal=True)
        v.av_pending = None


# ======================================================================================
# Bus agents
# ======================================================================================
class FixedBus:
    """Baseline bus behaviour: never ask for priority (mode 'none') or always ask ('always')."""

    def __init__(self, mode="none"):
        self.mode = mode
        self.log = None

    def begin(self, train, eps, record=False):
        self.log = [] if record else None

    def decide(self, env, v, i, a):
        act = 1 if self.mode == "always" else 0
        if self.log is not None:
            dev, load, hw, green, qc = env.bus_state_features(v, i, a)
            self.log.append((env.t, v.vid, v.bus.direction, i, bus_state(dev, load, hw, green,
                                                                         qc), act, dev, load))
        return act

    def learn_arrival(self, env, v, dev, terminal=False):
        pass


class RLBus:
    """Fleet-shared Q-table. Reward = -(passenger-minutes on the segment / 40
    + |schedule deviation at the next stop| in minutes + priority externality)."""

    def __init__(self, algo="qlearning", seed=0):
        hp = C.RL["bus"]
        self.learner = TabularLearner(N_BUS_STATES, 3, algo, hp["alpha"], hp["gamma"],
                                      seed * 13 + 5)
        self.train = False
        self.eps = 0.0
        self.pending_r = {}
        self.log = None

    def begin(self, train, eps, record=False):
        self.train, self.eps = train, eps
        self.pending_r = {}
        self.log = [] if record else None

    def decide(self, env, v, i, a):
        b = v.bus
        dev, load, hw, green, qc = env.bus_state_features(v, i, a)
        s = bus_state(dev, load, hw, green, qc)
        act = self.learner.select(s, (0, 1, 2), self.eps if self.train else 0.0,
                                  deploy=not self.train)
        if self.train and b.pending is not None and v.vid in self.pending_r:
            s0, a0 = b.pending[0], b.pending[1]
            self.learner.update(s0, a0, self.pending_r.pop(v.vid), s, act, (0, 1, 2))
        tsp_cost = (0.15 + 0.35 * (qc >= 8)) if act == 1 else 0.0
        b.pending = (s, act, env.t, load, tsp_cost)
        if self.log is not None:
            self.log.append((env.t, v.vid, b.direction, i, s, act, dev, load))
        return act

    def learn_arrival(self, env, v, dev, terminal=False):
        b = v.bus
        if b.pending is None:
            return
        s0, a0, t0, load0, cost0 = b.pending
        t_now = env.t if terminal else v.t_arr
        seg = max(0.0, t_now - t0)
        r = -(load0 * seg / 2400.0 + abs(dev) / 60.0 + cost0)
        if terminal:
            if self.train:
                self.learner.update(s0, a0, r, terminal=True)
            b.pending = None
        else:
            self.pending_r[v.vid] = r


# ======================================================================================
# The multi-agent system: one object that owns every controller
# ======================================================================================
POLICIES = {
    "Fixed-time": dict(signal="fixed", routing="static", bus="none"),
    "Actuated": dict(signal="actuated", routing="static", bus="always"),
    "Q-learning": dict(signal="qlearning", routing="qlearning", bus="qlearning"),
    "SARSA": dict(signal="sarsa", routing="sarsa", bus="sarsa"),
}
POLICY_ORDER = list(POLICIES)


class MultiAgentSystem:
    def __init__(self, signal="qlearning", routing="qlearning", bus="qlearning",
                 coop=True, seed=0, plan=None):
        self.spec = dict(signal=signal, routing=routing, bus=bus, coop=coop)
        if signal == "fixed":
            self.signals = FixedTimeSignals(plan)
        elif signal == "actuated":
            self.signals = ActuatedSignals()
        else:
            self.signals = RLSignals(signal, seed)
        self.routing = None if routing == "static" else RLRouting(routing, coop, seed)
        if bus in ("none", "always"):
            self.bus = FixedBus(bus)
        else:
            self.bus = RLBus(bus, seed)

    # ---- persistence -------------------------------------------------------------
    def snapshot(self):
        snap = {}
        if isinstance(self.signals, RLSignals):
            snap["signals"] = self.signals.snapshot()
        if self.routing is not None:
            snap["routing"] = self.routing.learner.snapshot()
        if isinstance(self.bus, RLBus):
            snap["bus"] = self.bus.learner.snapshot()
        return snap

    def load(self, snap):
        if "signals" in snap and isinstance(self.signals, RLSignals):
            self.signals.load(snap["signals"])
        if "routing" in snap and self.routing is not None:
            self.routing.learner.load(snap["routing"])
        if "bus" in snap and isinstance(self.bus, RLBus):
            self.bus.learner.load(snap["bus"])

    # ---- one episode --------------------------------------------------------------
    def run_episode(self, scenario="normal", seed=0, train=False, eps=0.0, record=False,
                    overrides=None):
        env = TE.TrafficEnv(scenario, seed, record=record, overrides=overrides)
        if self.routing is not None:
            self.routing.begin(train, eps, record)
            env.av_policy = self.routing.policy
            env.av_learn = self.routing.learn_terminal
        self.bus.begin(train, eps, record)
        env.bus_policy = self.bus
        if isinstance(self.signals, RLSignals):
            self.signals.begin(train, eps)
        while not env.done:
            chosen = self.signals.act(env)
            executed, codes = env.shield(chosen)
            self.signals.after_shield(env, executed)
            rewards = env.step(executed, chosen, codes)
            self.signals.store(env, rewards, executed)
        self.signals.end_episode(env)
        env.signal_return = float(env.log[:, :, TE.F["reward"]].sum())
        if record:
            env.av_log = self.routing.log if self.routing is not None else []
            env.bus_log = self.bus.log
        return env


def make_system(policy: str, tables=None, seed=0, plan=None, routing_override=None,
                coop=True):
    spec = dict(POLICIES[policy])
    if routing_override is not None:
        spec["routing"] = routing_override
    sysm = MultiAgentSystem(spec["signal"], spec["routing"], spec["bus"], coop=coop,
                            seed=seed, plan=plan)
    if tables is not None:
        sysm.load(tables)
    return sysm
