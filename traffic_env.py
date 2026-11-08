"""
traffic_env.py - the smart-city traffic environment.

A mesoscopic, vehicle-level "point queue with storage" model of a 3 x 3 grid of
signalised junctions:

* every vehicle is an object with a type, an origin-destination pair and a route;
* links have a free-flow travel time and a storage limit (spillback blocks upstream);
* at each stop line vehicles wait in a FIFO queue and are discharged during green at the
  saturation flow rate (turning, trucks, AV platooning and crossing pedestrians change the
  per-vehicle cost);
* pedestrians, scheduled buses with near-side stops, emergency vehicles, weather and
  incidents are modelled explicitly.

The environment is agent-agnostic: every 5 s it receives one signal action per junction
(0 = keep phase, 1 = change phase), applies the safety shield (min/max green, emergency
preemption, transit signal priority) and returns a reward per junction. Autonomous
vehicles and buses call back into their own policies when they reach a decision point.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

import config as C

# --------------------------------------------------------------------------------------
# Static network tables (computed once at import; plain lists are faster than numpy
# for the scalar look-ups inside the vehicle loops)
# --------------------------------------------------------------------------------------
N = C.GRID_N
N_INT = C.N_INT
N_LINKS = N_INT * 4          # link L = i*4 + a feeds approach a of junction i
PH_A = list(C.PHASE_OF_APPROACH)


def rc(i: int):
    return divmod(i, N)


def neighbour(i: int, h: int) -> int:
    r, c = rc(i)
    r2, c2 = r + C.DIR_DR[h], c + C.DIR_DC[h]
    if 0 <= r2 < N and 0 <= c2 < N:
        return r2 * N + c2
    return -1


# downstream link when leaving junction i with heading h (-1 = leaves the network)
DOWN = [[-1] * 4 for _ in range(N_INT)]
for _i in range(N_INT):
    for _h in range(4):
        _j = neighbour(_i, _h)
        if _j >= 0:
            DOWN[_i][_h] = _j * 4 + (_h + 2) % 4

LINK_LEN = [0.0] * N_LINKS
LINK_LANES = [0] * N_LINKS
LINK_UP = [-1] * N_LINKS     # upstream junction (-1 = boundary entry)
for _i in range(N_INT):
    _r, _c = rc(_i)
    for _a in range(4):
        _L = _i * 4 + _a
        _up = neighbour(_i, _a)              # vehicles come FROM side a
        LINK_UP[_L] = _up
        LINK_LEN[_L] = C.BLOCK_M if _up >= 0 else C.ENTRY_M
        _corr = (_r == C.CORRIDOR_ROW) and (_a in (1, 3))
        LINK_LANES[_L] = C.LANES_CORRIDOR if _corr else C.LANES_DEFAULT
LINK_STORAGE = [int(LINK_LEN[L] * LINK_LANES[L] // C.JAM_SPACING) for L in range(N_LINKS)]
ENTRY_LINKS = [L for L in range(N_LINKS) if LINK_UP[L] < 0]
EXITS = [(i, h) for i in range(N_INT) for h in range(4) if neighbour(i, h) < 0]

# turning cost / turn flag tables indexed [approach][new heading]
TURN_COST = [[0.0] * 4 for _ in range(4)]
IS_TURN = [[False] * 4 for _ in range(4)]
for _a in range(4):
    _th = (_a + 2) % 4
    for _h in range(4):
        if _h == _th:
            TURN_COST[_a][_h] = C.COST_THROUGH
        elif _h == (_th + 1) % 4:
            TURN_COST[_a][_h] = C.COST_RIGHT
        else:
            TURN_COST[_a][_h] = C.COST_LEFT
        IS_TURN[_a][_h] = _h != _th
EM = [C.EMISSION[k] for k in range(5)]
TYPE_COST = [1.0, C.COST_AV, C.COST_TRUCK, C.COST_TRUCK, 1.0]


def turn_cost(approach: int, heading: int) -> float:
    return TURN_COST[approach][heading]


def is_turn(approach: int, heading: int) -> bool:
    return IS_TURN[approach][heading]


def entry_heading(L: int) -> int:
    return (L % 4 + 2) % 4


def od_table():
    """Origin-destination probabilities for every boundary entry (gravity-style)."""
    table = {}
    for L in ENTRY_LINKS:
        i0 = L // 4
        h0 = entry_heading(L)
        r0, c0 = rc(i0)
        exits, w = [], []
        for (i, h) in EXITS:
            if h == (h0 + 2) % 4:      # no exit on the side the trip came from
                continue
            r, c = rc(i)
            dist = abs(r - r0) + abs(c - c0) + 1
            weight = math.exp(-0.35 * dist)
            if h == h0:
                weight *= 2.5          # most trips go straight across the grid
            exits.append((i, h))
            w.append(weight)
        w = np.array(w) / np.sum(w)
        table[L] = (exits, w)
    return table


OD = od_table()


def monotone_path(i0: int, dest_i: int, dest_h: int, x_first: bool):
    """Headings taken at each junction from i0 to the exit (no U-turns, no detours)."""
    r0, c0 = rc(i0)
    rd, cd = rc(dest_i)
    hx = 1 if cd > c0 else 3
    hy = 2 if rd > r0 else 0
    xs = [hx] * abs(cd - c0)
    ys = [hy] * abs(rd - r0)
    return (xs + ys if x_first else ys + xs) + [dest_h]


def remaining_moves(i: int, dest_i: int):
    r, c = rc(i)
    rd, cd = rc(dest_i)
    return cd - c, rd - r


# --------------------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------------------
class Vehicle:
    __slots__ = ("vid", "vtype", "dest_i", "dest_h", "path", "pidx", "t_spawn", "t_enter",
                 "link", "t_arr", "next_h", "stopped", "delay", "stops", "idle", "dist",
                 "ff", "co2", "nox", "av_pending", "bus")

    def __init__(self, vid, vtype, dest_i, dest_h, path, t_spawn):
        self.vid = vid
        self.vtype = vtype
        self.dest_i = dest_i
        self.dest_h = dest_h
        self.path = path          # fixed headings (None = AV deciding on the fly)
        self.pidx = 0
        self.t_spawn = t_spawn
        self.t_enter = -1.0
        self.link = -1
        self.t_arr = 0.0
        self.next_h = -1
        self.stopped = False
        self.delay = 0.0          # queueing + entry-buffer delay (s)
        self.stops = 0
        self.idle = 0.0
        self.dist = 0.0
        self.ff = 0.0             # free-flow time of the links travelled (s)
        self.co2 = 0.0
        self.nox = 0.0
        self.av_pending = None    # (state, action, t_decision, coop_penalty)
        self.bus = None           # BusInfo for buses


class BusInfo:
    __slots__ = ("route", "direction", "stops", "sched", "stop_idx", "load", "t_ready",
                 "pending", "tsp", "hold_done", "trip_id", "dev_log", "last_dep")

    def __init__(self, route, direction, stops, sched, load, trip_id):
        self.route = route
        self.direction = direction
        self.stops = stops        # junction indices with a near-side stop
        self.sched = sched        # scheduled arrival time at each stop + exit
        self.stop_idx = 0
        self.load = load
        self.t_ready = 0.0
        self.pending = None       # (state, action, t_decision, load, tsp_cost)
        self.tsp = False
        self.hold_done = False
        self.trip_id = trip_id
        self.dev_log = []         # schedule deviation at each stop (s)
        self.last_dep = None


BUS_ROUTES = {
    # route id: (entry link, path headings, stop junctions)
    "R1-E": (3 * 4 + 3, [1, 1, 1], [3, 4, 5]),    # Street 2 eastbound  J4 -> J5 -> J6
    "R1-W": (5 * 4 + 1, [3, 3, 3], [5, 4, 3]),    # Street 2 westbound  J6 -> J5 -> J4
    "R2-S": (1 * 4 + 0, [2, 2, 2], [1, 4, 7]),    # Avenue B southbound J2 -> J5 -> J8
    "R2-N": (7 * 4 + 2, [0, 0, 0], [7, 4, 1]),    # Avenue B northbound J8 -> J5 -> J2
}

# --------------------------------------------------------------------------------------
# Step-log fields (per junction per step)
# --------------------------------------------------------------------------------------
FIELDS = ["phase", "elapsed", "q_n", "q_e", "q_s", "q_w", "arrivals", "departures",
          "ped_wait_ns", "ped_wait_ew", "jaywalks", "conflicts", "co2", "nox", "delay",
          "ped_delay", "reward", "action", "executed", "override", "ev", "tsp", "blocked",
          "stops"]
F = {k: j for j, k in enumerate(FIELDS)}
N_FIELDS = len(FIELDS)
OVERRIDE_NAMES = ["none", "min-green", "max-green", "EV preemption", "TSP extension",
                  "TSP early green", "pedestrian clearance"]


class TrafficEnv:
    """One episode of the grid under a scenario. Call step() until done."""

    def __init__(self, scenario: str = "normal", seed: int = 0, record: bool = False,
                 overrides: dict | None = None):
        self.scenario_key = scenario
        sc = dict(C.SCENARIOS[scenario])
        if overrides:
            sc.update(overrides)
        self.sc = sc
        self.seed = seed
        self.record = record
        self.weather = sc["weather"]
        (self.w_sat, self.w_speed, self.w_risk, self.w_ped, self.w_demand,
         self.w_lost) = C.WEATHER[self.weather]
        self.lost_time = C.LOST_TIME + self.w_lost
        self.n_steps = int(C.EPISODE_S / C.DT)
        inc = sc.get("incident")
        if inc:
            self.inc_L = inc["junction"] * 4 + inc["approach"]
            self.inc_t0, self.inc_t1 = inc["t_start"], inc["t_end"]
            self.inc_sat, self.inc_speed = inc["sat_factor"], inc["speed_factor"]
        else:
            self.inc_L = -1
        # hooks set by the multi-agent system
        self.av_policy = None      # callable(env, veh, i, options, dx, dy) -> heading
        self.av_learn = None       # callable(env, veh, t_exit, terminal)
        self.bus_policy = None     # object with decide() and learn_arrival()
        self.reset()

    # ----------------------------------------------------------------------------------
    # Demand generation (common random numbers: identical for every controller)
    # ----------------------------------------------------------------------------------
    def _base_rate(self, L: int) -> float:
        i, a = divmod(L, 4)
        r, c = rc(i)
        if a in (1, 3):                                 # enters on an E-W street
            base = C.DEMAND_CORRIDOR if r == C.CORRIDOR_ROW else C.DEMAND_STREET
            if r == C.CORRIDOR_ROW and a == 3:          # eastbound corridor (towards CBD)
                base *= self.sc.get("corridor_mult", 1.0)
        else:
            base = C.DEMAND_AVENUE
        return base / 3600.0

    def _generate_demand(self):
        rng = np.random.default_rng(10_000 + self.seed)
        sc = self.sc
        T = C.EPISODE_S
        ev = sc.get("event")
        ev_entries = {tuple(e) for e in ev["entries"]} if ev else set()
        chunks = []
        for L in ENTRY_LINKS:
            base = self._base_rate(L) * sc["demand_mult"] * self.w_demand
            surge = ev["veh_mult"] if (ev and (L // 4, L % 4) in ev_entries) else 1.0
            lam_max = base * 1.5 * surge
            n = rng.poisson(lam_max * T)
            t = np.sort(rng.uniform(0.0, T, n))
            prof = np.array([C.demand_profile(sc["profile"], x) for x in t]) if \
                sc["profile"] != "flat" else np.ones(n)
            lam = base * prof
            if surge > 1.0:
                lam = np.where((t >= ev["t_start"]) & (t <= ev["t_end"]), lam * surge, lam)
            keep = rng.random(n) < lam / lam_max
            t = t[keep]
            m = len(t)
            exits, probs = OD[L]
            k = np.searchsorted(np.cumsum(probs), rng.random(m) * 0.999999)
            u = rng.random(m)
            vtype = np.where(u < C.AV_SHARE, C.VT_AV,
                             np.where(u < C.AV_SHARE + C.TRUCK_SHARE, C.VT_TRUCK, C.VT_CAR))
            xf = rng.random(m) < 0.5
            for j in range(m):
                di, dh = exits[k[j]]
                chunks.append((float(t[j]), L, int(vtype[j]), di, dh, bool(xf[j])))
        # emergency vehicles
        n_ev = rng.poisson(sc["ev_per_episode"])
        for _ in range(n_ev):
            t = float(rng.uniform(60.0, T - 300.0))
            L = ENTRY_LINKS[int(rng.integers(len(ENTRY_LINKS)))]
            exits, probs = OD[L]
            k = int(np.searchsorted(np.cumsum(probs), rng.random() * 0.999999))
            chunks.append((t, L, C.VT_EV, exits[k][0], exits[k][1], True))
        inc = sc.get("incident")
        if inc:   # ambulances / police driving through the incident junction
            for t in np.linspace(inc["t_start"] + 30, inc["t_start"] + 400, 3):
                chunks.append((float(t), 3 * 4 + 3, C.VT_EV, 5, 1, True))
        chunks.sort(key=lambda x: x[0])
        self.arrivals = deque(chunks)

        # pedestrians: Poisson counts per step, per junction, per crossing group
        lam = C.PED_PER_HOUR * sc["ped_mult"] * self.w_ped / 3600.0 * C.DT
        lam_arr = np.full((self.n_steps, N_INT, 2), lam)
        if ev:
            s0, s1 = int(ev["t_start"] / C.DT), int(ev["t_end"] / C.DT)
            for j in ev["junctions"]:
                lam_arr[s0:s1, j, :] *= ev["ped_mult"]
        if sc["profile"] == "peak":
            prof = np.array([C.demand_profile("peak", s * C.DT) for s in range(self.n_steps)])
            lam_arr *= prof[:, None, None]
        self.ped_arrivals = rng.poisson(lam_arr).tolist()

        # bus passengers + timetable
        self.pax_rate = C.BUS_PAX_PER_MIN * sc["pax_mult"] / 60.0
        self.rng_pax = np.random.default_rng(20_000 + self.seed)
        self.bus_headway = {r: (sc["bus_headway_s"] if (r.startswith("R1") or
                                                         not sc.get("bus_lane")) else 360)
                            for r in BUS_ROUTES}
        deps = []
        for route in BUS_ROUTES:
            hw = self.bus_headway[route]
            t = float(rng.uniform(0, hw))
            while t < T - 120:
                jitter = float(np.clip(rng.normal(0.0, C.BUS_DEPART_SD), -0.6 * hw, 0.6 * hw))
                deps.append((max(0.0, t + jitter), route, t))
                t += hw
        deps.sort()
        self.bus_departures = deque(deps)
        self.rng_noise = np.random.default_rng(30_000 + self.seed)

    # ----------------------------------------------------------------------------------
    def reset(self):
        self.t = 0.0
        self.step_idx = 0
        self.vid_counter = 0
        self.moving = [deque() for _ in range(N_LINKS)]   # vehicles travelling on link
        self.moving_ev = [[] for _ in range(N_LINKS)]     # emergency vehicles (overtake)
        self.queue = [deque() for _ in range(N_LINKS)]    # vehicles waiting at stop line
        self.bus_q = [deque() for _ in range(N_LINKS)]    # dedicated bus-lane queue
        self.ev_wait = [[] for _ in range(N_LINKS)]       # emergency vehicles at stop line
        self.bus_dwell = [[] for _ in range(N_LINKS)]     # buses at a near-side stop
        self.buffer = [deque() for _ in range(N_LINKS)]   # vehicles waiting to enter
        self.occ = [0] * N_LINKS
        self.credit = [0.0] * N_LINKS
        self.phase = [0, 1, 0, 1, 0, 1, 0, 1, 0]         # staggered start
        self.elapsed = [20.0] * N_INT
        self.switched = [False] * N_INT
        self.tsp_ext = [0.0] * N_INT
        self.tsp_req = [dict() for _ in range(N_INT)]     # approach -> bus vid
        self.ped_wait = [[deque(), deque()] for _ in range(N_INT)]
        self.ped_cross = [[0.0, 0.0] for _ in range(N_INT)]
        self.ped_started = [False] * N_INT     # pedestrians began crossing in this green
        self.active_ev = []
        self.finished = []
        self.unfinished = []
        self.bus_trips = []
        self.ped_stats = dict(served=0, wait_sum=0.0, over30=0, jaywalks=0)
        self.ev_trips = []
        self.av_decisions = 0
        self.log = np.zeros((self.n_steps, N_INT, N_FIELDS), dtype=np.float32)
        self.last_dep_stop = {}
        self.pax_waiting = {}
        self.pax_last_t = {}
        self.rec_segments = []     # (vid, vtype, link, t_enter, t_arr, t_dep)
        self.rec_signal = []       # (step, phases, switched flags)
        self.rec_ped = []          # (step, waiting counts per junction/group)
        self._generate_demand()
        self.done = False
        return self

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------
    def incident_factor(self, L: int, t: float):
        if L == self.inc_L and self.inc_t0 <= t < self.inc_t1:
            return self.inc_sat, self.inc_speed
        return 1.0, 1.0

    def incident_active(self, L: int) -> bool:
        return L == self.inc_L and self.inc_t0 <= self.t < self.inc_t1

    def occupancy_ratio(self, L: int) -> float:
        return self.occ[L] / LINK_STORAGE[L]

    def approaching(self, i: int, a: int, horizon: float) -> int:
        """Vehicles that will reach the stop line of approach (i, a) within `horizon` s."""
        n = 0
        lim = self.t + horizon
        for v in self.moving[i * 4 + a]:
            if v.t_arr <= lim:
                n += 1
            else:
                break
        return n

    def queue_len(self, i: int, a: int) -> int:
        L = i * 4 + a
        return len(self.queue[L]) + len(self.bus_q[L]) + len(self.ev_wait[L])

    def _new_vehicle(self, t, L, vtype, dest_i, dest_h, x_first):
        self.vid_counter += 1
        dynamic = vtype == C.VT_AV and self.av_policy is not None
        path = None if dynamic else monotone_path(L // 4, dest_i, dest_h, x_first)
        return Vehicle(self.vid_counter, vtype, dest_i, dest_h, path, t)

    def _enter_link(self, v: Vehicle, L: int, t: float):
        if v.vtype == C.VT_EV:
            speed = C.FREE_SPEED * C.EV_SPEED_FACTOR * max(self.w_speed, 0.85)
        else:
            speed = C.FREE_SPEED * self.w_speed
            if L == self.inc_L and self.inc_t0 <= t < self.inc_t1:
                speed *= self.inc_speed
        ln = LINK_LEN[L]
        v.link = L
        v.t_enter = t
        v.t_arr = t + ln / speed
        v.ff += ln / C.FREE_SPEED
        v.dist += ln
        em = EM[v.vtype]
        v.co2 += em[0] * ln * 0.001
        v.nox += em[3] * ln * 0.001
        self.occ[L] += 1
        if v.vtype == C.VT_EV:
            self.moving_ev[L].append(v)
        else:
            self.moving[L].append(v)

    def candidate_headings(self, v: Vehicle, i: int):
        dx, dy = remaining_moves(i, v.dest_i)
        opts = []
        if dx != 0:
            opts.append(1 if dx > 0 else 3)
        if dy != 0:
            opts.append(2 if dy > 0 else 0)
        if not opts:
            opts.append(v.dest_h)
        return opts, dx, dy

    def congestion_level(self, L: int) -> int:
        """0 free, 1 moderate, 2 heavy, 3 incident on the link (AV routing state)."""
        if L < 0:
            return 0
        if self.incident_active(L):
            return 3
        occ = self.occ[L] / LINK_STORAGE[L]
        q = len(self.queue[L])
        if occ > 0.45 or q > 14:
            return 2
        if occ > 0.20 or q > 6:
            return 1
        return 0

    # ----------------------------------------------------------------------------------
    # Arrival at a stop line
    # ----------------------------------------------------------------------------------
    def _arrive(self, v: Vehicle, L: int):
        i = L // 4
        if v.vtype == C.VT_BUS and v.bus.stop_idx < len(v.bus.stops) \
                and v.bus.stops[v.bus.stop_idx] == i:
            self._bus_arrive_stop(v, L, i)
            return
        if v.path is not None:
            v.next_h = v.path[v.pidx]
        else:
            opts, dx, dy = self.candidate_headings(v, i)
            if len(opts) == 1 or self.av_policy is None:
                v.next_h = opts[0]
            else:
                v.next_h = self.av_policy(self, v, i, opts, dx, dy)
                self.av_decisions += 1
        self.queue[L].append(v)

    # ----------------------------------------------------------------------------------
    # Buses
    # ----------------------------------------------------------------------------------
    def _spawn_bus(self, t: float, route: str, t_planned: float | None = None):
        L, path, stops = BUS_ROUTES[route]
        self.vid_counter += 1
        v = Vehicle(self.vid_counter, C.VT_BUS, stops[-1], path[-1], list(path), t)
        hw = self.bus_headway[route]
        t = t if t_planned is None else t_planned          # timetable uses the plan
        planned_dwell = C.BUS_DOOR_S + C.BUS_BOARD_S * self.pax_rate * hw
        sched, tt = [], t
        for k in range(len(stops)):
            tt += (C.ENTRY_M if k == 0 else C.BLOCK_M) / C.FREE_SPEED + (
                C.BUS_SCHEDULE_SLACK if k > 0 else 0.0)
            sched.append(tt)
            tt += planned_dwell
        sched.append(tt + C.BUS_SCHEDULE_SLACK)            # scheduled exit
        load = int(self.rng_noise.poisson(25 * self.sc["pax_mult"]))
        v.bus = BusInfo(route[:2], route, stops, sched, min(load, C.BUS_CAPACITY),
                        self.vid_counter)
        self.buffer[L].append(v)

    def _pax_board(self, key, t):
        last = self.pax_last_t.get(key, 0.0)
        new = self.rng_pax.poisson(self.pax_rate * max(t - last, 0.0))
        self.pax_waiting[key] = self.pax_waiting.get(key, 0) + int(new)
        self.pax_last_t[key] = max(t, last)

    def _bus_arrive_stop(self, v: Vehicle, L: int, i: int):
        b = v.bus
        key = (b.direction, i)
        self._pax_board(key, v.t_arr)
        alight = int(round(b.load * C.BUS_ALIGHT_SHARE)) if b.stop_idx > 0 else 0
        b.load -= alight
        board = min(self.pax_waiting[key], C.BUS_CAPACITY - b.load)
        self.pax_waiting[key] -= board
        b.load += board
        dwell = C.BUS_DOOR_S + max(C.BUS_BOARD_S * board, C.BUS_ALIGHT_S * alight)
        b.t_ready = v.t_arr + dwell
        dev = v.t_arr - b.sched[b.stop_idx]
        b.dev_log.append(dev)
        if b.pending is not None and self.bus_policy is not None:
            self.bus_policy.learn_arrival(self, v, dev)
        b.hold_done = False
        self.bus_dwell[L].append(v)

    def bus_state_features(self, v: Vehicle, i: int, a: int):
        b = v.bus
        dev = self.t - b.sched[b.stop_idx]
        last = self.last_dep_stop.get((b.direction, i))
        hw = self.bus_headway[b.direction]
        if last is None:
            hw_state = 1
        else:
            gap = self.t - last
            hw_state = 0 if gap < 0.5 * hw else (2 if gap > 1.5 * hw else 1)
        green = int(PH_A[a] == self.phase[i])
        qc = sum(self.queue_len(i, x) for x in range(4) if PH_A[x] != PH_A[a])
        return dev, b.load, hw_state, green, qc

    def _process_bus_dwell(self, L: int, t_end: float):
        i, a = divmod(L, 4)
        keep = []
        joined = 0
        for v in self.bus_dwell[L]:
            b = v.bus
            if b.t_ready >= t_end:
                keep.append(v)
                continue
            if not b.hold_done:
                action = self.bus_policy.decide(self, v, i, a) if self.bus_policy else 0
                if action == 2:                      # hold at the stop for headway control
                    b.tsp = False
                    b.t_ready += C.BUS_HOLD_S
                    b.hold_done = True
                    key = (b.direction, i)
                    self._pax_board(key, b.t_ready)
                    extra = min(self.pax_waiting[key], C.BUS_CAPACITY - b.load)
                    self.pax_waiting[key] -= extra
                    b.load += extra
                    keep.append(v)
                    continue
                b.tsp = action == 1
            self.last_dep_stop[(b.direction, i)] = b.t_ready
            b.stop_idx += 1
            v.t_arr = b.t_ready
            v.next_h = v.path[v.pidx]
            if b.tsp:
                self.tsp_req[i][a] = v.vid
            if self.sc.get("bus_lane") and b.route == "R1":
                self.bus_q[L].append(v)
            else:
                self.queue[L].append(v)
                joined += 1
        self.bus_dwell[L] = keep
        return joined

    # ----------------------------------------------------------------------------------
    # Safety shield
    # ----------------------------------------------------------------------------------
    def shield(self, actions):
        """Return (executed actions, override codes). Codes index OVERRIDE_NAMES."""
        executed = [int(x) for x in actions]
        codes = [0] * N_INT
        self.ev_ped_cut = [False] * N_INT
        ev_req = [None] * N_INT
        for (v, L) in self.active_ev:
            i = L // 4
            if v.t_arr - self.t <= C.EV_DETECT_S and ev_req[i] is None:
                ev_req[i] = PH_A[L % 4]
        for i in range(N_INT):
            el = self.elapsed[i]
            ph = self.phase[i]
            if ev_req[i] is not None:
                if ev_req[i] != ph and el >= C.EV_MIN_GREEN:
                    executed[i], codes[i] = 1, 3
                    if self.ped_started[i] and el < C.PED_MIN_GREEN:
                        self.ev_ped_cut[i] = True     # preemption cut a pedestrian clearance
                    continue
                if ev_req[i] == ph:
                    if executed[i] == 1:
                        codes[i] = 3
                    executed[i] = 0
                    continue
            if el < C.MIN_GREEN:
                if executed[i] == 1:
                    codes[i] = 1
                executed[i] = 0
                continue
            if self.ped_started[i] and el < C.PED_MIN_GREEN:
                if executed[i] == 1:
                    codes[i] = 6
                executed[i] = 0
                continue
            req = self.tsp_req[i]
            if req:
                bus_phases = {PH_A[a] for a in req}
                if ph in bus_phases and executed[i] == 1 and self.tsp_ext[i] < C.TSP_EXT_MAX:
                    executed[i], codes[i] = 0, 4
                    self.tsp_ext[i] += C.DT
                    continue
                if ph not in bus_phases and executed[i] == 0:
                    executed[i], codes[i] = 1, 5
                    continue
            if el >= C.MAX_GREEN and executed[i] == 0:
                executed[i], codes[i] = 1, 2
        return executed, codes

    @staticmethod
    def legal_actions(elapsed: float):
        return (0,) if elapsed < C.MIN_GREEN else (0, 1)

    def legal_actions_for(self, i: int):
        """Actions the shield will actually allow at junction i (min green, pedestrian
        clearance). Learning agents choose only among these."""
        el = self.elapsed[i]
        if el < C.MIN_GREEN or (self.ped_started[i] and el < C.PED_MIN_GREEN):
            return (0,)
        return (0, 1)

    # ----------------------------------------------------------------------------------
    # Main step
    # ----------------------------------------------------------------------------------
    def step(self, actions, chosen=None, codes=None):
        """Advance DT seconds. `actions` are the executed signal actions (after shield)."""
        t0 = self.t
        DT = C.DT
        t1 = t0 + DT
        s = self.step_idx
        lost = self.lost_time
        phase, elapsed, switched = self.phase, self.elapsed, self.switched
        occ, credit = self.occ, self.credit
        queue, moving = self.queue, self.moving

        # ---------------- signals
        conflicts_ev = [0.0] * N_INT
        for i in range(N_INT):
            if actions[i] == 1:
                phase[i] = 1 - phase[i]
                elapsed[i] = DT - lost
                switched[i] = True
                self.tsp_ext[i] = 0.0
                self.ped_started[i] = False
                if getattr(self, "ev_ped_cut", None) and self.ev_ped_cut[i]:
                    conflicts_ev[i] = 0.05 * self.w_risk
            else:
                elapsed[i] += DT
                switched[i] = False
        if self.record:
            self.rec_signal.append((s, list(phase), list(switched)))

        conflicts = list(conflicts_ev)
        # dilemma-zone exposure when a phase terminates
        for i in range(N_INT):
            if not switched[i]:
                continue
            old = 1 - phase[i]
            for a in range(4):
                if PH_A[a] != old:
                    continue
                n_dz = 0
                lim = t0 + C.DILEMMA_WINDOW
                for v in moving[i * 4 + a]:
                    if v.t_arr <= lim:
                        n_dz += 1
                    else:
                        break
                if n_dz:
                    cross_q = sum(self.queue_len(i, x) for x in range(4)
                                  if PH_A[x] == phase[i])
                    p_conf = C.CONFLICT_PER_RLR * min(1.0, 0.4 + 0.06 * cross_q)
                    conflicts[i] += n_dz * C.P_RED_LIGHT_RUN * p_conf * self.w_risk

        # ---------------- new demand into entry buffers, then into the network
        arr = self.arrivals
        while arr and arr[0][0] < t1:
            (ta, L, vtype, di, dh, xf) = arr.popleft()
            self.buffer[L].append(self._new_vehicle(ta, L, vtype, di, dh, xf))
        bd = self.bus_departures
        while bd and bd[0][0] < t1:
            tb, route, tp = bd.popleft()
            self._spawn_bus(tb, route, tp)
        for L in ENTRY_LINKS:
            buf = self.buffer[L]
            stor = LINK_STORAGE[L]
            while buf and (occ[L] < stor or buf[0].vtype == C.VT_EV):
                v = buf.popleft()
                t_in = v.t_spawn if v.t_spawn > t0 else t0
                v.delay += t_in - v.t_spawn
                self._enter_link(v, L, t_in)
                if v.vtype == C.VT_EV:
                    self.active_ev.append((v, L))

        # ---------------- arrivals at stop lines
        arrivals = [0] * N_INT
        old_len = [0] * N_LINKS
        n_new = [0] * N_LINKS
        for L in range(N_LINKS):
            q = queue[L]
            old_len[L] = len(q)
            mv = moving[L]
            if mv and mv[0].t_arr < t1:
                while mv and mv[0].t_arr < t1:
                    self._arrive(mv.popleft(), L)
            if self.moving_ev[L]:
                still = []
                for v in self.moving_ev[L]:
                    if v.t_arr < t1:
                        self.ev_wait[L].append(v)
                    else:
                        still.append(v)
                self.moving_ev[L] = still
            if self.bus_dwell[L]:
                self._process_bus_dwell(L, t1)
            n_new[L] = len(q) - old_len[L]
            arrivals[L // 4] += n_new[L]

        # ---------------- pedestrians: arrivals and WALK service
        # Pedestrians may only start crossing during the first WALK_S seconds of the
        # parallel green; anyone arriving later waits for the next cycle (HCM logic).
        ped_arr = self.ped_arrivals[s]
        ped_now = [[0.0, 0.0] for _ in range(N_INT)]
        stats = self.ped_stats
        for i in range(N_INT):
            for g in range(2):
                wq = self.ped_wait[i][g]
                k = ped_arr[i][g]
                if k:
                    tm = t0 + 0.5 * DT
                    wq.extend([tm] * k)
                started = 0
                if phase[i] == 1 - g and wq:     # group 0 crosses the avenue in EW green
                    g_start = t1 - elapsed[i]    # time the current green began
                    walk_end = g_start + C.WALK_S
                    if walk_end > t0:
                        t_ok = min(t1, walk_end)
                        wsum = 0.0
                        over = 0
                        while wq and wq[0] <= t_ok:
                            ta = wq.popleft()
                            w = g_start - ta if g_start > ta else 0.0
                            wsum += w
                            if w > C.PED_PATIENCE_1:
                                over += 1
                            started += 1
                        stats["served"] += started
                        stats["wait_sum"] += wsum
                        stats["over30"] += over
                        if started:
                            self.ped_started[i] = True
                ped_now[i][g] = started + 0.6 * self.ped_cross[i][g]
        self.ped_cross = ped_now

        # ---------------- departures
        delay_s = [0.0] * N_INT
        co2_step = [0.0] * N_INT
        nox_step = [0.0] * N_INT
        stops_step = [0] * N_INT
        deps = [0] * N_INT
        blocked = [0] * N_INT
        sat_base = C.SAT_FLOW * self.w_sat
        for i in range(N_INT):
            ph = phase[i]
            g_eff = DT - (lost if switched[i] else 0.0)
            t_green0 = t1 - g_eff
            walkers = ped_now[i][1 - ph]
            down_i = DOWN[i]
            for a in range(4):
                L = i * 4 + a
                if PH_A[a] != ph:
                    credit[L] = 0.0
                    if self.ev_wait[L]:
                        delay_s[i] += DT * len(self.ev_wait[L])
                    continue
                if self.ev_wait[L]:                 # emergency vehicles go first
                    for v in self.ev_wait[L]:
                        t_dep = v.t_arr if v.t_arr > t_green0 else t_green0
                        v.delay += t_dep - v.t_arr
                        self._depart(v, L, i, v.path[v.pidx], t_dep)
                    self.ev_wait[L] = []
                bq = self.bus_q[L]
                if bq:                               # dedicated bus lane
                    k = 0
                    while bq and k < max(1, int(g_eff // 2)):
                        v = bq[0]
                        t_dep = max(v.t_arr, t_green0 + 2.0 * k)
                        if t_dep >= t1:
                            break
                        Ld = down_i[v.next_h]
                        if Ld >= 0 and occ[Ld] >= LINK_STORAGE[Ld]:
                            break
                        bq.popleft()
                        self._finish_queue_wait(v, t_dep)
                        self._depart(v, L, i, v.next_h, t_dep)
                        k += 1
                q = queue[L]
                if not q:
                    if credit[L] > 1.0:
                        credit[L] = 1.0
                    continue
                if L == self.inc_L and self.inc_t0 <= t0 < self.inc_t1:
                    rate = LINK_LANES[L] * sat_base * self.inc_sat
                else:
                    rate = LINK_LANES[L] * sat_base
                cr = credit[L] + rate * g_eff
                used = 0.0
                tc = TURN_COST[a]
                it = IS_TURN[a]
                ped_extra = C.COST_PED_TURN * min(1.0, walkers / 8.0) if walkers > 0 else 0.0
                while q:
                    v = q[0]
                    h = v.next_h
                    cost = tc[h] * TYPE_COST[v.vtype]
                    if ped_extra and it[h]:
                        cost += ped_extra
                    if cr < cost:
                        break
                    Ld = down_i[h]
                    if Ld >= 0 and occ[Ld] >= LINK_STORAGE[Ld]:
                        blocked[i] += 1
                        break
                    t_dep = t_green0 + used / rate
                    if v.t_arr > t_dep:
                        t_dep = v.t_arr
                    cr -= cost
                    used += cost
                    q.popleft()
                    self._finish_queue_wait(v, t_dep)
                    self._depart(v, L, i, h, t_dep)
                    deps[i] += 1
                credit[L] = cr if q else min(cr, 1.0)

        # ---------------- queue delay / stops / idling for vehicles still waiting
        for L in range(N_LINKS):
            q = queue[L]
            n = len(q)
            i = L // 4
            if n:
                departed = old_len[L] + n_new[L] - n
                remaining_old = old_len[L] - departed
                if remaining_old < 0:
                    remaining_old = 0
                d = DT * remaining_old
                for k in range(remaining_old, n):
                    v = q[k]
                    d += t1 - v.t_arr
                    if not v.stopped:
                        v.stopped = True
                        v.stops += 1
                        em = EM[v.vtype]
                        co2_step[i] += em[2]
                        nox_step[i] += em[5]
                        stops_step[i] += 1
                delay_s[i] += d
                co2_step[i] += 0.9 * d
                nox_step[i] += 0.001 * d
            bq = self.bus_q[L]
            if bq:
                for v in bq:
                    if not v.stopped:
                        v.stopped = True
                        v.stops += 1
                delay_s[i] += DT * len(bq)

        # ---------------- pedestrian waiting and jaywalking
        ped_delay = [0.0] * N_INT
        jay = [0] * N_INT
        p1 = 1.0 - math.exp(-C.PED_HAZARD_1 * DT)
        p2 = 1.0 - math.exp(-C.PED_HAZARD_2 * DT)
        for i in range(N_INT):
            for g in range(2):
                wq = self.ped_wait[i][g]
                if not wq:
                    continue
                n1 = n2 = 0
                for ta in wq:
                    w = t1 - ta
                    if w > C.PED_PATIENCE_2:
                        n2 += 1
                    elif w > C.PED_PATIENCE_1:
                        n1 += 1
                    else:
                        break
                ped_delay[i] += DT * len(wq)
                if n1 or n2:
                    exp_j = n1 * p1 + n2 * p2
                    flow = min(1.0, deps[i] / 6.0)
                    conflicts[i] += exp_j * C.CONFLICT_PER_JAYWALK * (0.3 + 0.7 * flow) \
                        * self.w_risk
                    k = int(self.rng_noise.binomial(n2, p2)) + \
                        int(self.rng_noise.binomial(n1, p1))
                    for _ in range(min(k, len(wq))):
                        ta = wq.popleft()
                        stats["jaywalks"] += 1
                        stats["served"] += 1
                        stats["over30"] += 1
                        stats["wait_sum"] += t1 - ta
                        jay[i] += 1
        for i in range(N_INT):
            conflicts[i] += stops_step[i] * C.CONFLICT_PER_STOP * self.w_risk

        # ---------------- emergency vehicles still in the network (for the shield)
        self.active_ev = [(v, v.link) for (v, L) in self.active_ev if v.link >= 0]
        ev_flag = [0] * N_INT
        for (v, L) in self.active_ev:
            ev_flag[L // 4] = 1

        # ---------------- rewards + log
        rewards = [-(C.W_DELAY * delay_s[i] + C.W_PED * ped_delay[i]
                     + C.W_CONFLICT * conflicts[i] + C.W_CO2 * co2_step[i]) / C.REWARD_SCALE
                   for i in range(N_INT)]
        rows = []
        for i in range(N_INT):
            rows.append([phase[i], elapsed[i], self.queue_len(i, 0), self.queue_len(i, 1),
                         self.queue_len(i, 2), self.queue_len(i, 3), arrivals[i], deps[i],
                         len(self.ped_wait[i][0]), len(self.ped_wait[i][1]), jay[i],
                         conflicts[i], co2_step[i], nox_step[i], delay_s[i], ped_delay[i],
                         rewards[i], actions[i] if chosen is None else chosen[i], actions[i],
                         0 if codes is None else codes[i], ev_flag[i],
                         1 if self.tsp_req[i] else 0, blocked[i], stops_step[i]])
        self.log[s] = rows
        if self.record:
            self.rec_ped.append([[len(self.ped_wait[i][0]), len(self.ped_wait[i][1])]
                                 for i in range(N_INT)])
        self.t = t1
        self.step_idx += 1
        if self.step_idx >= self.n_steps:
            self.done = True
            self._close_episode()
        return rewards

    def _finish_queue_wait(self, v: Vehicle, t_dep: float):
        w = t_dep - v.t_arr
        if w < 0.0:
            w = 0.0
        v.delay += w
        if v.stopped:
            em = EM[v.vtype]
            v.idle += w
            v.co2 += em[1] * w + em[2]
            v.nox += em[4] * w + em[5]
            v.stopped = False

    def _depart(self, v: Vehicle, L: int, i: int, heading: int, t_dep: float):
        self.occ[L] -= 1
        if self.record:
            self.rec_segments.append((v.vid, v.vtype, L, round(v.t_enter, 1),
                                      round(v.t_arr, 1), round(t_dep, 1)))
        if v.vtype == C.VT_BUS:
            self.tsp_req[i].pop(L % 4, None)
            v.bus.last_dep = t_dep
        if v.path is not None:
            v.pidx += 1
        Ld = DOWN[i][heading]
        if Ld < 0:
            v.link = -1
            self._finish_trip(v, t_dep)
            return
        self._enter_link(v, Ld, t_dep)

    def _finish_trip(self, v: Vehicle, t_exit: float):
        if v.av_pending is not None and self.av_learn is not None:
            self.av_learn(self, v, t_exit, terminal=True)
        if v.vtype == C.VT_BUS:
            b = v.bus
            dev = t_exit - b.sched[-1]
            b.dev_log.append(dev)
            if b.pending is not None and self.bus_policy is not None:
                self.bus_policy.learn_arrival(self, v, dev, terminal=True)
            self.bus_trips.append(dict(trip=b.trip_id, route=b.direction,
                                       t_start=v.t_spawn, t_end=t_exit,
                                       mean_abs_dev=float(np.mean(np.abs(b.dev_log))),
                                       final_dev=float(dev), load=b.load, stops=v.stops,
                                       delay=float(v.delay)))
        if v.vtype == C.VT_EV:
            self.ev_trips.append(dict(t_start=v.t_spawn, delay=float(v.delay),
                                      tt=float(t_exit - v.t_spawn)))
        self.finished.append((v.vtype, v.t_spawn, t_exit, v.delay, v.stops, v.idle,
                              v.dist, v.co2, v.nox, v.ff, 1))

    def _close_episode(self):
        """Count vehicles still in the network with their delay so far (no survivor bias)."""
        t_end = self.t
        out = []
        for L in range(N_LINKS):
            for v in self.queue[L]:
                w = max(0.0, t_end - v.t_arr)
                out.append((v.vtype, v.t_spawn, t_end, v.delay + w, v.stops,
                            v.idle + w, v.dist, v.co2 + EM[v.vtype][1] * w, v.nox, v.ff, 0))
            for v in list(self.moving[L]) + list(self.bus_q[L]) + self.ev_wait[L] \
                    + self.bus_dwell[L] + self.moving_ev[L]:
                out.append((v.vtype, v.t_spawn, t_end, v.delay, v.stops, v.idle,
                            v.dist, v.co2, v.nox, v.ff, 0))
            for v in self.buffer[L]:
                out.append((v.vtype, v.t_spawn, t_end, t_end - v.t_spawn, 0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0))
            if self.record:
                for v in list(self.queue[L]) + list(self.moving[L]):
                    self.rec_segments.append((v.vid, v.vtype, L, round(v.t_enter, 1),
                                              round(v.t_arr, 1), round(t_end + 60.0, 1)))
        self.unfinished = out

    # ----------------------------------------------------------------------------------
    # Observation for a signal agent
    # ----------------------------------------------------------------------------------
    def signal_obs(self, i: int):
        """Local observation of junction i used by every signal controller.

        Returns (phase, green elapsed, queue on green, queue on red, vehicles arriving on
        green within 6 s, pedestrian flag, spillback flag, per-approach queues)."""
        q = [self.queue_len(i, a) for a in range(4)]
        ph = self.phase[i]
        q_ns, q_ew = q[0] + q[2], q[1] + q[3]
        qg, qr = (q_ns, q_ew) if ph == 0 else (q_ew, q_ns)
        arr_g = 0
        lim = self.t + 6.0
        for a in range(4):
            if PH_A[a] == ph:
                for v in self.moving[i * 4 + a]:
                    if v.t_arr <= lim:
                        arr_g += 1
                    else:
                        break
        # pedestrians waiting > 20 s in the group that is red for them now (group == phase)
        wq = self.ped_wait[i][ph]
        ped_flag = 1 if (wq and self.t - wq[0] > 20.0) else 0
        spill = 0
        for a in range(4):
            if PH_A[a] == ph:
                Ld = DOWN[i][(a + 2) % 4]
                if Ld >= 0 and self.occ[Ld] > 0.85 * LINK_STORAGE[Ld]:
                    spill = 1
        return ph, self.elapsed[i], qg, qr, arr_g, ped_flag, spill, q

    # ----------------------------------------------------------------------------------
    # KPIs
    # ----------------------------------------------------------------------------------
    def kpis(self) -> dict:
        rows = [r for r in (self.finished + self.unfinished) if r[1] >= C.WARMUP_S]
        arr = np.array(rows, dtype=float) if rows else np.zeros((0, 11))
        gen = arr[arr[:, 0] != C.VT_EV] if len(arr) else arr
        fin = gen[gen[:, 10] == 1] if len(gen) else gen
        log = self.log
        w0 = int(C.WARMUP_S / C.DT)
        conflicts = float(log[w0:, :, F["conflicts"]].sum())
        n_gen = max(1.0, float(len(gen)))
        tt = fin[:, 2] - fin[:, 1] if len(fin) else np.array([0.0])
        km = gen[:, 6].sum() / 1000.0 if len(gen) else 0.0
        av = gen[gen[:, 0] == C.VT_AV] if len(gen) else gen
        hu = gen[gen[:, 0] == C.VT_CAR] if len(gen) else gen
        bt = self.bus_trips
        out = dict(
            vehicles=int(len(gen)),
            completed=int(len(fin)),
            throughput_vph=float(len(fin)) / ((C.EPISODE_S - C.WARMUP_S) / 3600.0),
            avg_delay_s=float(gen[:, 3].mean()) if len(gen) else 0.0,
            avg_travel_time_s=float(tt.mean()),
            avg_stops=float(gen[:, 4].mean()) if len(gen) else 0.0,
            idle_hours=float(gen[:, 5].sum() / 3600.0) if len(gen) else 0.0,
            co2_kg=float(gen[:, 7].sum() / 1000.0) if len(gen) else 0.0,
            nox_g=float(gen[:, 8].sum()) if len(gen) else 0.0,
            co2_g_per_km=float(gen[:, 7].sum() / max(1.0, km)) if len(gen) else 0.0,
            conflicts=conflicts,
            conflicts_per_1000=1000.0 * conflicts / n_gen,
            jaywalks=int(self.ped_stats["jaywalks"]),
            ped_avg_wait_s=self.ped_stats["wait_sum"] / max(1, self.ped_stats["served"]),
            ped_over30_pct=100.0 * self.ped_stats["over30"] / max(1, self.ped_stats["served"]),
            ev_avg_delay_s=float(np.mean([e["delay"] for e in self.ev_trips]))
            if self.ev_trips else 0.0,
            ev_trips=len(self.ev_trips),
            bus_abs_dev_s=float(np.mean([b["mean_abs_dev"] for b in bt])) if bt else 0.0,
            bus_on_time_pct=float(100.0 * np.mean([abs(b["final_dev"]) <= 60.0 for b in bt]))
            if bt else 0.0,
            bus_avg_delay_s=float(np.mean([b["delay"] for b in bt])) if bt else 0.0,
            av_avg_delay_s=float(av[:, 3].mean()) if len(av) else 0.0,
            human_avg_delay_s=float(hu[:, 3].mean()) if len(hu) else 0.0,
            switches=int(log[:, :, F["executed"]].sum()),
            blocked=float(log[:, :, F["blocked"]].sum()),
        )
        out["safety_score"] = 100.0 / (1.0 + out["conflicts_per_1000"] / 10.0)
        return out
