"""
config.py - network geometry, physical parameters and the six evaluation scenarios.

Every number that drives the simulator lives here so that the requirement
specification, the notebook and the dashboard all quote the same values.
Sources are noted next to each parameter; values marked ASSUMPTION are
modelling choices that the group should be ready to defend.
"""
from __future__ import annotations

import math

# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------
DT = 5.0              # control / simulation step (s). Signals decide every 5 s.
EPISODE_S = 1800.0    # one episode = 30 simulated minutes
WARMUP_S = 300.0      # KPIs ignore trips that start in the first 5 minutes

# --------------------------------------------------------------------------------------
# Network: a 3 x 3 grid of signalised junctions (J1..J9, row-major, north to south),
# block structure modelled on Downtown Dubai / Business Bay (approx. 400 m blocks).
# Rows run East-West ("Streets"), columns run North-South ("Avenues").
# Row 1 (Street 2) is the main arterial and the bus-priority corridor.
# --------------------------------------------------------------------------------------
GRID_N = 3
N_INT = GRID_N * GRID_N
BLOCK_M = 400.0       # internal link length (m)
ENTRY_M = 300.0       # boundary entry link length (m)
FREE_SPEED = 60 / 3.6 # free-flow speed 60 km/h (urban arterial)

STREET_NAMES = ["Street 1", "Street 2 (corridor)", "Street 3"]      # E-W rows
AVENUE_NAMES = ["Avenue A", "Avenue B", "Avenue C"]                  # N-S columns
JUNCTION_NAMES = [f"J{i + 1}" for i in range(N_INT)]
CORRIDOR_ROW = 1

# Headings / approach sides: 0 = North, 1 = East, 2 = South, 3 = West
HEADING_NAMES = ["N", "E", "S", "W"]
DIR_DC = [0, 1, 0, -1]   # column change when moving in heading h
DIR_DR = [-1, 0, 1, 0]   # row change (rows increase southwards)
# An approach is named after the side vehicles come FROM. Vehicles on approach a travel
# with heading (a + 2) % 4. Phase 0 = North-South green, phase 1 = East-West green.
PHASE_OF_APPROACH = [0, 1, 0, 1]
PHASE_NAMES = ["NS green", "EW green"]

# --------------------------------------------------------------------------------------
# Signal timing constraints (safety shield, applied to every controller)
# --------------------------------------------------------------------------------------
LOST_TIME = 4.0        # 3 s yellow + 1 s all-red per phase change
MIN_GREEN = 10.0       # vehicle minimum green; switching is masked before this
WALK_S = 7.0           # WALK indication at the start of each green (MUTCD minimum 7 s)
PED_CLEAR_S = 12.0     # flashing DON'T WALK: ~14 m crossing / 1.2 m/s
PED_MIN_GREEN = WALK_S + PED_CLEAR_S   # green cannot end while pedestrians are still crossing
MAX_GREEN = 90.0       # forced change after 90 s of green
EV_DETECT_S = 15.0     # emergency vehicle detected 15 s before the stop line
EV_MIN_GREEN = 5.0     # preemption may truncate the running green down to 5 s
TSP_EXT_MAX = 15.0     # transit signal priority: max green extension (s)

# --------------------------------------------------------------------------------------
# Capacity. Base saturation flow 1,800 veh/h/lane (HCM base is 1,900 pc/h/ln; 1,800 is a
# common field value for mixed urban traffic).
# --------------------------------------------------------------------------------------
SAT_FLOW = 1800.0 / 3600.0   # veh / s / lane
JAM_SPACING = 7.5            # m of road per queued vehicle
LANES_CORRIDOR = 3           # Street 2, each direction
LANES_DEFAULT = 2            # all other approaches

# Departure "cost" in saturation-flow units (1.0 = one through car)
COST_THROUGH = 1.0
COST_RIGHT = 1.15
COST_LEFT = 1.40             # permissive left turn yields to opposing traffic
COST_PED_TURN = 0.60         # extra cost for a turning vehicle when pedestrians are crossing
COST_TRUCK = 1.8             # heavy vehicle passenger-car equivalent
COST_AV = 0.85               # ASSUMPTION: cooperative platooning (CACC) shortens AV headways

# --------------------------------------------------------------------------------------
# Vehicle mix
# --------------------------------------------------------------------------------------
VT_CAR, VT_AV, VT_TRUCK, VT_BUS, VT_EV = 0, 1, 2, 3, 4
VTYPE_NAMES = ["car", "av", "truck", "bus", "emergency"]
AV_SHARE = 0.25              # autonomous / connected vehicle penetration
TRUCK_SHARE = 0.05

# --------------------------------------------------------------------------------------
# Emission factors (per vehicle type). CO2 cruise for cars: EPA "typical passenger vehicle"
# 400 g/mile = 249 g/km. Idling: Argonne National Laboratory, a car wastes ~0.3 gal/h
# (x 8,887 g CO2/gal = 0.74 g/s), a heavy vehicle ~1 gal/h diesel (x 10,180 g/gal = 2.8 g/s).
# Stop penalty: kinetic energy of a 1.5 t car at 50 km/h (145 kJ) at ~24 % engine
# efficiency = ~18 mL fuel = ~41 g CO2; same method for a 15 t bus = ~290 g.
# NOx factors are order-of-magnitude values (Euro 6 petrol car / Euro VI diesel bus).
# --------------------------------------------------------------------------------------
EMISSION = {
    #            co2_km  co2_idle co2_stop  nox_km  nox_idle  nox_stop
    VT_CAR:   (249.0,   0.74,    41.0,     0.06,   0.0005,   0.010),
    VT_AV:    (237.0,   0.74,    25.0,     0.06,   0.0005,   0.006),   # eco-driving, smoother starts
    VT_TRUCK: (900.0,   2.80,   250.0,     2.00,   0.0200,   0.100),
    VT_BUS:   (1100.0,  2.80,   290.0,     1.00,   0.0150,   0.080),
    VT_EV:    (400.0,   1.20,    60.0,     0.50,   0.0050,   0.030),
}

# --------------------------------------------------------------------------------------
# Safety model (expected conflicts; a conflict is a near-miss that could become a crash)
# --------------------------------------------------------------------------------------
P_RED_LIGHT_RUN = 0.08       # ASSUMPTION: share of dilemma-zone vehicles that enter on late yellow/red
DILEMMA_WINDOW = 3.0         # s: vehicles this close to the stop line when the phase ends
CONFLICT_PER_RLR = 0.30      # conflict probability per red-light runner (scaled by cross traffic)
CONFLICT_PER_STOP = 0.002    # rear-end exposure per stop event
PED_PATIENCE_1 = 30.0        # HCM pedestrian LOS: non-compliance becomes likely beyond 30 s
PED_PATIENCE_2 = 60.0        # ... and very likely beyond 60 s (LOS F)
PED_HAZARD_1 = 0.010         # jaywalking hazard per second of waiting, 30-60 s
PED_HAZARD_2 = 0.030         # jaywalking hazard per second of waiting, > 60 s
CONFLICT_PER_JAYWALK = 0.15  # base conflict probability per jaywalker

# --------------------------------------------------------------------------------------
# Weather. Rain: saturation flow -6 %, speed -10 % (Perrin et al. 2002, reported in
# Sullivan et al. 2010, UTCA Report 07211); crash rate +71 % (Qiu & Nixon 2008).
# Sandstorm: HCM low-visibility capacity reductions of 10-12 % (HCM Exhibit 10-15 via
# FHWA-HOP-13-042); speed -20 % and crash multiplier 2.0 are ASSUMPTIONS.
# --------------------------------------------------------------------------------------
WEATHER_NAMES = ["Clear", "Rain", "Sandstorm"]
WEATHER = {
    #              sat_factor speed_factor risk_mult ped_factor demand_factor lost_time_add
    "Clear":     (1.00,      1.00,        1.00,     1.00,      1.00,         0.0),
    "Rain":      (0.94,      0.90,        1.71,     0.60,      0.95,         0.2),
    "Sandstorm": (0.88,      0.80,        2.00,     0.30,      0.85,         0.4),
}

# --------------------------------------------------------------------------------------
# Buses. Two routes: R1 along Street 2 (both directions), R2 along Avenue B (both
# directions). Near-side stops at the three junctions of each route.
# Dwell: 8 s door time + max(2.5 s per boarding, 1.5 s per alighting) (TCQSM ranges).
# --------------------------------------------------------------------------------------
BUS_CAPACITY = 80
BUS_DOOR_S = 8.0
BUS_BOARD_S = 2.5
BUS_ALIGHT_S = 1.5
BUS_ALIGHT_SHARE = 0.30
BUS_HOLD_S = 20.0
BUS_SCHEDULE_SLACK = 12.0     # planned signal delay per junction in the timetable (s)
BUS_DEPART_SD = 45.0          # s, deviation of buses arriving from upstream of the network
BUS_PAX_PER_MIN = 2.0         # passengers arriving at each stop per minute (normal)

# --------------------------------------------------------------------------------------
# Emergency vehicles
# --------------------------------------------------------------------------------------
EV_SPEED_FACTOR = 1.3

# --------------------------------------------------------------------------------------
# Demand (vehicles per hour entering at each boundary entry, before scenario multipliers)
# --------------------------------------------------------------------------------------
DEMAND_CORRIDOR = 950.0       # Street 2 entries (east and west ends)
DEMAND_STREET = 520.0         # Streets 1 and 3 entries
DEMAND_AVENUE = 560.0         # Avenue entries (north and south ends)
PED_PER_HOUR = 150.0          # pedestrians per hour per crossing group per junction

# --------------------------------------------------------------------------------------
# Scenarios. Each scenario is a dict of modifiers applied by traffic_env.TrafficEnv.
# --------------------------------------------------------------------------------------
SCENARIOS = {
    "normal": dict(
        label="Normal flow", weather="Clear", demand_mult=1.0, profile="flat",
        ped_mult=1.0, ev_per_episode=3, bus_headway_s=360, bus_lane=False,
        pax_mult=1.0, incident=None, event=None, corridor_mult=1.0,
        description="Typical off-peak weekday: moderate demand, clear weather, "
                    "buses every 6 minutes, ~3 emergency calls per 30 minutes."),
    "rush_hour": dict(
        label="Rush hour", weather="Clear", demand_mult=1.30, profile="peak",
        ped_mult=1.3, ev_per_episode=3, bus_headway_s=360, bus_lane=False,
        pax_mult=1.8, incident=None, event=None, corridor_mult=1.15,
        description="Morning peak: demand ramps up to ~1.5x normal mid-episode, heavier "
                    "eastbound flow on the Street 2 corridor, crowded buses."),
    "incident": dict(
        label="Incident / blockage", weather="Clear", demand_mult=1.05, profile="flat",
        ped_mult=1.0, ev_per_episode=6, bus_headway_s=360, bus_lane=False,
        pax_mult=1.0, event=None, corridor_mult=1.0,
        incident=dict(junction=4, approach=3, t_start=300.0, t_end=1500.0,
                      sat_factor=0.25, speed_factor=0.4),
        description="A crash blocks two of three eastbound lanes on Street 2 just before J5 "
                    "from minute 5 to minute 25 (capacity -75 %); extra emergency vehicles "
                    "respond."),
    "rain": dict(
        label="Rain", weather="Rain", demand_mult=1.0, profile="flat",
        ped_mult=1.0, ev_per_episode=4, bus_headway_s=360, bus_lane=False,
        pax_mult=1.2, incident=None, event=None, corridor_mult=1.0,
        description="Rain: saturation flow -6 %, speeds -10 %, crash risk x1.71, fewer "
                    "pedestrians."),
    "sandstorm": dict(
        label="Sandstorm", weather="Sandstorm", demand_mult=1.0, profile="flat",
        ped_mult=1.0, ev_per_episode=4, bus_headway_s=360, bus_lane=False,
        pax_mult=1.0, incident=None, event=None, corridor_mult=1.0,
        description="Sandstorm (shamal): visibility-driven capacity loss -12 %, speeds -20 %, "
                    "crash risk x2.0, few pedestrians, 15 % of trips cancelled."),
    "event_surge": dict(
        label="Public event surge", weather="Clear", demand_mult=1.0, profile="flat",
        ped_mult=1.0, ev_per_episode=3, bus_headway_s=360, bus_lane=False,
        pax_mult=1.5, incident=None, corridor_mult=1.0,
        event=dict(junctions=[5, 8, 4], t_start=480.0, t_end=1500.0,
                   veh_mult=2.6, ped_mult=8.0, entries=[(5, 1), (8, 1)]),
        description="An arena event ends at minute 8: a surge of cars leaves from the east "
                    "side (J6/J9) and thousands of pedestrians cross at J5, J6 and J9."),
    "bus_corridor": dict(
        label="Bus-priority corridor", weather="Clear", demand_mult=1.10, profile="flat",
        ped_mult=1.0, ev_per_episode=3, bus_headway_s=180, bus_lane=True,
        pax_mult=2.0, incident=None, event=None, corridor_mult=1.0,
        description="Street 2 runs a bus every 3 minutes in a dedicated bus lane with "
                    "transit signal priority available; high passenger demand."),
}
SCENARIO_ORDER = ["normal", "rush_hour", "incident", "rain", "sandstorm",
                  "event_surge", "bus_corridor"]
# The brief lists six scenario families; "Rain / sandstorm" is run as two variants.
SCENARIO_FAMILY = {"normal": "Normal flow", "rush_hour": "Rush hour",
                   "incident": "Incident/blockage", "rain": "Rain/sandstorm",
                   "sandstorm": "Rain/sandstorm", "event_surge": "Public event surge",
                   "bus_corridor": "Bus-priority corridor"}


def demand_profile(kind: str, t: float, T: float = EPISODE_S) -> float:
    """Time-of-day multiplier inside an episode."""
    if kind == "peak":
        # rises from 1.0 to ~1.45 at the middle of the episode and eases off
        return 1.0 + 0.45 * math.sin(math.pi * min(max(t / T, 0.0), 1.0))
    return 1.0


# --------------------------------------------------------------------------------------
# Reward weights (delay-equivalent seconds). Used by the signal agents.
# --------------------------------------------------------------------------------------
W_DELAY = 1.0            # per vehicle-second of queueing
W_PED = 0.5              # per pedestrian-second of waiting
W_CONFLICT = 300.0       # one expected conflict ~ 5 vehicle-minutes of delay
W_CO2 = 0.20             # per gram of CO2 from idling and stop-and-go
REWARD_SCALE = 100.0

# --------------------------------------------------------------------------------------
# Default RL hyper-parameters
# --------------------------------------------------------------------------------------
RL = dict(
    signal=dict(alpha=0.10, gamma=0.90, eps_start=0.30, eps_end=0.02),
    av=dict(alpha=0.10, gamma=0.98, eps_start=0.30, eps_end=0.02, coop_beta=1.0),
    bus=dict(alpha=0.10, gamma=0.90, eps_start=0.30, eps_end=0.02),
)
