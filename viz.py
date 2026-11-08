"""
viz.py - Plotly figures for the dashboard (one visual system for every chart).

Colour roles (validated for colour-vision deficiency, adjacent order):
  Fixed-time #eda100 | Actuated #1baf7a | Q-learning #2a78d6 | SARSA #eb6834
Scatter plots never show all four at once: results are indexed to the Fixed-time
baseline (origin), so only three policies appear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as C

POLICY_COLORS = {"Fixed-time": "#eda100", "Actuated": "#1baf7a",
                 "Q-learning": "#2a78d6", "SARSA": "#eb6834"}
POLICY_SYMBOLS = {"Fixed-time": "square", "Actuated": "diamond", "Q-learning": "circle",
                  "SARSA": "triangle-up"}
SURFACE = "#fcfcfb"
GRID = "#e8e6df"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a887f"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
NODE_COLORS = {"chance": "#cde2fb", "decision": "#fde3c8", "utility": "#d7f0e3"}
NODE_EDGES = {"chance": "#256abf", "decision": "#c2591f", "utility": "#16865f"}


def base_layout(fig, height=360, title=None, legend=True, margin=None):
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=13,
                  color=INK),
        title=dict(text=title, font=dict(size=15, color=INK), x=0.0, xanchor="left")
        if title else None,
        margin=margin or dict(l=10, r=10, t=46 if title else 16, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
                    font=dict(color=INK2, size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(color=INK)),
        bargap=0.35, bargroupgap=0.08,
    )
    try:
        fig.update_layout(barcornerradius=4)
    except Exception:          # older plotly
        pass
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK2),
                     title_font=dict(color=INK2), zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor=GRID, tickfont=dict(color=INK2),
                     title_font=dict(color=INK2), zeroline=False, automargin=True)
    return fig


# ======================================================================================
# City overview: network map
# ======================================================================================
def _jxy(i):
    r, c = divmod(i, C.GRID_N)
    return c, -r


def network_map(queues_step, phases_step, title=None, incident=None, show_values=True,
                height=470):
    """queues_step: [9][4] queue per approach; phases_step: [9] current phase."""
    fig = go.Figure()
    # roads
    for r in range(3):
        fig.add_trace(go.Scatter(x=[-0.6, 2.6], y=[-r, -r], mode="lines",
                                 line=dict(color="#c9c6bc", width=16 if r == 1 else 11),
                                 hoverinfo="skip", showlegend=False))
    for c in range(3):
        fig.add_trace(go.Scatter(x=[c, c], y=[0.6, -2.6], mode="lines",
                                 line=dict(color="#c9c6bc", width=11), hoverinfo="skip",
                                 showlegend=False))
    # approach queues as short coloured stubs
    vmax = 30.0
    dirs = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}   # approach side offsets
    xs, ys, cols, txt = [], [], [], []
    for i in range(9):
        x0, y0 = _jxy(i)
        for a in range(4):
            q = queues_step[i][a]
            dx, dy = dirs[a]
            lat = {0: (-0.06, 0), 1: (0, 0.06), 2: (0.06, 0), 3: (0, -0.06)}[a]
            xs.append(x0 + dx * 0.24 + lat[0])
            ys.append(y0 + dy * 0.24 + lat[1])
            cols.append(min(q, vmax))
            txt.append(f"{C.JUNCTION_NAMES[i]} approach from {C.HEADING_NAMES[a]}<br>"
                       f"queue {q:.0f} veh")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", hovertext=txt, hoverinfo="text", showlegend=False,
        marker=dict(size=[10 + min(q, vmax) * 0.9 for q in cols], color=cols,
                    colorscale=[[0, SEQ_BLUE[0]], [0.5, SEQ_BLUE[3]], [1, SEQ_BLUE[6]]],
                    cmin=0, cmax=vmax, line=dict(color=SURFACE, width=2),
                    colorbar=dict(title=dict(text="queue<br>(veh)", font=dict(size=11)),
                                  thickness=10, len=0.6, tickfont=dict(size=10)))))
    # junction phase badges
    for i in range(9):
        x0, y0 = _jxy(i)
        ph = phases_step[i]
        fig.add_trace(go.Scatter(
            x=[x0], y=[y0], mode="markers+text",
            marker=dict(size=26, color="#ffffff", line=dict(color="#4a4d52", width=1.5),
                        symbol="square"),
            text=["NS" if ph == 0 else "EW"], textfont=dict(size=10, color=INK),
            hovertext=f"{C.JUNCTION_NAMES[i]}: {C.PHASE_NAMES[ph]}", hoverinfo="text",
            showlegend=False))
        fig.add_annotation(x=x0 + 0.17, y=y0 + 0.17, text=C.JUNCTION_NAMES[i],
                           showarrow=False, font=dict(size=11, color=INK2))
    if incident:
        i = incident["junction"]
        x0, y0 = _jxy(i)
        fig.add_trace(go.Scatter(x=[x0 - 0.5], y=[y0 - 0.06], mode="markers+text",
                                 marker=dict(symbol="triangle-up", size=16, color="#d03b3b"),
                                 text=["incident"], textposition="bottom center",
                                 textfont=dict(size=11, color=INK2),
                                 hoverinfo="skip", showlegend=False))
    for r, name in enumerate(C.STREET_NAMES):
        fig.add_annotation(x=-0.66, y=-r + 0.1, text=name.split(" (")[0], showarrow=False,
                           xanchor="left", font=dict(size=10, color=MUTED))
    for c, name in enumerate(C.AVENUE_NAMES):
        fig.add_annotation(x=c + 0.04, y=0.62, text=name, showarrow=False, xanchor="left",
                           font=dict(size=10, color=MUTED))
    base_layout(fig, height=height, title=title, legend=False)
    fig.update_xaxes(visible=False, range=[-0.75, 2.75])
    fig.update_yaxes(visible=False, range=[-2.75, 0.75], scaleanchor="x", scaleratio=1)
    return fig


# ======================================================================================
# Generic charts
# ======================================================================================
def policy_bars(df, metric, label, policies=None, title=None, height=330, fmt=".1f"):
    policies = policies or [p for p in POLICY_COLORS if p in df.policy.unique()]
    g = df.groupby(["scenario", "policy"])[metric].agg(["mean", "std"]).reset_index()
    order = [s for s in C.SCENARIO_ORDER if s in g.scenario.unique()]
    fig = go.Figure()
    for p in policies:
        sub = g[g.policy == p].set_index("scenario").reindex(order)
        fig.add_trace(go.Bar(
            name=p, x=[C.SCENARIOS[s]["label"] for s in order], y=sub["mean"],
            marker=dict(color=POLICY_COLORS[p], line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", array=sub["std"], color="#b9b7ae", thickness=1,
                         width=3),
            hovertemplate=f"%{{x}}<br>{p}: %{{y:{fmt}}} {label}<extra></extra>"))
    base_layout(fig, height=height, title=title)
    fig.update_yaxes(title_text=label)
    return fig


def improvement_heatmap(df, metric, higher_is_better=False, title=None, height=320):
    """% change vs Fixed-time per scenario x policy (diverging, blue = better)."""
    g = df.groupby(["scenario", "policy"])[metric].mean().unstack()
    order = [s for s in C.SCENARIO_ORDER if s in g.index]
    base = g["Fixed-time"]
    pols = ["Actuated", "Q-learning", "SARSA"]
    z, text = [], []
    for s in order:
        row, trow = [], []
        for p in pols:
            v = 100.0 * (g.loc[s, p] - base[s]) / base[s] if base[s] else 0.0
            good = -v if not higher_is_better else v
            row.append(good)
            trow.append(f"{v:+.1f}%")
        z.append(row)
        text.append(trow)
    lim = max(5.0, float(np.nanmax(np.abs(z))))
    fig = go.Figure(go.Heatmap(
        z=z, x=pols, y=[C.SCENARIOS[s]["label"] for s in order], text=text,
        texttemplate="%{text}", textfont=dict(size=12, color=INK),
        colorscale=[[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]],
        zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="better →", font=dict(size=11)), thickness=10,
                      tickfont=dict(size=10)),
        hovertemplate="%{y} · %{x}: %{text} vs Fixed-time<extra></extra>"))
    base_layout(fig, height=height, title=title, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def tradeoff_scatter(df, scenario=None, height=420, title=None):
    """Delay change (x) vs conflict change (y) relative to Fixed-time; bubble = CO2 change."""
    g = df.groupby(["scenario", "policy"])[["avg_delay_s", "conflicts_per_1000", "co2_kg"]] \
        .mean().reset_index()
    rows = []
    for s, sub in g.groupby("scenario"):
        b = sub[sub.policy == "Fixed-time"].iloc[0]
        for _, r in sub.iterrows():
            if r.policy == "Fixed-time":
                continue
            rows.append(dict(scenario=s, policy=r.policy,
                             d_delay=100 * (r.avg_delay_s - b.avg_delay_s) / b.avg_delay_s,
                             d_conf=100 * (r.conflicts_per_1000 - b.conflicts_per_1000)
                             / b.conflicts_per_1000,
                             d_co2=100 * (r.co2_kg - b.co2_kg) / b.co2_kg))
    t = pd.DataFrame(rows)
    if scenario:
        t = t[t.scenario == scenario]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#b9b7ae", width=1))
    fig.add_vline(x=0, line=dict(color="#b9b7ae", width=1))
    for p in ["Actuated", "Q-learning", "SARSA"]:
        sub = t[t.policy == p]
        fig.add_trace(go.Scatter(
            x=sub.d_delay, y=sub.d_conf, mode="markers", name=p,
            marker=dict(color=POLICY_COLORS[p], symbol=POLICY_SYMBOLS[p],
                        size=np.clip(12 + (-sub.d_co2) * 1.5, 9, 30),
                        line=dict(color=SURFACE, width=2)),
            customdata=np.stack([sub.scenario.map(lambda s: C.SCENARIOS[s]["label"]),
                                 sub.d_co2], axis=-1) if len(sub) else None,
            hovertemplate=(f"<b>{p}</b> · %{{customdata[0]}}<br>delay %{{x:+.1f}}%<br>"
                           "conflicts %{y:+.1f}%<br>CO₂ %{customdata[1]:+.1f}%<extra></extra>")))
    fig.add_annotation(x=0, y=0, text="Fixed-time baseline", showarrow=False, xanchor="left",
                       yanchor="bottom", font=dict(size=11, color=MUTED), xshift=4, yshift=4)
    base_layout(fig, height=height, title=title)
    fig.update_xaxes(title_text="Delay vs fixed-time (%), left = better",
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text="Conflicts vs fixed-time (%), lower = safer")
    return fig, t


def learning_curves(curves, metric="avg_delay_s", stage="pre-train", scenario=None,
                    window=15, height=340, title=None, label="Average delay (s/veh)"):
    fig = go.Figure()
    for p in ["Q-learning", "SARSA"]:
        sub = curves[(curves.policy == p) & (curves.stage == stage)]
        if scenario:
            sub = sub[sub.scenario == scenario]
        if sub.empty:
            continue
        y = sub[metric].rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=sub.episode, y=sub[metric], mode="markers", name=p,
                                 marker=dict(color=POLICY_COLORS[p], size=4, opacity=0.25),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=sub.episode, y=y, mode="lines", name=p,
                                 line=dict(color=POLICY_COLORS[p], width=2),
                                 hovertemplate=f"{p} episode %{{x}}: %{{y:.1f}}<extra></extra>"))
    base_layout(fig, height=height, title=title)
    fig.update_xaxes(title_text="Training episode (30 simulated minutes each)")
    fig.update_yaxes(title_text=label)
    return fig


def box_by_policy(df, metric, label, height=320, title=None):
    fig = go.Figure()
    for p in POLICY_COLORS:
        sub = df[df.policy == p]
        if sub.empty:
            continue
        fig.add_trace(go.Box(y=sub[metric], name=p, marker=dict(color=POLICY_COLORS[p], size=6),
                             line=dict(width=1.5), boxpoints="all", jitter=0.35, pointpos=0,
                             fillcolor="rgba(0,0,0,0)",
                             hovertemplate=f"{p}: %{{y:.1f}} {label}<extra></extra>"))
    base_layout(fig, height=height, title=title, legend=False)
    fig.update_yaxes(title_text=label)
    return fig


def timeseries(sim_log, scenario, metric, label, junction=None, height=320, title=None,
               agg="sum"):
    sub = sim_log[sim_log.scenario == scenario]
    if junction and junction != "All junctions":
        sub = sub[sub.junction == junction]
    g = sub.groupby(["policy", "t_start_s"])[metric].agg(agg).reset_index()
    fig = go.Figure()
    for p in POLICY_COLORS:
        s = g[g.policy == p]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.t_start_s / 60.0, y=s[metric], mode="lines", name=p,
                                 line=dict(color=POLICY_COLORS[p], width=2),
                                 hovertemplate=f"{p} minute %{{x:.1f}}: %{{y:.1f}}<extra></extra>"))
    inc = C.SCENARIOS[scenario].get("incident")
    ev = C.SCENARIOS[scenario].get("event")
    for blk, name in ((inc, "incident"), (ev, "event surge")):
        if blk:
            fig.add_vrect(x0=blk["t_start"] / 60, x1=blk["t_end"] / 60,
                          fillcolor="rgba(227,73,72,0.06)", line_width=0)
            fig.add_annotation(x=blk["t_start"] / 60, y=1.0, yref="paper", text=name,
                               showarrow=False, xanchor="left", yanchor="top",
                               font=dict(size=11, color=MUTED))
    base_layout(fig, height=height, title=title)
    fig.update_xaxes(title_text="Minute of episode")
    fig.update_yaxes(title_text=label)
    return fig


def radar(utilities: pd.DataFrame, height=380, title=None):
    """utilities: index = policy, columns = attribute names, values 0-100."""
    fig = go.Figure()
    cats = list(utilities.columns)
    for p in POLICY_COLORS:
        if p not in utilities.index:
            continue
        v = utilities.loc[p].tolist()
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=cats + cats[:1], name=p,
                                      line=dict(color=POLICY_COLORS[p], width=2),
                                      fill="toself", fillcolor="rgba(0,0,0,0)",
                                      hovertemplate=f"{p} · %{{theta}}: %{{r:.0f}}<extra></extra>"))
    base_layout(fig, height=height, title=title)
    fig.update_layout(polar=dict(bgcolor=SURFACE,
                                 radialaxis=dict(range=[0, 100], gridcolor=GRID,
                                                 tickfont=dict(size=10, color=MUTED)),
                                 angularaxis=dict(gridcolor=GRID,
                                                  tickfont=dict(size=12, color=INK2))))
    return fig


# ======================================================================================
# Decision network drawing
# ======================================================================================
DN_LAYOUT = {
    "Traffic density": (0.4, 2.6, "chance"),
    "Weather": (1.9, 2.6, "chance"),
    "Pedestrian flow": (3.4, 2.6, "chance"),
    "Routing": (-0.5, 1.3, "decision"),
    "Congestion": (0.9, 1.3, "chance"),
    "Collision risk": (2.6, 1.3, "chance"),
    "Signal timing": (-0.5, 0.0, "decision"),
    "Emissions": (1.4, 0.0, "chance"),
    "Health impact": (3.2, 0.0, "chance"),
    "U_efficiency": (0.0, -1.4, "utility"),
    "U_safety": (2.0, -1.4, "utility"),
    "U_environment": (3.6, -1.4, "utility"),
}
U_LABELS = {"U_efficiency": "Efficiency utility", "U_safety": "Safety utility",
            "U_environment": "Environment utility"}


def influence_diagram(dn, posteriors=None, decision=None, utilities=None, height=560):
    fig = go.Figure()
    edges = []
    for n, nd in dn.chance.items():
        for p in nd["parents"]:
            edges.append((p, n))
    for u, ud in dn.utilities.items():
        for p in ud["parents"]:
            edges.append((p, u))
    for (a, b) in edges:
        x0, y0, _ = DN_LAYOUT[a]
        x1, y1, _ = DN_LAYOUT[b]
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x",
                           ayref="y", showarrow=True, arrowhead=3, arrowsize=1.1,
                           arrowwidth=1.2, arrowcolor="#a19f96", standoff=34,
                           startstandoff=30, text="")
    for n, (x, y, kind) in DN_LAYOUT.items():
        if kind == "chance":
            shape = dict(type="circle", x0=x - 0.55, x1=x + 0.55, y0=y - 0.3, y1=y + 0.3)
        elif kind == "decision":
            shape = dict(type="rect", x0=x - 0.55, x1=x + 0.55, y0=y - 0.27, y1=y + 0.27)
        else:
            shape = dict(type="path",
                         path=f"M {x - 0.62} {y} L {x} {y + 0.33} L {x + 0.62} {y} "
                              f"L {x} {y - 0.33} Z")
        fig.add_shape(**shape, fillcolor=NODE_COLORS[kind],
                      line=dict(color=NODE_EDGES[kind], width=1.5), layer="below")
        label = U_LABELS.get(n, n)
        sub = ""
        if kind == "chance" and posteriors is not None and n in posteriors:
            st = dn.states(n)
            pr = posteriors[n]
            k = int(np.argmax(pr))
            sub = f"{st[k]} {pr[k]:.0%}"
        elif kind == "decision" and decision is not None:
            sub = decision.get(n, "")
        elif kind == "utility" and utilities is not None:
            sub = f"E[U] = {utilities.get(n, 0):.1f}"
        fig.add_annotation(x=x, y=y + (0.07 if sub else 0), text=f"<b>{label}</b>",
                           showarrow=False, font=dict(size=12, color=INK))
        if sub:
            fig.add_annotation(x=x, y=y - 0.12, text=sub, showarrow=False,
                               font=dict(size=11, color=INK2))
    # invisible hover points
    fig.add_trace(go.Scatter(
        x=[v[0] for v in DN_LAYOUT.values()], y=[v[1] for v in DN_LAYOUT.values()],
        mode="markers", marker=dict(size=40, color="rgba(0,0,0,0)"),
        hovertext=[_node_hover(dn, n, posteriors) for n in DN_LAYOUT], hoverinfo="text",
        showlegend=False))
    base_layout(fig, height=height, legend=False)
    fig.update_xaxes(visible=False, range=[-1.15, 4.3])
    fig.update_yaxes(visible=False, range=[-1.85, 3.0])
    return fig


def _node_hover(dn, n, posteriors):
    if n in dn.chance:
        par = dn.chance[n]["parents"]
        s = f"<b>{n}</b> (chance)<br>parents: {', '.join(par) if par else 'none'}"
        if posteriors is not None and n in posteriors:
            s += "<br>" + "<br>".join(f"P({st}) = {p:.2f}" for st, p in
                                      zip(dn.states(n), posteriors[n]))
        return s
    if n in dn.decisions:
        return f"<b>{n}</b> (decision)<br>options: {', '.join(dn.decisions[n])}"
    return f"<b>{U_LABELS.get(n, n)}</b> (utility)<br>depends on: " \
           f"{', '.join(dn.utilities[n]['parents'])}"


def posterior_bars(dn, posteriors, nodes, height=300, title=None):
    fig = go.Figure()
    shades = {"Low": SEQ_BLUE[2], "Medium": SEQ_BLUE[4], "High": SEQ_BLUE[6]}
    labels = [n for n in nodes]
    for k, st in enumerate(["Low", "Medium", "High"]):
        vals = []
        for n in nodes:
            states = dn.states(n)
            vals.append(float(posteriors[n][states.index(st)]) if st in states else 0.0)
        fig.add_trace(go.Bar(y=labels, x=vals, name=st, orientation="h",
                             marker=dict(color=shades[st], line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y}: P(" + st + ") = %{x:.2f}<extra></extra>"))
    base_layout(fig, height=height, title=title)
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def eu_bars(table, height=340, title=None):
    t = table.copy()
    t["label"] = t["Signal timing"] + " · " + t["Routing"]
    t = t.sort_values("EU")
    fig = go.Figure(go.Bar(
        y=t["label"], x=t["EU"], orientation="h",
        marker=dict(color=[POLICY_COLORS[s] for s in t["Signal timing"]],
                    line=dict(color=SURFACE, width=2)),
        text=[f"{v:.1f}" for v in t["EU"]], textposition="outside",
        textfont=dict(color=INK2, size=11),
        hovertemplate="%{y}<br>expected utility %{x:.2f}<extra></extra>"))
    base_layout(fig, height=height, title=title, legend=False)
    lo = max(0.0, float(t["EU"].min()) - 8)
    fig.update_xaxes(range=[lo, float(t["EU"].max()) + 4], showgrid=True, gridcolor=GRID,
                     title_text="Expected utility (0-100)")
    fig.update_yaxes(showgrid=False)
    return fig


def ternary_map(ws: pd.DataFrame, height=420, title=None):
    fig = go.Figure()
    for p in POLICY_COLORS:
        sub = ws[ws.best_signal == p]
        if sub.empty:
            continue
        fig.add_trace(go.Scatterternary(
            a=sub.w_efficiency, b=sub.w_safety, c=sub.w_environment, mode="markers",
            name=p, marker=dict(color=POLICY_COLORS[p], symbol=POLICY_SYMBOLS[p], size=9,
                                line=dict(color=SURFACE, width=1.5)),
            hovertemplate=(f"best: {p}<br>efficiency %{{a:.2f}}<br>safety %{{b:.2f}}"
                           "<br>environment %{c:.2f}<extra></extra>")))
    base_layout(fig, height=height, title=title, margin=dict(l=40, r=40, t=70, b=40))
    fig.update_layout(ternary=dict(
        bgcolor=SURFACE, sum=1,
        aaxis=dict(title=dict(text="Efficiency weight", font=dict(size=11)), gridcolor=GRID,
                   linecolor=GRID, tickfont=dict(size=10, color=MUTED)),
        baxis=dict(title=dict(text="Safety weight", font=dict(size=11)), gridcolor=GRID,
                   linecolor=GRID, tickfont=dict(size=10, color=MUTED)),
        caxis=dict(title=dict(text="Environment weight", font=dict(size=11)), gridcolor=GRID,
                   linecolor=GRID, tickfont=dict(size=10, color=MUTED))))
    return fig


def policy_heatmap(prob_switch, x_labels, y_labels, title=None, height=300):
    fig = go.Figure(go.Heatmap(
        z=prob_switch, x=x_labels, y=y_labels, zmin=0, zmax=1, xgap=2, ygap=2,
        colorscale=[[0, "#f0efec"], [1, "#2a78d6"]],
        text=[["–" if np.isnan(v) else ("switch" if v >= 0.5 else "keep") for v in row]
              for row in prob_switch],
        texttemplate="%{text}", textfont=dict(size=11, color=INK),
        colorbar=dict(title=dict(text="share of states<br>choosing switch",
                                 font=dict(size=10)), thickness=10, tickformat=".0%"),
        hovertemplate="queue on green %{y} · queue on red %{x}<br>"
                      "switch in %{z:.0%} of states<extra></extra>"))
    base_layout(fig, height=height, title=title, legend=False)
    fig.update_xaxes(title_text="Queue on red approaches", side="bottom")
    fig.update_yaxes(title_text="Queue on green approaches", autorange="reversed",
                     showgrid=False)
    return fig


def combo_bars(df, metric="avg_delay_s", label="Average delay (s/veh)", height=330, title=None):
    """Signal x routing pairs for one scenario, sorted; colour = signal policy, the label names
    the routing (no second colour channel)."""
    g = df.groupby("variant")[metric].agg(["mean", "std"]).reset_index()
    g["signal"] = g.variant.str.split(" + ", regex=False).str[0]
    g = g.sort_values("mean", ascending=False)
    fig = go.Figure(go.Bar(
        y=g.variant, x=g["mean"], orientation="h",
        marker=dict(color=[POLICY_COLORS[s] for s in g.signal],
                    line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", array=g["std"], color="#b9b7ae", thickness=1, width=3),
        text=[f"{v:.1f}" for v in g["mean"]], textposition="outside",
        textfont=dict(color=INK2, size=11),
        hovertemplate="%{y}<br>%{x:.1f} " + label + "<extra></extra>"))
    base_layout(fig, height=height, title=title, legend=False)
    fig.update_xaxes(title_text=label, showgrid=True, gridcolor=GRID,
                     range=[0, float((g["mean"] + g["std"].fillna(0)).max()) * 1.15])
    fig.update_yaxes(showgrid=False)
    return fig
