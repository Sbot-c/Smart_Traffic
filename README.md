# Smart City Traffic Management — Multi-Agent RL + Decision Networks

Group 2 project for *Reasoning and Decision Making under Uncertainty* (MAIB DSC 103).

Traffic lights, autonomous vehicles and buses on a 3 × 3 Downtown-Dubai-style grid learn
with **Q-learning** and **SARSA**; a **decision network (influence diagram)** learned from
the simulator scores efficiency, safety and emissions under uncertainty; six scenario
families are compared; a six-tab **Streamlit dashboard** shows everything.

## Deliverables (Blackboard, 8 Nov 2026, 10 pm)

| Deliverable | Where |
|---|---|
| Requirement specification (5 pages) | Claude doc *Requirement Specification – Smart City Traffic Management (Group 2)*, <https://claude.ai/artifact/NcNxM9NPofCuKRQMF5Km59> — export it as **PDF, A4** |
| Data CSV file | `data/smart_traffic_simulation_log.csv` (main file; the other CSVs support it) |
| Colab code link | upload `Smart_Traffic_MARL_DecisionNetwork.ipynb` to Google Drive, open with Colab, share *Anyone with the link* |
| Dashboard | `app.py` (run locally or deploy, below) |
| PPT | still to be made by the group |
| GenAI prompts | `GenAI_disclosure_and_prompts.md` |

| Agent | State | Actions | Reward |
|---|---|---|---|
| Traffic light (×9) | phase, green time, queue on green / red, arrivals on green, pedestrian / spillback flag (2,250 states) | keep / change phase every 5 s | −(vehicle delay + 0.5 × pedestrian delay + 300 × expected conflicts + 0.2 × g CO₂) / 100 |
| Autonomous vehicle (fleet) | remaining moves, congestion on the two candidate links (64) | next link along x or y | −travel time − cooperative congestion externality |
| Bus (fleet) | schedule deviation, load, headway, signal ahead, cross queue (144) | go / request priority / hold 20 s | −(passenger-minutes + schedule deviation + priority externality) |

## Files

```
app.py                  Streamlit dashboard (6 tabs)
viz.py                  Plotly figures
sim_view.py             animated canvas replay (Play / Pause / speed / scrubber, side-by-side)
config.py               every parameter, scenario definition and source
traffic_env.py          the simulator (vehicles, queues, signals, pedestrians, buses, EVs, weather)
agents.py               tabular Q-learning / SARSA, the three agent types, Webster and actuated baselines
decision_network.py     influence-diagram engine, CPT learning, expected utility, VOI, sensitivity
experiments.py          training pipeline, evaluation, component studies, data export
run_experiments.py      regenerate ./data (full ~8 min, --quick ~2 min)
robustness.py           re-adapt with different seeds -> data/training_robustness.csv
build_notebook.py       rebuild the Colab notebook from these files
Smart_Traffic_MARL_DecisionNetwork.ipynb   the Colab notebook (code + results)
data/                   CSV outputs, trained Q-tables, animation recordings
```

## Data (`data/`)

| File | Rows | Content |
|---|---|---|
| `smart_traffic_simulation_log.csv` | 15,120 | **main dataset** — one row per junction per 30 s window, every scenario × policy |
| `kpi_results.csv` | 224 | KPIs per evaluation episode (7 scenarios × 4 policies × 8 seeds) |
| `decision_network_samples.csv` | 11,664 | junction × 10-minute samples used to learn the decision-network CPTs |
| `learning_curves.csv` | 1,680 | per-episode training metrics |
| `ablation_results.csv` | 400 | component studies: AV routing, bus priority, and all eight signal × routing pairs |
| `agent_decisions.csv` | 7,225 | recorded bus and AV decisions with their states |
| `training_robustness.csv` | 18 | re-adaptation with three other random seeds |
| `scenario_parameters.csv` | 7 | scenario definitions |
| `q_tables.npz` | — | trained Q-tables (general + per-scenario) |
| `animations.json.gz` | — | vehicle trajectories for the replay |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate every result: `python run_experiments.py` (then `python robustness.py`).

## Deploy (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repository** (these files at the repo root, including `data/`).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Pick the repository, branch `main`, main file `app.py`, then **Deploy**.

## Google Colab

Upload `Smart_Traffic_MARL_DecisionNetwork.ipynb` to Google Drive, open it with Colab, and
share it with *Anyone with the link → Viewer*. *Runtime → Run all* reproduces the results.
