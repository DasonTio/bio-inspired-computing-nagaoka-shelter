# Task 2 – GA & PSO: Earthquake shelter assignment in Nagaoka

Status: **paused (archived 2026-10-29)**. Data, model, baselines, GA and PSO are built and tested.
Not done yet: parameter tuning results, the 30-run experiments, statistics/plots, the report.

## The problem
After a large earthquake, assign the evacuees of each populated 500 m cell in Nagaoka to one
shelter, so that walking stays short and shelters are not overcrowded.

- Decision: for each of 1,403 cells, choose one of its 5 nearest shelters (within 3 km road distance).
- Score (lower is better): total walking (person-km) + 10 × crowding,
  where crowding = Σ (people above capacity)² / capacity  (convex: 200% full is much worse than 110%).
- Capacity: floor area ÷ 3.5 m² per person (Nagaoka shelter operations manual, FY2026).
- Scenarios: 20%, 30%, 40% of residents evacuate (2004 Chūetsu quake: ~103,000 evacuees in 603 shelters).
- Road distance ≈ straight-line distance × 1.3.

## Data
| File | Source | Notes |
|---|---|---|
| data/raw/P20-12_15_GML.zip | MLIT National Land Numerical Information, evacuation facilities P20 (2012, Niigata) | Shift-JIS; 281 Nagaoka rows |
| data/raw/tblT001101S55xx/56xx.zip | e-Stat, 2020 census, 500 m mesh, table T001101 (population & households) | Shift-JIS; sheets 5538, 5539, 5638, 5639 |
| data/processed/nagaoka_shelters_raw.csv | built by code/inspect_p20.py | Nagaoka rows only (city code 15202) |
| data/processed/nagaoka_cells_2020.csv | built by code/build_cells.py | 1,403 cells, 266,975 people |

## Cleaning log (findings)
1. Earthquake flag (P20_007) is 0 for all 281 Nagaoka shelters, so it cannot be used. Shelters are selected by type instead.
2. 12 flood-only sites (信濃川等のはんらん時避難場所) excluded.
3. 23 sites have no capacity and no floor area (−1), mostly outdoor spaces: excluded → 246 shelters.
4. Official 2012 capacities imply 0.57–2.0 m² per person; recomputed as floor area ÷ 3.5 m². Official values kept for a sensitivity scenario.
5. Location accuracy: level 1 = 214, level 3 = 60, level 4 = 7.
6. The 1 km table (T001100) was downloaded by mistake at first; the 500 m table is T001101.
7. Privacy-merged cells (flag 2) still report total population; the 65+ breakdown is hidden for 232 cells (not used).
8. Cells assigned to Nagaoka by nearest facility: total 266,975 vs official 2020 census 266,936 (difference 0.015%).

## Results so far
Nearest-shelter rule (current practice), capacity at 3.5 m²:
| Scenario | Evacuees | Overloaded shelters | People above capacity |
|---|---|---|---|
| 20% | 53,395 | 67 / 246 | 14,177 (27%) |
| 30% | 80,092 | 107 / 246 | 30,238 (38%) |
| 40% | 106,790 | 127 / 246 | 49,685 (47%) |

Why: shelter size is barely related to nearby population (Spearman 0.17). Urban shelters run at 148% load,
rural ones at 56% (30% scenario).

Linear-penalty version is solved by an exact solver (HiGHS) in ~1 min → switched to the convex crowding
score so a metaheuristic is justified; the linear exact result is kept as a reference.

First single runs, 30% scenario, 50,000 evaluations each:
| Method | Score | People above capacity | Worst shelter load |
|---|---|---|---|
| Nearest | 1,287,378 | 30,238 | 16.8× |
| Greedy | 116,358 | 10,972 | 2.1× |
| GA (1 run) | 109,350 | 8,929 | 2.2× |
| PSO (1 run, untuned) | 558,710 | 23,104 | 9.9× |

## Code
| File | Purpose |
|---|---|
| code/inspect_p20.py | Read the shelter shapefile, keep Nagaoka |
| code/build_cells.py | Combine the 4 mesh files, compute cell centres, keep Nagaoka cells |
| code/explore_nearest_baseline.py | First exploration of the nearest-shelter baseline |
| code/shelter_model.py | Shared problem + evaluator (counts evaluations) |
| code/baselines.py | Nearest, greedy, LP lower bound, exact ILP (linear version) |
| code/algorithms.py | GA (integer) and PSO (continuous + floor decoding) |
| code/test_model.py | Checks against hand calculation and census total |
| code/tune.py | Parameter sensitivity (seeds 1000–1004) |
| code/run_experiments.py | 30 seeds × GA/PSO × 3 scenarios (needs results/chosen_params.json) |
| code/analyze.py | Summary table, Mann-Whitney + A12, convergence and box plots |

Run from a folder containing the two processed CSVs:
`python test_model.py`, then `python tune.py`, then write results/chosen_params.json, then
`python run_experiments.py` and `python analyze.py`.
(build_cells.py expects the raw files unzipped into mesh/ and p20/.)

## Next steps
1. Finish tuning; choose GA and PSO parameters.
2. 30-run experiments + statistics + plots.
3. One-page proposal, 8–12 page report, data dictionary, AI disclosure table.
4. Stakeholder input: one short question to Nagaoka City's disaster office or dorm staff.
