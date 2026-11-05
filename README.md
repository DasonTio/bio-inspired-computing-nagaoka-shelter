# Task 2 – GA & PSO: Earthquake shelter assignment in Nagaoka

Status: **experiments finished**. Data (GSI 2026 shelter list), model, baselines, tuning, the 30-run
GA/PSO experiments, statistics, sensitivity checks and figures are done.
Not done yet: the written report, one-page proposal and AI disclosure table.

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
Everything in `dataset/` is rebuilt from the untouched downloads in `dataset/raw/` by
`python dataset/build_dataset.py` (numpy, pandas, pyshp). See `dataset/data_dictionary.csv` for columns.

| Raw file | Source | Notes |
|---|---|---|
| 15202_1.zip | GSI (hinanmap.gsi.go.jp), designated shelters 指定避難所, Nagaoka, current registration | UTF-8; 285 rows, no floor area |
| 15202_2.zip | GSI, designated emergency evacuation sites 指定緊急避難場所 | UTF-8; per hazard |
| P20-12_15_GML.zip | MLIT National Land Numerical Information, evacuation facilities P20 (2012, Niigata) | Shift-JIS; supplies floor area. 281 Nagaoka rows |
| tblT001101S5538/5539/5638/5639.zip | e-Stat, 2020 census, 500 m mesh, table T001101 (population & households) | Shift-JIS |
| dataset/manual/floor_area_estimates.csv | Hand-collected, source URL per row | Floor area for 9 shelters with no P20 match |

Outputs: `cells.csv` (1,403 cells, 266,975 people), `shelters.csv` (223 shelters, capacity 102,025 at 3.5 m²),
`candidates.csv` (5 nearest shelters per cell), `distance_road_km.npz/.csv`, `scenarios.json`.
`index.html` (made by `dataset/make_index.py`) is a map and table viewer for all of it; it opens offline except for map tiles.

## Cleaning log (findings)
1. The shelter list now comes from GSI (the city's current registration). GSI has no floor area, so each
   shelter is matched to a P20 (2012) site: by name first (173), then by location within 100 m for renamed
   buildings (41). 9 shelters have no match and use a documented estimate (`floor_area_estimates.csv`).
2. GSI exclusions: 59 welfare rooms (要配慮者 only), 2 sites in the same P20 building as an earlier shelter,
   1 whose P20 site has no floor area → 285 GSI rows give **223 shelters**.
3. The P20 earthquake flag is 0 for all 281 Nagaoka sites, so it is not used.
4. Official 2012 capacities imply 0.57–2.0 m² per person; recomputed as floor area ÷ 3.5 m². Official values are kept
   (`capacity_official`) for a sensitivity scenario.
5. Privacy-merged census cells still report total population; the 65+ breakdown is hidden for 232 cells (not used).
6. Cells assigned to Nagaoka by nearest facility: total 266,975 vs official 2020 census 266,936 (difference 0.015%).
7. Shelter size is barely related to nearby population (Spearman 0.17), so urban shelters overflow and rural ones stay empty.

## Method
- **Baselines:** nearest shelter (current practice), greedy, and an exact MILP (HiGHS) for the linear-penalty
  version. The linear version is solved in about 1 min, so the convex crowding score is used to justify a metaheuristic;
  the linear result is kept as a reference (`results/linear_reference.csv`).
- **GA:** integer vector of 1,403 values in 0..4, tournament selection, crossover, mutation, elitism.
- **PSO:** continuous positions decoded by flooring to 0..4, ring topology.
- **Budget:** 50,000 evaluations per run, 30 seeds (0–29), 3 scenarios, same budget for both algorithms.
- **Tuning:** seeds 1000–1019 (separate from the final seeds), 30% scenario. Chosen settings are in `results/chosen_params.json`:

| | Setting |
|---|---|
| GA | population 30, tournament 3, crossover 0.9, elite 2, mutation rate 0.0014 per gene |
| PSO | swarm 20, w 0.729, c1 = c2 = 1.49445, vmax 0.2 of range, ring topology |

## Results
Nearest-shelter rule (current practice), capacity at 3.5 m²:

| Scenario | Evacuees | Overloaded shelters | People above capacity |
|---|---|---|---|
| 20% | 53,395 | 62 / 223 | 13,565 (25%) |
| 30% | 80,092 | 95 / 223 | 28,780 (36%) |
| 40% | 106,790 | 112 / 223 | 47,396 (44%) |

Final score (median of 30 runs, lower is better):

| Scenario | Nearest | Greedy | GA | PSO | GA vs nearest | GA vs greedy |
|---|---|---|---|---|---|---|
| 20% | 457,694 | 45,271 | 45,838 | 79,666 | −90.0% | +1.3% (GA beats greedy in 13% of runs) |
| 30% | 1,244,037 | 115,097 | 109,817 | 238,370 | −91.2% | −4.6% (100% of runs) |
| 40% | 2,471,123 | 297,244 | 275,299 | 564,397 | −88.9% | −7.4% (100% of runs) |

- GA beats PSO in every scenario (Mann-Whitney p = 3.0e-11, A12 = 1.0), as in all 30 vs 30 comparisons.
- GA is clearly better than greedy at 30% and 40%, but not at 20%, where crowding is mild and greedy is already near optimal.
- PSO is much worse and has a wide spread (std 9k–68k vs GA 0.6k–5k). It is the less suitable method for this integer problem.
- Both methods take about 2.6 s per run. No run removes all overflow: 30% scenario GA leaves about 8,200 people above capacity (median).
- Sensitivity (30%, GA 10 seeds, median score): penalty weight 1 / 3 / 10 / 30 / 100 → 71,125 / 85,116 / 109,982 / 164,247 / 346,349;
  official capacities → 57,980; without the 9 estimated floor areas → 118,101. The ranking nearest ≫ greedy ≳ GA holds in all cases
  except weight 3, where greedy and GA are about equal.
- On the linear objective GA reaches a median of 158,203, against 127,477 for the exact MILP (gap ~0.7%).

Figures are in `results/` (`fig_convergence.png`, `fig_final_scores.png`, `fig_demand_profile.png`) and map screenshots in `results/screens/`.
The best assignment per method and scenario is in `results/best_assign.npz`; all runs are in `results/runs.csv`; `results/summary.csv` and `results/tests.csv` hold the tables above.

## Code
| File | Purpose |
|---|---|
| code/shelter_model.py | Shared problem + evaluator (counts evaluations) |
| code/baselines.py | Nearest, greedy, LP lower bound, exact ILP (linear version) |
| code/algorithms.py | GA (integer) and PSO (continuous + floor decoding) |
| code/test_model.py | Checks against hand calculation and census total |
| code/tune.py, tune_ga.py, tune_pso.py, tune_confirm.py | Parameter study (screening, then 20-seed confirmation) |
| code/run_experiments.py | 30 seeds × GA/PSO × 3 scenarios (reads results/chosen_params.json) |
| code/analyze.py | Summary table, Mann-Whitney + A12, convergence and box plots |
| code/extra_checks.py | Sensitivity cases, PSO step sizes, GA vs MILP on the linear objective |
| code/export_final.py, export_solutions.py | Export results as JSON for the viewer in index.html |
| code/fig_demand.py | Demand profile figure |
| code/inspect_p20.py, build_cells.py, explore_nearest_baseline.py | Early exploration scripts (data/processed, 2012 P20 only) |
| dataset/build_dataset.py, gsi_shelters.py | Build the dataset from dataset/raw |
| dataset/make_index.py | Build index.html |

## How to reproduce
```
python dataset/build_dataset.py        # rebuild the dataset
python dataset/make_index.py           # rebuild index.html
cd code
python test_model.py
python tune_ga.py summary ; python tune_pso.py summary ; python tune_confirm.py   # tuning (see each file's header)
python run_experiments.py              # or run in parts: run_experiments.py 0.3 GA, then run_experiments.py merge
python analyze.py                      # run from the project root
python extra_checks.py sens base       # also w1, w3, w30, w100, official, noest; then pso_steps, linear, summary
python export_final.py
```
`code/build_cells.py` expects the raw files unzipped into `mesh/` and `p20/`.

## Limitations
- Shelter locations and names come from GSI 2026, but floor areas come from the 2012 P20 data (and 9 estimates), so capacity may be outdated.
- Distances are straight-line × 1.3, not real walking routes, and ignore rivers, bridges and earthquake damage.
- Residents are placed at the centre of their 500 m cell.
- City membership of border cells is approximated by the nearest facility.
- Evacuation rates are scenarios, not forecasts.

## Next steps
1. 8–12 page report with the figures and tables above.
2. One-page proposal, data dictionary check, AI disclosure table.
3. Stakeholder input: one short question to Nagaoka City's disaster office or dorm staff.
