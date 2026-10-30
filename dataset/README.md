# Nagaoka earthquake shelter dataset (Task 2: GA and PSO)

Data for assigning the residents of Nagaoka City to evacuation shelters after a large earthquake.
Every populated 500 m census cell must send its evacuees to one of its 5 nearest shelters, so that
walking stays short and shelters are not overcrowded.

Everything here is built from public data by `build_dataset.py`, which reads only the untouched
downloads in `raw/`. Run `python build_dataset.py` (numpy, pandas, pyshp) to rebuild every file.

## At a glance

| | |
|---|---|
| Area | Nagaoka City, Niigata (city code 15202) |
| Demand points | 1,403 populated 500 m cells, 266,975 residents (2020 census) |
| Shelters | 246 usable of 281 listed, total capacity 101,305 people at 3.5 m² per person |
| Decision | for each cell, one of its 5 nearest shelters |
| Scenarios | 20%, 30% or 40% of residents evacuate (53,395 / 80,092 / 106,790 people) |

Check: the cell population (266,975) differs from the official 2020 census count for Nagaoka (266,936) by 0.015%.

## Files

| File | Rows × columns | What it is | Use it for |
|---|---|---|---|
| `cells.csv` | 1,403 × 20 | Demand points: location, population, age groups, households, evacuees per scenario | Model input (demand) |
| `shelters.csv` | 246 × 19 | Usable shelters: location, floor area, capacity | Model input (supply) |
| `candidates.csv` | 7,015 × 5 | The 5 nearest shelters of every cell with road distance | Model input (the options per cell) |
| `distance_road_km.npz` | 1,403 × 246 | Full distance matrix, float32 (`np.load(...)["distance_road_km"]`) | Model input, fast |
| `distance_road_km.csv` | 1,403 × 246 | Same matrix as CSV, rows = `cell_id`, columns = `shelter_id` | Spreadsheets, other languages |
| `scenarios.json` | | Evacuation rates, 3.5 m²/person, road factor 1.3, k = 5, 3 km limit, penalty 10 | Model settings |
| `shelters_all.csv` | 281 × 20 | All Nagaoka sites with `included` and `exclusion_reason` | Audit of the cleaning |
| `census_mesh_nagaoka_full.csv` | 1,403 × 56 | All 50 census variables for the Nagaoka cells, English column names | Extra features (elderly, households, foreigners) |
| `data_dictionary.csv` | | Meaning of every column | Reference |
| `raw/` | | Original downloads, unchanged | Rebuilding |

IDs: `cell_id` and `shelter_id` are 0-based and match the row and column order of the distance matrix.

## Sources

| Raw file | Source | Notes |
|---|---|---|
| `raw/P20-12_15_GML.zip` | MLIT National Land Numerical Information, evacuation facilities (P20), Niigata, 2012 | Shapefile, Shift-JIS (cp932) |
| `raw/tblT001101S5538.zip`, `S5539`, `S5638`, `S5639` | e-Stat, 2020 Population Census, 500 m mesh, table T001101 (population and households) | Four first-level mesh sheets that cover Nagaoka; Shift-JIS |

## How the data was cleaned

**Cells**
1. The four census sheets are combined (18,851 cells), and each 9-digit mesh code is turned into its cell centre.
2. A cell belongs to Nagaoka when it lies in the box 37.15–37.80 N, 138.55–139.20 E and its nearest Niigata evacuation facility is a Nagaoka one. There is no city boundary file, so the facilities stand in for the city territory.
3. Only populated cells are kept (1,403).
4. Privacy: e-Stat hides the detail of very small cells. `privacy_flag = 2` (232 cells): the age breakdown is hidden (`*` → empty) and merged into the cell in `merged_into`. `privacy_flag = 1` (206 cells): the age counts include those merged neighbours, so they can exceed the cell's own population. Total population, sex and households are always the cell's own, which is why the model only uses `population`. The column `age_detail_scope` says which case applies.
5. Evacuees per scenario: `evacuees_XXpct = population × rate`.

**Shelters**
1. Keep the 281 Nagaoka rows (city code 15202).
2. The earthquake flag (`flag_earthquake`) is 0 for all 281 sites, so it cannot select earthquake shelters. Sites are selected by type instead.
3. Drop 12 flood-only sites (信濃川等のはんらん時避難場所) and 23 sites with no floor area or capacity (−1 in the source, mostly outdoor spaces). That leaves 246 shelters.
4. Capacity = floor(floor area ÷ 3.5 m²), following Nagaoka's shelter operations manual. The official 2012 capacities imply only 0.57–2.0 m² per person, which is unrealistically crowded. They are kept in `capacity_official` for a sensitivity check.
5. Location accuracy (`location_level`): level 1 for 214 sites, level 3 for 60 and level 4 for 7.

**Distances**
- Road km ≈ straight-line (haversine) km × 1.3. No road network is used.
- Candidates are the 5 nearest shelters by road km. A candidate beyond 3 km has `within_walk_limit = False`. The nearest shelter is always allowed, so every cell has at least one option. On average a cell has 3.9 options within 3 km.

## The optimisation problem this data is for

For each cell *i*, choose a candidate shelter *a(i)* from `candidates.csv`. Then:

- load of shelter *j*: L_j = Σ evacuees of the cells sent to *j*
- score to minimise = Σ_i e_i · d_{i,a(i)} + 10 · Σ_j max(0, L_j − C_j)² / C_j

where e_i = `evacuees_XXpct`, d = `road_km` and C_j = `capacity_3p5m2`. The first part is total walking in person-km. The second part is a crowding penalty that grows faster the more a shelter is over capacity.

A solution is a vector of 1,403 integers between 0 and 4 (the `rank` in `candidates.csv`). GA uses it directly. PSO uses real numbers rounded down to an integer.

## Quick start

```python
import json, numpy as np, pandas as pd
cells = pd.read_csv("cells.csv")
shelters = pd.read_csv("shelters.csv")
cand = pd.read_csv("candidates.csv")
cfg = json.load(open("scenarios.json"))

n, k = len(cells), cfg["k_nearest_candidates"]
cand_id = cand.pivot(index="cell_id", columns="rank", values="shelter_id").to_numpy()   # (1403, 5)
cand_km = cand.pivot(index="cell_id", columns="rank", values="road_km").to_numpy()
e = cells["evacuees_30pct"].to_numpy()
C = shelters["capacity_3p5m2"].to_numpy()

def score(x):                                   # x: int array, length 1403, values 0..4
    j, d = cand_id[np.arange(n), x], cand_km[np.arange(n), x]
    load = np.bincount(j, weights=e, minlength=len(C))
    over = np.maximum(load - C, 0)
    return e @ d + cfg["crowding_penalty_weight"] * (over ** 2 / C).sum()

nearest = np.zeros(n, dtype=int)                 # everyone goes to the nearest shelter
print(score(nearest))                            # 1,287,377 for the 30% scenario
```

## Reference numbers (nearest-shelter rule, which is current practice)

| Scenario | Evacuees | Overloaded shelters | People above capacity |
|---|---|---|---|
| 20% | 53,395 | 67 / 246 | 14,178 |
| 30% | 80,092 | 107 / 246 | 30,238 |
| 40% | 106,790 | 127 / 246 | 49,685 |

Shelter size is barely related to the population around it (Spearman 0.17), so urban shelters overflow while rural ones stay half empty.

## Limitations

- The shelter list is from 2012. Some sites may have been renamed, closed or added since.
- Distances are straight-line × 1.3, not real walking routes, and ignore rivers, bridges and earthquake damage.
- Residents are placed at the centre of their 500 m cell.
- City membership of border cells is approximated by the nearest facility.
- Evacuation rates are scenarios. For scale, the 2004 Chūetsu earthquake had about 103,000 evacuees in 603 shelters across the region.

## Licence and credit

- Census mesh data: Statistics Bureau of Japan, e-Stat (Government of Japan Standard Terms of Use, compatible with CC BY 4.0).
- Evacuation facilities: MLIT National Land Numerical Information (P20).
- Credit both sources when you use or share these files.
