"""Build the Task 2 dataset (Nagaoka earthquake shelter assignment) from the raw downloads.

Inputs  (raw/, unchanged downloads):
    P20-12_15_GML.zip          MLIT evacuation facilities, Niigata, 2012 (shapefile, Shift-JIS)
    tblT001101S5538/5539/5638/5639.zip   e-Stat 2020 census, 500 m mesh, table T001101 (Shift-JIS)
Outputs (in this folder):
    cells.csv, shelters.csv, shelters_all.csv, candidates.csv,
    distance_road_km.npz, distance_road_km.csv, census_mesh_nagaoka_full.csv,
    scenarios.json, data_dictionary.csv

Run:  python build_dataset.py        (needs numpy, pandas, pyshp)
"""
import csv, io, json, os, sys, zipfile
import numpy as np, pandas as pd, shapefile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
os.chdir(HERE)

CITY = "15202"                      # Nagaoka city code
ROAD_FACTOR = 1.3                   # straight-line km -> approximate road km
M2_PER_PERSON = 3.5                 # Nagaoka shelter operations manual (FY2026)
K, MAX_WALK_KM = 5, 3.0
EVAC_RATES = [0.2, 0.3, 0.4]
FLOOD_ONLY = "信濃川等のはんらん時避難場所"
TYPE_EN = {"指定避難所": "designated shelter",
           "指定避難所、信濃川等のはんらん時避難場所": "designated shelter + Shinano River flood site",
           FLOOD_ONLY: "Shinano River flood site only"}


def km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def mesh_centre(k):
    """Centre of a 500 m (1/2 third-level) JIS mesh from its 9-digit code."""
    p, u = int(k[:2]), int(k[2:4]); q, v, r, w, m = map(int, k[4:9])
    lat = p / 1.5 + q * 5 / 60 + r * 30 / 3600 + (15 / 3600 if m in (3, 4) else 0) + 7.5 / 3600
    lon = u + 100 + v * 7.5 / 60 + w * 45 / 3600 + (22.5 / 3600 if m in (2, 4) else 0) + 11.25 / 3600
    return lat, lon


# ------------------------------------------------------------------ census mesh
CENSUS_EN = ["pop_total", "pop_male", "pop_female", "pop_0_14", "pop_0_14_male", "pop_0_14_female",
             "pop_15plus", "pop_15plus_male", "pop_15plus_female", "pop_15_64", "pop_15_64_male", "pop_15_64_female",
             "pop_18plus", "pop_18plus_male", "pop_18plus_female", "pop_20plus", "pop_20plus_male", "pop_20plus_female",
             "pop_65plus", "pop_65plus_male", "pop_65plus_female", "pop_75plus", "pop_75plus_male", "pop_75plus_female",
             "pop_85plus", "pop_85plus_male", "pop_85plus_female", "pop_95plus", "pop_95plus_male", "pop_95plus_female",
             "foreign_total", "foreign_male", "foreign_female", "households_total", "households_general",
             "hh_1person", "hh_2person", "hh_3person", "hh_4person", "hh_5person", "hh_6person", "hh_7plus",
             "hh_relatives_only", "hh_nuclear", "hh_non_nuclear", "hh_with_child_under6", "hh_with_65plus",
             "hh_single_age20s", "hh_elderly_single", "hh_elderly_couple"]
rows, labels_ja = [], None
for sheet in ["5538", "5539", "5638", "5639"]:
    with zipfile.ZipFile(f"{RAW}/tblT001101S{sheet}.zip") as z:
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        r = list(csv.reader(io.TextIOWrapper(z.open(name), encoding="cp932")))
    labels_ja = [x.strip() for x in r[1][4:]]
    for x in r[2:]:
        rec = dict(mesh_key=x[0], privacy_flag=int(x[1] or 0), merged_into=x[2] or "", merged_from=x[3] or "")
        for col, val in zip(CENSUS_EN, x[4:]):
            rec[col] = int(val) if val.isdigit() else np.nan     # '*' = hidden for privacy
        rec["lat"], rec["lon"] = mesh_centre(x[0])
        rows.append(rec)
mesh = pd.DataFrame(rows)

# ------------------------------------------------------------------ shelters (P20)
with zipfile.ZipFile(f"{RAW}/P20-12_15_GML.zip") as z:
    z.extractall("/tmp/_p20")
shp = [os.path.join(dp, f) for dp, _, fs in os.walk("/tmp/_p20") for f in fs if f.endswith(".shp")][0]
sh = shapefile.Reader(shp, encoding="cp932")
fields = [f[0] for f in sh.fields[1:]]
allfac = pd.DataFrame([list(rec) for rec in sh.records()], columns=fields)
allfac["lon"] = [s.points[0][0] for s in sh.shapes()]
allfac["lat"] = [s.points[0][1] for s in sh.shapes()]

# cells belonging to Nagaoka: inside a bounding box and nearest Niigata facility is a Nagaoka one
box = mesh[mesh.lat.between(37.15, 37.80) & mesh.lon.between(138.55, 139.20)].copy()
D = km(box.lat.values[:, None], box.lon.values[:, None], allfac.lat.values[None], allfac.lon.values[None])
box["nearest_city"] = allfac["P20_001"].astype(str).values[D.argmin(1)]
box["nearest_facility_km"] = D.min(1)
ng = box[box.nearest_city == CITY].drop(columns="nearest_city").reset_index(drop=True)

full = ng[["mesh_key", "lat", "lon", "privacy_flag", "merged_into", "merged_from"] + CENSUS_EN].copy()
full.to_csv("census_mesh_nagaoka_full.csv", index=False)

# ------------------------------------------------------------------ shelters table
s = allfac[allfac["P20_001"].astype(str) == CITY].copy().reset_index(drop=True)
num = lambda c: pd.to_numeric(s[c], errors="coerce")
shel = pd.DataFrame({
    "name_ja": s["P20_002"], "address_ja": s["P20_003"], "type_ja": s["P20_004"],
    "type_en": s["P20_004"].map(TYPE_EN),
    "lat": s["lat"], "lon": s["lon"],
    "floor_area_m2": num("P20_006"), "capacity_official": num("P20_005"),
    "flag_earthquake": num("P20_007").astype(int), "flag_tsunami": num("P20_008").astype(int),
    "flag_flood": num("P20_009").astype(int), "flag_volcano": num("P20_010").astype(int),
    "flag_other": num("P20_011").astype(int), "flag_none": num("P20_012").astype(int),
    "location_level": pd.to_numeric(s.get("レベル"), errors="coerce"),
})
shel.loc[shel.floor_area_m2 <= 0, "floor_area_m2"] = np.nan          # -1 = unknown
shel.loc[shel.capacity_official <= 0, "capacity_official"] = np.nan
shel["capacity_3p5m2"] = np.floor(shel.floor_area_m2 / M2_PER_PERSON)
shel["official_m2_per_person"] = (shel.floor_area_m2 / shel.capacity_official).round(2)
reason = np.where(shel.type_ja == FLOOD_ONLY, "flood-only site",
                  np.where(shel.floor_area_m2.isna(), "no floor area / capacity (mostly outdoor)", ""))
shel["included"] = reason == ""
shel["exclusion_reason"] = reason
shel.insert(0, "source_row", range(len(shel)))
shel.to_csv("shelters_all.csv", index=False)
use = shel[shel.included].copy().reset_index(drop=True)
use.insert(0, "shelter_id", range(len(use)))
use["capacity_3p5m2"] = use["capacity_3p5m2"].astype(int)
use = use.drop(columns=["included", "exclusion_reason"])
use.to_csv("shelters.csv", index=False)

# ------------------------------------------------------------------ cells table (model input)
c = ng[ng.pop_total > 0].reset_index(drop=True)
cells = pd.DataFrame({
    "cell_id": range(len(c)), "mesh_key": c.mesh_key, "lat": c.lat.round(6), "lon": c.lon.round(6),
    "population": c.pop_total.astype(int), "pop_male": c.pop_male, "pop_female": c.pop_female,
    "pop_0_14": c.pop_0_14, "pop_15_64": c.pop_15_64, "pop_65plus": c.pop_65plus, "pop_75plus": c.pop_75plus,
    "households": c.households_total,
    "privacy_flag": c.privacy_flag, "merged_into": c.merged_into, "merged_from": c.merged_from,
    "age_detail_scope": np.select([c.privacy_flag == 2, c.privacy_flag == 1], ["hidden", "includes_merged_cells"], "own_cell"),
    "nearest_facility_km": c.nearest_facility_km.round(3),
})
for r in EVAC_RATES:
    cells[f"evacuees_{int(r*100)}pct"] = (cells.population * r).round(1)
cells.to_csv("cells.csv", index=False)

# ------------------------------------------------------------------ distances and candidate shelters
Dr = km(cells.lat.values[:, None], cells.lon.values[:, None], use.lat.values[None], use.lon.values[None]) * ROAD_FACTOR
np.savez_compressed("distance_road_km.npz", distance_road_km=Dr.astype(np.float32),
                    cell_id=cells.cell_id.values, shelter_id=use.shelter_id.values)
pd.DataFrame(Dr.round(3), index=cells.cell_id.rename("cell_id"), columns=use.shelter_id.values).to_csv("distance_road_km.csv")
order = np.argsort(Dr, axis=1)[:, :K]
cand = []
for i in range(len(cells)):
    for rank, j in enumerate(order[i]):
        d = Dr[i, j]
        cand.append(dict(cell_id=i, rank=rank, shelter_id=int(j), road_km=round(float(d), 3),
                         within_walk_limit=bool(d <= MAX_WALK_KM or rank == 0)))
pd.DataFrame(cand).to_csv("candidates.csv", index=False)

# ------------------------------------------------------------------ scenarios and dictionary
json.dump({
    "evacuation_rates": EVAC_RATES, "m2_per_person": M2_PER_PERSON, "road_factor": ROAD_FACTOR,
    "k_nearest_candidates": K, "max_walk_km": MAX_WALK_KM, "crowding_penalty_weight": 10.0,
    "objective": "minimise sum_i e_i * d_i,a(i) + 10 * sum_j max(0, load_j - C_j)^2 / C_j",
    "notes": "nearest candidate is always allowed, even if beyond max_walk_km",
}, open("scenarios.json", "w"), indent=2)

dd = [
 ("cells.csv", "cell_id", "int", "0-based row id used by the model"),
 ("cells.csv", "mesh_key", "str", "9-digit JIS 500 m mesh code"),
 ("cells.csv", "lat / lon", "float", "cell centre, decimal degrees (JGD2000 ~ WGS84)"),
 ("cells.csv", "population", "int", "residents, 2020 census (T001101001)"),
 ("cells.csv", "pop_male / pop_female", "int", "residents by sex"),
 ("cells.csv", "pop_0_14 / pop_15_64 / pop_65plus / pop_75plus", "int or empty", "age groups; empty when hidden for privacy (see age_detail_scope)"),
 ("cells.csv", "households", "int", "total households (T001101034)"),
 ("cells.csv", "privacy_flag", "int", "0 = normal; 1 = cell absorbed small neighbours' detail (age counts include them); 2 = small cell, detail hidden and merged into merged_into"),
 ("cells.csv", "merged_into / merged_from", "str", "mesh keys involved in the privacy merge"),
 ("cells.csv", "age_detail_scope", "str", "own_cell, includes_merged_cells or hidden"),
 ("cells.csv", "nearest_facility_km", "float", "straight-line km to the nearest Niigata evacuation facility (used to assign cells to Nagaoka)"),
 ("cells.csv", "evacuees_20pct / 30pct / 40pct", "float", "demand e_i = population x evacuation rate"),
 ("shelters.csv", "shelter_id", "int", "0-based id used by the model (246 usable shelters)"),
 ("shelters.csv", "source_row", "int", "row in shelters_all.csv"),
 ("shelters.csv", "name_ja / address_ja", "str", "facility name and address (P20_002, P20_003)"),
 ("shelters.csv", "type_ja / type_en", "str", "facility type (P20_004)"),
 ("shelters.csv", "lat / lon", "float", "location, decimal degrees"),
 ("shelters.csv", "floor_area_m2", "float", "floor area m2 (P20_006); empty if unknown"),
 ("shelters.csv", "capacity_official", "float", "official capacity in 2012 (P20_005); empty if unknown"),
 ("shelters.csv", "capacity_3p5m2", "int", "capacity used by the model = floor(floor_area_m2 / 3.5)"),
 ("shelters.csv", "official_m2_per_person", "float", "floor_area_m2 / capacity_official (0.57-2.0 in the data)"),
 ("shelters.csv", "flag_earthquake ... flag_none", "0/1", "hazard designations P20_007-P20_012; earthquake is 0 for every Nagaoka site"),
 ("shelters.csv", "location_level", "int", "geocoding accuracy level from the source (1 = most precise; 3 and 4 are coarser)"),
 ("shelters_all.csv", "included / exclusion_reason", "bool / str", "all 281 Nagaoka sites with the reason a site was dropped"),
 ("candidates.csv", "cell_id, rank, shelter_id", "int", "the 5 nearest shelters of each cell, rank 0 = nearest (the decision options)"),
 ("candidates.csv", "road_km", "float", "straight-line km x 1.3"),
 ("candidates.csv", "within_walk_limit", "bool", "road_km <= 3 km (rank 0 always allowed)"),
 ("distance_road_km.npz/.csv", "distance_road_km", "float", "full matrix cells x shelters, approximate road km"),
 ("census_mesh_nagaoka_full.csv", "all T001101 columns", "int or empty", "all 50 census variables for the Nagaoka cells, English names, '*' -> empty"),
]
pd.DataFrame(dd, columns=["file", "column", "type", "meaning"]).to_csv("data_dictionary.csv", index=False)

print(f"census mesh rows {len(mesh)} | Nagaoka cells {len(ng)} (populated {len(cells)}, people {int(cells.population.sum())})")
print(f"shelters: {len(shel)} Nagaoka sites, {len(use)} usable, capacity {int(use.capacity_3p5m2.sum())} people")
print("exclusions:", shel.exclusion_reason.replace('', 'kept').value_counts().to_dict())
print("candidates within 3 km per cell (mean):", round(pd.DataFrame(cand).within_walk_limit.mean() * K, 2))
