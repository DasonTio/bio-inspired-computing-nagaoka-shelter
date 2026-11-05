"""Write ../index.html: a map and table viewer for every CSV in this folder.

Run after build_dataset.py:  python make_index.py
The page embeds the data, so it opens offline except for the map tiles.
"""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

FILES = [
	("cells.csv", "Demand: populated 500 m census cells (model input)"),
	("shelters.csv", "Supply: shelters used by the model (GSI list + P20 floor area or estimate)"),
	("candidates.csv", "Decision options: the 5 nearest shelters of every cell"),
	("shelters_gsi.csv", "Audit: all 285 GSI designated shelters with match and exclusion reason"),
	("shelters_all.csv", "Audit: all 281 Nagaoka sites in the 2012 MLIT P20 list"),
	("census_mesh_nagaoka_full.csv", "Extra: all 50 census variables per Nagaoka cell"),
	("manual/floor_area_estimates.csv", "Hand-entered floor areas for new shelters, with sources"),
	("data_dictionary.csv", "Meaning of every column"),
]


def table(path):
	df = pd.read_csv(path)
	df = df.astype(object).where(df.notna(), None)
	return {"columns": list(df.columns), "rows": df.values.tolist()}


data = {f: dict(desc=d, **table(f)) for f, d in FILES}
data["_scenarios"] = json.load(open("scenarios.json"))
# GA vs PSO: final 30-seed results (code/export_final.py) if present, else the preview (export_solutions.py)
import glob
data["_preview"] = {}
final = sorted(glob.glob(os.path.join(HERE, "..", "results", "final_*.json")))
for f in final or sorted(glob.glob(os.path.join(HERE, "..", "results", "preview_*.json"))):
	r = json.load(open(f))
	data["_preview"][str(r["rate"])] = r
dd = pd.read_csv("data_dictionary.csv")
data["_dict"] = dd.astype(object).where(dd.notna(), None).values.tolist()

html = open(os.path.join(HERE, "index_template.html"), encoding="utf-8").read()
html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
open(os.path.join(HERE, "..", "index.html"), "w", encoding="utf-8").write(html)
print("wrote ../index.html", round(len(html) / 1e6, 2), "MB")
