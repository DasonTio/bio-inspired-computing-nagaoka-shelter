"""Current shelter list from GSI (国土地理院), matched to the 2012 MLIT P20 floor areas.

GSI files (raw/, unchanged downloads from hinanmap.gsi.go.jp, city code 15202):
	15202_1.zip  指定避難所 (designated shelters, where evacuees stay)
	15202_2.zip  指定緊急避難場所 (designated emergency evacuation sites, per hazard)

GSI is the city's current registration but has no floor area or capacity, so each
general shelter is matched to a P20 site: first by name, then by location (within
MATCH_M metres, for renamed buildings). The P20 floor area gives the capacity.
"""
import io, os, unicodedata, zipfile
import numpy as np, pandas as pd

MATCH_M = 100                       # max distance for a location-only match
WELFARE = "要配慮者"                 # 受入対象者 value of welfare rooms


def _read_zip_csv(path):
	with zipfile.ZipFile(path) as z:
		name = [n for n in z.namelist() if n.endswith(".csv")][0]
		return pd.read_csv(io.BytesIO(z.read(name)), encoding="utf-8-sig")


def _norm(s):
	s = unicodedata.normalize("NFKC", str(s))
	return s.replace("(福祉)", "").replace(" ", "").replace("　", "")


def _metres(lat1, lon1, lat2, lon2):
	r = np.pi / 180
	a = np.sin((lat2 - lat1) * r / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2
	return 12742000 * np.arcsin(np.sqrt(a))


def build(raw_dir, p20, m2_per_person, estimates=None):
	"""p20: the shelters_all table (name_ja, lat, lon, floor_area_m2, capacity_official, source_row).
	estimates: optional table (name_ja, floor_area_m2, basis, source) for shelters with no P20 site.
	Returns (gsi table with match and inclusion columns, emergency-site hazard counts)."""
	g = _read_zip_csv(os.path.join(raw_dir, "15202_1.zip"))
	e = _read_zip_csv(os.path.join(raw_dir, "15202_2.zip"))
	g = g.rename(columns={"共通ID": "gsi_id", "施設・場所名": "name_ja", "住所": "address_ja",
		"受入対象者": "accepts", "緯度": "lat", "経度": "lon", "指定緊急避難場所との住所同一": "also_emergency_site"})
	g["welfare_only"] = g.accepts.eq(WELFARE)

	D = _metres(g.lat.values[:, None], g.lon.values[:, None], p20.lat.values[None], p20.lon.values[None])
	pn = p20.name_ja.map(_norm).values
	rows, how, dist = [], [], []
	for i, n in enumerate(g.name_ja.map(_norm)):
		same = np.where(pn == n)[0]
		if len(same):
			j = same[np.argmin(D[i, same])]; h = "name"
		else:
			j = int(np.argmin(D[i])); h = "location" if D[i, j] <= MATCH_M else "none"
		rows.append(j); how.append(h); dist.append(D[i, j])
	g["match"] = how
	g["match_dist_m"] = np.round(dist, 1)
	g["p20_source_row"] = np.where(g.match != "none", p20.source_row.values[rows], -1)
	g["p20_name_ja"] = np.where(g.match != "none", p20.name_ja.values[rows], "")
	g["floor_area_m2"] = np.where(g.match != "none", p20.floor_area_m2.values[rows], np.nan)
	g["capacity_official"] = np.where(g.match != "none", p20.capacity_official.values[rows], np.nan)
	g["area_source"] = np.where(g.match != "none", "P20 2012", "")
	if estimates is not None:
		est = estimates.set_index(estimates.name_ja.map(_norm))
		for i in np.where((g.match == "none") & ~g.welfare_only)[0]:
			n = _norm(g.name_ja.iat[i])
			if n in est.index:
				g.loc[g.index[i], ["match", "match_dist_m", "floor_area_m2", "area_source"]] = ["estimate", np.nan, est.loc[n, "floor_area_m2"], "estimate: " + est.loc[n, "basis"]]
	g["capacity_3p5m2"] = np.floor(g.floor_area_m2 / m2_per_person)

	reason = np.select(
		[g.welfare_only, g.match.eq("none"), g.floor_area_m2.isna()],
		["welfare room (要配慮者 only)", f"no P20 site by name or within {MATCH_M} m", "P20 site has no floor area"], "")
	# two GSI shelters on one P20 building: keep the first, the building is counted once
	ok = (reason == "") & (g.p20_source_row.values >= 0)      # estimates have no P20 building
	dup = np.zeros(len(g), bool)
	dup[ok] = pd.Series(g.p20_source_row.values[ok]).duplicated().values
	reason = np.where(dup, "same P20 building as an earlier shelter", reason)
	g["included"] = reason == ""
	g["exclusion_reason"] = reason

	hazards = ["洪水", "崖崩れ、土石流及び地滑り", "高潮", "地震", "津波", "大規模な火事", "内水氾濫", "火山現象"]
	hz = {h: int(e[h].fillna(0).sum()) for h in hazards}
	cols = ["gsi_id", "name_ja", "address_ja", "lat", "lon", "accepts", "welfare_only", "also_emergency_site",
		"match", "match_dist_m", "p20_source_row", "p20_name_ja", "floor_area_m2", "capacity_official",
		"area_source", "capacity_3p5m2", "included", "exclusion_reason"]
	return g[cols], hz, len(e)
