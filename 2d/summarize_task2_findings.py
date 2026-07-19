"""Generate a compact Task 2d mechanism-readout from saved attribution JSON."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "2d/results"
OUT = ROOT / "2d/TASK2_FINDINGS.md"

DRIVERS = ["blh", "tcc", "u10", "v10", "ssrd", "d2m"]
MODEL_ORDER = [
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "randomforest",
    "extratrees",
    "histgradientboosting",
    "xgboost",
]
STATIC = [
    "road_density",
    "BCR",
    "nightlight",
    "mean_height",
    "dem",
    "distance_to_waterbody",
    "water_ratio",
    "ndvi",
    "wind_exposure_proxy",
    "poi_density",
]


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) and v == v else "-"


def load(name):
    return json.loads((RESULTS / name).read_text())["results"]


def count_top3(res, mode):
    c = Counter()
    for r in res.values():
        c.update(r["arm_a"][mode]["models"]["xgboost"]["rank"][:3])
    return c


def count_top3_model(res, mode, model):
    c = Counter()
    for r in res.values():
        models = r["arm_a"][mode]["models"]
        if model in models:
            c.update(models[model]["rank"][:3])
    return c


def mean_model_mae(res, mode, model):
    vals = []
    for r in res.values():
        try:
            v = r["arm_a"][mode]["models"][model]["eval_metrics"]["MAE"]
        except KeyError:
            continue
        if isinstance(v, (int, float)) and v == v:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def static_top3(res):
    c = Counter()
    for r in res.values():
        st = r["arm_b"]["targets"]["anomaly"]["dynamic_plus_static_attribution"]["static"]
        c.update(st["rank"][:3])
    return c


def summarize_dynamic(res, mode="all"):
    vals = {}
    top = count_top3(res, mode)
    frac = defaultdict(list)
    response = defaultdict(list)
    for r in res.values():
        m = r["arm_a"][mode]["models"]["xgboost"]
        for k, v in m["fraction"].items():
            frac[k].append(v)
        for k, payload in m.get("current_driver_response", {}).items():
            response[k].append(payload["high_minus_low_pred"])
    for k in DRIVERS:
        rs = response.get(k, [])
        vals[k] = {
            "top3": top[k],
            "fraction": sum(frac[k]) / len(frac[k]) if frac[k] else None,
            "response": sum(rs) / len(rs) if rs else None,
            "pos": sum(v > 0 for v in rs),
            "neg": sum(v < 0 for v in rs),
        }
    return vals


def summarize_static(res):
    vals = {}
    top = static_top3(res)
    frac = defaultdict(list)
    response = defaultdict(list)
    for r in res.values():
        st = r["arm_b"]["targets"]["anomaly"]["dynamic_plus_static_attribution"]["static"]
        for k, v in st["fraction"].items():
            frac[k].append(v)
        for k, payload in st.get("response", {}).items():
            response[k].append(payload["high_minus_low_pred"])
    for k in STATIC:
        rs = response.get(k, [])
        vals[k] = {
            "top3": top[k],
            "fraction": sum(frac[k]) / len(frac[k]) if frac[k] else None,
            "response": sum(rs) / len(rs) if rs else None,
            "pos": sum(v > 0 for v in rs),
            "neg": sum(v < 0 for v in rs),
        }
    return vals


def top_counts_line(c):
    return ", ".join(f"{k}({v}/16)" for k, v in c.most_common())


def dynamic_table(title, res, mode="all"):
    s = [f"### {title}", "", "| driver | top3 cities | mean XGB fraction | high-low response | sign consistency |", "|---|---:|---:|---:|---|"]
    dyn = summarize_dynamic(res, mode)
    order = sorted(DRIVERS, key=lambda k: (-dyn[k]["top3"], -(dyn[k]["fraction"] or 0), k))
    for k in order:
        d = dyn[k]
        s.append(f"| {k} | {d['top3']}/16 | {fmt(d['fraction'])} | {fmt(d['response'])} K | {d['pos']}+ / {d['neg']}- |")
    return "\n".join(s) + "\n"


def static_table(title, res):
    s = [f"### {title}", "", "| static feature | top3 cities | mean static fraction | high-low response | sign consistency |", "|---|---:|---:|---:|---|"]
    st = summarize_static(res)
    order = sorted(STATIC, key=lambda k: (-st[k]["top3"], -(st[k]["fraction"] or 0), k))
    for k in order:
        d = st[k]
        s.append(f"| {k} | {d['top3']}/16 | {fmt(d['fraction'])} | {fmt(d['response'])} K | {d['pos']}+ / {d['neg']}- |")
    return "\n".join(s) + "\n"


def temporal_table(title, res):
    s = [f"### {title}", "", "| mode | top3 frequency | interpretation |", "|---|---|---|"]
    interpretations = {
        "ta-daytime": "Daytime Ta still has BLH as universal, but tcc/u10/ssrd compete; shortwave appears only as a secondary daytime term.",
        "ta-nighttime": "Nighttime Ta is cleaner: BLH and tcc are universal, u10 is also frequent; stable boundary layer and ventilation dominate.",
        "lst-daytime": "Daytime LST is diffuse across v10/BLH/tcc/d2m/u10/ssrd, consistent with radiative forcing plus cloud/sampling and wind-regime interactions.",
        "lst-nighttime": "Nighttime LST shifts toward tcc/d2m/wind components; surface cooling, humidity/cloud and air movement become more important than ssrd.",
    }
    key = "ta" if title.startswith("Ta") else "lst"
    for mode in ["daytime", "nighttime"]:
        c = count_top3(res, mode)
        s.append(f"| {mode} | {top_counts_line(c)} | {interpretations[f'{key}-{mode}']} |")
    return "\n".join(s) + "\n"


def model_robustness_table(title, res):
    s = [f"### {title}", "", "| model | mean all-hour MAE | all top3 frequency | daytime top3 frequency | nighttime top3 frequency |", "|---|---:|---|---|---|"]
    for model in MODEL_ORDER:
        if not any(model in r["arm_a"]["all"]["models"] for r in res.values()):
            continue
        mae = mean_model_mae(res, "all", model)
        all_c = top_counts_line(count_top3_model(res, "all", model))
        day_c = top_counts_line(count_top3_model(res, "daytime", model))
        night_c = top_counts_line(count_top3_model(res, "nighttime", model))
        s.append(f"| {model} | {fmt(mae)} | {all_c} | {day_c} | {night_c} |")
    return "\n".join(s) + "\n"


def city_table(title, res):
    s = [f"### {title}", "", "| city | XGB all top3 | BLH response | TCC response | wind response | SSRD response | static top3 |", "|---|---|---:|---:|---:|---:|---|"]
    for city, r in res.items():
        m = r["arm_a"]["all"]["models"]["xgboost"]
        resp = m.get("current_driver_response", {})
        wind = None
        if "u10" in resp and "v10" in resp:
            wind = (resp["u10"]["high_minus_low_pred"] + resp["v10"]["high_minus_low_pred"]) / 2
        st = r["arm_b"]["targets"]["anomaly"]["dynamic_plus_static_attribution"]["static"]["rank"][:3]
        s.append(
            f"| {city} | {', '.join(m['rank'][:3])} | "
            f"{fmt(resp.get('blh', {}).get('high_minus_low_pred'))} | "
            f"{fmt(resp.get('tcc', {}).get('high_minus_low_pred'))} | "
            f"{fmt(wind)} | "
            f"{fmt(resp.get('ssrd', {}).get('high_minus_low_pred'))} | "
            f"{', '.join(st)} |"
        )
    return "\n".join(s) + "\n"


def main():
    ta = load("task2_attribution.json")
    lst = load("task2_lst_attribution.json")
    md = [
        "# Task 2d Mechanism Findings",
        "",
        "Generated from `benchmark/2d/results/task2_attribution.json` and `benchmark/2d/results/task2_lst_attribution.json`.",
        "",
        "Direction columns use the XGBoost diagnostic `E[pred | feature high] - E[pred | feature low]` on sampled evaluation rows. This is model-response evidence, not a causal estimate. Wind is represented by ERA5 `u10/v10` components, so it is only a ventilation/advection proxy.",
        "",
        "## Main Conclusions",
        "",
        "- Ta-UHI anomaly is cross-city stable: `blh` and `tcc` enter the XGBoost top-3 drivers in 16/16 cities. High BLH, high cloud cover, and stronger wind generally reduce Ta-UHI anomaly.",
        "- Cross-climate transfer is partial, not complete: the DE8 consensus mechanism is `blh/tcc/u10`, while Cairo/Riyadh/Lagos keep `blh/tcc` but often replace wind with `ssrd`; Lagos also flips the BLH response sign. See `benchmark/2d/TASK2_TRANSFER.md`.",
        "- This Ta conclusion is not only an XGBoost artifact: RandomForest and ExtraTrees also rank `blh` and `tcc` in the top-3 for almost all cities. Linear/Ridge coefficients are less reliable for mechanism ranking because collinear ERA5 lag blocks push `d2m`/wind components upward.",
        "- LST-UHI anomaly is much more heterogeneous: no driver is top-3 in all cities, and the day/night rank structure is less stable. LST should not be treated as a direct proxy for Air-T UHI mechanisms.",
        "- LST remains heterogeneous across model families. Linear models emphasize `ssrd`/`d2m`/wind, tree ensembles emphasize mixtures of `tcc`/`d2m`/wind/`blh`, and no single dynamic driver has Ta-like cross-city dominance.",
        "- Static features affect anomaly mostly as spatial modifiers. For Ta, road/building/activity proxies dominate. For LST, elevation and distance to water become more prominent, while NDVI/water/BCR are not consistently dominant after pixel-hour-month climatology removal.",
        "- Static feature mechanisms are also summarized as grouped responses (urban morphology, anthropogenic/impervious, blue-green cooling, ventilation, topography) in `benchmark/2d/TASK2_STATIC_MECHANISMS.md`.",
        "- Hour-specific static mechanism diagnostics for 06/12/18 are recorded in `benchmark/2d/TASK2_STATIC_BY_HOUR.md`: Ta static responses are small and can flip sign across the day, while LST responses are larger and more surface/geography dominated.",
        "",
        dynamic_table("Ta-UHI dynamic drivers, all hours", ta),
        model_robustness_table("Ta-UHI Arm A model-family robustness", ta),
        static_table("Ta-UHI static modifiers, dynamic+static model", ta),
        temporal_table("Ta-UHI temporal split", ta),
        city_table("Ta-UHI city-level consistency", ta),
        dynamic_table("LST-UHI dynamic drivers, all hours", lst),
        model_robustness_table("LST-UHI Arm A model-family robustness", lst),
        static_table("LST-UHI static modifiers, dynamic+static model", lst),
        temporal_table("LST-UHI temporal split", lst),
        city_table("LST-UHI city-level consistency", lst),
        "## Conservative Wording",
        "",
        "Use: `Ta-UHI anomalies are controlled by robust atmospheric drivers (BLH/cloud/wind), whereas LST-UHI anomalies show lower cross-city stability and stronger surface/geographic modulation.`",
        "",
        "Avoid: `LST is always dominated by SSRD/NDVI/water/BCR.` The current anomaly experiment does not support that absolute statement.",
        "",
    ]
    OUT.write_text("\n".join(md))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
