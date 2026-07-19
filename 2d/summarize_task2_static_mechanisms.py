"""Summarize Task 2d static features by interpretable mechanism groups.

Inputs are the saved Task 2d attribution JSON files. The main quantity is the
XGBoost high-minus-low response:

    E[pred | mechanism high] - E[pred | mechanism low]

on sampled evaluation rows. Units are K anomaly. This is an association/model
response diagnostic, not a causal estimate.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "2d/TASK2_STATIC_MECHANISMS.md"

MECHANISMS = {
    "urban_morphology_storage": ["BCR", "mean_height"],
    "anthropogenic_impervious": ["road_density", "poi_density", "nightlight"],
    "blue_green_raw": ["water_ratio", "ndvi", "distance_to_waterbody"],
    "ventilation_boundary_exchange": ["wind_exposure_proxy"],
    "topographic_context": ["dem"],
}

# Oriented score where positive means stronger warming-risk / weaker cooling:
# farther from water is positive; more water/NDVI are inverted because they are
# expected cooling features.
BLUE_GREEN_ORIENTATION = {
    "water_ratio": -1.0,
    "ndvi": -1.0,
    "distance_to_waterbody": 1.0,
}

LABELS = {
    "urban_morphology_storage": "Urban morphology / storage",
    "anthropogenic_impervious": "Anthropogenic activity / impervious proxy",
    "blue_green_warming_risk": "Blue-green cooling, oriented as warming risk",
    "ventilation_boundary_exchange": "Ventilation / boundary exchange",
    "topographic_context": "Topographic context",
}


def load(name: str) -> dict:
    return json.loads((ROOT / "2d/results" / name).read_text())["results"]


def static_block(row: dict) -> dict:
    return row["arm_b"]["targets"]["anomaly"]["dynamic_plus_static_attribution"]["static"]


def feature_response(row: dict, feature: str) -> float | None:
    response = static_block(row).get("response", {})
    payload = response.get(feature)
    if not payload:
        return None
    value = payload.get("high_minus_low_pred")
    return float(value) if isinstance(value, (int, float)) else None


def feature_fraction(row: dict, feature: str) -> float:
    return float(static_block(row).get("fraction", {}).get(feature, 0.0))


def mechanism_values(row: dict) -> dict:
    out = {}
    for mech, features in MECHANISMS.items():
        vals = [feature_response(row, f) for f in features]
        vals = [v for v in vals if v is not None]
        frac = sum(feature_fraction(row, f) for f in features)
        out[mech] = {
            "response": sum(vals) / len(vals) if vals else None,
            "fraction": frac,
        }
    vals = []
    for feature, direction in BLUE_GREEN_ORIENTATION.items():
        value = feature_response(row, feature)
        if value is not None:
            vals.append(direction * value)
    out["blue_green_warming_risk"] = {
        "response": sum(vals) / len(vals) if vals else None,
        "fraction": out["blue_green_raw"]["fraction"],
    }
    return out


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for mech in LABELS:
        responses = []
        fractions = []
        for row in rows:
            vals = mechanism_values(row)[mech]
            if vals["response"] is not None:
                responses.append(vals["response"])
            fractions.append(vals["fraction"])
        out[mech] = {
            "response": sum(responses) / len(responses) if responses else None,
            "fraction": sum(fractions) / len(fractions) if fractions else None,
            "n_pos": sum(1 for v in responses if v > 0),
            "n_neg": sum(1 for v in responses if v < 0),
            "n": len(responses),
        }
    return out


def fmt(value: float | None) -> str:
    return "na" if value is None else f"{value:+.4f}"


def frac(value: float | None) -> str:
    return "na" if value is None else f"{100.0 * value:.1f}%"


def climate_table(title: str, results: dict) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    group_cities: dict[str, list[str]] = defaultdict(list)
    for city, row in results.items():
        groups[row["climate_group"]].append(row)
        group_cities[row["climate_group"]].append(city)
    lines = [f"## {title} by climate group", ""]
    lines.append("| climate group | cities | mechanism | response K | static share | sign consistency |")
    lines.append("|---|---:|---|---:|---:|---|")
    for group, rows in groups.items():
        agg = aggregate(rows)
        for mech, payload in agg.items():
            lines.append(
                f"| {group} | {len(rows)} | {LABELS[mech]} | {fmt(payload['response'])} | "
                f"{frac(payload['fraction'])} | {payload['n_pos']}+/{payload['n_neg']}- |"
            )
    lines.append("")
    return "\n".join(lines)


def city_table(title: str, results: dict, cities: list[str]) -> str:
    lines = [f"## {title} selected cities", ""]
    lines.append("| city | Koppen | mechanism | response K | static share |")
    lines.append("|---|---|---|---:|---:|")
    for city in cities:
        if city not in results:
            continue
        row = results[city]
        vals = mechanism_values(row)
        for mech in LABELS:
            payload = vals[mech]
            lines.append(
                f"| {city} | {row['koppen']} | {LABELS[mech]} | "
                f"{fmt(payload['response'])} | {frac(payload['fraction'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ta = load("task2_attribution.json")
    lst = load("task2_lst_attribution.json")
    selected = ["berlin", "munich", "cologne", "sao_paulo", "buenos_aires", "lagos", "cairo", "riyadh"]

    md = [
        "# Task 2d Static Mechanism Groups",
        "",
        "Generated from Task 2d XGBoost dynamic+static attribution. `response K` is the"
        " mean high-minus-low model response within each mechanism group. `static share`"
        " is the summed contribution fraction inside the static-feature block, not the"
        " total model importance. Values are model-response diagnostics, not causal effects.",
        "",
        "For blue-green features, the grouped response is oriented as warming risk:"
        " `-water_ratio`, `-ndvi`, and `+distance_to_waterbody`. Positive means less"
        " blue/green cooling is associated with higher predicted UHI anomaly.",
        "",
        climate_table("Ta-UHI", ta),
        city_table("Ta-UHI", ta, selected),
        climate_table("LST-UHI", lst),
        city_table("LST-UHI", lst, selected),
    ]
    OUT.write_text("\n".join(md))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
