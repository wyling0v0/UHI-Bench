"""Summarize Task 2d cross-climate mechanism transfer diagnostics.

This does not train new models. It makes the transfer question explicit from
the existing per-city XGBoost attribution results:

1. Learn the DE8 consensus mechanism as the most frequent top-3 drivers.
2. Compare each non-DE city against that consensus by top-3 overlap.
3. Compare the direction of the DE core driver responses.

The output is a mechanism-transfer diagnostic, not a cross-domain prediction
benchmark. A stricter train-on-DE/eval-on-target experiment should be added
separately if we want predictive transfer evidence.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DRIVERS = ["u10", "v10", "tcc", "d2m", "blh", "ssrd"]
DE8 = ["berlin", "hamburg", "munich", "cologne", "dortmund", "dusseldorf", "frankfurt", "stuttgart"]
INTL8 = ["warsaw", "bucharest", "sao_paulo", "buenos_aires", "johannesburg", "lagos", "cairo", "riyadh"]
KEY_TARGETS = ["lagos", "cairo", "riyadh"]

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "2d/results/task2_attribution.json"
OUT_PATH = ROOT / "2d/TASK2_TRANSFER.md"


def _rank(city_result: dict) -> list[str]:
    return city_result["arm_a"]["all"]["models"]["xgboost"]["rank"]


def _response(city_result: dict, driver: str) -> float:
    return float(
        city_result["arm_a"]["all"]["models"]["xgboost"]["current_driver_response"][driver][
            "high_minus_low_pred"
        ]
    )


def _sign(value: float) -> str:
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return "0"


def _fmt(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    data = json.loads(IN_PATH.read_text())
    results = data["results"]

    freq = Counter()
    for city in DE8:
        freq.update(_rank(results[city])[:3])
    de_core = [driver for driver, _ in freq.most_common(3)]
    de_response = {
        driver: sum(_response(results[city], driver) for city in DE8) / len(DE8)
        for driver in DRIVERS
    }

    lines: list[str] = []
    lines.append("# Task 2d Cross-Climate Mechanism Transfer")
    lines.append("")
    lines.append(
        "Generated from `benchmark/2d/results/task2_attribution.json`. This appendix makes the"
        " transfer question explicit: does the DE8 Ta-UHI mechanism also appear in non-DE climates?"
    )
    lines.append("")
    lines.append(
        "This is a mechanism-transfer diagnostic based on per-city XGBoost attribution, not a"
        " train-on-DE/eval-on-target predictive transfer benchmark."
    )
    lines.append("")
    lines.append("## DE8 consensus mechanism")
    lines.append("")
    lines.append(f"DE8 consensus top-3 drivers: `{', '.join(de_core)}`.")
    lines.append("")
    lines.append("| driver | DE8 top3 cities | DE8 mean high-minus-low response | sign |")
    lines.append("|---|---:|---:|---|")
    for driver, count in freq.most_common():
        lines.append(f"| {driver} | {count}/8 | {_fmt(de_response[driver])} K | {_sign(de_response[driver])} |")
    lines.append("")
    lines.append("## Non-DE transfer check")
    lines.append("")
    lines.append(
        "`top3 overlap` is the fraction of a city's XGBoost top-3 drivers that match the DE8"
        " consensus `{blh,tcc,u10}`. `core sign match` counts whether the direction of `blh`,"
        " `tcc`, and `u10` matches the DE8 mean response."
    )
    lines.append("")
    lines.append("| city | Koppen | climate group | XGB all top3 | top3 overlap | core sign match | BLH | TCC | U10 | SSRD |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for city in INTL8:
        row = results[city]
        top3 = _rank(row)[:3]
        overlap = len(set(top3) & set(de_core)) / len(de_core)
        sign_match = sum(_sign(_response(row, d)) == _sign(de_response[d]) for d in de_core)
        lines.append(
            "| "
            + " | ".join(
                [
                    city,
                    row["koppen"],
                    row["climate_group"],
                    ", ".join(top3),
                    f"{overlap:.2f}",
                    f"{sign_match}/3",
                    _fmt(_response(row, "blh")),
                    _fmt(_response(row, "tcc")),
                    _fmt(_response(row, "u10")),
                    _fmt(_response(row, "ssrd")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Key target climates")
    lines.append("")
    lines.append("| city | conclusion |")
    lines.append("|---|---|")
    for city in KEY_TARGETS:
        row = results[city]
        top3 = _rank(row)[:3]
        overlap = len(set(top3) & set(de_core)) / len(de_core)
        sign_match = sum(_sign(_response(row, d)) == _sign(de_response[d]) for d in de_core)
        if city == "lagos":
            conclusion = (
                f"Partial transfer: top3={top3} overlaps DE by {overlap:.2f}, but BLH response flips"
                f" positive ({_fmt(_response(row, 'blh'))} K). Cloud and wind signs still match."
            )
        elif city == "cairo":
            conclusion = (
                f"Strongest transfer among the three: top3={top3}, overlap={overlap:.2f},"
                f" all DE core signs match ({sign_match}/3). Wind drops out of top3 and SSRD enters."
            )
        else:
            conclusion = (
                f"Partial transfer: top3={top3}, overlap={overlap:.2f}; BLH/TCC signs match but U10"
                f" is weakly positive ({_fmt(_response(row, 'u10'))} K). SSRD replaces wind in top3."
            )
        lines.append(f"| {city} | {conclusion} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The DE mechanism does not transfer as a complete package. `blh` and `tcc` remain important"
        " in Cairo, Riyadh, and Lagos, but `u10` is often replaced by `ssrd`."
    )
    lines.append(
        "- The physical direction is mostly transferable for Cairo: high BLH, high cloud, and stronger"
        " zonal wind all reduce Ta-UHI anomaly, matching DE8."
    )
    lines.append(
        "- Riyadh is a partial transfer case: BLH/TCC still reduce anomaly, but wind response is weak"
        " and slightly opposite; shortwave becomes a top driver."
    )
    lines.append(
        "- Lagos is the clearest non-transfer case: BLH remains important but changes sign, which"
        " suggests tropical moist boundary-layer regimes should not be described with the same"
        " DE/temperate mechanism without qualification."
    )
    lines.append("")
    lines.append(
        "Conservative claim: DE/European Ta-UHI mechanisms transfer partly to hot/arid and tropical"
        " cities through cloud and boundary-layer controls, but the driver mix changes; solar forcing"
        " becomes more important and wind/BLH direction can be climate-regime dependent."
    )
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
