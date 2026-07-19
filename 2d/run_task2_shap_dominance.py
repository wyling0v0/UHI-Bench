"""Task 2d TreeSHAP dominance diagnostics.

This script reuses the Task 2d anomaly protocol and trains the same
dynamic+static XGBoost model used in Arm B. It then computes:

  * XGBoost TreeSHAP feature contributions (pred_contribs=True)
  * XGBoost pairwise TreeSHAP interactions (pred_interactions=True)

Outputs are written as CSV/JSON/Markdown plus a compact factor heatmap. The
results are model diagnostics for the sampled anomaly experiment, not causal
effects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

import run_2_attribution as task2  # noqa: E402


DYNAMIC_FACTORS = task2.DRIVERS
STATIC_FACTORS = [
    "BCR",
    "road_density",
    "poi_density",
    "nightlight",
    "ndvi",
    "water_ratio",
    "distance_to_waterbody",
    "mean_height",
    "dem",
    "wind_exposure_proxy",
]
FACTOR_ORDER = DYNAMIC_FACTORS + STATIC_FACTORS


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def factor_for_feature(name: str) -> str:
    for driver in DYNAMIC_FACTORS:
        if name.startswith(f"{driver}_"):
            return driver
    return name


def factor_family(factor: str) -> str:
    return "dynamic" if factor in DYNAMIC_FACTORS else "static"


def aggregate_feature_shap(mean_abs: np.ndarray, mean_signed: np.ndarray, feature_names: list[str]) -> dict:
    out = {
        factor: {"shap_abs": 0.0, "shap_signed": 0.0}
        for factor in FACTOR_ORDER
    }
    for i, name in enumerate(feature_names):
        factor = factor_for_feature(name)
        if factor not in out:
            out[factor] = {"shap_abs": 0.0, "shap_signed": 0.0}
        out[factor]["shap_abs"] += float(mean_abs[i])
        out[factor]["shap_signed"] += float(mean_signed[i])
    total = sum(v["shap_abs"] for v in out.values()) + 1e-12
    for factor, vals in out.items():
        vals["shap_fraction"] = float(vals["shap_abs"] / total)
        vals["family"] = factor_family(factor)
    return out


def summarize_response(x_eval: np.ndarray, pred: np.ndarray, feature_names: list[str]) -> dict[str, float | None]:
    dynamic = task2.high_low_response(
        x_eval,
        pred,
        feature_names,
        {driver: [f"{driver}_cur"] for driver in DYNAMIC_FACTORS},
    )
    static = task2.high_low_response(
        x_eval,
        pred,
        feature_names,
        {name: [name] for name in STATIC_FACTORS},
    )
    merged = {}
    for factor in FACTOR_ORDER:
        payload = dynamic.get(factor) or static.get(factor)
        merged[factor] = None if payload is None else float(payload["high_minus_low_pred"])
    return merged


def aggregate_interactions(interactions: np.ndarray, feature_names: list[str]) -> list[dict]:
    """Aggregate upper-triangle pairwise TreeSHAP interactions by factor pair."""
    n_feat = len(feature_names)
    factors = [factor_for_feature(name) for name in feature_names]
    pair_abs: dict[tuple[str, str], float] = {}
    pair_signed: dict[tuple[str, str], float] = {}

    mat = interactions[:, :n_feat, :n_feat]
    for i in range(n_feat):
        fi = factors[i]
        for j in range(i + 1, n_feat):
            fj = factors[j]
            if fi == fj:
                continue
            pair = tuple(sorted((fi, fj)))
            vals = mat[:, i, j]
            pair_abs[pair] = pair_abs.get(pair, 0.0) + float(np.mean(np.abs(vals)))
            pair_signed[pair] = pair_signed.get(pair, 0.0) + float(np.mean(vals))

    total = sum(pair_abs.values()) + 1e-12
    rows = []
    for pair, val in pair_abs.items():
        rows.append({
            "factor_a": pair[0],
            "factor_b": pair[1],
            "interaction_abs": float(val),
            "interaction_signed": float(pair_signed[pair]),
            "interaction_fraction": float(val / total),
        })
    return sorted(rows, key=lambda r: (-r["interaction_abs"], r["factor_a"], r["factor_b"]))


def train_city_shap(args, city: str, target: str) -> dict:
    years = list(range(args.start_year, args.end_year + 1))
    print(f"[load] {target} {city}", flush=True)
    if args.sample_windows_per_year > 0:
        data = task2.load_windowed_city_data(
            city,
            years,
            args.n_pixels,
            args.seed,
            windows_per_year=args.sample_windows_per_year,
            window_days=args.window_days,
            target=target,
        )
    else:
        data = task2.load_city_data(city, years, args.n_pixels, args.seed, target=target)

    train_mask, eval_mask = task2.split_masks(data.times, data.sampleable)
    clim = task2.fit_climatology(data.uhi[train_mask], data.times[train_mask], min_count=args.clim_min_count)
    clim_pred = task2.predict_climatology(clim, data.times)
    anomaly = (data.uhi - clim_pred).astype(np.float32)
    era5_z, _, _ = task2.fit_era5_stats(data.era5, train_mask)
    static_z, _, _ = task2.fit_static_stats(data.static)

    t_tr, p_tr = task2.sample_rows(anomaly, train_mask, None, args.max_train_samples, args.seed + 101)
    t_ev, p_ev = task2.sample_rows(anomaly, eval_mask, None, args.max_eval_samples, args.seed + 202)
    xd_tr, dyn_names = task2.build_dynamic_features(era5_z, t_tr, p_tr)
    xd_ev, _ = task2.build_dynamic_features(era5_z, t_ev, p_ev)
    xs_tr = task2.build_static_rows(static_z, p_tr)
    xs_ev = task2.build_static_rows(static_z, p_ev)
    x_tr = np.column_stack([xd_tr, xs_tr]).astype(np.float32)
    x_ev = np.column_stack([xd_ev, xs_ev]).astype(np.float32)
    feature_names = dyn_names + data.static_names
    y_tr = task2.target_values(anomaly, t_tr, p_tr)
    y_ev = task2.target_values(anomaly, t_ev, p_ev)

    print(f"[fit] {target} {city}: train={len(y_tr)} eval={len(y_ev)} features={len(feature_names)}", flush=True)
    model = task2.fit_xgb_regressor(x_tr, y_tr, seed=args.seed)
    pred = model.predict(x_ev)
    metrics = task2.regression_metrics(y_ev, pred)
    response = summarize_response(x_ev, pred, feature_names)

    booster = model.get_booster()
    contrib = booster.predict(xgb.DMatrix(x_ev), pred_contribs=True)
    mean_abs = np.mean(np.abs(contrib[:, :-1]), axis=0)
    mean_signed = np.mean(contrib[:, :-1], axis=0)
    factor_shap = aggregate_feature_shap(mean_abs, mean_signed, feature_names)

    if len(x_ev) > args.max_interaction_samples:
        rng = np.random.default_rng(args.seed + 303)
        take = np.sort(rng.choice(len(x_ev), size=args.max_interaction_samples, replace=False))
        x_inter = x_ev[take]
    else:
        x_inter = x_ev
    print(f"[interactions] {target} {city}: n={len(x_inter)}", flush=True)
    interactions = booster.predict(xgb.DMatrix(x_inter), pred_interactions=True)
    interaction_rows = aggregate_interactions(interactions, feature_names)

    koppen, group, dist = task2.CITIES[city]
    ranked = sorted(
        [
            {
                "factor": factor,
                **vals,
                "high_minus_low_pred": response.get(factor),
            }
            for factor, vals in factor_shap.items()
        ],
        key=lambda r: (-r["shap_abs"], r["factor"]),
    )
    total_abs = sum(r["shap_abs"] for r in ranked) + 1e-12
    dynamic_abs = sum(r["shap_abs"] for r in ranked if r["family"] == "dynamic")
    static_abs = sum(r["shap_abs"] for r in ranked if r["family"] == "static")
    for row in ranked:
        row["shap_fraction"] = float(row["shap_abs"] / total_abs)

    return {
        "city": city,
        "target": target,
        "label_kind": data.label_kind,
        "koppen": koppen,
        "climate_group": group,
        "climate_dist_from_DE": dist,
        "metrics": metrics,
        "samples": {"train": int(len(y_tr)), "eval": int(len(y_ev)), "interaction": int(len(x_inter))},
        "dominance": {
            "dynamic_shap_fraction": float(dynamic_abs / total_abs),
            "static_shap_fraction": float(static_abs / total_abs),
            "top_factors": ranked[:8],
            "top_interactions": interaction_rows[:12],
        },
        "all_factors": ranked,
    }


def write_tables(results: list[dict], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    factor_rows = []
    interaction_rows = []

    for res in results:
        top = res["dominance"]["top_factors"]
        inter = res["dominance"]["top_interactions"]
        row = {
            "target": res["target"],
            "city": res["city"],
            "koppen": res["koppen"],
            "climate_group": res["climate_group"],
            "mae": res["metrics"]["MAE"],
            "rmse": res["metrics"]["RMSE"],
            "dynamic_shap_fraction": res["dominance"]["dynamic_shap_fraction"],
            "static_shap_fraction": res["dominance"]["static_shap_fraction"],
        }
        for i in range(3):
            if i < len(top):
                row[f"top{i + 1}_factor"] = top[i]["factor"]
                row[f"top{i + 1}_family"] = top[i]["family"]
                row[f"top{i + 1}_fraction"] = top[i]["shap_fraction"]
                row[f"top{i + 1}_signed"] = top[i]["shap_signed"]
                row[f"top{i + 1}_high_minus_low"] = top[i]["high_minus_low_pred"]
        if inter:
            row["top_interaction"] = f"{inter[0]['factor_a']} x {inter[0]['factor_b']}"
            row["top_interaction_fraction"] = inter[0]["interaction_fraction"]
        summary_rows.append(row)

        for frow in res["all_factors"]:
            factor_rows.append({
                "target": res["target"],
                "city": res["city"],
                "koppen": res["koppen"],
                "climate_group": res["climate_group"],
                **frow,
            })
        for irow in res["dominance"]["top_interactions"]:
            interaction_rows.append({
                "target": res["target"],
                "city": res["city"],
                "koppen": res["koppen"],
                "climate_group": res["climate_group"],
                **irow,
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(["target", "city"])
    factor_df = pd.DataFrame(factor_rows).sort_values(["target", "city", "shap_abs"], ascending=[True, True, False])
    interaction_df = pd.DataFrame(interaction_rows).sort_values(["target", "city", "interaction_abs"], ascending=[True, True, False])

    summary_df.to_csv(out_dir / "task2_shap_city_dominance_summary.csv", index=False)
    factor_df.to_csv(out_dir / "task2_shap_factor_importance_long.csv", index=False)
    interaction_df.to_csv(out_dir / "task2_shap_interactions_long.csv", index=False)
    return summary_df, factor_df, interaction_df


def plot_heatmaps(factor_df: pd.DataFrame, out_dir: Path) -> None:
    for target in sorted(factor_df["target"].unique()):
        sub = factor_df[factor_df["target"] == target]
        cities = sorted(sub["city"].unique())
        mat = np.zeros((len(cities), len(FACTOR_ORDER)), dtype=float)
        for i, city in enumerate(cities):
            csub = sub[sub["city"] == city].set_index("factor")
            for j, factor in enumerate(FACTOR_ORDER):
                if factor in csub.index:
                    mat[i, j] = float(csub.loc[factor, "shap_fraction"])
        fig_w = max(12, len(FACTOR_ORDER) * 0.7)
        fig_h = max(6, len(cities) * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.25, float(mat.max())))
        ax.set_title(f"Task 2d TreeSHAP factor dominance ({target.upper()}-UHI anomaly)")
        ax.set_yticks(np.arange(len(cities)))
        ax.set_yticklabels(cities)
        ax.set_xticks(np.arange(len(FACTOR_ORDER)))
        ax.set_xticklabels(FACTOR_ORDER, rotation=45, ha="right")
        ax.set_xlabel("Factor")
        ax.set_ylabel("City")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Mean |SHAP| fraction")
        fig.savefig(out_dir / f"task2_shap_factor_heatmap_{target}.png", dpi=300)
        fig.savefig(out_dir / f"task2_shap_factor_heatmap_{target}.pdf")
        plt.close(fig)


def write_markdown(summary_df: pd.DataFrame, factor_df: pd.DataFrame, interaction_df: pd.DataFrame, out_dir: Path) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        text = df.astype(str).fillna("")
        header = "| " + " | ".join(text.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
        rows = [
            "| " + " | ".join(str(v).replace("|", "/") for v in row) + " |"
            for row in text.to_numpy()
        ]
        return "\n".join([header, sep, *rows])

    lines = [
        "# Task 2d TreeSHAP dominance diagnostics",
        "",
        "TreeSHAP is computed from the dynamic+static XGBoost model on the sampled UHI anomaly target.",
        "The target is UHI minus pixel x hour-of-day x month train climatology. Values are model diagnostics, not causal effects.",
        "",
    ]
    for target in sorted(summary_df["target"].unique()):
        lines.extend([f"## {target.upper()}-UHI", ""])
        cols = [
            "city",
            "koppen",
            "climate_group",
            "top1_factor",
            "top1_fraction",
            "top2_factor",
            "top2_fraction",
            "top3_factor",
            "top3_fraction",
            "dynamic_shap_fraction",
            "static_shap_fraction",
            "top_interaction",
        ]
        table = summary_df[summary_df["target"] == target][cols].copy()
        for col in table.columns:
            if table[col].dtype.kind in "fc":
                table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        lines.append(markdown_table(table))
        lines.append("")

        counts = (
            summary_df[summary_df["target"] == target]["top1_factor"]
            .value_counts()
            .rename_axis("top1_factor")
            .reset_index(name="city_count")
        )
        lines.append("Top-1 factor frequency:")
        lines.append(markdown_table(counts))
        lines.append("")

    lines.extend([
        "## Output files",
        "",
        "- `task2_shap_city_dominance_summary.csv`: one row per city-target.",
        "- `task2_shap_factor_importance_long.csv`: all factor-level SHAP importances.",
        "- `task2_shap_interactions_long.csv`: top pairwise TreeSHAP interactions.",
        "- `task2_shap_factor_heatmap_ta.png/pdf` and `task2_shap_factor_heatmap_lst.png/pdf`: city x factor heatmaps.",
        "",
    ])
    (out_dir / "TASK2_SHAP_DOMINANCE.md").write_text("\n".join(lines))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["all16"])
    ap.add_argument("--targets", nargs="+", choices=["ta", "lst"], default=["ta", "lst"])
    ap.add_argument("--n_pixels", type=int, default=64)
    ap.add_argument("--max_train_samples", type=int, default=10000)
    ap.add_argument("--max_eval_samples", type=int, default=3000)
    ap.add_argument("--max_interaction_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_year", type=int, default=2015)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--sample_windows_per_year", type=int, default=12)
    ap.add_argument("--window_days", type=int, default=5)
    ap.add_argument("--clim_min_count", type=int, default=20)
    ap.add_argument("--out_dir", default=str(THIS_DIR / "results" / "shap_dominance"))
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cities = task2.expand_cities(args.cities)
    results = []
    skipped = []
    for target in args.targets:
        for city in cities:
            try:
                results.append(train_city_shap(args, city, target))
            except Exception as exc:  # Keep long all-city runs from failing completely.
                skipped.append({"target": target, "city": city, "reason": repr(exc)})
                print(f"[skip] {target} {city}: {exc!r}", flush=True)

    payload = {
        "task": "Task 2d TreeSHAP dominance diagnostics",
        "targets": args.targets,
        "cities": cities,
        "protocol": {
            "n_pixels": args.n_pixels,
            "train_years": [args.start_year, 2022],
            "eval_years": [2023, args.end_year],
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_interaction_samples": args.max_interaction_samples,
            "sample_windows_per_year": args.sample_windows_per_year,
            "window_days": args.window_days,
            "model": "XGBoost dynamic_plus_static, n_estimators=40, max_depth=3",
            "shap": "XGBoost TreeSHAP via pred_contribs=True and pred_interactions=True",
        },
        "results": results,
        "skipped": skipped,
    }
    (out_dir / "task2_shap_dominance.json").write_text(json.dumps(payload, indent=2, default=_json_default))
    summary_df, factor_df, interaction_df = write_tables(results, out_dir)
    plot_heatmaps(factor_df, out_dir)
    write_markdown(summary_df, factor_df, interaction_df, out_dir)
    print(f"[write] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
