"""Task 2b — +meteo / +meteo+static configs for the feature-based ML methods,
using the EXACT main-table sampling of run_1b.py (Munich, same seed/timestamps/
masks/hidden pixels).

Replays run_1b.py's rng sequence verbatim (identical rng.choice(full_idx) +
identical per-(ti,split,bin) random_mask_for_bin + rng.choice hide subsample),
then adds +meteo / +meteo+static for RandomForest / XGBoost / RegressionKriging
on the SAME observed/hidden pixels.

Fairness check: recomputes XGBoost base and XGBoost+static; these MUST match the
published main-table values (e.g. 0-25%% XGBoost base=0.0356). If they match the
rng replay is correct and the +meteo numbers are comparable.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import Field, _load_static, load_ta_field, ta_cache_root
from common.masks import BIN_LABELS, random_mask_for_bin
from common.baselines import BASELINES, standardize_feats
from common.covariates import (
    DRIVERS,
    fit_era5_stats_stream,
    load_era5_aligned as load_era5_aligned_fast,
    standardize_era5,
)

NBINS = len(BIN_LABELS)
DEFAULTS = dict(years=[2023, 2024, 2025], n_times=40, n_splits=4,
                max_pred=500, seed=42,
                stat_years=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])


def load_ta_field_fast(city: str, years) -> Field:
    """Prefer HOSTRADA v7 npy caches, fallback to the parquet loader.

    The common loader rebuilds a long-table merge/pivot from monthly parquet files.
    The v7 cache already stores the same full grid as dense arrays and is the
    stable path used by the DL baselines.
    """
    years = [int(y) for y in years]
    cache_root = ta_cache_root(city)
    derived = cache_root / ("bench_1b_field_" + "_".join(map(str, years)))
    if ((derived / "done.json").exists() and (derived / "values.npy").exists()
            and (derived / "times.npy").exists() and (derived / "xy.npy").exists()
            and (derived / "pixel_ids.npy").exists()):
        values = np.load(derived / "values.npy", mmap_mode="r")
        xy_m = np.asarray(np.load(derived / "xy.npy", mmap_mode="r"), dtype=np.float64)
        pixel_ids = np.asarray(np.load(derived / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64).ravel()
        dt = pd.to_datetime(np.load(derived / "times.npy", mmap_mode="r"), unit="s").to_numpy()
        static_xy, feats, names, _ = _load_static(city)
        srow = {tuple(map(float, xy)): i for i, xy in enumerate(static_xy)}
        order = np.asarray([srow[tuple(map(float, xy))] for xy in xy_m], dtype=np.int64)
        return Field(values, (xy_m / 1000.0).astype(np.float32), feats[order],
                     names, dt, pixel_ids)

    cache_dirs = [cache_root / f"v7_{y}" for y in years]
    if not all((d / "done.json").exists() and (d / "uhi.npy").exists()
               and (d / "times.npy").exists() and (d / "xy.npy").exists()
               and (d / "pixel_ids.npy").exists()
               for d in cache_dirs):
        return load_ta_field(city, years=years)

    derived.mkdir(parents=True, exist_ok=True)
    shapes = [np.load(d / "uhi.npy", mmap_mode="r").shape for d in cache_dirs]
    total_t = int(sum(s[0] for s in shapes))
    n_pix = int(shapes[0][1])
    values_mm = np.lib.format.open_memmap(
        derived / "values.npy", mode="w+", dtype=np.float32, shape=(total_t, n_pix)
    )
    times_parts = []
    xy_m = np.asarray(np.load(cache_dirs[0] / "xy.npy", mmap_mode="r"), dtype=np.float64)
    pixel_ids = np.asarray(np.load(cache_dirs[0] / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64).ravel()
    off = 0
    for d in cache_dirs:
        uhi = np.load(d / "uhi.npy", mmap_mode="r")
        nt = uhi.shape[0]
        print(f"    [cache] building {derived.name}: {d.name} ({nt}h)", flush=True)
        values_mm[off:off + nt] = uhi[:, :, 1]
        off += nt
        times_parts.append(np.asarray(np.load(d / "times.npy", mmap_mode="r"), dtype=np.int64))
    values_mm.flush()
    times_raw = np.concatenate(times_parts)
    np.save(derived / "times.npy", times_raw)
    np.save(derived / "xy.npy", xy_m.astype(np.float32))
    np.save(derived / "pixel_ids.npy", pixel_ids)
    (derived / "done.json").write_text(json.dumps(
        {"city": city, "years": years, "shape": [total_t, n_pix], "source": "hostrada_v7"},
        indent=2,
    ))
    values = np.load(derived / "values.npy", mmap_mode="r")
    dt = pd.to_datetime(times_raw, unit="s").to_numpy()

    static_xy, feats, names, _ = _load_static(city)
    srow = {tuple(map(float, xy)): i for i, xy in enumerate(static_xy)}
    order = np.asarray([srow[tuple(map(float, xy))] for xy in xy_m], dtype=np.int64)
    return Field(values, (xy_m / 1000.0).astype(np.float32), feats[order],
                 names, dt, pixel_ids)


def run_one(city, years, stat_years, n_times, n_splits, max_pred, seed, methods):
    rng = np.random.default_rng(seed)
    print(f"\n{'='*60}\n[1b/meteo] {city}  loading AirT {years} ...", flush=True)
    t0 = time.time()
    fld = load_ta_field_fast(city, years=years)
    feats_std = standardize_feats(fld.feats)
    xy = fld.xy_km.astype(np.float64)
    N = fld.values.shape[1]
    print(f"    values {fld.values.shape}  ({time.time()-t0:.0f}s)", flush=True)

    cov = np.isnan(fld.values).mean(1)
    full_idx = np.where(cov < 0.01)[0]
    if len(full_idx) > n_times:
        full_idx = rng.choice(full_idx, n_times, replace=False)   # IDENTICAL to run_1b.py
    eval_times = fld.times[full_idx]
    print(f"    eval ts: {len(full_idx)}  fitting ERA5 scaler ...", flush=True)
    era_stats = fit_era5_stats_stream(city, stat_years, fld.pixel_ids)
    print(f"    loading eval ERA5 ({years}) ...", flush=True)
    era = load_era5_aligned_fast(city, years, fld.pixel_ids, pd.DatetimeIndex(eval_times))
    era_std, _ = standardize_era5(era, era_stats)

    METEO_M = [m for m in ["RandomForest", "XGBoost", "RegressionKriging"] if m in methods]
    print(f"    methods={METEO_M}", flush=True)
    ae = {m: {cfg: {str(b): [] for b in range(NBINS)} for cfg in ("meteo", "meteo+static")}
          for m in METEO_M}
    chk = {"XGBoost": {str(b): [] for b in range(NBINS)},
           "XGBoost_static": {str(b): [] for b in range(NBINS)}}

    for ei, ti in enumerate(full_idx):
        scene = fld.values[ti].astype(np.float64)
        if not np.isfinite(scene).all():
            scene = np.where(np.isfinite(scene), scene, np.nanmean(scene))
        era_scene = era_std[ei]
        for s in range(n_splits):
            for b in range(NBINS):
                hide_idx = np.where(random_mask_for_bin(N, b, rng))[0]   # IDENTICAL rng
                obs_idx = np.setdiff1d(np.arange(N), hide_idx)
                if len(hide_idx) > max_pred:
                    hide_eval = rng.choice(hide_idx, max_pred, replace=False)  # IDENTICAL rng
                else:
                    hide_eval = hide_idx
                co, vo = xy[obs_idx], scene[obs_idx]
                cp = xy[hide_eval]; yt = scene[hide_eval]
                fo, fp = feats_std[obs_idx], feats_std[hide_eval]
                eo, ep = era_scene[obs_idx], era_scene[hide_eval]

                chk["XGBoost"][str(b)].append(np.abs(BASELINES["XGBoost"](co, vo, cp) - yt))
                chk["XGBoost_static"][str(b)].append(np.abs(
                    BASELINES["XGBoost"](co, vo, cp, feat_obs=fo, feat_pred=fp) - yt))
                for name in METEO_M:
                    f = BASELINES[name]
                    ae[name]["meteo"][str(b)].append(np.abs(f(co, vo, cp, feat_obs=eo, feat_pred=ep) - yt))
                    eo_s = np.hstack([fo, eo]); ep_s = np.hstack([fp, ep])
                    ae[name]["meteo+static"][str(b)].append(np.abs(f(co, vo, cp, feat_obs=eo_s, feat_pred=ep_s) - yt))
        if (ei + 1) % 5 == 0:
            print(f"    timestamps done: {ei+1}/{len(full_idx)}", flush=True)

    def agg(store, key):
        return {str(b): (lambda a: {"MAE": float(a.mean()) if a.size else None,
                                    "RMSE": float(np.sqrt((a**2).mean())) if a.size else None,
                                    "N": int(a.size)})(np.concatenate(store[key][str(b)]) if store[key][str(b)] else np.array([]))
                for b in range(NBINS)}

    out = {"city": city, "bins": BIN_LABELS, "axis": "missing%", "drivers": DRIVERS,
           "protocol": {"years": list(map(int, years)), "n_times": int(len(full_idx)),
                        "eval_years": list(map(int, years)),
                        "stat_years": list(map(int, stat_years)),
                        "mode": "scene-wise transductive",
                        "note": "Each eval scene uses only visible pixels at that timestamp.",
                        "requested_n_times": int(n_times), "n_splits": int(n_splits),
                        "max_pred": int(max_pred), "seed": int(seed)},
           "sanity": {"XGBoost": agg(chk, "XGBoost"), "XGBoost_static": agg(chk, "XGBoost_static")},
           "methods": {}}
    for name in METEO_M:
        out["methods"][name] = {cfg: agg(ae[name], cfg) if False else
                                {str(b): (lambda a: {"MAE": float(a.mean()) if a.size else None,
                                                     "RMSE": float(np.sqrt((a**2).mean())) if a.size else None,
                                                     "N": int(a.size)})(np.concatenate(ae[name][cfg][str(b)]) if ae[name][cfg][str(b)] else np.array([]))
                                 for b in range(NBINS)} for cfg in ("meteo", "meteo+static")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="munich")
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULTS["years"])
    ap.add_argument("--stat_years", type=int, nargs="+", default=DEFAULTS["stat_years"])
    ap.add_argument("--n_times", type=int, default=DEFAULTS["n_times"])
    ap.add_argument("--n_splits", type=int, default=DEFAULTS["n_splits"])
    ap.add_argument("--max_pred", type=int, default=DEFAULTS["max_pred"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--methods", default="XGBoost,RandomForest,RegressionKriging",
                    help="subset of covariate-capable ML methods")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]

    res = run_one(a.city, a.years, a.stat_years, a.n_times, a.n_splits,
                  a.max_pred, a.seed, methods)
    out_path = out_dir / f"1b_{a.city}_meteo.json"
    if out_path.exists():                                  # merge across incremental --methods runs
        prev = json.loads(out_path.read_text())
        if prev.get("protocol") == res.get("protocol"):
            prev.setdefault("methods", {}).update(res["methods"])
            prev["sanity"] = res["sanity"]
            res = prev
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n--- {a.city}  sanity (must match main table) ---", flush=True)
    print(f"{'config':>16}" + "".join(f"{BIN_LABELS[b]:>12}" for b in range(NBINS)), flush=True)
    for cfg in ("XGBoost", "XGBoost_static"):
        print(f"{cfg:>16}" + "".join(f"{res['sanity'][cfg][str(b)]['MAE'] or float('nan'):>12.4f}"
                                     for b in range(NBINS)), flush=True)
    print(f"\n--- {a.city}  +meteo / +meteo+static MAE ---", flush=True)
    for name in res["methods"]:
        for cfg in ("meteo", "meteo+static"):
            print(f"{name+'+'+cfg:>22}" + "".join(
                f"{res['methods'][name][cfg][str(b)]['MAE'] or float('nan'):>12.4f}" for b in range(NBINS)), flush=True)
    print(f"\n[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
