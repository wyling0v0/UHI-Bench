"""Task 2a — +meteo / +meteo+static configs for the feature-based ML methods,
using the EXACT main-table sampling of run_1a.py (same seed/pairs/obs/hide pixels).

Why a companion script: the main table and earlier ERA5 ablations use different
samplings, so their numbers are not directly comparable. This script replays
run_1a.py's rng sequence verbatim
(identical build_cloud_eval_pairs call + identical per-pair rng.choice subsample),
then adds the +meteo and +meteo+static configs for RandomForest / XGBoost /
RegressionKriging on the SAME observed/hidden pixels.

Fairness check: it also recomputes XGBoost base and XGBoost+static; these MUST
match the published main-table values (e.g. Bucharest 0-25%% XGBoost base=1.006).
If they match, the rng replay is correct and the +meteo numbers are comparable.

Configs (DRIVERS = u10,v10,tcc,d2m,blh,ssrd; t2m excluded = temperature itself):
  +meteo        : coords + 6 ERA5 drivers (per-pixel, scene timestamp)
  +meteo+static : coords + 6 ERA5 drivers + 10 static features
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import Field, LST_BASE, _load_static, load_lst_field
from common.masks import build_cloud_eval_pairs, BIN_LABELS
from common.baselines import BASELINES, standardize_feats
from common.covariates import (
    DRIVERS,
    fit_era5_stats_stream,
    load_era5_aligned as load_era5_aligned_fast,
    standardize_era5,
)

# MUST mirror run_1a.py defaults exactly
DEFAULTS = dict(years=[2023, 2024, 2025], n_clear=80, masks_per_bin=20,
                max_pred=800, seed=42,
                stat_years=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])


def load_lst_field_fast(city: str, years) -> Field:
    """Prefer annual LST v7 npy caches, fallback to the parquet pivot loader."""
    years = [int(y) for y in years]
    cache_root = LST_BASE / city / "cache"
    derived = cache_root / ("bench_1a_field_" + "_".join(map(str, years)))
    if ((derived / "done.json").exists() and (derived / "values.npy").exists()
            and (derived / "times.npy").exists() and (derived / "xy_km.npy").exists()
            and (derived / "pixel_ids.npy").exists()):
        values = np.load(derived / "values.npy", mmap_mode="r")
        xy_km = np.asarray(np.load(derived / "xy_km.npy", mmap_mode="r"), dtype=np.float32)
        pixel_ids = np.asarray(np.load(derived / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64).ravel()
        dt = pd.to_datetime(np.load(derived / "times.npy", mmap_mode="r"), unit="s").to_numpy()
        _, feats, names, spids = _load_static(city)
        pid2row = {int(pid): i for i, pid in enumerate(spids)}
        order = np.asarray([pid2row[int(p)] for p in pixel_ids], dtype=np.int64)
        return Field(values, xy_km, feats[order], names, dt, pixel_ids)

    cache_dirs = [cache_root / f"v7_{y}" for y in years]
    if not all((d / "done.json").exists() and (d / "uhi.npy").exists()
               and (d / "times.npy").exists() and (d / "xy_km.npy").exists()
               and (d / "pixel_ids.npy").exists()
               for d in cache_dirs):
        return load_lst_field(city, years=years)

    derived.mkdir(parents=True, exist_ok=True)
    shapes = [np.load(d / "uhi.npy", mmap_mode="r").shape for d in cache_dirs]
    total_t = int(sum(s[0] for s in shapes))
    n_pix = int(shapes[0][1])
    values_mm = np.lib.format.open_memmap(
        derived / "values.npy", mode="w+", dtype=np.float32, shape=(total_t, n_pix)
    )
    xy_km = np.asarray(np.load(cache_dirs[0] / "xy_km.npy", mmap_mode="r"), dtype=np.float32)
    pixel_ids = np.asarray(np.load(cache_dirs[0] / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64).ravel()
    times_parts = []
    off = 0
    for d in cache_dirs:
        uhi = np.load(d / "uhi.npy", mmap_mode="r")
        nt = uhi.shape[0]
        print(f"    [cache] building {derived.name}: {d.name} ({nt}h)", flush=True)
        values_mm[off:off + nt] = uhi[:, :, 0]
        off += nt
        times_parts.append(np.asarray(np.load(d / "times.npy", mmap_mode="r"), dtype=np.int64))
    values_mm.flush()
    times_raw = np.concatenate(times_parts)
    np.save(derived / "times.npy", times_raw)
    np.save(derived / "xy_km.npy", xy_km)
    np.save(derived / "pixel_ids.npy", pixel_ids)
    (derived / "done.json").write_text(json.dumps(
        {"city": city, "years": years, "shape": [total_t, n_pix], "source": "lstuhi_v7"},
        indent=2,
    ))
    values = np.load(derived / "values.npy", mmap_mode="r")
    dt = pd.to_datetime(times_raw, unit="s").to_numpy()
    _, feats, names, spids = _load_static(city)
    pid2row = {int(pid): i for i, pid in enumerate(spids)}
    order = np.asarray([pid2row[int(p)] for p in pixel_ids], dtype=np.int64)
    return Field(values, xy_km, feats[order], names, dt, pixel_ids)


def run_one(city, years, stat_years, n_clear, masks_per_bin, max_pred, seed, methods, use_v7_cache=False):
    rng = np.random.default_rng(seed)
    print(f"\n{'='*60}\n[1a/meteo] {city}  loading LST {years} ...", flush=True)
    t0 = time.time()
    fld = load_lst_field_fast(city, years=years) if use_v7_cache else load_lst_field(city, years=years)
    feats_std = standardize_feats(fld.feats)
    xy = fld.xy_km.astype(np.float64)
    print(f"    values {fld.values.shape}  ({time.time()-t0:.0f}s)")

    # === IDENTICAL pair construction as run_1a.py (same rng) ===
    pairs = build_cloud_eval_pairs(fld.values, fld.times, rng,
                                   n_clear=n_clear, masks_per_bin=masks_per_bin)
    clear_idx = sorted({int(p["clear_t"]) for p in pairs})
    clear_times = fld.times[np.array(clear_idx)]
    print(f"    pairs={len(pairs)}  clear_ts={len(clear_times)}  fitting ERA5 scaler ...", flush=True)
    era_stats = fit_era5_stats_stream(city, stat_years, fld.pixel_ids)
    print(f"    loading eval ERA5 ({years}) ...", flush=True)
    era = load_era5_aligned_fast(city, years, fld.pixel_ids, pd.DatetimeIndex(clear_times))
    era_std, _ = standardize_era5(era, era_stats)
    ct_to_row = {ct: i for i, ct in enumerate(clear_idx)}

    METEO_M = [m for m in ["RandomForest", "XGBoost", "RegressionKriging"] if m in methods]
    print(f"    pairs={len(pairs)}  methods={METEO_M}  ERA5 for {len(clear_times)} clear ts ...", flush=True)
    ae = {m: {cfg: {b: [] for b in range(4)} for cfg in ("meteo", "meteo+static")}
          for m in METEO_M}
    # sanity-check configs (must reproduce main table)
    chk = {"XGBoost": {b: [] for b in range(4)},
           "XGBoost_static": {b: [] for b in range(4)}}

    for pi, pr in enumerate(pairs):
        ct = pr["clear_t"]; m = pr["mask"]; b = pr["bin_idx"]
        scene = fld.values[ct].astype(np.float64)
        era_scene = era_std[ct_to_row[int(ct)]]
        obs_idx = np.where(~m & np.isfinite(scene))[0]
        hide_idx = np.where(m & np.isfinite(scene))[0]
        if len(obs_idx) < 5 or len(hide_idx) < 1:
            continue
        # === IDENTICAL hide subsample as run_1a.py (same rng call) ===
        if len(hide_idx) > max_pred:
            hide_idx = rng.choice(hide_idx, max_pred, replace=False)
        co, vo = xy[obs_idx], scene[obs_idx]
        cp = xy[hide_idx]; yt = scene[hide_idx]
        fo, fp = feats_std[obs_idx], feats_std[hide_idx]
        eo, ep = era_scene[obs_idx], era_scene[hide_idx]

        # sanity: base + static (coords-only / static-only)
        chk["XGBoost"][b].append(np.abs(BASELINES["XGBoost"](co, vo, cp) - yt))
        chk["XGBoost_static"][b].append(np.abs(
            BASELINES["XGBoost"](co, vo, cp, feat_obs=fo, feat_pred=fp) - yt))

        for name in METEO_M:
            f = BASELINES[name]
            ae[name]["meteo"][b].append(np.abs(f(co, vo, cp, feat_obs=eo, feat_pred=ep) - yt))
            eo_s = np.hstack([fo, eo]); ep_s = np.hstack([fp, ep])
            ae[name]["meteo+static"][b].append(np.abs(f(co, vo, cp, feat_obs=eo_s, feat_pred=ep_s) - yt))

        if (pi + 1) % 50 == 0:
            print(f"    [{pi+1}/{len(pairs)}] done", flush=True)

    def agg(store, key):
        return {str(b): (lambda a: {"MAE": float(a.mean()) if a.size else None,
                                    "RMSE": float(np.sqrt((a**2).mean())) if a.size else None,
                                    "N": int(a.size)})(np.concatenate(store[key][b]) if store[key][b] else np.array([]))
                for b in range(4)}

    out = {"city": city, "bins": BIN_LABELS, "drivers": DRIVERS,
           "protocol": {"years": list(map(int, years)), "n_clear": int(n_clear),
                        "eval_years": list(map(int, years)),
                        "stat_years": list(map(int, stat_years)),
                        "mode": "scene-wise transductive",
                        "note": "Each eval scene uses only visible pixels at that timestamp.",
                        "masks_per_bin": int(masks_per_bin), "max_pred": int(max_pred),
                        "seed": int(seed), "n_pairs": int(len(pairs))},
           "sanity": {"XGBoost": agg(chk, "XGBoost"),
                      "XGBoost_static": agg(chk, "XGBoost_static")},
           "methods": {}}
    for name in METEO_M:
        out["methods"][name] = {cfg: {str(b): (lambda a: {
            "MAE": float(a.mean()) if a.size else None,
            "RMSE": float(np.sqrt((a**2).mean())) if a.size else None,
            "N": int(a.size)})(np.concatenate(ae[name][cfg][b]) if ae[name][cfg][b] else np.array([]))
            for b in range(4)} for cfg in ("meteo", "meteo+static")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["cairo", "bucharest", "lagos"])
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULTS["years"])
    ap.add_argument("--stat_years", type=int, nargs="+", default=DEFAULTS["stat_years"])
    ap.add_argument("--n_clear", type=int, default=DEFAULTS["n_clear"])
    ap.add_argument("--masks_per_bin", type=int, default=DEFAULTS["masks_per_bin"])
    ap.add_argument("--max_pred", type=int, default=DEFAULTS["max_pred"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--methods", default="XGBoost,RandomForest,RegressionKriging",
                    help="subset of covariate-capable ML methods")
    ap.add_argument("--use_v7_cache", action="store_true",
                    help="use dense LST cache if available instead of the main-table parquet source")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]

    all_res = {}
    for c in a.cities:
        res = run_one(c, a.years, a.stat_years, a.n_clear, a.masks_per_bin,
                      a.max_pred, a.seed, methods, a.use_v7_cache)
        out_path = out_dir / f"1a_{c}_meteo.json"
        if out_path.exists():                              # merge across incremental runs
            prev = json.loads(out_path.read_text())
            if prev.get("protocol") == res.get("protocol"):
                prev.setdefault("methods", {}).update(res["methods"])
                prev["sanity"] = res["sanity"]
                res = prev
        all_res[c] = res
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n--- {c}  sanity (must match main table) ---")
        print(f"{'config':>16}" + "".join(f"{BIN_LABELS[b]:>10}" for b in range(4)))
        for cfg in ("XGBoost", "XGBoost_static"):
            print(f"{cfg:>16}" + "".join(f"{res['sanity'][cfg][str(b)]['MAE'] or float('nan'):>10.3f}"
                                         for b in range(4)))
        print(f"\n--- {c}  +meteo / +meteo+static MAE ---")
        for name in res["methods"]:
            for cfg in ("meteo", "meteo+static"):
                print(f"{name+'+'+cfg:>22}" + "".join(
                    f"{res['methods'][name][cfg][str(b)]['MAE'] or float('nan'):>10.3f}" for b in range(4)))
    (out_dir / "1a_all_meteo.json").write_text(json.dumps(all_res, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out_dir}/1a_all_meteo.json")


if __name__ == "__main__":
    main()
