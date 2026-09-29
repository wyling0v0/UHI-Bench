"""Task 1a cross-source diagnostics for matched LST-UHI and Air-T UHI.

This script answers the RQ1 bridge that the separate Task 2a/2b tables cannot:

1. same-city/same-hour LST-UHI vs Air-T UHI correlation;
2. lagged city-mean correlation;
3. day/night stratification;
4. whether one source helps spatially impute the other when the target source is
   missing or sparse.

The default is Munich, but the loader supports the 16 cities that have both
LST-UHI and either HOStrADA Air-T UHI (DE8) or corrected model-derived Air-T UHI
(Intl8).  Outputs are one JSON per city plus an optional batch summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.baselines import BASELINES  # noqa: E402
from common.data import Field, LST_BASE, TA_BASE, _load_static, load_ta_field  # noqa: E402
from common.masks import BIN_LABELS, cloud_bin_of, random_mask_for_bin  # noqa: E402


DEFAULTS = {
    "city": "munich",
    "cities": [
        "berlin", "cologne", "dortmund", "dusseldorf", "frankfurt", "hamburg", "munich", "stuttgart",
        "bucharest", "buenos_aires", "cairo", "johannesburg", "lagos", "riyadh", "sao_paulo", "warsaw",
    ],
    "years": [2023, 2024, 2025],
    "seed": 42,
    "min_spatial_pixels": 500,
    "mean_pixel_sample": 1024,
    "corr_sample_hours": 1500,
    "mask_search_hours": 6000,
    "lag_max": 48,
    "lst_clear_thr": 0.02,
    "lst_feature_min_valid": 0.75,
    "adaptive_anchor_min_valid": 0.50,
    "n_clear": 6,
    "masks_per_bin": 2,
    "n_times": 8,
    "n_splits": 1,
    "max_pred": 500,
}

TWO_SOURCE_CITIES = DEFAULTS["cities"]


def _ts_key(t) -> int:
    return int(pd.Timestamp(t).value)


def _corr(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return None
    x = x[ok]
    y = y[ok]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _zscore_scene(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    ok = np.isfinite(v)
    out = np.zeros_like(v, dtype=np.float32)
    if int(ok.sum()) == 0:
        return out
    mu = float(np.nanmean(v[ok]))
    sd = float(np.nanstd(v[ok]))
    if sd < 1e-8:
        sd = 1.0
    out[ok] = (v[ok] - mu) / sd
    return out


def _rowwise_corr_nan(x: np.ndarray, y: np.ndarray, min_pixels: int) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum(axis=1).astype(np.float64)
    x0 = np.where(mask, x, 0.0)
    y0 = np.where(mask, y, 0.0)
    sx = x0.sum(axis=1)
    sy = y0.sum(axis=1)
    sxx = (x0 * x0).sum(axis=1)
    syy = (y0 * y0).sum(axis=1)
    sxy = (x0 * y0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        r = cov / np.sqrt(vx * vy)
    r[(n < min_pixels) | (vx <= 0) | (vy <= 0)] = np.nan
    return r.astype(np.float64)


def _identity_order(order: np.ndarray) -> bool:
    return np.array_equal(order, np.arange(len(order), dtype=order.dtype))


def _take_rows(values, rows, col_order: np.ndarray | None = None) -> np.ndarray:
    block = np.asarray(values[rows], dtype=np.float64)
    if col_order is not None and not _identity_order(col_order):
        block = block[:, col_order]
    return block


def _take_rows_cols(values, rows, cols: np.ndarray) -> np.ndarray:
    return np.asarray(values[rows][:, cols], dtype=np.float64)


def _row_nanmean_count_cols(values, cols: np.ndarray, chunk: int = 2048):
    means = np.empty(values.shape[0], dtype=np.float64)
    counts = np.empty(values.shape[0], dtype=np.int64)
    for start in range(0, values.shape[0], chunk):
        end = min(start + chunk, values.shape[0])
        block = _take_rows_cols(values, slice(start, end), cols)
        ok = np.isfinite(block)
        cnt = ok.sum(axis=1)
        total = np.where(ok, block, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[start:end] = total / cnt
        means[start:end][cnt == 0] = np.nan
        counts[start:end] = cnt
    return means, counts


def _load_lst_cached(city: str, years: list[int]) -> Field:
    """Load LST-UHI parquet into a dense memmap cache aligned to static pixels."""
    v7_dirs = [LST_BASE / city / "cache" / f"v7_{int(y)}" for y in years]
    if all((d / "done.json").exists() and (d / "uhi.npy").exists()
           and (d / "times.npy").exists() and (d / "pixel_ids.npy").exists()
           for d in v7_dirs):
        values = np.ascontiguousarray(np.concatenate([
            np.asarray(np.load(d / "uhi.npy", mmap_mode="r")[:, :, 0], dtype=np.float32)
            for d in v7_dirs
        ], axis=0))
        times_raw = np.concatenate([
            np.asarray(np.load(d / "times.npy", mmap_mode="r"), dtype=np.int64)
            for d in v7_dirs
        ])
        pixel_ids = np.asarray(np.load(v7_dirs[0] / "pixel_ids.npy", mmap_mode="r"),
                               dtype=np.int64).ravel()
        xy_m, feats, names, spids = _load_static(city)
        pid2row = {int(pid): i for i, pid in enumerate(spids)}
        order = np.asarray([pid2row[int(p)] for p in pixel_ids], dtype=np.int64)
        xy_km = (xy_m[order] / 1000.0).astype(np.float32)
        return Field(values, xy_km, feats[order], names,
                     pd.to_datetime(times_raw, unit="s").to_numpy(), pixel_ids)

    cache_dir = LST_BASE / city / "cache" / ("bench_1ab_field_" + "_".join(map(str, years)))
    required = ["done.json", "values.npy", "times.npy", "xy_km.npy", "pixel_ids.npy"]
    if all((cache_dir / p).exists() for p in required):
        values = np.load(cache_dir / "values.npy", mmap_mode="r")
        times = pd.to_datetime(np.load(cache_dir / "times.npy", mmap_mode="r"), unit="s").to_numpy()
        xy_km = np.asarray(np.load(cache_dir / "xy_km.npy", mmap_mode="r"), dtype=np.float32)
        pixel_ids = np.asarray(np.load(cache_dir / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64)
        _, feats, names, spids = _load_static(city)
        pid2row = {int(pid): i for i, pid in enumerate(spids)}
        order = np.asarray([pid2row[int(p)] for p in pixel_ids], dtype=np.int64)
        return Field(values, xy_km, feats[order], names, times, pixel_ids)

    files = [LST_BASE / city / f"lst_uhi_1km_hourly_{y}.parquet" for y in years]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing LST parquet files: {missing}")

    xy_m, feats, names, spids = _load_static(city)
    static_pids = np.asarray(spids, dtype=np.int64)
    t0 = time.time()
    unique_parts = []
    for p in files:
        print(f"    [LST] indexing {p.name}", flush=True)
        pf = pq.ParquetFile(p)
        file_parts = []
        for batch in pf.iter_batches(batch_size=1_000_000, columns=["datetime"]):
            arr = batch.column(batch.schema.get_field_index("datetime"))
            dt_ns = arr.to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype(np.int64)
            file_parts.append(np.unique(dt_ns))
        file_times = np.unique(np.concatenate(file_parts))
        unique_parts.append(file_times)
        print(f"    [LST] {p.name}: {len(file_times)} hours", flush=True)
    times_ns = np.unique(np.concatenate(unique_parts))

    cache_dir.mkdir(parents=True, exist_ok=True)
    values_mm = np.lib.format.open_memmap(
        cache_dir / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(times_ns), len(static_pids)),
    )
    values_mm[:] = np.nan

    pid_max = int(static_pids.max())
    col_map = np.full(pid_max + 1, -1, dtype=np.int64)
    col_map[static_pids] = np.arange(len(static_pids), dtype=np.int64)
    for p in files:
        print(f"    [LST] filling {p.name}", flush=True)
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=1_000_000, columns=["pixel_id", "datetime", "lst_uhi_K"]):
            schema = batch.schema
            pid = batch.column(schema.get_field_index("pixel_id")).to_numpy(zero_copy_only=False).astype(np.int64)
            dt_ns = (
                batch.column(schema.get_field_index("datetime"))
                .to_numpy(zero_copy_only=False)
                .astype("datetime64[ns]")
                .astype(np.int64)
            )
            val = batch.column(schema.get_field_index("lst_uhi_K")).to_numpy(zero_copy_only=False).astype(np.float32)
            rows = np.searchsorted(times_ns, dt_ns)
            cols = col_map[pid]
            ok = cols >= 0
            values_mm[rows[ok], cols[ok]] = val[ok]
    values_mm.flush()
    times_s = (times_ns // 1_000_000_000).astype(np.int64)
    xy_km = (xy_m / 1000.0).astype(np.float32)
    np.save(cache_dir / "times.npy", times_s)
    np.save(cache_dir / "xy_km.npy", xy_km)
    np.save(cache_dir / "pixel_ids.npy", static_pids)
    (cache_dir / "done.json").write_text(
        json.dumps(
            {
                "city": city,
                "years": years,
                "shape": [int(len(times_ns)), int(len(static_pids))],
                "source": "lstuhi_1km_hourly_parquet",
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    values = np.load(cache_dir / "values.npy", mmap_mode="r")
    return Field(values, xy_km, feats, names, pd.to_datetime(times_s, unit="s").to_numpy(), static_pids)


def _load_ta_cached(city: str, years: list[int]) -> Field:
    cache_dir = TA_BASE / city / "cache" / ("bench_1b_field_" + "_".join(map(str, years)))
    required = ["done.json", "values.npy", "times.npy", "xy.npy", "pixel_ids.npy"]
    if all((cache_dir / p).exists() for p in required):
        values = np.load(cache_dir / "values.npy", mmap_mode="r")
        times = pd.to_datetime(np.load(cache_dir / "times.npy", mmap_mode="r"), unit="s").to_numpy()
        xy_m = np.asarray(np.load(cache_dir / "xy.npy", mmap_mode="r"), dtype=np.float64)
        pixel_ids = np.asarray(np.load(cache_dir / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64)
        static_xy, feats, names, _ = _load_static(city)
        srow = {tuple(map(float, xy)): i for i, xy in enumerate(static_xy)}
        order = np.asarray([srow[tuple(map(float, xy))] for xy in xy_m], dtype=np.int64)
        return Field(values, (xy_m / 1000.0).astype(np.float32), feats[order], names, times, pixel_ids)

    files = [TA_BASE / city / "monthly_uhi" / f"uhi_{int(y)}{month:02d}.parquet"
             for y in years for month in range(1, 13)]
    files = [p for p in files if p.exists()]
    if not files:
        fld = load_ta_field(city, years=years)
        xy_m, feats, names, spids = _load_static(city)
        pid2row = {int(pid): i for i, pid in enumerate(spids)}
        order = np.asarray([pid2row[int(p)] for p in fld.pixel_ids], dtype=np.int64)
        return Field(fld.values, (xy_m[order] / 1000.0).astype(np.float32),
                     feats[order], names, fld.times, fld.pixel_ids)

    xy_m, feats, names, spids = _load_static(city)
    xy_i = np.rint(xy_m).astype(np.int64)
    coord_map = pd.DataFrame({
        "x_epsg3034": xy_i[:, 0],
        "y_epsg3034": xy_i[:, 1],
        "_col": np.arange(len(xy_i), dtype=np.int64),
    })

    t0 = time.time()
    unique_parts = []
    print(f"    [AirT] building dense HOSTRADA cache {cache_dir.name} from {len(files)} files", flush=True)
    for p in files:
        pf = pq.ParquetFile(p)
        file_parts = []
        for batch in pf.iter_batches(batch_size=1_000_000, columns=["datetime"]):
            arr = batch.column(batch.schema.get_field_index("datetime"))
            dt_ns = arr.to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype(np.int64)
            file_parts.append(np.unique(dt_ns))
        if file_parts:
            file_times = np.unique(np.concatenate(file_parts))
            unique_parts.append(file_times)
            print(f"    [AirT] {p.name}: {len(file_times)} hours", flush=True)
    if not unique_parts:
        raise RuntimeError(f"no HOSTRADA timestamps for {city} years={years}")
    times_ns = np.unique(np.concatenate(unique_parts))

    cache_dir.mkdir(parents=True, exist_ok=True)
    values_mm = np.lib.format.open_memmap(
        cache_dir / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(times_ns), len(spids)),
    )
    values_mm[:] = np.nan

    for p in files:
        print(f"    [AirT] filling {p.name}", flush=True)
        d = pd.read_parquet(p, columns=["x_epsg3034", "y_epsg3034", "datetime", "uhi"])
        d = d.merge(coord_map, on=["x_epsg3034", "y_epsg3034"], how="inner", sort=False)
        if d.empty:
            continue
        d["datetime"] = pd.to_datetime(d["datetime"])
        piv = d.pivot_table(index="datetime", columns="_col", values="uhi",
                            aggfunc="first").sort_index()
        rows = np.searchsorted(times_ns, piv.index.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        cols = piv.columns.to_numpy(dtype=np.int64)
        values_mm[np.ix_(rows, cols)] = piv.to_numpy(dtype=np.float32)
    values_mm.flush()

    times_s = (times_ns // 1_000_000_000).astype(np.int64)
    pixel_ids = np.asarray(spids, dtype=np.int64).ravel()
    np.save(cache_dir / "times.npy", times_s)
    np.save(cache_dir / "xy.npy", xy_m.astype(np.float64))
    np.save(cache_dir / "pixel_ids.npy", pixel_ids)
    (cache_dir / "done.json").write_text(json.dumps({
        "city": city,
        "years": years,
        "shape": [int(len(times_ns)), int(len(pixel_ids))],
        "source": "hostrada_v7",
        "seconds": round(time.time() - t0, 1),
    }, indent=2))

    values = np.load(cache_dir / "values.npy", mmap_mode="r")
    return Field(values, (xy_m / 1000.0).astype(np.float32), feats, names,
                 pd.to_datetime(times_s, unit="s").to_numpy(), pixel_ids)


def _write_city_result(res: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"1ab_{res['city']}_cross_source.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return out_path


def _summary_row(res: dict) -> dict[str, object]:
    corr = res["correlation"]
    imp = res["imputation"]

    def g(path, default=None):
        cur = res
        for key in path:
            if cur is None:
                return default
            cur = cur.get(key) if isinstance(cur, dict) else default
        return cur

    row = {
        "city": res["city"],
        "years": ",".join(map(str, res["years"])),
        "n_lst_from_air_pairs": g(["imputation", "lst_from_air_t", "n_pairs"]),
        "n_air_from_lst_masks": g(["imputation", "air_t_from_lst", "n_masks"]),
        "lst_from_air_base_mae": g(["imputation", "lst_from_air_t", "base", "overall", "MAE"]),
        "lst_from_air_plus_mae": g(["imputation", "lst_from_air_t", "plus_air_t", "overall", "MAE"]),
        "lst_from_air_gain_pct": g(["imputation", "lst_from_air_t", "mae_improvement_pct", "overall"]),
        "air_from_lst_base_mae": g(["imputation", "air_t_from_lst", "base", "overall", "MAE"]),
        "air_from_lst_plus_mae": g(["imputation", "air_t_from_lst", "plus_lst", "overall", "MAE"]),
        "air_from_lst_gain_pct": g(["imputation", "air_t_from_lst", "mae_improvement_pct", "overall"]),
    }
    for subset in ("all", "day", "night"):
        row[f"{subset}_temporal_r"] = corr["same_hour"][subset].get("temporal_mean_r")
        row[f"{subset}_spatial_median_r"] = corr["same_hour"][subset].get("spatial_map_r_median")
        row[f"{subset}_best_lag_h"] = corr["lagged"][subset].get("best_lag_h")
        row[f"{subset}_best_lag_r"] = corr["lagged"][subset].get("best_r")
    return row


def _write_summary(results: list[dict], out_dir: Path) -> None:
    if not results:
        return
    df = pd.DataFrame([_summary_row(r) for r in results])
    csv_path = out_dir / "1ab_cross_source_summary.csv"
    tex_path = out_dir / "1ab_cross_source_summary.tex"
    md_path = out_dir / "1ab_cross_source_summary.md"
    df.to_csv(csv_path, index=False)
    csv_lines = df.to_csv(index=False).splitlines()
    header = csv_lines[0].split(",")
    rows = [line.split(",") for line in csv_lines[1:]]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    md_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    md_path.write_text("\n".join(md_lines) + "\n")
    cols = [
        "city", "all_temporal_r", "all_spatial_median_r", "all_best_lag_h", "all_best_lag_r",
        "lst_from_air_base_mae", "lst_from_air_plus_mae", "lst_from_air_gain_pct",
        "air_from_lst_base_mae", "air_from_lst_plus_mae", "air_from_lst_gain_pct",
    ]
    df[cols].to_latex(tex_path, index=False, float_format=lambda x: f"{x:.3f}", escape=True)
    print(f"[summary] {csv_path}", flush=True)
    print(f"[summary] {md_path}", flush=True)
    print(f"[summary] {tex_path}", flush=True)


def _align_fields(lst: Field, ta: Field) -> tuple[np.ndarray, np.ndarray]:
    ta_pid_to_col = {int(pid): i for i, pid in enumerate(ta.pixel_ids)}
    ta_order = np.asarray([ta_pid_to_col[int(pid)] for pid in lst.pixel_ids], dtype=np.int64)
    xy_delta = np.abs(lst.xy_km - ta.xy_km[ta_order]).max()
    if float(xy_delta) > 1e-6:
        raise RuntimeError(f"LST/Air-T grids are not exactly aligned (max delta {xy_delta})")
    return np.arange(len(lst.pixel_ids), dtype=np.int64), ta_order


def _subset_name(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    hour = index.hour.to_numpy()
    return {
        "all": np.ones(len(index), dtype=bool),
        "day": (hour >= 6) & (hour < 18),
        "night": (hour < 6) | (hour >= 18),
    }


def compute_correlations(
    lst: Field,
    ta: Field,
    ta_order: np.ndarray,
    min_pixels: int,
    lag_max: int,
    rng: np.random.Generator,
    mean_pixel_sample: int,
    corr_sample_hours: int,
):
    print("    [corr] sampled-pixel city means", flush=True)
    n_pix = len(lst.pixel_ids)
    if n_pix > mean_pixel_sample:
        mean_cols = np.sort(rng.choice(n_pix, size=mean_pixel_sample, replace=False))
    else:
        mean_cols = np.arange(n_pix, dtype=np.int64)
    ta_mean_cols = ta_order[mean_cols]
    lst_means, lst_counts = _row_nanmean_count_cols(lst.values, mean_cols)
    ta_means, _ = _row_nanmean_count_cols(ta.values, ta_mean_cols)

    lst_times = pd.DatetimeIndex(lst.times)
    ta_times = pd.DatetimeIndex(ta.times)
    lst_s = pd.Series(lst_means, index=lst_times)
    ta_s = pd.Series(ta_means, index=ta_times)
    lst_count_s = pd.Series(lst_counts, index=lst_times)

    common = lst_s.index.intersection(ta_s.index)
    mean_min_pixels = max(10, int(0.2 * len(mean_cols)))
    common = common[lst_count_s.loc[common].to_numpy() >= mean_min_pixels]
    same_df = pd.DataFrame({"lst": lst_s.loc[common], "ta": ta_s.loc[common]}).dropna()
    subsets = _subset_name(pd.DatetimeIndex(same_df.index))

    same_hour = {}
    for name, mask in subsets.items():
        d = same_df.iloc[np.where(mask)[0]]
        same_hour[name] = {
            "temporal_mean_r": _corr(d["lst"].to_numpy(), d["ta"].to_numpy()),
            "n_hours": int(len(d)),
        }

    print("    [corr] same-hour spatial maps", flush=True)
    lst_i = {_ts_key(t): i for i, t in enumerate(lst.times)}
    ta_i = {_ts_key(t): i for i, t in enumerate(ta.times)}
    spatial_times = pd.DatetimeIndex(common)
    if len(spatial_times) > corr_sample_hours:
        take = np.sort(rng.choice(len(spatial_times), size=corr_sample_hours, replace=False))
        spatial_times = spatial_times[take]
    common_lst_rows = np.asarray([lst_i[_ts_key(t)] for t in spatial_times], dtype=np.int64)
    common_ta_rows = np.asarray([ta_i[_ts_key(t)] for t in spatial_times], dtype=np.int64)
    common_hours = pd.DatetimeIndex(spatial_times).hour.to_numpy()
    spatial_r = np.full(len(spatial_times), np.nan, dtype=np.float64)
    chunk = 512
    for start in range(0, len(common), chunk):
        end = min(start + chunk, len(common))
        x = _take_rows(lst.values, common_lst_rows[start:end])
        y = _take_rows(ta.values, common_ta_rows[start:end], ta_order)
        spatial_r[start:end] = _rowwise_corr_nan(x, y, min_pixels)
    spatial_by_subset = {
        "all": spatial_r[np.isfinite(spatial_r)],
        "day": spatial_r[((common_hours >= 6) & (common_hours < 18)) & np.isfinite(spatial_r)],
        "night": spatial_r[((common_hours < 6) | (common_hours >= 18)) & np.isfinite(spatial_r)],
    }
    for name, vals in spatial_by_subset.items():
        a = np.asarray(vals, dtype=np.float64)
        same_hour.setdefault(name, {})
        same_hour[name].update(
            {
                "spatial_map_r_median": float(np.nanmedian(a)) if a.size else None,
                "spatial_map_r_mean": float(np.nanmean(a)) if a.size else None,
                "spatial_map_n_hours": int(a.size),
            }
        )

    print("    [corr] lag scan", flush=True)
    lagged = {}
    for subset in ("all", "day", "night"):
        rows = []
        for lag in range(-lag_max, lag_max + 1):
            ta_lag = ta_s.copy()
            ta_lag.index = ta_lag.index - pd.Timedelta(hours=lag)
            d = pd.DataFrame({"lst": lst_s, "ta": ta_lag, "n": lst_count_s}).dropna()
            d = d[d["n"] >= mean_min_pixels]
            if subset != "all":
                hour = pd.DatetimeIndex(d.index).hour
                keep = ((hour >= 6) & (hour < 18)) if subset == "day" else ((hour < 6) | (hour >= 18))
                d = d.iloc[np.where(keep)[0]]
            r = _corr(d["lst"].to_numpy(), d["ta"].to_numpy())
            rows.append({"lag_h": int(lag), "r": r, "n_hours": int(len(d))})
        finite = [x for x in rows if x["r"] is not None]
        best = max(finite, key=lambda x: abs(x["r"])) if finite else None
        lagged[subset] = {
            "best_lag_h": None if best is None else int(best["lag_h"]),
            "best_r": None if best is None else float(best["r"]),
            "same_hour_r": next((x["r"] for x in rows if x["lag_h"] == 0), None),
            "scan": rows,
        }

    return {
        "lag_definition": "corr(LST(t), AirT(t+lag_h)); positive lag means Air-T lags LST",
        "day_definition": "clock-hour day = 06:00--17:59; night = 18:00--05:59",
        "temporal_mean_note": f"city-mean series use a fixed {len(mean_cols)}-pixel sample for speed; spatial-map correlations use full-grid sampled hours",
        "min_spatial_pixels": int(min_pixels),
        "mean_pixel_sample": int(len(mean_cols)),
        "corr_sample_hours": int(len(spatial_times)),
        "same_hour": same_hour,
        "lagged": lagged,
    }


def _agg_errors(errors: dict[str, list[np.ndarray]]) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for b in range(len(BIN_LABELS)):
        key = str(b)
        if errors[key]:
            a = np.concatenate(errors[key]).astype(np.float64)
        else:
            a = np.array([], dtype=np.float64)
        out[key] = {
            "bin": BIN_LABELS[b],
            "MAE": float(np.mean(a)) if a.size else None,
            "RMSE": float(np.sqrt(np.mean(a * a))) if a.size else None,
            "N": int(a.size),
        }
    all_a = np.concatenate([np.concatenate(v) for v in errors.values() if v]) if any(errors.values()) else np.array([])
    out["overall"] = {
        "bin": "overall",
        "MAE": float(np.mean(all_a)) if all_a.size else None,
        "RMSE": float(np.sqrt(np.mean(all_a * all_a))) if all_a.size else None,
        "N": int(all_a.size),
    }
    return out


def _improvement(base, assist):
    out = {}
    for k, b in base.items():
        bm = b.get("MAE")
        am = assist[k].get("MAE")
        out[k] = None if bm in (None, 0) or am is None else float((bm - am) / bm * 100.0)
    return out


def run_lst_from_air(
    lst: Field,
    ta: Field,
    ta_order: np.ndarray,
    rng: np.random.Generator,
    n_clear: int,
    masks_per_bin: int,
    max_pred: int,
    clear_thr: float,
    mask_search_hours: int,
    adaptive: bool = False,
    adaptive_anchor_min_valid: float = 0.50,
):
    mode = "adaptive" if adaptive else "strict"
    print(f"    [impute] LST cloud gaps with same-hour Air-T feature ({mode})", flush=True)
    xgb = BASELINES["XGBoost"]
    time_to_ta = {_ts_key(t): i for i, t in enumerate(ta.times)}
    candidate = np.asarray([i for i, t in enumerate(lst.times) if _ts_key(t) in time_to_ta], dtype=np.int64)
    if len(candidate) > mask_search_hours:
        candidate = rng.choice(candidate, size=mask_search_hours, replace=False)
    candidate = candidate[rng.permutation(len(candidate))]
    masks = {str(b): [] for b in range(len(BIN_LABELS))}
    clear_rows = []
    for ti in candidate:
        row = np.asarray(lst.values[int(ti)], dtype=np.float32)
        mask = ~np.isfinite(row)
        cov = float(mask.mean())
        valid = 1.0 - cov
        if adaptive:
            if 0.0 < valid and valid >= adaptive_anchor_min_valid:
                clear_rows.append((cov, int(ti)))
            if 0.0 < cov < 1.0:
                b = cloud_bin_of(cov)
                if len(masks[str(b)]) < masks_per_bin:
                    masks[str(b)].append(mask)
        else:
            if cov < clear_thr and len(clear_rows) < n_clear:
                clear_rows.append((cov, int(ti)))
            elif clear_thr <= cov < 1.0:
                b = cloud_bin_of(cov)
                if len(masks[str(b)]) < masks_per_bin:
                    masks[str(b)].append(mask)
        if (not adaptive) and len(clear_rows) >= n_clear and all(len(masks[str(b)]) >= masks_per_bin for b in range(len(BIN_LABELS))):
            break

    if adaptive:
        if len(clear_rows) < n_clear:
            fallback = []
            chosen = {ti for _, ti in clear_rows}
            for ti in candidate:
                if int(ti) in chosen:
                    continue
                row = np.asarray(lst.values[int(ti)], dtype=np.float32)
                cov = float((~np.isfinite(row)).mean())
                if cov < 1.0:
                    fallback.append((cov, int(ti)))
            clear_rows.extend(fallback)
        clear_rows = sorted(clear_rows, key=lambda x: x[0])[:n_clear]

    clear_idx = np.asarray([ti for _, ti in clear_rows], dtype=np.int64)

    base_err = {str(b): [] for b in range(len(BIN_LABELS))}
    assist_err = {str(b): [] for b in range(len(BIN_LABELS))}
    pairs = 0
    for b in range(len(BIN_LABELS)):
        key = str(b)
        bin_masks = masks[key]
        if not bin_masks:
            continue
        for mask in bin_masks:
            for ct in clear_idx:
                scene = np.asarray(lst.values[ct], dtype=np.float64)
                ta_scene = np.asarray(ta.values[time_to_ta[_ts_key(lst.times[ct])], ta_order], dtype=np.float64)
                obs_idx = np.where((~mask) & np.isfinite(scene) & np.isfinite(ta_scene))[0]
                hide_idx = np.where(mask & np.isfinite(scene) & np.isfinite(ta_scene))[0]
                if len(obs_idx) < 5 or len(hide_idx) < 1:
                    continue
                if len(hide_idx) > max_pred:
                    hide_idx = rng.choice(hide_idx, size=max_pred, replace=False)
                z = _zscore_scene(ta_scene)[:, None]
                co = lst.xy_km[obs_idx].astype(np.float64)
                cp = lst.xy_km[hide_idx].astype(np.float64)
                vo = scene[obs_idx]
                yt = scene[hide_idx]
                base_err[key].append(np.abs(xgb(co, vo, cp) - yt))
                assist_err[key].append(
                    np.abs(xgb(co, vo, cp, feat_obs=z[obs_idx], feat_pred=z[hide_idx]) - yt)
                )
                pairs += 1

    base = _agg_errors(base_err)
    assist = _agg_errors(assist_err)
    return {
        "condition": (
            "adaptive: real LST cloud masks transferred to each city's clearest available LST anchor scenes; "
            "+AirT uses same-hour Air-T UHI as one per-pixel feature; MAE is computed only where anchor LST is observed"
            if adaptive else
            "real LST cloud masks transferred to clear LST scenes; +AirT uses same-hour Air-T UHI as one per-pixel feature"
        ),
        "mode": "adaptive" if adaptive else "strict",
        "n_clear": int(len(clear_idx)),
        "anchor_valid_min": float(adaptive_anchor_min_valid) if adaptive else None,
        "anchor_cloud_cover": [float(cov) for cov, _ in clear_rows],
        "masks_per_bin": int(masks_per_bin),
        "searched_hours": int(len(candidate)),
        "n_pairs": int(pairs),
        "base": base,
        "plus_air_t": assist,
        "mae_improvement_pct": _improvement(base, assist),
    }


def run_air_from_lst(
    lst: Field,
    ta: Field,
    ta_order: np.ndarray,
    rng: np.random.Generator,
    n_times: int,
    n_splits: int,
    max_pred: int,
    lst_feature_min_valid: float,
    mask_search_hours: int,
):
    print("    [impute] Air-T sparse gaps with same-hour clear-LST feature", flush=True)
    xgb = BASELINES["XGBoost"]
    lst_time_to_i = {_ts_key(t): i for i, t in enumerate(lst.times)}
    common_ta = []
    valid_fracs = []
    candidate = np.asarray([ti for ti, t in enumerate(ta.times) if _ts_key(t) in lst_time_to_i], dtype=np.int64)
    if len(candidate) > mask_search_hours:
        candidate = rng.choice(candidate, size=mask_search_hours, replace=False)
    candidate = candidate[rng.permutation(len(candidate))]
    for ti in candidate:
        t = ta.times[int(ti)]
        li = lst_time_to_i.get(_ts_key(t))
        if li is None:
            continue
        valid_frac = float(np.isfinite(lst.values[li]).mean())
        if valid_frac >= lst_feature_min_valid:
            common_ta.append(int(ti))
            valid_fracs.append(valid_frac)
            if len(common_ta) >= n_times:
                break
    common_ta = np.asarray(common_ta, dtype=np.int64)

    base_err = {str(b): [] for b in range(len(BIN_LABELS))}
    assist_err = {str(b): [] for b in range(len(BIN_LABELS))}
    feature_cover = []
    masks_done = 0
    n_pix = ta.values.shape[1]
    all_idx = np.arange(n_pix, dtype=np.int64)
    for ta_ti in common_ta:
        li = lst_time_to_i[_ts_key(ta.times[ta_ti])]
        scene = np.asarray(ta.values[ta_ti, ta_order], dtype=np.float64)
        lst_feat = np.asarray(lst.values[li], dtype=np.float64)
        finite_feat = np.isfinite(scene) & np.isfinite(lst_feat)
        feature_cover.append(float(finite_feat.mean()))
        z = _zscore_scene(lst_feat)[:, None]
        for _ in range(n_splits):
            for b in range(len(BIN_LABELS)):
                key = str(b)
                mask = random_mask_for_bin(n_pix, b, rng)
                hide_idx = np.where(mask & finite_feat)[0]
                obs_idx = all_idx[(~mask) & finite_feat]
                if len(obs_idx) < 5 or len(hide_idx) < 1:
                    continue
                if len(hide_idx) > max_pred:
                    hide_idx = rng.choice(hide_idx, size=max_pred, replace=False)
                co = ta.xy_km[ta_order][obs_idx].astype(np.float64)
                cp = ta.xy_km[ta_order][hide_idx].astype(np.float64)
                vo = scene[obs_idx]
                yt = scene[hide_idx]
                base_err[key].append(np.abs(xgb(co, vo, cp) - yt))
                assist_err[key].append(
                    np.abs(xgb(co, vo, cp, feat_obs=z[obs_idx], feat_pred=z[hide_idx]) - yt)
                )
                masks_done += 1

    base = _agg_errors(base_err)
    assist = _agg_errors(assist_err)
    return {
        "condition": "Air-T random sparse masks on timestamps where same-hour LST is at least 75% clear; both base and +LST are trained/evaluated only where LST feature is available",
        "n_times": int(len(common_ta)),
        "n_splits": int(n_splits),
        "searched_hours": int(len(candidate)),
        "n_masks": int(masks_done),
        "mean_lst_feature_coverage": float(np.mean(feature_cover)) if feature_cover else None,
        "base": base,
        "plus_lst": assist,
        "mae_improvement_pct": _improvement(base, assist),
    }


def run(args):
    years = [int(y) for y in args.years]
    rng = np.random.default_rng(args.seed)
    print(f"[1ab] {args.city} years={years}", flush=True)
    t0 = time.time()
    lst = _load_lst_cached(args.city, years)
    ta = _load_ta_cached(args.city, years)
    _, ta_order = _align_fields(lst, ta)
    print(f"    LST {lst.values.shape}; Air-T {ta.values.shape}; load {time.time() - t0:.1f}s", flush=True)

    corr = compute_correlations(
        lst,
        ta,
        ta_order,
        args.min_spatial_pixels,
        args.lag_max,
        rng,
        args.mean_pixel_sample,
        args.corr_sample_hours,
    )
    lst_from_air = run_lst_from_air(
        lst,
        ta,
        ta_order,
        rng,
        args.n_clear,
        args.masks_per_bin,
        args.max_pred,
        args.lst_clear_thr,
        args.mask_search_hours,
        args.adaptive_imputation,
        args.adaptive_anchor_min_valid,
    )
    air_from_lst = run_air_from_lst(
        lst,
        ta,
        ta_order,
        rng,
        args.n_times,
        args.n_splits,
        args.max_pred,
        args.lst_feature_min_valid,
        args.mask_search_hours,
    )

    return {
        "city": args.city,
        "years": years,
        "model": "XGBoost",
        "protocol": {
            "seed": int(args.seed),
            "min_spatial_pixels": int(args.min_spatial_pixels),
            "mean_pixel_sample": int(args.mean_pixel_sample),
            "corr_sample_hours": int(args.corr_sample_hours),
            "mask_search_hours": int(args.mask_search_hours),
            "lag_max_h": int(args.lag_max),
            "lst_clear_thr": float(args.lst_clear_thr),
            "lst_feature_min_valid": float(args.lst_feature_min_valid),
            "adaptive_imputation": bool(args.adaptive_imputation),
            "adaptive_anchor_min_valid": float(args.adaptive_anchor_min_valid),
            "n_clear": int(args.n_clear),
            "masks_per_bin": int(args.masks_per_bin),
            "n_times": int(args.n_times),
            "n_splits": int(args.n_splits),
            "max_pred": int(args.max_pred),
        },
        "correlation": corr,
        "imputation": {
            "lst_from_air_t": lst_from_air,
            "air_t_from_lst": air_from_lst,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=DEFAULTS["city"])
    ap.add_argument("--cities", nargs="+",
                    help="batch mode; default with --all_cities is the 16 two-source cities")
    ap.add_argument("--all_cities", action="store_true",
                    help="run all 16 cities with both LST and observed/model-derived Air-T sources")
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULTS["years"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--min_spatial_pixels", type=int, default=DEFAULTS["min_spatial_pixels"])
    ap.add_argument("--mean_pixel_sample", type=int, default=DEFAULTS["mean_pixel_sample"])
    ap.add_argument("--corr_sample_hours", type=int, default=DEFAULTS["corr_sample_hours"])
    ap.add_argument("--mask_search_hours", type=int, default=DEFAULTS["mask_search_hours"])
    ap.add_argument("--lag_max", type=int, default=DEFAULTS["lag_max"])
    ap.add_argument("--lst_clear_thr", type=float, default=DEFAULTS["lst_clear_thr"])
    ap.add_argument("--lst_feature_min_valid", type=float, default=DEFAULTS["lst_feature_min_valid"])
    ap.add_argument("--adaptive_imputation", action="store_true",
                    help="Use city-adaptive clearest LST anchors for LST<-AirT imputation.")
    ap.add_argument("--adaptive_anchor_min_valid", type=float, default=DEFAULTS["adaptive_anchor_min_valid"],
                    help="Minimum valid LST fraction for adaptive anchor scenes before falling back to the clearest available scenes.")
    ap.add_argument("--n_clear", type=int, default=DEFAULTS["n_clear"])
    ap.add_argument("--masks_per_bin", type=int, default=DEFAULTS["masks_per_bin"])
    ap.add_argument("--n_times", type=int, default=DEFAULTS["n_times"])
    ap.add_argument("--n_splits", type=int, default=DEFAULTS["n_splits"])
    ap.add_argument("--max_pred", type=int, default=DEFAULTS["max_pred"])
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_cities or args.cities:
        cities = list(args.cities or TWO_SOURCE_CITIES)
        results = []
        failures = []
        for city in cities:
            args.city = city
            try:
                res = run(args)
                out_path = _write_city_result(res, out_dir)
                results.append(res)
                print(f"[saved] {out_path}", flush=True)
            except Exception as exc:
                failures.append({"city": city, "error": repr(exc)})
                print(f"[fail] {city}: {exc!r}", flush=True)
        _write_summary(results, out_dir)
        if failures:
            fail_path = out_dir / "1ab_cross_source_failures.json"
            fail_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False))
            print(f"[failures] {fail_path}", flush=True)
            raise SystemExit(2)
        return

    res = run(args)
    out_path = _write_city_result(res, out_dir)

    print("\n--- correlation summary ---", flush=True)
    def _fmt(x, nd=3):
        return "nan" if x is None else f"{x:.{nd}f}"

    for subset in ("all", "day", "night"):
        sh = res["correlation"]["same_hour"][subset]
        lg = res["correlation"]["lagged"][subset]
        lag = "nan" if lg["best_lag_h"] is None else f"{lg['best_lag_h']}h"
        print(
            f"{subset:>5}: mean-r={_fmt(sh['temporal_mean_r'])} "
            f"spatial-med-r={_fmt(sh['spatial_map_r_median'])} "
            f"best-lag={lag} r={_fmt(lg['best_r'])}",
            flush=True,
        )
    print("\n--- imputation overall MAE ---", flush=True)
    lfa = res["imputation"]["lst_from_air_t"]
    afl = res["imputation"]["air_t_from_lst"]
    print(
        f"LST<-AirT: base={_fmt(lfa['base']['overall']['MAE'], 4)} "
        f"+AirT={_fmt(lfa['plus_air_t']['overall']['MAE'], 4)} "
        f"impr={_fmt(lfa['mae_improvement_pct']['overall'], 1)}%",
        flush=True,
    )
    print(
        f"AirT<-LST: base={_fmt(afl['base']['overall']['MAE'], 4)} "
        f"+LST={_fmt(afl['plus_lst']['overall']['MAE'], 4)} "
        f"impr={_fmt(afl['mae_improvement_pct']['overall'], 1)}%",
        flush=True,
    )
    print(f"\n[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
