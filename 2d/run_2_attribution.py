"""Task 2d — sampled driver-attribution prototype.

Builds UHI anomaly targets (Ta-UHI main, LST-UHI appendix), then runs:
  Arm A: 8 model-family dynamic-driver attribution baselines.
  Arm B: XGBoost contribution ablations over climatology/dynamic/static/both.

This is intentionally sampled. A full city-year tensor is too large for the
first reproducible pass; the sampling seed and sample counts are written into
the JSON output.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "results"

sys.path.append(str(BENCH))
from common.climate_zones import CITIES, DE_SOURCE, INTL_TARGET  # noqa: E402
from common.data import _load_static  # noqa: E402
from common.paths import ATUHI_BASE, ERA5_BASE, LST_BASE, STATIC_BASE, TA_BASE  # noqa: E402


LSTUHI_BASE = LST_BASE

DRIVERS = ["u10", "v10", "tcc", "d2m", "blh", "ssrd"]
LAG_SPECS = [
    ("cur", (0,)),
    ("lag1", (1,)),
    ("lag3", (3,)),
    ("lag6", (6,)),
    ("lag24", (24,)),
    ("roll3", (0, 1, 2)),
    ("roll6", (0, 1, 2, 3, 4, 5)),
]
ARM_A_MODELS = [
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "randomforest",
    "extratrees",
    "histgradientboosting",
    "xgboost",
]
TASK2_CITIES = DE_SOURCE + INTL_TARGET


class MissingTask2Data(RuntimeError):
    pass


@dataclass
class CityData:
    city: str
    label_kind: str
    times: pd.DatetimeIndex
    uhi: np.ndarray
    era5: np.ndarray
    static: np.ndarray
    static_names: list[str]
    pixel_ids: np.ndarray
    sampleable: np.ndarray


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def choose_pixels(city: str, years: list[int], n_pixels: int, seed: int, target: str) -> np.ndarray:
    _, _, _, static_pids = _load_static(city, n_static=10)
    available = set(static_pids.astype(int).tolist())
    if target == "ta" and city in DE_SOURCE:
        d = TA_BASE / city / "cache" / f"v7_{years[0]}"
        if (d / "pixel_ids.npy").exists():
            cache_pids = np.load(d / "pixel_ids.npy").astype(int)
            available &= set(cache_pids.tolist())
    pids = np.array(sorted(available), dtype=np.int64)
    if len(pids) == 0:
        raise ValueError(f"no candidate pixels for {city}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pids, size=min(n_pixels, len(pids)), replace=False))


def missing_era5_drivers(city: str, year: int):
    import pyarrow.parquet as pq

    p = ERA5_BASE / city / f"era5_hourly_{year}.parquet"
    if not p.exists():
        return [f"missing_file:{p}"]
    names = set(pq.read_schema(p).names)
    return [v for v in DRIVERS if v not in names]


def load_static_for_pixels(city: str, pixel_ids: np.ndarray):
    _, feats, names, pids = _load_static(city, n_static=10)
    row = {int(pid): i for i, pid in enumerate(pids)}
    order = np.array([row[int(pid)] for pid in pixel_ids], dtype=np.int64)
    return feats[order].astype(np.float32), names


def load_de_target(city: str, years: list[int], pixel_ids: np.ndarray):
    parts, time_parts = [], []
    for y in years:
        d = TA_BASE / city / "cache" / f"v7_{y}"
        if not (d / "done.json").exists():
            raise FileNotFoundError(f"missing HOSTRADA v7 cache: {d}")
        pids = np.load(d / "pixel_ids.npy").astype(np.int64)
        pid2row = {int(pid): i for i, pid in enumerate(pids)}
        order = np.array([pid2row[int(pid)] for pid in pixel_ids], dtype=np.int64)
        uhi = np.load(d / "uhi.npy", mmap_mode="r")
        times = np.load(d / "times.npy", mmap_mode="r")
        parts.append(np.asarray(uhi[:, order, 1], dtype=np.float32))
        time_parts.append(np.asarray(times, dtype=np.int64))
    values = np.ascontiguousarray(np.concatenate(parts, axis=0))
    times = pd.to_datetime(np.concatenate(time_parts), unit="s")
    return times, values


def _read_pixel_parquet(
    path: Path,
    columns: list[str],
    pixel_ids: np.ndarray,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    filt = [("pixel_id", "in", [int(p) for p in pixel_ids])]
    if start is not None:
        filt.append(("datetime", ">=", start.to_pydatetime()))
    if end is not None:
        filt.append(("datetime", "<", end.to_pydatetime()))
    return pd.read_parquet(path, columns=columns, filters=filt)


def _pivot_year(df: pd.DataFrame, times: pd.DatetimeIndex, pixel_ids: np.ndarray, value_cols: list[str]):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index(["datetime", "pixel_id"]).sort_index()
    out = []
    for col in value_cols:
        piv = df[col].unstack("pixel_id").reindex(index=times, columns=pixel_ids)
        out.append(piv.to_numpy(dtype=np.float32))
    return np.stack(out, axis=-1)


def load_intl_target(city: str, years: list[int], pixel_ids: np.ndarray):
    parts, time_parts = [], []
    for y in years:
        p = ATUHI_BASE / city / f"atuhi_ood_1km_hourly_{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        df = _read_pixel_parquet(p, ["datetime", "pixel_id", "uhi"], pixel_ids)
        times = pd.DatetimeIndex(pd.to_datetime(df["datetime"].drop_duplicates()).sort_values())
        arr = _pivot_year(df, times, pixel_ids, ["uhi"])[:, :, 0]
        parts.append(arr)
        time_parts.append(times)
    return pd.DatetimeIndex(np.concatenate(time_parts)), np.ascontiguousarray(np.concatenate(parts, axis=0))


def load_era5(city: str, years: list[int], pixel_ids: np.ndarray, target_times: pd.DatetimeIndex):
    parts = []
    offset = 0
    for y in years:
        p = ERA5_BASE / city / f"era5_hourly_{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        n = 8784 if pd.Timestamp(y, 1, 1).is_leap_year else 8760
        times_y = target_times[offset:offset + n]
        offset += n
        df = _read_pixel_parquet(p, ["datetime", "pixel_id", *DRIVERS], pixel_ids)
        arr = _pivot_year(df, times_y, pixel_ids, DRIVERS)
        parts.append(arr)
    return np.ascontiguousarray(np.concatenate(parts, axis=0))


def sampled_windows(years: list[int], windows_per_year: int, window_days: int, seed: int):
    """Stratified monthly windows with 24h lag warmup.

    Returns tuples (year, warmup_start, sample_start, sample_end), all UTC-like
    naive pandas timestamps. sample_end is exclusive.
    """
    rng = np.random.default_rng(seed)
    windows = []
    months = list(range(1, 13))
    for y in years:
        chosen = months if windows_per_year >= 12 else [months[i] for i in np.linspace(0, 11, windows_per_year, dtype=int)]
        for m in chosen:
            m0 = pd.Timestamp(y, m, 1)
            m1 = m0 + pd.offsets.MonthBegin(1)
            latest = m1 - pd.Timedelta(days=window_days)
            earliest = max(m0, pd.Timestamp(y, 1, 2))
            if latest < earliest:
                continue
            n_days = int((latest - earliest).days)
            start = earliest + pd.Timedelta(days=int(rng.integers(0, n_days + 1)))
            end = start + pd.Timedelta(days=window_days)
            windows.append((y, start - pd.Timedelta(hours=24), start, end))
    return windows


def load_de_target_window(city: str, year: int, pixel_ids: np.ndarray, start: pd.Timestamp, end: pd.Timestamp):
    d = TA_BASE / city / "cache" / f"v7_{year}"
    if not (d / "pixel_ids.npy").exists():
        return load_de_target_window_from_monthly(city, pixel_ids, start, end)
    pids = np.load(d / "pixel_ids.npy").astype(np.int64)
    pid2row = {int(pid): i for i, pid in enumerate(pids)}
    order = np.array([pid2row[int(pid)] for pid in pixel_ids], dtype=np.int64)
    uhi = np.load(d / "uhi.npy", mmap_mode="r")
    raw_times = np.load(d / "times.npy", mmap_mode="r")
    times = pd.to_datetime(np.asarray(raw_times, dtype=np.int64), unit="s")
    mask = (times >= start) & (times < end)
    return pd.DatetimeIndex(times[mask]), np.asarray(uhi[mask][:, order, 1], dtype=np.float32)


def _months_between(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1)
    stop = pd.Timestamp((end - pd.Timedelta(hours=1)).year, (end - pd.Timedelta(hours=1)).month, 1)
    out = []
    while cur <= stop:
        out.append((cur.year, cur.month))
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def load_de_target_window_from_monthly(city: str, pixel_ids: np.ndarray, start: pd.Timestamp, end: pd.Timestamp):
    xy_m, _, _, static_pids = _load_static(city, n_static=10)
    pid2row = {int(pid): i for i, pid in enumerate(static_pids)}
    rows = np.array([pid2row[int(pid)] for pid in pixel_ids], dtype=np.int64)
    coords = pd.DataFrame({
        "x_epsg3034": np.rint(xy_m[rows, 0]).astype(np.int64),
        "y_epsg3034": np.rint(xy_m[rows, 1]).astype(np.int64),
        "pixel_id": pixel_ids.astype(np.int64),
    })
    frames = []
    for y, m in _months_between(start, end):
        p = TA_BASE / city / "monthly_uhi" / f"uhi_{y}{m:02d}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["datetime", "x_epsg3034", "y_epsg3034", "uhi"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        if len(df):
            frames.append(df.merge(coords, on=["x_epsg3034", "y_epsg3034"], how="inner"))
    times = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    if frames:
        df = pd.concat(frames, ignore_index=True)
        arr = _pivot_year(df[["datetime", "pixel_id", "uhi"]], times, pixel_ids, ["uhi"])[:, :, 0]
    else:
        arr = np.full((len(times), len(pixel_ids)), np.nan, np.float32)
    return pd.DatetimeIndex(times), arr.astype(np.float32)


def load_intl_target_window(city: str, year: int, pixel_ids: np.ndarray, times: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    p = ATUHI_BASE / city / f"atuhi_ood_1km_hourly_{year}.parquet"
    df = _read_pixel_parquet(p, ["datetime", "pixel_id", "uhi"], pixel_ids, start=start, end=end)
    arr = _pivot_year(df, times, pixel_ids, ["uhi"])[:, :, 0]
    return arr


def load_lst_target_window(city: str, year: int, pixel_ids: np.ndarray, times: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    p = LSTUHI_BASE / city / f"lst_uhi_1km_hourly_{year}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = _read_pixel_parquet(p, ["datetime", "pixel_id", "lst_uhi_K"], pixel_ids, start=start, end=end)
    arr = _pivot_year(df, times, pixel_ids, ["lst_uhi_K"])[:, :, 0]
    return arr


def load_era5_window(city: str, year: int, pixel_ids: np.ndarray, times: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    p = ERA5_BASE / city / f"era5_hourly_{year}.parquet"
    df = _read_pixel_parquet(p, ["datetime", "pixel_id", *DRIVERS], pixel_ids, start=start, end=end)
    return _pivot_year(df, times, pixel_ids, DRIVERS)


def load_windowed_city_data(city: str, years: list[int], n_pixels: int, seed: int, windows_per_year: int, window_days: int, target: str) -> CityData:
    pixel_ids = choose_pixels(city, years, n_pixels=n_pixels, seed=seed, target=target)
    static, static_names = load_static_for_pixels(city, pixel_ids)
    windows = sampled_windows(years, windows_per_year, window_days, seed)
    time_parts, uhi_parts, era5_parts, sample_parts = [], [], [], []
    for wi, (year, warmup_start, sample_start, sample_end) in enumerate(windows):
        expected_times = pd.date_range(warmup_start, sample_end - pd.Timedelta(hours=1), freq="h")
        if target == "lst":
            times = expected_times
            uhi = load_lst_target_window(city, year, pixel_ids, times, warmup_start, sample_end)
        elif city in DE_SOURCE:
            times, uhi = load_de_target_window(city, year, pixel_ids, warmup_start, sample_end)
            times = pd.DatetimeIndex(times)
            if len(times) != len(expected_times):
                uhi = pd.DataFrame(uhi, index=times, columns=pixel_ids).reindex(expected_times).to_numpy(dtype=np.float32)
                times = expected_times
        else:
            times = expected_times
            uhi = load_intl_target_window(city, year, pixel_ids, times, warmup_start, sample_end)
        era5 = load_era5_window(city, year, pixel_ids, times, warmup_start, sample_end)
        sampleable = (times >= sample_start) & (times < sample_end)
        time_parts.append(times.to_numpy())
        uhi_parts.append(uhi)
        era5_parts.append(era5)
        sample_parts.append(np.asarray(sampleable, dtype=bool))
        if wi == 0 or (wi + 1) % 12 == 0 or wi == len(windows) - 1:
            print(f"  window {wi + 1:03d}/{len(windows)} {year} {sample_start.date()} {city}", flush=True)
    if target == "lst":
        label_kind = "LST_TRUE_MSG_1KM"
    else:
        label_kind = "DE_TRUE_HOSTRADA" if city in DE_SOURCE else "INTL_PSEUDO_ATUHI_CORRECTED"
    return CityData(
        city=city,
        label_kind=label_kind,
        times=pd.DatetimeIndex(np.concatenate(time_parts)),
        uhi=np.ascontiguousarray(np.concatenate(uhi_parts, axis=0)),
        era5=np.ascontiguousarray(np.concatenate(era5_parts, axis=0)),
        static=static,
        static_names=static_names,
        pixel_ids=pixel_ids,
        sampleable=np.concatenate(sample_parts),
    )


def load_city_data(city: str, years: list[int], n_pixels: int, seed: int, target: str) -> CityData:
    pixel_ids = choose_pixels(city, years, n_pixels=n_pixels, seed=seed, target=target)
    static, static_names = load_static_for_pixels(city, pixel_ids)
    if target == "lst":
        parts, time_parts = [], []
        for y in years:
            p = LSTUHI_BASE / city / f"lst_uhi_1km_hourly_{y}.parquet"
            df = _read_pixel_parquet(p, ["datetime", "pixel_id", "lst_uhi_K"], pixel_ids)
            times_y = pd.DatetimeIndex(pd.to_datetime(df["datetime"].drop_duplicates()).sort_values())
            parts.append(_pivot_year(df, times_y, pixel_ids, ["lst_uhi_K"])[:, :, 0])
            time_parts.append(times_y)
        times = pd.DatetimeIndex(np.concatenate(time_parts))
        uhi = np.ascontiguousarray(np.concatenate(parts, axis=0))
        label_kind = "LST_TRUE_MSG_1KM"
    elif city in DE_SOURCE:
        times, uhi = load_de_target(city, years, pixel_ids)
        label_kind = "DE_TRUE_HOSTRADA"
    else:
        times, uhi = load_intl_target(city, years, pixel_ids)
        label_kind = "INTL_PSEUDO_ATUHI_CORRECTED"
    era5 = load_era5(city, years, pixel_ids, times)
    if len(times) != len(uhi) or len(times) != len(era5):
        raise ValueError(f"{city}: time length mismatch")
    return CityData(city, label_kind, times, uhi, era5, static, static_names, pixel_ids, np.ones(len(times), dtype=bool))


def season_id(month: np.ndarray) -> np.ndarray:
    return ((month % 12) // 3).astype(np.int16)


def fit_climatology(train_values: np.ndarray, train_times: pd.DatetimeIndex, min_count: int):
    months = train_times.month.to_numpy()
    hours = train_times.hour.to_numpy()
    seasons = season_id(months)
    n_pix = train_values.shape[1]

    month_hod = np.full((12, 24, n_pix), np.nan, np.float32)
    season_hod = np.full((4, 24, n_pix), np.nan, np.float32)
    hod = np.full((24, n_pix), np.nan, np.float32)
    global_pix = np.nanmean(train_values, axis=0).astype(np.float32)
    counts = np.zeros((12, 24, n_pix), np.int32)

    for m in range(1, 13):
        for h in range(24):
            mask = (months == m) & (hours == h)
            if mask.any():
                vals = train_values[mask]
                month_hod[m - 1, h] = np.nanmean(vals, axis=0)
                counts[m - 1, h] = np.isfinite(vals).sum(axis=0)
    for s in range(4):
        for h in range(24):
            mask = (seasons == s) & (hours == h)
            if mask.any():
                season_hod[s, h] = np.nanmean(train_values[mask], axis=0)
    for h in range(24):
        mask = hours == h
        if mask.any():
            hod[h] = np.nanmean(train_values[mask], axis=0)
    return {"month_hod": month_hod, "season_hod": season_hod, "hod": hod, "global": global_pix, "counts": counts, "min_count": min_count}


def predict_climatology(clim: dict, times: pd.DatetimeIndex) -> np.ndarray:
    months = times.month.to_numpy()
    hours = times.hour.to_numpy()
    seasons = season_id(months)
    out = clim["month_hod"][months - 1, hours].copy()
    low = clim["counts"][months - 1, hours] < int(clim["min_count"])
    season_vals = clim["season_hod"][seasons, hours]
    hod_vals = clim["hod"][hours]
    out = np.where(low | ~np.isfinite(out), season_vals, out)
    out = np.where(~np.isfinite(out), hod_vals, out)
    out = np.where(~np.isfinite(out), clim["global"][None, :], out)
    return out.astype(np.float32)


def split_masks(times: pd.DatetimeIndex, sampleable: np.ndarray | None = None):
    year = times.year.to_numpy()
    base = np.ones(len(times), dtype=bool) if sampleable is None else np.asarray(sampleable, dtype=bool)
    return (year <= 2022) & base, (year >= 2023) & base


def fit_era5_stats(era5: np.ndarray, train_mask: np.ndarray):
    sample = era5[train_mask]
    mu = np.nanmean(sample, axis=(0, 1), keepdims=True)
    sd = np.nanstd(sample, axis=(0, 1), keepdims=True) + 1e-6
    z = (era5 - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), mu.reshape(-1), sd.reshape(-1)


def fit_static_stats(static: np.ndarray):
    valid = np.isfinite(static)
    count = valid.sum(axis=0)
    safe = np.where(valid, static, 0.0)
    mu = np.divide(safe.sum(axis=0), np.maximum(count, 1), dtype=np.float64).astype(np.float32)
    centered = np.where(valid, static - mu[None, :], 0.0)
    sd = np.sqrt(np.divide((centered * centered).sum(axis=0), np.maximum(count, 1))).astype(np.float32)
    mu = np.where(count > 0, mu, 0.0).astype(np.float32)
    sd = np.where((count > 1) & (sd > 1e-6), sd, 1.0).astype(np.float32)
    z = (static - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), mu, sd


def sample_rows(
    y: np.ndarray,
    time_mask: np.ndarray,
    day_mask: np.ndarray | None,
    max_samples: int,
    seed: int,
    min_lag: int = 24,
):
    valid_t = time_mask.copy()
    valid_t[:min_lag] = False
    if day_mask is not None:
        valid_t &= day_mask
    valid = np.isfinite(y) & valid_t[:, None]
    rows = np.flatnonzero(valid.ravel())
    if len(rows) == 0:
        raise ValueError("no valid sampled rows")
    rng = np.random.default_rng(seed)
    if len(rows) > max_samples:
        rows = rng.choice(rows, size=max_samples, replace=False)
    rows = np.sort(rows)
    return rows // y.shape[1], rows % y.shape[1]


def build_dynamic_features(era5_z: np.ndarray, t_idx: np.ndarray, p_idx: np.ndarray):
    cols = []
    names = []
    for vi, var in enumerate(DRIVERS):
        for suffix, offsets in LAG_SPECS:
            vals = np.zeros(len(t_idx), np.float32)
            for off in offsets:
                vals += era5_z[t_idx - off, p_idx, vi]
            vals /= float(len(offsets))
            cols.append(vals)
            names.append(f"{var}_{suffix}")
    return np.column_stack(cols).astype(np.float32), names


def build_static_rows(static_z: np.ndarray, p_idx: np.ndarray):
    return static_z[p_idx].astype(np.float32)


def target_values(y: np.ndarray, t_idx: np.ndarray, p_idx: np.ndarray):
    return y[t_idx, p_idx].astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"MAE": None, "RMSE": None, "N": 0}
    err = y_pred[ok] - y_true[ok]
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err * err))), "N": int(ok.sum())}


def aggregate_by_driver(values: np.ndarray, feature_names: list[str]):
    out = {}
    for var in DRIVERS:
        idx = [i for i, name in enumerate(feature_names) if name.startswith(f"{var}_")]
        out[var] = float(np.sum(np.abs(values[idx]))) if len(idx) else 0.0
    total = sum(out.values()) + 1e-12
    frac = {k: float(v / total) for k, v in out.items()}
    rank = sorted(DRIVERS, key=lambda k: (-out[k], k))
    return {"importance": out, "fraction": frac, "rank": rank}


def aggregate_driver_values(values: dict[str, float]):
    out = {k: float(values.get(k, 0.0)) for k in DRIVERS}
    total = sum(abs(v) for v in out.values()) + 1e-12
    frac = {k: float(abs(v) / total) for k, v in out.items()}
    rank = sorted(DRIVERS, key=lambda k: (-abs(out[k]), k))
    return {"importance": {k: abs(v) for k, v in out.items()}, "fraction": frac, "rank": rank}


def aggregate_signed_by_driver(values: np.ndarray, feature_names: list[str]):
    out = {}
    for var in DRIVERS:
        idx = [i for i, name in enumerate(feature_names) if name.startswith(f"{var}_")]
        out[var] = float(np.sum(values[idx])) if len(idx) else 0.0
    pos_rank = sorted(DRIVERS, key=lambda k: (-out[k], k))
    neg_rank = sorted(DRIVERS, key=lambda k: (out[k], k))
    return {"signed": out, "positive_rank": pos_rank, "negative_rank": neg_rank}


def grouped_permutation_importance(model, x_eval: np.ndarray, y_eval: np.ndarray, feature_names: list[str], seed: int, n_repeats: int = 2):
    """Driver-level permutation importance: MAE increase after shuffling all lags for a driver."""
    base_pred = model.predict(x_eval)
    base_mae = float(np.mean(np.abs(base_pred - y_eval)))
    rng = np.random.default_rng(seed)
    out = {}
    for var in DRIVERS:
        idx = [i for i, name in enumerate(feature_names) if name.startswith(f"{var}_")]
        if not idx:
            out[var] = 0.0
            continue
        scores = []
        for _ in range(n_repeats):
            x_perm = x_eval.copy()
            order = rng.permutation(x_perm.shape[0])
            x_perm[:, idx] = x_perm[order][:, idx]
            pred = model.predict(x_perm)
            scores.append(float(np.mean(np.abs(pred - y_eval)) - base_mae))
        out[var] = max(0.0, float(np.mean(scores)))
    agg = aggregate_driver_values(out)
    agg["permutation_metric"] = "MAE increase after grouped driver shuffle"
    agg["permutation_repeats"] = n_repeats
    return agg


def high_low_response(x_eval: np.ndarray, pred: np.ndarray, feature_names: list[str], group_to_features: dict[str, list[str]]):
    """Observed-response diagnostic: E[pred | feature high] - E[pred | feature low].

    This is not causal PDP; it is a compact direction check on the sampled eval
    distribution. Inputs are standardized, so high/low use 80/20 percentiles.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    out = {}
    for group, names in group_to_features.items():
        idx = [name_to_idx[n] for n in names if n in name_to_idx]
        if not idx:
            continue
        score = x_eval[:, idx].mean(axis=1)
        finite = np.isfinite(score) & np.isfinite(pred)
        if finite.sum() < 20:
            continue
        lo, hi = np.nanquantile(score[finite], [0.2, 0.8])
        low = finite & (score <= lo)
        high = finite & (score >= hi)
        if low.sum() == 0 or high.sum() == 0:
            continue
        out[group] = {
            "high_minus_low_pred": float(np.mean(pred[high]) - np.mean(pred[low])),
            "low_q20": float(lo),
            "high_q80": float(hi),
            "n_low": int(low.sum()),
            "n_high": int(high.sum()),
        }
    return out


def _attach_response(agg: dict, x_eval: np.ndarray, pred: np.ndarray, feature_names: list[str]):
    agg["current_driver_response"] = high_low_response(
        x_eval,
        pred,
        feature_names,
        {v: [f"{v}_cur"] for v in DRIVERS},
    )
    return agg


def fit_arm_a_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, feature_names: list[str], seed: int):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_mu = float(np.mean(y_train))
    y_sd = float(np.std(y_train) + 1e-6)
    y_train_z = (y_train - y_mu) / y_sd
    if model_name in {"linear", "ridge", "lasso", "elasticnet"}:
        estimators = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0, random_state=seed),
            "lasso": Lasso(alpha=0.002, max_iter=5000, random_state=seed),
            "elasticnet": ElasticNet(alpha=0.002, l1_ratio=0.5, max_iter=5000, random_state=seed),
        }
        model = make_pipeline(StandardScaler(), estimators[model_name])
        model.fit(x_train, y_train_z)
        step_name = "linearregression" if model_name == "linear" else model_name
        coefs = model.named_steps[step_name].coef_.astype(np.float64)
        agg = aggregate_by_driver(coefs, feature_names)
        agg["signed_effect"] = aggregate_signed_by_driver(coefs, feature_names)
        pred = model.predict(x_eval) * y_sd + y_mu
        _attach_response(agg, x_eval, pred, feature_names)
        return agg, pred
    if model_name in {"randomforest", "extratrees"}:
        cls = RandomForestRegressor if model_name == "randomforest" else ExtraTreesRegressor
        model = cls(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=4,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_eval)
        agg = aggregate_by_driver(model.feature_importances_.astype(np.float64), feature_names)
        agg["attribution"] = f"{model_name} feature_importances_ (mean decrease impurity)"
        _attach_response(agg, x_eval, pred, feature_names)
        return agg, pred
    if model_name == "histgradientboosting":
        model = HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_eval)
        agg = grouped_permutation_importance(model, x_eval, y_eval, feature_names, seed=seed, n_repeats=2)
        agg["attribution"] = "grouped permutation importance over ERA5 driver lag blocks"
        _attach_response(agg, x_eval, pred, feature_names)
        return agg, pred
    if model_name == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_eval)
        contrib = model.get_booster().predict(xgb.DMatrix(x_eval), pred_contribs=True)
        mean_abs = np.mean(np.abs(contrib[:, :-1]), axis=0)
        mean_signed = np.mean(contrib[:, :-1], axis=0)
        agg = aggregate_by_driver(mean_abs, feature_names)
        agg["signed_effect"] = aggregate_signed_by_driver(mean_signed, feature_names)
        _attach_response(agg, x_eval, pred, feature_names)
        agg["pred_contribs"] = "xgboost pred_contribs=True"
        return agg, pred
    raise ValueError(model_name)


def run_arm_a(
    era5_z: np.ndarray,
    anomaly: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    ssrd_raw: np.ndarray,
    max_train: int,
    max_eval: int,
    seed: int,
):
    day_by_time = np.nanmean(ssrd_raw, axis=1) > 10.0
    modes = {"all": None, "daytime": day_by_time, "nighttime": ~day_by_time}
    out = {}
    mode_offsets = {"all": 0, "daytime": 100, "nighttime": 200}
    for mode, day_mask in modes.items():
        offset = mode_offsets[mode]
        t_tr, p_tr = sample_rows(anomaly, train_mask, day_mask, max_train, seed + offset)
        t_ev, p_ev = sample_rows(anomaly, eval_mask, day_mask, max_eval, seed + 17 + offset)
        x_tr, names = build_dynamic_features(era5_z, t_tr, p_tr)
        x_ev, _ = build_dynamic_features(era5_z, t_ev, p_ev)
        y_tr = target_values(anomaly, t_tr, p_tr)
        y_ev = target_values(anomaly, t_ev, p_ev)
        mode_out = {"samples": {"train": int(len(y_tr)), "eval": int(len(y_ev))}, "models": {}}
        for model_name in ARM_A_MODELS:
            agg, pred = fit_arm_a_model(model_name, x_tr, y_tr, x_ev, y_ev, names, seed=seed)
            agg["eval_metrics"] = regression_metrics(y_ev, pred)
            mode_out["models"][model_name] = agg
        out[mode] = mode_out
    return out


def fit_xgb_regressor(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=40,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    return model


def run_arm_b(
    era5_z: np.ndarray,
    static_z: np.ndarray,
    raw: np.ndarray,
    clim_pred: np.ndarray,
    anomaly: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    max_train: int,
    max_eval: int,
    seed: int,
    static_names: list[str],
):
    t_tr, p_tr = sample_rows(anomaly, train_mask, None, max_train, seed + 101)
    t_ev, p_ev = sample_rows(anomaly, eval_mask, None, max_eval, seed + 202)
    xd_tr, dyn_names = build_dynamic_features(era5_z, t_tr, p_tr)
    xd_ev, _ = build_dynamic_features(era5_z, t_ev, p_ev)
    xs_tr = build_static_rows(static_z, p_tr)
    xs_ev = build_static_rows(static_z, p_ev)

    targets = {
        "anomaly": {
            "train": target_values(anomaly, t_tr, p_tr),
            "eval": target_values(anomaly, t_ev, p_ev),
            "clim_pred": np.zeros(len(t_ev), np.float32),
        },
        "raw_reference": {
            "train": target_values(raw, t_tr, p_tr),
            "eval": target_values(raw, t_ev, p_ev),
            "clim_pred": target_values(clim_pred, t_ev, p_ev),
        },
    }
    arms = {
        "climatology_only": (None, None),
        "dynamic_only": (xd_tr, xd_ev),
        "static_only": (xs_tr, xs_ev),
        "dynamic_plus_static": (np.column_stack([xd_tr, xs_tr]), np.column_stack([xd_ev, xs_ev])),
    }

    out = {"samples": {"train": int(len(t_tr)), "eval": int(len(t_ev))}, "targets": {}}
    for target_name, target in targets.items():
        target_out = {}
        for arm, (x_tr, x_ev) in arms.items():
            if arm == "climatology_only":
                pred = target["clim_pred"]
            else:
                model = fit_xgb_regressor(x_tr, target["train"], seed=seed)
                pred = model.predict(x_ev)
                if target_name == "anomaly" and arm == "dynamic_plus_static":
                    import xgboost as xgb

                    feature_names = dyn_names + static_names
                    contrib = model.get_booster().predict(xgb.DMatrix(x_ev), pred_contribs=True)
                    mean_abs = np.mean(np.abs(contrib[:, :-1]), axis=0)
                    mean_signed = np.mean(contrib[:, :-1], axis=0)
                    dyn_abs = mean_abs[:len(dyn_names)]
                    dyn_signed = mean_signed[:len(dyn_names)]
                    st_abs = mean_abs[len(dyn_names):]
                    st_signed = mean_signed[len(dyn_names):]
                    st_importance = {name: float(st_abs[i]) for i, name in enumerate(static_names)}
                    st_signed_map = {name: float(st_signed[i]) for i, name in enumerate(static_names)}
                    total_st = float(np.sum(st_abs) + 1e-12)
                    target_out["dynamic_plus_static_attribution"] = {
                        "dynamic": {
                            **aggregate_by_driver(dyn_abs, dyn_names),
                            "signed_effect": aggregate_signed_by_driver(dyn_signed, dyn_names),
                            "current_driver_response": high_low_response(
                                x_ev,
                                pred,
                                feature_names,
                                {v: [f"{v}_cur"] for v in DRIVERS},
                            ),
                        },
                        "static": {
                            "importance": st_importance,
                            "fraction": {k: float(v / total_st) for k, v in st_importance.items()},
                            "signed": st_signed_map,
                            "rank": sorted(static_names, key=lambda k: (-st_importance[k], k)),
                            "positive_rank": sorted(static_names, key=lambda k: (-st_signed_map[k], k)),
                            "negative_rank": sorted(static_names, key=lambda k: (st_signed_map[k], k)),
                            "response": high_low_response(
                                x_ev,
                                pred,
                                feature_names,
                                {name: [name] for name in static_names},
                            ),
                        },
                        "note": "Signed contributions and high-low response are model diagnostics, not causal effects.",
                    }
            target_out[arm] = regression_metrics(target["eval"], pred)
        base = target_out["climatology_only"]["MAE"]
        if base is not None:
            for arm in arms:
                metrics = target_out[arm]
                metrics["delta_MAE_vs_climatology"] = None if metrics["MAE"] is None else float(metrics["MAE"] - base)
        out["targets"][target_name] = target_out
    return out


def spearman_matrix(city_results: dict, mode: str, model: str, cities: list[str]):
    from scipy.stats import spearmanr

    ranks = {}
    for city in cities:
        try:
            rank = city_results[city]["arm_a"][mode]["models"][model]["rank"]
        except KeyError:
            continue
        # Smaller rank number = more important.
        ranks[city] = np.array([rank.index(v) for v in DRIVERS], dtype=np.float32)
    out = {}
    for a in cities:
        if a not in ranks:
            continue
        out[a] = {}
        for b in cities:
            if b not in ranks:
                continue
            out[a][b] = float(spearmanr(ranks[a], ranks[b]).correlation)
    return out


def kendall_w(city_results: dict, mode: str, model: str, cities: list[str]):
    mats = []
    for city in cities:
        try:
            rank = city_results[city]["arm_a"][mode]["models"][model]["rank"]
        except KeyError:
            continue
        mats.append([rank.index(v) + 1 for v in DRIVERS])
    if len(mats) < 2:
        return None
    arr = np.asarray(mats, np.float64)
    m, n = arr.shape
    rank_sums = arr.sum(axis=0)
    s = np.sum((rank_sums - rank_sums.mean()) ** 2)
    return float(12 * s / (m * m * (n**3 - n) + 1e-12))


def run_city(args, city: str):
    years = list(range(args.start_year, args.end_year + 1))
    missing = missing_era5_drivers(city, years[0])
    if missing:
        raise MissingTask2Data(f"{city}: ERA5 file lacks required drivers {missing}")
    print(f"[load] {city}: pixels={args.n_pixels} years={years[0]}-{years[-1]}", flush=True)
    if args.sample_windows_per_year > 0:
        data = load_windowed_city_data(
            city,
            years,
            args.n_pixels,
            args.seed,
            windows_per_year=args.sample_windows_per_year,
            window_days=args.window_days,
            target=args.target,
        )
    else:
        data = load_city_data(city, years, args.n_pixels, args.seed, target=args.target)
    train_mask, eval_mask = split_masks(data.times, data.sampleable)
    clim = fit_climatology(data.uhi[train_mask], data.times[train_mask], min_count=args.clim_min_count)
    clim_pred = predict_climatology(clim, data.times)
    anomaly = (data.uhi - clim_pred).astype(np.float32)
    era5_z, era5_mu, era5_sd = fit_era5_stats(data.era5, train_mask)
    static_z, static_mu, static_sd = fit_static_stats(data.static)

    print(f"[arm_a] {city}", flush=True)
    arm_a = run_arm_a(
        era5_z,
        anomaly,
        train_mask,
        eval_mask,
        ssrd_raw=data.era5[:, :, DRIVERS.index("ssrd")],
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
        seed=args.seed,
    )
    print(f"[arm_b] {city}", flush=True)
    arm_b = run_arm_b(
        era5_z,
        static_z,
        raw=data.uhi,
        clim_pred=clim_pred,
        anomaly=anomaly,
        train_mask=train_mask,
        eval_mask=eval_mask,
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
        seed=args.seed,
        static_names=data.static_names,
    )
    koppen, group, dist = CITIES[city]
    out = {
        "task": "Task 2d Driver Attribution",
        "target": args.target,
        "city": city,
        "label_kind": data.label_kind,
        "koppen": koppen,
        "climate_group": group,
        "climate_dist_from_DE": dist,
        "years": {"train": [args.start_year, 2022], "eval": [2023, args.end_year]},
        "protocol": {
            "n_pixels": int(len(data.pixel_ids)),
            "n_hours_loaded": int(len(data.times)),
            "n_sampleable_hours": int(data.sampleable.sum()),
            "pixel_seed": args.seed,
            "sample_windows_per_year": args.sample_windows_per_year,
            "window_days": args.window_days,
            "max_train_samples_per_mode": args.max_train_samples,
            "max_eval_samples_per_mode": args.max_eval_samples,
            "target_main": f"{args.target.upper()}-UHI anomaly = UHI - pixel x hour-of-day x month train climatology",
            "climatology_fallback": f"month/hod min_count={args.clim_min_count}, then season/hod, hod, pixel mean",
            "drivers": DRIVERS,
            "lag_specs": [name for name, _ in LAG_SPECS],
            "daytime": "mean selected-pixel ssrd > 10",
            "static_features": data.static_names,
            "arm_a_models": ARM_A_MODELS,
            "arm_a_model_notes": {
                "ridge": "regularized linear coefficients",
                "elasticnet": "sparse + grouped regularized linear coefficients",
                "randomforest": "mean decrease impurity feature importance",
                "extratrees": "mean decrease impurity feature importance",
                "histgradientboosting": "grouped permutation importance over driver lag blocks",
            },
            "xgboost_attribution": "Booster.predict(pred_contribs=True), mean abs/signed contribution, lag-aggregated to 6 drivers; Arm B also reports static10 diagnostics.",
            "model_config": {
                "xgboost": {"arm_a_estimators": 30, "arm_b_estimators": 40, "max_depth": 3, "n_jobs": 4},
                "randomforest_extratrees": {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 5, "n_jobs": 4},
                "histgradientboosting": {"max_iter": 80, "learning_rate": 0.05, "max_leaf_nodes": 31, "permutation_repeats": 2},
            },
        },
        "pixel_ids": data.pixel_ids.astype(int).tolist(),
        "era5_standardization": {
            "mean": {k: float(v) for k, v in zip(DRIVERS, era5_mu)},
            "std": {k: float(v) for k, v in zip(DRIVERS, era5_sd)},
        },
        "arm_a": arm_a,
        "arm_b": arm_b,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "task2" if args.target == "ta" else f"task2_{args.target}"
    path = OUT_DIR / f"{prefix}_{city}.json"
    path.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"[write] {path}", flush=True)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["munich"], help="city names, 'de8', 'intl8', or 'all16'")
    ap.add_argument("--target", choices=["ta", "lst"], default="ta")
    ap.add_argument("--n_pixels", type=int, default=64)
    ap.add_argument("--max_train_samples", type=int, default=10000)
    ap.add_argument("--max_eval_samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_year", type=int, default=2015)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--sample_windows_per_year", type=int, default=12,
                    help="Monthly sampled windows per year; set 0 to load full years.")
    ap.add_argument("--window_days", type=int, default=5)
    ap.add_argument("--clim_min_count", type=int, default=20)
    ap.add_argument("--combined_out", default=None)
    args = ap.parse_args()
    if args.combined_out is None:
        name = "task2_attribution.json" if args.target == "ta" else f"task2_{args.target}_attribution.json"
        args.combined_out = str(OUT_DIR / name)
    return args


def expand_cities(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item == "de8":
            out.extend(DE_SOURCE)
        elif item == "intl8":
            out.extend(INTL_TARGET)
        elif item == "all16":
            out.extend(TASK2_CITIES)
        else:
            out.append(item.lower())
    seen = set()
    ordered = []
    for city in out:
        if city not in CITIES:
            raise ValueError(f"unknown city: {city}")
        if city not in seen:
            ordered.append(city)
            seen.add(city)
    return ordered


def main():
    args = parse_args()
    cities = expand_cities(args.cities)
    results = {}
    skipped = {}
    for city in cities:
        try:
            results[city] = run_city(args, city)
        except MissingTask2Data as exc:
            skipped[city] = {"reason": str(exc)}
            print(f"[skip] {exc}", flush=True)
    matrices = {}
    for mode in ["all", "daytime", "nighttime"]:
        matrices[mode] = {}
        for model in ARM_A_MODELS:
            matrices[mode][model] = {
                "spearman_all_available": spearman_matrix(results, mode, model, cities),
                "kendall_w_all_available": kendall_w(results, mode, model, cities),
                "spearman_de8": spearman_matrix(results, mode, model, [c for c in cities if c in DE_SOURCE]),
                "kendall_w_de8": kendall_w(results, mode, model, [c for c in cities if c in DE_SOURCE]),
            }
    combined = {
        "task": "Task 2d Driver Attribution",
        "target": args.target,
        "cities": cities,
        "note": (
            "DE8 true Ta is the core result; Intl8 corrected AtUHI is pseudo-label extension."
            if args.target == "ta" else
            "LST-UHI appendix uses MSG/TsHARP LST-UHI true surface-temperature target for all core 16 cities."
        ),
        "results": results,
        "skipped": skipped,
        "rank_similarity": matrices,
    }
    Path(args.combined_out).write_text(json.dumps(combined, indent=2, default=_json_default))
    print(f"[write] {args.combined_out}", flush=True)


if __name__ == "__main__":
    main()
