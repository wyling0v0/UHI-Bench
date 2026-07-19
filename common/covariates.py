"""Shared covariate loaders for public UHI-Bench baselines."""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import _load_static
from .paths import CACHE_ROOT, ERA5_BASE

DRIVERS = ["u10", "v10", "tcc", "d2m", "blh", "ssrd"]
CONFIGS = ["L1", "+static", "+meteo", "+static+meteo"]
N_STATIC = 10
N_DRIVERS = len(DRIVERS)
ERA5_CACHE = CACHE_ROOT / "era5_covariates"


def in_dim(cfg: str) -> int:
    c = 2
    if "+static" in cfg:
        c += N_STATIC
    if "+meteo" in cfg:
        c += N_DRIVERS
    return c


def load_static_raw(city: str, pixel_ids: np.ndarray) -> np.ndarray:
    """Return raw static features [n, 10] aligned to ``pixel_ids``."""
    _, feats, _, pids = _load_static(city, n_static=N_STATIC)
    pids = np.asarray(pids, dtype=np.int64)
    order = {int(p): i for i, p in enumerate(pids)}
    idx = np.asarray([order[int(p)] for p in pixel_ids], dtype=np.int64)
    return np.ascontiguousarray(feats[idx], dtype=np.float32)


def standardize_static(arr: np.ndarray, stats=None):
    """Fill NaNs by column median, then z-score using train statistics."""
    if stats is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(arr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        filled = np.where(np.isnan(arr), med, arr)
        mu = filled.mean(axis=0)
        sd = filled.std(axis=0) + 1e-6
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        stats = (med.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32))
    med, mu, sd = stats
    filled = np.where(np.isnan(arr), med, arr)
    z = ((filled - mu) / sd).astype(np.float32)
    return np.nan_to_num(z, nan=0.0), stats


def _hash_arrays(*arrays) -> str:
    h = hashlib.sha1()
    for arr in arrays:
        a = np.asarray(arr)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()[:16]


def _era5_cache_path(city: str, years, pixel_ids: np.ndarray,
                     time_index: pd.DatetimeIndex) -> Path:
    years_key = "-".join(str(int(y)) for y in years)
    t_ns = pd.DatetimeIndex(time_index).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    key = _hash_arrays(np.asarray(pixel_ids, dtype=np.int64), t_ns)
    return ERA5_CACHE / f"{city}_{years_key}_{key}.npy"


def _era5_stats_cache_path(city: str, years, pixel_ids: np.ndarray) -> Path:
    years_key = "-".join(str(int(y)) for y in years)
    key = _hash_arrays(np.asarray(pixel_ids, dtype=np.int64))
    return ERA5_CACHE / f"{city}_{years_key}_{key}_stats.npz"


def _year_hours(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{int(year)}-01-01", f"{int(year) + 1}-01-01", freq="h")[:-1]


def load_era5_pixel_fast(city: str, year: int, pixel_ids: np.ndarray) -> np.ndarray:
    """Load one year of ERA5 as [hours, N, drivers]."""
    path = ERA5_BASE / city / f"era5_hourly_{int(year)}.parquet"
    times = _year_hours(year)
    out = np.full((len(times), len(pixel_ids), N_DRIVERS), np.nan, dtype=np.float32)
    if len(pixel_ids) == 0:
        return out

    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    dense_n = n_rows // len(times)
    if (n_rows == len(times) * dense_n
            and dense_n > int(np.max(pixel_ids))
            and len(pixel_ids) >= int(0.75 * dense_n)):
        print(f"      ERA5 {city} {int(year)}: dense read {dense_n} px -> select {len(pixel_ids)}", flush=True)
        df = pd.read_parquet(path, columns=DRIVERS)
        raw = df.to_numpy(dtype=np.float32, copy=False).reshape(len(times), dense_n, N_DRIVERS)
        out[:] = raw[:, np.asarray(pixel_ids, dtype=np.int64), :]
        return out

    filters = [("pixel_id", "in", [int(p) for p in pixel_ids])]
    cols = ["datetime", "pixel_id"] + DRIVERS
    df = pd.read_parquet(path, columns=cols, filters=filters)
    if df.empty:
        return out

    start = np.datetime64(f"{int(year)}-01-01T00:00:00", "ns")
    dt = pd.to_datetime(df["datetime"]).to_numpy(dtype="datetime64[ns]")
    t_idx = ((dt - start) / np.timedelta64(1, "h")).astype(np.int64)

    pids = df["pixel_id"].to_numpy(dtype=np.int64)
    if pids.min() >= 0 and pids.max() < 10_000_000:
        lut = np.full(int(max(pids.max(), np.max(pixel_ids))) + 1, -1, dtype=np.int32)
        lut[np.asarray(pixel_ids, dtype=np.int64)] = np.arange(len(pixel_ids), dtype=np.int32)
        p_idx = lut[pids]
    else:
        pos = {int(p): i for i, p in enumerate(pixel_ids)}
        p_idx = np.asarray([pos.get(int(p), -1) for p in pids], dtype=np.int32)
    ok = (t_idx >= 0) & (t_idx < len(times)) & (p_idx >= 0)
    vals = df.loc[ok, DRIVERS].to_numpy(dtype=np.float32, copy=False)
    out[t_idx[ok], p_idx[ok], :] = vals
    return out


def load_era5_selected_times(city: str, years, pixel_ids: np.ndarray,
                             time_index: pd.DatetimeIndex) -> np.ndarray:
    """Load ERA5 only for requested timestamps as [len(time_index), N, drivers]."""
    time_index = pd.DatetimeIndex(time_index)
    pixel_ids = np.asarray(pixel_ids, dtype=np.int64)
    out = np.full((len(time_index), len(pixel_ids), N_DRIVERS), np.nan, dtype=np.float32)
    if len(time_index) == 0 or len(pixel_ids) == 0:
        return out

    t_ns = time_index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    t_pos = {int(ns): i for i, ns in enumerate(t_ns)}
    if pixel_ids.min(initial=0) >= 0 and pixel_ids.max(initial=0) < 10_000_000:
        lut = np.full(int(pixel_ids.max(initial=0)) + 1, -1, dtype=np.int32)
        lut[pixel_ids] = np.arange(len(pixel_ids), dtype=np.int32)
        use_lut = True
    else:
        p_pos = {int(p): i for i, p in enumerate(pixel_ids)}
        use_lut = False

    for y in years:
        mask_y = time_index.year == int(y)
        if not bool(np.any(mask_y)):
            continue
        year_times = pd.DatetimeIndex(time_index[mask_y]).unique()
        path = ERA5_BASE / city / f"era5_hourly_{int(y)}.parquet"
        filters = [("datetime", "in", [pd.Timestamp(t).to_pydatetime() for t in year_times])]
        cols = ["datetime", "pixel_id"] + DRIVERS
        df = pd.read_parquet(path, columns=cols, filters=filters)
        if df.empty:
            continue
        pids = df["pixel_id"].to_numpy(dtype=np.int64)
        if use_lut:
            p_idx = np.full(len(pids), -1, dtype=np.int32)
            in_range = (pids >= 0) & (pids < len(lut))
            p_idx[in_range] = lut[pids[in_range]]
        else:
            p_idx = np.asarray([p_pos.get(int(p), -1) for p in pids], dtype=np.int32)
        ok_p = p_idx >= 0
        if not bool(ok_p.any()):
            continue
        dt_ns = pd.to_datetime(df.loc[ok_p, "datetime"]).astype("int64").to_numpy()
        row_t = np.asarray([t_pos.get(int(ns), -1) for ns in dt_ns], dtype=np.int64)
        ok_t = row_t >= 0
        vals = df.loc[ok_p, DRIVERS].to_numpy(dtype=np.float32, copy=False)
        out[row_t[ok_t], p_idx[ok_p][ok_t], :] = vals[ok_t]
    return out


def load_era5_aligned(city: str, years, pixel_ids: np.ndarray, time_index: pd.DatetimeIndex) -> np.ndarray:
    """ERA5 [T, N, 6] reindexed onto the UHI time index."""
    ERA5_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _era5_cache_path(city, years, pixel_ids, time_index)
    if cache.exists():
        print(f"    ERA5 cache hit: {cache.name}", flush=True)
        return np.load(cache)

    print(f"    ERA5 cache miss: {cache.name}", flush=True)
    if len(time_index) <= 2048:
        out = load_era5_selected_times(city, years, pixel_ids, time_index)
    else:
        parts = [load_era5_pixel_fast(city, int(y), pixel_ids) for y in years]
        era = np.concatenate(parts, axis=0)
        full = []
        for y in years:
            full.append(_year_hours(y))
        full = pd.DatetimeIndex(np.concatenate([np.asarray(f) for f in full]))
        if len(full) != era.shape[0]:
            raise RuntimeError(f"era5 rows {era.shape[0]} != full index {len(full)}")
        out = np.empty((len(time_index), len(pixel_ids), N_DRIVERS), dtype=np.float32)
        pos = {t: i for i, t in enumerate(full)}
        src_idx = []
        keep_mask = np.zeros(len(time_index), dtype=bool)
        for i, t in enumerate(pd.DatetimeIndex(time_index)):
            if t in pos:
                src_idx.append(pos[t])
                keep_mask[i] = True
            else:
                src_idx.append(0)
        src_idx = np.asarray(src_idx, dtype=np.int64)
        out[keep_mask] = era[src_idx[keep_mask]]
        out[~keep_mask] = np.nan

    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.save(f, out)
    tmp.replace(cache)
    print(f"    ERA5 cache saved: {cache.name} {out.shape}", flush=True)
    return out


def standardize_era5(arr: np.ndarray, stats=None, chunk: int = 2048):
    """Per-driver NaN-to-median fill and z-score over train samples."""
    n = int(arr.shape[0])
    chunk = int(chunk)
    verbose = n >= 10_000
    if stats is None:
        total = np.zeros(N_DRIVERS, dtype=np.float64)
        total2 = np.zeros(N_DRIVERS, dtype=np.float64)
        count = np.zeros(N_DRIVERS, dtype=np.float64)
        saw_nan = False
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            x = np.asarray(arr[s:e], dtype=np.float32).reshape(-1, N_DRIVERS)
            ok = np.isfinite(x)
            saw_nan = saw_nan or not bool(ok.all())
            vals = np.where(ok, x, 0.0).astype(np.float64, copy=False)
            total += vals.sum(axis=0)
            total2 += (vals * vals).sum(axis=0)
            count += ok.sum(axis=0)
            if verbose and (s == 0 or e == n or (e // chunk) % 8 == 0):
                print(f"    ERA5 stats {e}/{n}", flush=True)
        count = np.maximum(count, 1.0)
        mu = total / count
        var = np.maximum(total2 / count - mu * mu, 0.0)
        sd = np.sqrt(var) + 1e-6
        if saw_nan:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(arr.reshape(-1, N_DRIVERS), axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
        else:
            med = mu
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        stats = (med.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32))
    med, mu, sd = stats
    z = np.empty(arr.shape, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x = np.asarray(arr[s:e], dtype=np.float32)
        if np.isnan(x).any():
            x = np.where(np.isnan(x), med, x)
        y = ((x - mu) / sd).astype(np.float32, copy=False)
        z[s:e] = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose and (s == 0 or e == n or (e // chunk) % 8 == 0):
            print(f"    ERA5 z {e}/{n}", flush=True)
    return z, stats


def fit_era5_stats_stream(city: str, years, pixel_ids: np.ndarray):
    """Fit per-driver ERA5 mean/std over years without materializing all years."""
    ERA5_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _era5_stats_cache_path(city, years, pixel_ids)
    if cache.exists():
        d = np.load(cache)
        print(f"    ERA5 stats cache hit: {cache.name}", flush=True)
        return (d["med"].astype(np.float32), d["mu"].astype(np.float32), d["sd"].astype(np.float32))

    total = np.zeros(N_DRIVERS, dtype=np.float64)
    total2 = np.zeros(N_DRIVERS, dtype=np.float64)
    count = np.zeros(N_DRIVERS, dtype=np.float64)
    pixel_ids = np.asarray(pixel_ids, dtype=np.int64)
    for y in years:
        path = ERA5_BASE / city / f"era5_hourly_{int(y)}.parquet"
        pf = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows
        hours = len(_year_hours(y))
        dense_n = n_rows // hours
        full_city = (
            len(pixel_ids) == dense_n
            and pixel_ids.min(initial=0) == 0
            and pixel_ids.max(initial=-1) == dense_n - 1
        )
        cols = DRIVERS if full_city else ["pixel_id"] + DRIVERS
        print(f"    ERA5 stats {city} {int(y)}: stream rows={n_rows} full_city={full_city}", flush=True)
        for batch in pf.iter_batches(batch_size=262_144, columns=cols):
            if full_city:
                keep = None
            else:
                pids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                keep = np.isin(pids, pixel_ids)
                if not bool(keep.any()):
                    continue
            cols_np = []
            offset = 0 if full_city else 1
            for j in range(N_DRIVERS):
                col = batch.column(offset + j).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                cols_np.append(col if keep is None else col[keep])
            arr = np.column_stack(cols_np)
            ok = np.isfinite(arr)
            vals = np.where(ok, arr, 0.0).astype(np.float64, copy=False)
            total += vals.sum(axis=0)
            total2 += (vals * vals).sum(axis=0)
            count += ok.sum(axis=0)

    count = np.maximum(count, 1.0)
    mu = total / count
    var = np.maximum(total2 / count - mu * mu, 0.0)
    sd = np.sqrt(var) + 1e-6
    med = mu.astype(np.float32)
    mu = mu.astype(np.float32)
    sd = sd.astype(np.float32)
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, med=med, mu=mu, sd=sd)
    tmp.replace(cache)
    print(f"    ERA5 stats cache saved: {cache.name}", flush=True)
    return med, mu, sd
