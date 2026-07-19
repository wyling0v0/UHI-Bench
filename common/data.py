"""
Benchmark data loaders for Task 1a (LST-UHI) and Task 1b (Air-T UHI).

All loaders return a unified `Field` namedtuple:
    values : float32 [T, N]   (NaN = missing/cloud)
    xy_km  : float32 [N, 2]   (EPSG:3034 metres / 1000, for distance math)
    feats  : float32 [N, F]   (static features, NaN-filled/standardized upstream)
    feat_names : list[str]
    times  : datetime64[ns, T]
    pixel_ids : int64 [N]
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd

from .paths import LST_BASE, PSEUDO_TA_BASE, STATIC_BASE, TA_BASE

Field = namedtuple("Field", "values xy_km feats feat_names times pixel_ids")

def _load_static(city: str, n_static=10):
    """Return (xy_m[N,2], feats[N,n_static] float32, feat_names list, pixel_ids[N]).

    n_static=10 by default → only the Tier-1 features common to ALL cities
    (German npz have 16 incl. 6 GBA; we slice the first 10 for cross-city fairness)."""
    p = os.path.join(STATIC_BASE, city, "static_features.npz")
    d = np.load(p, allow_pickle=True)
    xy = np.asarray(d["xy"], np.float64)            # metres, EPSG:3034
    feats = np.asarray(d["features"], np.float32)
    names = [str(s) for s in d["feat_names"]]
    feats = feats[:, :n_static]                      # slice to Tier-1 (cross-city fair)
    names = names[:n_static]
    pids = np.asarray(d["pixel_ids"]).astype(np.int64).ravel()
    return xy, feats, names, pids


def _align_static_by_pixel_ids(city: str, pixel_ids: np.ndarray, n_static=10):
    xy_m, feats, names, spids = _load_static(city, n_static=n_static)
    pid2row = {int(pid): i for i, pid in enumerate(spids)}
    order = np.array([pid2row[int(p)] for p in pixel_ids], dtype=np.int64)
    return (xy_m[order] / 1000.0).astype(np.float32), feats[order], names


def has_hostrada_ta(city: str) -> bool:
    return bool(
        glob.glob(str(TA_BASE / city / "monthly_uhi" / "uhi_*.parquet"))
        or glob.glob(str(TA_BASE / city / "cache" / "v7_*" / "done.json"))
    )


def ta_source_base(city: str) -> Path:
    """Return the Ta source root for a city.

    German cities use HOSTRADA. International cities fall back to ATUHI-OOD
    AirT-UHI so Task 1b/1c/1d can be extended beyond Germany while
    keeping the source explicit in protocol metadata.
    """
    if has_hostrada_ta(city):
        return TA_BASE
    if (PSEUDO_TA_BASE / city).exists():
        return PSEUDO_TA_BASE
    return TA_BASE


def ta_cache_root(city: str) -> Path:
    return ta_source_base(city) / city / "cache"


def _load_v7_ta_cache(city: str, years) -> Field | None:
    """Load dense v7 Ta-UHI cache when available.

    v7 caches are the canonical fast path used by DL/FMs. German HOSTRADA caches
    store coordinates as metres in ``xy.npy``; international pseudo-Ta caches
    store kilometres in ``xy_km.npy``.
    """
    years = [int(y) for y in years]
    derived = ta_cache_root(city) / ("bench_1b_field_" + "_".join(map(str, years)))
    if ((derived / "done.json").exists() and (derived / "values.npy").exists()
            and (derived / "times.npy").exists() and (derived / "pixel_ids.npy").exists()):
        values = np.load(derived / "values.npy", mmap_mode="r")
        times_raw = np.asarray(np.load(derived / "times.npy", mmap_mode="r"),
                               dtype=np.int64)
        pixel_ids = np.asarray(np.load(derived / "pixel_ids.npy", mmap_mode="r"),
                               dtype=np.int64).ravel()
        if (derived / "xy_km.npy").exists():
            xy_km = np.asarray(np.load(derived / "xy_km.npy", mmap_mode="r"),
                               dtype=np.float32)
        elif (derived / "xy.npy").exists():
            xy_km = (np.asarray(np.load(derived / "xy.npy", mmap_mode="r"),
                                dtype=np.float32) / 1000.0)
        else:
            xy_km, _, _ = _align_static_by_pixel_ids(city, pixel_ids)
        _, feats, names = _align_static_by_pixel_ids(city, pixel_ids)
        return Field(values, xy_km.astype(np.float32), feats, names,
                     pd.to_datetime(times_raw, unit="s").to_numpy(), pixel_ids)

    cache_dirs = [ta_cache_root(city) / f"v7_{y}" for y in years]
    if not all((d / "done.json").exists() and (d / "uhi.npy").exists()
               and (d / "times.npy").exists() and (d / "pixel_ids.npy").exists()
               for d in cache_dirs):
        return None

    first = cache_dirs[0]
    if (first / "xy_km.npy").exists():
        xy_km = np.asarray(np.load(first / "xy_km.npy", mmap_mode="r"),
                           dtype=np.float32)
    elif (first / "xy.npy").exists():
        xy_km = (np.asarray(np.load(first / "xy.npy", mmap_mode="r"),
                            dtype=np.float32) / 1000.0)
    else:
        xy_km = None

    shapes = [np.load(d / "uhi.npy", mmap_mode="r").shape for d in cache_dirs]
    total_t = int(sum(shape[0] for shape in shapes))
    n_pix = int(shapes[0][1])
    derived.mkdir(parents=True, exist_ok=True)
    values_mm = np.lib.format.open_memmap(
        derived / "values.npy", mode="w+", dtype=np.float32, shape=(total_t, n_pix)
    )
    times_parts = []
    off = 0
    for d in cache_dirs:
        uhi = np.load(d / "uhi.npy", mmap_mode="r")
        nt = int(uhi.shape[0])
        values_mm[off:off + nt] = uhi[:, :, 1]
        off += nt
        times_parts.append(np.asarray(np.load(d / "times.npy", mmap_mode="r"),
                                      dtype=np.int64))
    values_mm.flush()
    times_raw = np.concatenate(times_parts)
    pixel_ids = np.asarray(np.load(first / "pixel_ids.npy", mmap_mode="r"),
                           dtype=np.int64).ravel()
    np.save(derived / "times.npy", times_raw)
    np.save(derived / "pixel_ids.npy", pixel_ids)
    if xy_km is not None:
        np.save(derived / "xy_km.npy", xy_km.astype(np.float32))
    (derived / "done.json").write_text(json.dumps({
        "city": city,
        "years": years,
        "shape": [total_t, n_pix],
        "source": "v7_ta_cache",
    }, indent=2))

    values = np.load(derived / "values.npy", mmap_mode="r")
    static_xy_km, feats, names = _align_static_by_pixel_ids(city, pixel_ids)
    if xy_km is None:
        xy_km = static_xy_km
    return Field(values, xy_km.astype(np.float32), feats, names,
                 pd.to_datetime(times_raw, unit="s").to_numpy(), pixel_ids)


def load_lst_field(city: str, years=range(2023, 2026)) -> Field:
    """LST-UHI grid for non-German cities (Cairo/Bucharest/Lagos).
    pixel_id in parquet aligns 1:1 with static_features pixel_ids."""
    files = []
    for y in years:
        files += sorted(glob.glob(os.path.join(LST_BASE, city, f"lst_uhi_1km_hourly_{y}.parquet")))
    if not files:
        raise FileNotFoundError(f"no LST parquet for {city} years={list(years)}")
    frames = [pd.read_parquet(f, columns=["pixel_id", "datetime", "lst_uhi_K"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    piv = df.pivot_table(index="datetime", columns="pixel_id", values="lst_uhi_K",
                         aggfunc="first").sort_index()
    pixel_ids = piv.columns.to_numpy().astype(np.int64)
    values = np.ascontiguousarray(piv.to_numpy(dtype=np.float32))   # [T, N]

    xy_km, feats, names = _align_static_by_pixel_ids(city, pixel_ids)
    times = piv.index.to_numpy()
    return Field(values, xy_km, feats, names, times, pixel_ids)


def load_pseudo_ta_field(city: str, years=range(2023, 2026)) -> Field:
    """ATUHI-OOD AirT-UHI grid for non-German cities.

    Schema mirrors LST (`pixel_id`, `datetime`, `uhi`) and aligns directly to
    static feature pixel IDs. Dense per-year pivots are cached under the ATUHI
    source tree because these parquet files are large.
    """
    years = [int(y) for y in years]
    base = PSEUDO_TA_BASE / city
    cached = _load_v7_ta_cache(city, years)
    if cached is not None:
        return cached

    cache = base / "cache" / ("bench_ta_field_" + "_".join(map(str, years)))
    if ((cache / "done.json").exists() and (cache / "values.npy").exists()
            and (cache / "times.npy").exists() and (cache / "pixel_ids.npy").exists()):
        values = np.load(cache / "values.npy", mmap_mode="r")
        times = pd.to_datetime(np.load(cache / "times.npy", mmap_mode="r"), unit="s").to_numpy()
        pixel_ids = np.asarray(np.load(cache / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64).ravel()
        xy_km, feats, names = _align_static_by_pixel_ids(city, pixel_ids)
        return Field(values, xy_km, feats, names, times, pixel_ids)

    files = []
    for y in years:
        files += sorted(glob.glob(str(base / f"*{y}*.parquet")))
    if not files:
        raise FileNotFoundError(f"no ATUHI-OOD parquet for {city} years={years}")

    print(f"    [pseudo-ta] building dense field cache {cache.name} from {len(files)} files", flush=True)
    frames = []
    for f in files:
        d = pd.read_parquet(f, columns=["pixel_id", "datetime", "uhi"])
        d["datetime"] = pd.to_datetime(d["datetime"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    piv = df.pivot_table(index="datetime", columns="pixel_id", values="uhi",
                         aggfunc="first").sort_index()
    pixel_ids = piv.columns.to_numpy().astype(np.int64)
    values = np.ascontiguousarray(piv.to_numpy(dtype=np.float32))
    times = pd.DatetimeIndex(piv.index)

    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "values.npy", values)
    np.save(cache / "times.npy", times.asi8 // 1_000_000_000)
    np.save(cache / "pixel_ids.npy", pixel_ids)
    (cache / "done.json").write_text(json.dumps({
        "city": city,
        "years": years,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "source": "atuhi_ood_1km_hourly",
    }, indent=2))

    xy_km, feats, names = _align_static_by_pixel_ids(city, pixel_ids)
    return Field(values, xy_km, feats, names, times.to_numpy(), pixel_ids)


def load_ta_field(city: str, years=range(2023, 2026)) -> Field:
    """Air-T UHI grid for German cities (HOSTRADA). No pixel_id column →
    derive pixel index from unique (x_epsg3034, y_epsg3034) coordinate pairs,
    then align static_features by matching xy."""
    cached = _load_v7_ta_cache(city, years)
    if cached is not None:
        return cached

    files = []
    for y in years:
        files += sorted(glob.glob(os.path.join(TA_BASE, city, "monthly_uhi", f"uhi_{y}*.parquet")))
    if not files:
        if (PSEUDO_TA_BASE / city).exists():
            return load_pseudo_ta_field(city, years=years)
        raise FileNotFoundError(f"no HOSTRADA or ATUHI-OOD parquet for {city}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    # canonical pixel id = sorted unique coord pairs
    keys = df[["x_epsg3034", "y_epsg3034"]].drop_duplicates()
    keys = keys.sort_values(["x_epsg3034", "y_epsg3034"]).reset_index(drop=True)
    keys["pid"] = np.arange(len(keys))
    df = df.merge(keys, on=["x_epsg3034", "y_epsg3034"])
    piv = df.pivot_table(index="datetime", columns="pid", values="uhi", aggfunc="first").sort_index()
    pixel_ids = piv.columns.to_numpy()
    values = np.ascontiguousarray(piv.to_numpy(dtype=np.float32))

    xy_m, feats, names, spids = _load_static(city)
    sxy = xy_m
    # map static rows to the pivot pixel order by matching xy
    coord2pid = {tuple(xy): pid for pid, xy in zip(keys["pid"], keys[["x_epsg3034", "y_epsg3034"]].to_numpy())}
    srow_pid = np.array([coord2pid[tuple(xy)] for xy in sxy])
    # invert: for each pivot pid -> static row
    pid2srow = {pid: i for i, pid in enumerate(srow_pid)}
    order = np.array([pid2srow[p] for p in pixel_ids])
    xy_km = (xy_m[order] / 1000.0).astype(np.float32)
    feats = feats[order]
    times = piv.index.to_numpy()
    return Field(values, xy_km, feats, names, times, pixel_ids)
