"""Task 3 - LST-UHI OOD transfer baselines.

Sampled first-pass runner for RQ3:
  source city set size/diversity -> unseen OOD4 LST-UHI forecasting.

Implemented models:
  - Persistence: last observed value in the 168h input window.
  - SourceClimatology: source-train month x hour mean, no target-city labels.
  - XGBoost: supervised source-city transfer with L1/L2/L3 feature layers.

The script is deliberately sampled. Full 20-city pixel-hour tensors are too
large for an initial reproducible benchmark pass; all sampling settings are
written to the output JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "results"

sys.path.insert(0, str(BENCH))
from common.data import _load_static  # noqa: E402
from common.paths import ATUHI_BASE, CACHE_ROOT, ERA5_BASE, HOSTRADA_BASE, LST_BASE  # noqa: E402


CACHE_DIR = CACHE_ROOT / "task3"

DRIVERS = ["u10", "v10", "tcc", "d2m", "blh", "ssrd"]
HIST_LAGS = [0, 1, 3, 6, 12, 24, 48, 72, 167]
ERA5_LAGS = [0, 1, 3, 6, 24]
LOOKBACK = 168

SOURCE_SETS = {
    "cfb4": ["stuttgart", "frankfurt", "munich", "cologne"],
    "diverse4": ["stuttgart", "cairo", "lagos", "riyadh"],
    "cfb7": ["stuttgart", "frankfurt", "munich", "cologne",
             "dortmund", "dusseldorf", "berlin"],
    "diverse7": ["stuttgart", "cairo", "lagos", "riyadh",
                 "bucharest", "sao_paulo", "johannesburg"],
    "diverse8": ["stuttgart", "frankfurt", "cairo", "lagos", "riyadh",
                 "bucharest", "sao_paulo", "johannesburg"],
}
OOD_CITIES = ["hamburg", "warsaw", "buenos_aires", "casablanca", "tehran", "khartoum"]
LAYERS = ["L1", "L1S", "L2", "L3"]


@dataclass
class CityYearData:
    city: str
    year: int
    times: pd.DatetimeIndex
    values: np.ndarray
    era5: np.ndarray
    static: np.ndarray
    pixel_ids: np.ndarray


@dataclass
class SampleBatch:
    x_l1: np.ndarray
    x_l1s: np.ndarray
    x_l2: np.ndarray
    x_l3: np.ndarray
    y: np.ndarray
    persistence: np.ndarray
    target_times: np.ndarray
    city: np.ndarray
    valid_ratio: np.ndarray


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def stable_city_seed(seed: int, city: str, year: int = 0, extra: int = 0) -> int:
    seed_key = f"{seed}:{city}:{year}:{extra}".encode("utf-8")
    return int(hashlib.md5(seed_key).hexdigest()[:8], 16)


def choose_pixels(city: str, n_pixels: int, seed: int) -> np.ndarray:
    _, _, _, pids = _load_static(city, n_static=10)
    pids = np.asarray(pids, dtype=np.int64)
    rng = np.random.default_rng(stable_city_seed(seed, city))
    if len(pids) <= n_pixels:
        return np.sort(pids)
    return np.sort(rng.choice(pids, size=n_pixels, replace=False))


def static_for_pixels(city: str, pixel_ids: np.ndarray) -> np.ndarray:
    _, feats, _, pids = _load_static(city, n_static=10)
    row = {int(pid): i for i, pid in enumerate(pids)}
    order = np.array([row[int(pid)] for pid in pixel_ids], dtype=np.int64)
    return feats[order].astype(np.float32)


def _read_pixel_parquet(path: Path, columns: list[str], pixel_ids: np.ndarray) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    filters = [("pixel_id", "in", [int(p) for p in pixel_ids])]
    return pd.read_parquet(path, columns=columns, filters=filters)


def _year_hours(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        pd.Timestamp(year, 1, 1),
        pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=1),
        freq="h",
    )


def _pivot_hourly(
    df: pd.DataFrame,
    times: pd.DatetimeIndex,
    pixel_ids: np.ndarray,
    value_cols: list[str],
) -> np.ndarray:
    if len(df) == 0:
        return np.full((len(times), len(pixel_ids), len(value_cols)), np.nan, np.float32)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index(["datetime", "pixel_id"]).sort_index()
    arrays = []
    for col in value_cols:
        piv = df[col].unstack("pixel_id").reindex(index=times, columns=pixel_ids)
        arrays.append(piv.to_numpy(dtype=np.float32))
    return np.stack(arrays, axis=-1)


def _load_value_grid(
    city: str,
    year: int,
    times: pd.DatetimeIndex,
    pixel_ids: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    if target_kind == "lst":
        path = LST_BASE / city / f"lst_uhi_1km_hourly_{year}.parquet"
        df = _read_pixel_parquet(path, ["datetime", "pixel_id", "lst_uhi_K"], pixel_ids)
        return _pivot_hourly(df, times, pixel_ids, ["lst_uhi_K"])[:, :, 0]

    if target_kind != "airt":
        raise ValueError(f"unknown target_kind={target_kind!r}")

    intl_path = ATUHI_BASE / city / f"atuhi_ood_1km_hourly_{year}.parquet"
    if intl_path.exists():
        df = _read_pixel_parquet(intl_path, ["datetime", "pixel_id", "uhi"], pixel_ids)
        return _pivot_hourly(df, times, pixel_ids, ["uhi"])[:, :, 0]

    files = sorted((HOSTRADA_BASE / city / "monthly_uhi").glob(f"uhi_{year}*.parquet"))
    if not files:
        raise FileNotFoundError(f"no AirT-UHI parquet for {city} {year}")
    pix_hash = hashlib.md5(np.asarray(pixel_ids, dtype=np.int64).tobytes()).hexdigest()[:10]
    cache_path = CACHE_DIR / "hostrada_airtuhi" / city / f"airtuhi_{year}_pix{len(pixel_ids)}_{pix_hash}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path, columns=["datetime", "pixel_id", "uhi"])
        return _pivot_hourly(df, times, pixel_ids, ["uhi"])[:, :, 0]

    xy_m, _, _, _ = _load_static(city, n_static=10)
    coords = pd.DataFrame({
        "x_epsg3034": xy_m[pixel_ids, 0].astype(np.int32),
        "y_epsg3034": xy_m[pixel_ids, 1].astype(np.int32),
        "pixel_id": pixel_ids.astype(np.int64),
    })
    frames = []
    x_vals = [int(x) for x in coords["x_epsg3034"].unique().tolist()]
    y_vals = [int(y) for y in coords["y_epsg3034"].unique().tolist()]
    filters = [("x_epsg3034", "in", x_vals), ("y_epsg3034", "in", y_vals)]
    for f in files:
        df = pd.read_parquet(
            f,
            columns=["datetime", "x_epsg3034", "y_epsg3034", "uhi"],
            filters=filters,
        )
        frames.append(df.merge(coords, on=["x_epsg3034", "y_epsg3034"], how="inner"))
    df = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df[["datetime", "pixel_id", "uhi"]].to_parquet(cache_path, index=False)
    return _pivot_hourly(df, times, pixel_ids, ["uhi"])[:, :, 0]


def load_city_year(city: str, year: int, n_pixels: int, seed: int, target_kind: str = "lst") -> CityYearData:
    pixel_ids = choose_pixels(city, n_pixels, seed)
    times = _year_hours(year)
    static = static_for_pixels(city, pixel_ids)

    values = _load_value_grid(city, year, times, pixel_ids, target_kind)

    era5_path = ERA5_BASE / city / f"era5_hourly_{year}.parquet"
    era5_df = _read_pixel_parquet(era5_path, ["datetime", "pixel_id", *DRIVERS], pixel_ids)
    era5 = _pivot_hourly(era5_df, times, pixel_ids, DRIVERS)
    if not np.isfinite(era5).all():
        # ERA5 should be complete after reindexing. Fill rare holes locally so
        # feature construction is robust, while preserving LST cloud NaNs.
        flat = era5.reshape(len(times), -1)
        flat = pd.DataFrame(flat, index=times).ffill().bfill().to_numpy(dtype=np.float32)
        era5 = flat.reshape(len(times), len(pixel_ids), len(DRIVERS))

    return CityYearData(city, year, times, values.astype(np.float32), era5.astype(np.float32), static, pixel_ids)


def candidate_rows(
    values: np.ndarray,
    horizon: int,
    min_valid_ratio: float,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sampled (target_t, pixel, valid_ratio) rows for one city-year."""
    t_min = LOOKBACK - 1 + horizon
    if t_min >= values.shape[0]:
        raise ValueError("lookback+horizon exceeds series length")
    threshold = int(np.ceil(LOOKBACK * min_valid_ratio))
    rows_t, rows_p, ratios = [], [], []
    for p in range(values.shape[1]):
        valid = np.isfinite(values[:, p]).astype(np.int16)
        csum = np.concatenate([[0], np.cumsum(valid)])
        target_t = np.arange(t_min, values.shape[0], dtype=np.int64)
        input_end = target_t - horizon
        start = input_end - LOOKBACK + 1
        counts = csum[input_end + 1] - csum[start]
        ok = (counts >= threshold) & np.isfinite(values[target_t, p])
        if np.any(ok):
            rows_t.append(target_t[ok])
            rows_p.append(np.full(int(ok.sum()), p, dtype=np.int64))
            ratios.append((counts[ok] / float(LOOKBACK)).astype(np.float32))
    if not rows_t:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    t_idx = np.concatenate(rows_t)
    p_idx = np.concatenate(rows_p)
    valid_ratio = np.concatenate(ratios)
    rng = np.random.default_rng(seed)
    if len(t_idx) > max_samples:
        sel = rng.choice(len(t_idx), size=max_samples, replace=False)
        t_idx, p_idx, valid_ratio = t_idx[sel], p_idx[sel], valid_ratio[sel]
    order = np.lexsort((p_idx, t_idx))
    return t_idx[order], p_idx[order], valid_ratio[order]


def _hist_fill_and_stats(series: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    hist = series[start : end + 1]
    finite = np.isfinite(hist)
    if not finite.any():
        return np.zeros(6, np.float32), np.array([0.0, 0.0, 0.0], np.float32)
    vals = hist[finite].astype(np.float32)
    mean = float(vals.mean())
    std = float(vals.std())
    vmin = float(vals.min())
    vmax = float(vals.max())
    last_pos = int(np.flatnonzero(finite)[-1])
    last_val = float(hist[last_pos])
    last_age = float((len(hist) - 1) - last_pos)
    stats = np.array([mean, std, vmin, vmax, last_val, last_age], np.float32)
    return stats, np.array([mean, last_val, last_age], np.float32)


def build_samples(data: CityYearData, horizon: int, max_samples: int, min_valid_ratio: float, seed: int) -> SampleBatch:
    t_idx, p_idx, valid_ratio = candidate_rows(data.values, horizon, min_valid_ratio, max_samples, seed)
    if len(t_idx) == 0:
        empty = np.empty((0, 0), np.float32)
        return SampleBatch(empty, empty, empty, empty, np.empty(0, np.float32), np.empty(0, np.float32),
                           np.empty(0, "datetime64[ns]"), np.empty(0, object), np.empty(0, np.float32))

    l1_rows, l1s_rows, l2_rows, l3_rows = [], [], [], []
    y = np.empty(len(t_idx), np.float32)
    persistence = np.empty(len(t_idx), np.float32)
    for i, (tt, pp) in enumerate(zip(t_idx, p_idx)):
        input_end = int(tt - horizon)
        start = input_end - LOOKBACK + 1
        series = data.values[:, pp]
        stats, fallback = _hist_fill_and_stats(series, start, input_end)
        hist_mean, last_val, last_age = fallback
        lag_vals, lag_mask = [], []
        for lag in HIST_LAGS:
            idx = input_end - lag
            val = series[idx] if idx >= start else np.nan
            ok = np.isfinite(val)
            lag_vals.append(float(val) if ok else float(hist_mean))
            lag_mask.append(1.0 if ok else 0.0)
        l1 = np.array([*lag_vals, *lag_mask, *stats, float(valid_ratio[i])], np.float32)

        era = []
        for vi in range(len(DRIVERS)):
            era.append(float(data.era5[tt, pp, vi]))
            for lag in ERA5_LAGS:
                era.append(float(data.era5[input_end - lag, pp, vi]))
            era.append(float(data.era5[input_end - 5 : input_end + 1, pp, vi].mean()))
            era.append(float(data.era5[input_end - 23 : input_end + 1, pp, vi].mean()))
        l1s = np.concatenate([l1, data.static[pp].astype(np.float32)])
        l2 = np.concatenate([l1, np.asarray(era, np.float32)])
        l3 = np.concatenate([l2, data.static[pp].astype(np.float32)])

        l1_rows.append(l1)
        l1s_rows.append(l1s)
        l2_rows.append(l2)
        l3_rows.append(l3)
        y[i] = data.values[tt, pp]
        persistence[i] = last_val if np.isfinite(last_val) else hist_mean

    target_times = data.times[t_idx].to_numpy()
    cities = np.full(len(t_idx), data.city, dtype=object)
    return SampleBatch(
        x_l1=np.vstack(l1_rows).astype(np.float32),
        x_l1s=np.vstack(l1s_rows).astype(np.float32),
        x_l2=np.vstack(l2_rows).astype(np.float32),
        x_l3=np.vstack(l3_rows).astype(np.float32),
        y=y,
        persistence=persistence,
        target_times=target_times,
        city=cities,
        valid_ratio=valid_ratio,
    )


def concat_batches(batches: list[SampleBatch]) -> SampleBatch:
    batches = [b for b in batches if len(b.y)]
    if not batches:
        empty = np.empty((0, 0), np.float32)
        return SampleBatch(empty, empty, empty, empty, np.empty(0, np.float32), np.empty(0, np.float32),
                           np.empty(0, "datetime64[ns]"), np.empty(0, object), np.empty(0, np.float32))
    return SampleBatch(
        x_l1=np.vstack([b.x_l1 for b in batches]),
        x_l1s=np.vstack([b.x_l1s for b in batches]),
        x_l2=np.vstack([b.x_l2 for b in batches]),
        x_l3=np.vstack([b.x_l3 for b in batches]),
        y=np.concatenate([b.y for b in batches]),
        persistence=np.concatenate([b.persistence for b in batches]),
        target_times=np.concatenate([b.target_times for b in batches]),
        city=np.concatenate([b.city for b in batches]),
        valid_ratio=np.concatenate([b.valid_ratio for b in batches]),
    )


def layer_matrix(batch: SampleBatch, layer: str) -> np.ndarray:
    if layer == "L1":
        return batch.x_l1
    if layer == "L1S":
        return batch.x_l1s
    if layer == "L2":
        return batch.x_l2
    if layer == "L3":
        return batch.x_l3
    raise ValueError(layer)


def fit_feature_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x)
    counts = valid.sum(axis=0)
    filled = np.where(valid, x, 0.0)
    mu = filled.sum(axis=0) / np.maximum(counts, 1)
    centered = np.where(valid, x - mu[None, :], 0.0)
    sd = np.sqrt((centered * centered).sum(axis=0) / np.maximum(counts, 1))
    mu = np.where(counts > 0, mu, 0.0).astype(np.float32)
    sd = np.where((counts > 1) & (sd > 1e-6), sd, 1.0).astype(np.float32)
    return mu, sd


def apply_feature_scaler(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mu[None, :]) / sd[None, :], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_source_climatology(times: np.ndarray, y: np.ndarray) -> dict:
    dt = pd.to_datetime(times)
    month = dt.month.to_numpy()
    hour = dt.hour.to_numpy()
    mh = np.full((12, 24), np.nan, np.float32)
    m_only = np.full(12, np.nan, np.float32)
    h_only = np.full(24, np.nan, np.float32)
    for m in range(1, 13):
        m_mask = month == m
        if m_mask.any():
            m_only[m - 1] = np.nanmean(y[m_mask])
        for h in range(24):
            mask = m_mask & (hour == h)
            if mask.any():
                mh[m - 1, h] = np.nanmean(y[mask])
    for h in range(24):
        mask = hour == h
        if mask.any():
            h_only[h] = np.nanmean(y[mask])
    return {"month_hour": mh, "month": m_only, "hour": h_only, "global": float(np.nanmean(y))}


def predict_source_climatology(clim: dict, times: np.ndarray) -> np.ndarray:
    dt = pd.to_datetime(times)
    month = dt.month.to_numpy()
    hour = dt.hour.to_numpy()
    pred = clim["month_hour"][month - 1, hour].astype(np.float32)
    pred = np.where(np.isfinite(pred), pred, clim["month"][month - 1])
    pred = np.where(np.isfinite(pred), pred, clim["hour"][hour])
    pred = np.where(np.isfinite(pred), pred, clim["global"])
    return pred.astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"MAE": None, "RMSE": None, "N": 0}
    err = y_pred[ok] - y_true[ok]
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "N": int(ok.sum()),
    }


def metrics_by_city(y_true: np.ndarray, y_pred: np.ndarray, cities: np.ndarray) -> dict:
    out = {}
    for city in sorted(set(cities.tolist())):
        mask = cities == city
        out[city] = regression_metrics(y_true[mask], y_pred[mask])
    return out


def sample_counts_by_city(cities: np.ndarray) -> dict:
    return {city: int((cities == city).sum()) for city in sorted(set(cities.tolist()))}


def summarize_city_year_availability(cities: list[str], years: list[int], target_kind: str = "lst") -> dict:
    out = {}
    for city in cities:
        if target_kind == "lst":
            value_years = [y for y in years if (LST_BASE / city / f"lst_uhi_1km_hourly_{y}.parquet").exists()]
        else:
            value_years = []
            for y in years:
                intl = ATUHI_BASE / city / f"atuhi_ood_1km_hourly_{y}.parquet"
                hostrada = list((HOSTRADA_BASE / city / "monthly_uhi").glob(f"uhi_{y}*.parquet"))
                if intl.exists() or hostrada:
                    value_years.append(y)
        out[city] = {
            f"{target_kind}_years": value_years,
            "era5_years": [y for y in years if (ERA5_BASE / city / f"era5_hourly_{y}.parquet").exists()],
        }
    return out


def train_xgb(x_train: np.ndarray, y_train: np.ndarray, seed: int, n_estimators: int, max_depth: int):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
        verbosity=0,
    )
    model.fit(x_train, y_train)
    return model


def build_source_batches(args, source_cities: list[str], horizon: int) -> SampleBatch:
    batches = []
    for city in source_cities:
        for year in args.train_years:
            print(f"    source {city} {year} H={horizon}", flush=True)
            data = load_city_year(city, year, args.n_pixels, args.seed, args.target_kind)
            batch = build_samples(
                data,
                horizon=horizon,
                max_samples=args.train_samples_per_city_year,
                min_valid_ratio=args.min_valid_ratio,
                seed=stable_city_seed(args.seed, city, year, horizon),
            )
            batches.append(batch)
    return concat_batches(batches)


def build_ood_batches(args, horizon: int, ood_cities: list[str]) -> SampleBatch:
    batches = []
    for city in ood_cities:
        for year in args.eval_years:
            print(f"    ood {city} {year} H={horizon}", flush=True)
            data = load_city_year(city, year, args.n_pixels, args.seed, args.target_kind)
            batch = build_samples(
                data,
                horizon=horizon,
                max_samples=args.eval_samples_per_city_year,
                min_valid_ratio=args.min_valid_ratio,
                seed=stable_city_seed(args.seed, city, year, horizon + 1000),
            )
            batches.append(batch)
    return concat_batches(batches)


def run_source_set(args, source_name: str, source_cities: list[str], ood_cities: list[str]) -> dict:
    out = {
        "source_cities": source_cities,
        "horizons": {},
    }
    for horizon in args.horizons:
        print(f"[horizon] source={source_name} H=+{horizon}h", flush=True)
        needs_source = ("xgboost" in args.models) or ("climatology" in args.models)
        source_batch = build_source_batches(args, source_cities, horizon) if needs_source else None
        ood_batch = build_ood_batches(args, horizon, ood_cities)
        h_out = {
            "samples": {
                "source_train": 0 if source_batch is None else int(len(source_batch.y)),
                "ood_eval": int(len(ood_batch.y)),
                "source_by_city": {} if source_batch is None else sample_counts_by_city(source_batch.city),
                "ood_by_city": sample_counts_by_city(ood_batch.city),
            },
            "models": {},
        }
        if len(ood_batch.y) == 0:
            out["horizons"][f"{horizon}h"] = h_out
            continue

        if "persistence" in args.models:
            pred = ood_batch.persistence
            h_out["models"]["Persistence"] = {
                "overall": regression_metrics(ood_batch.y, pred),
                "by_city": metrics_by_city(ood_batch.y, pred, ood_batch.city),
                "input_layer": "L1",
            }

        if "climatology" in args.models and source_batch is not None and len(source_batch.y):
            clim = fit_source_climatology(source_batch.target_times, source_batch.y)
            pred = predict_source_climatology(clim, ood_batch.target_times)
            h_out["models"]["SourceClimatology"] = {
                "overall": regression_metrics(ood_batch.y, pred),
                "by_city": metrics_by_city(ood_batch.y, pred, ood_batch.city),
                "input_layer": "source month x hour, no OOD labels",
            }

        if "xgboost" in args.models and source_batch is not None and len(source_batch.y):
            for layer in args.layers:
                x_train = layer_matrix(source_batch, layer)
                x_eval = layer_matrix(ood_batch, layer)
                mu, sd = fit_feature_scaler(x_train)
                x_train_z = apply_feature_scaler(x_train, mu, sd)
                x_eval_z = apply_feature_scaler(x_eval, mu, sd)
                model = train_xgb(
                    x_train_z,
                    source_batch.y,
                    seed=args.seed + horizon + LAYERS.index(layer),
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                )
                pred = model.predict(x_eval_z)
                name = f"XGBoost_{layer}"
                h_out["models"][name] = {
                    "overall": regression_metrics(ood_batch.y, pred),
                    "by_city": metrics_by_city(ood_batch.y, pred, ood_batch.city),
                    "input_layer": layer,
                    "n_features": int(x_train.shape[1]),
                }

        out["horizons"][f"{horizon}h"] = h_out
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-set", choices=["cfb4", "diverse4", "cfb7", "diverse7", "diverse8", "all"], default="diverse4")
    ap.add_argument("--source-cities", nargs="+", default=None,
                    help="Override --source-set with an explicit source-city list.")
    ap.add_argument("--ood-cities", nargs="+", default=None,
                    help="Override the default OOD4 evaluation cities.")
    ap.add_argument("--models", nargs="+", choices=["persistence", "climatology", "xgboost"],
                    default=["persistence", "climatology", "xgboost"])
    ap.add_argument("--layers", nargs="+", choices=LAYERS, default=["L1", "L2", "L3"])
    ap.add_argument("--target-kind", choices=["lst", "airt"], default="lst")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    ap.add_argument("--train-years", type=int, nargs="+", default=list(range(2015, 2023)))
    ap.add_argument("--eval-years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n-pixels", type=int, default=128)
    ap.add_argument("--train-samples-per-city-year", type=int, default=1500)
    ap.add_argument("--eval-samples-per-city-year", type=int, default=800)
    ap.add_argument("--min-valid-ratio", type=float, default=0.70)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_DIR / "3_ood_transfer.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    if args.source_cities:
        selected = {"custom": args.source_cities}
    elif args.source_set == "all":
        selected = SOURCE_SETS
    else:
        selected = {args.source_set: SOURCE_SETS[args.source_set]}
    ood_cities = args.ood_cities if args.ood_cities else OOD_CITIES

    result = {
        "task": "Task 3 - LST-UHI OOD Transfer",
        "protocol": {
            "target": "lst_uhi_K" if args.target_kind == "lst" else "air_temperature_uhi_K",
            "target_kind": args.target_kind,
            "lookback_hours": LOOKBACK,
            "horizons": args.horizons,
            "source_sets": selected,
            "ood_cities": ood_cities,
            "train_years": args.train_years,
            "eval_years": args.eval_years,
            "input_layers": {
                "L1": "historical LST-UHI only: lag values, lag masks, window stats",
                "L1S": "L1 + raw static10; model feature scaler is fit on source train only",
                "L2": "L1 + ERA5 six drivers at target/input/lags/rolling windows",
                "L3": "L2 + raw static10; model feature scaler is fit on source train only",
            },
            "min_valid_ratio": args.min_valid_ratio,
            "n_pixels_per_city": args.n_pixels,
            "train_samples_per_city_year": args.train_samples_per_city_year,
            "eval_samples_per_city_year": args.eval_samples_per_city_year,
            "seed": args.seed,
            "xgboost": {"n_estimators": args.n_estimators, "max_depth": args.max_depth},
            "note": "Sampled OOD benchmark. OOD4 labels are used for evaluation only.",
        },
        "availability": summarize_city_year_availability(
            sorted(set(sum(selected.values(), []) + ood_cities)),
            sorted(set(args.train_years + args.eval_years)),
            args.target_kind,
        ),
        "results": {},
    }

    for source_name, cities in selected.items():
        print(f"[source-set] {source_name}: {cities}", flush=True)
        result["results"][source_name] = run_source_set(args, source_name, cities, ood_cities)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=_json_default))
    print(f"[wrote] {out}", flush=True)


if __name__ == "__main__":
    main()
