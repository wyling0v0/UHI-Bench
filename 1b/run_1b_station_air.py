"""Task 1b station-format Air-T extreme detection for Rome and Temuco.

Rome and Temuco are OOD station datasets, not gridded city-mean products.  Their
processed arrays are station anomalies, i.e. each station minus the same-hour
station-network mean.  A city mean would therefore be close to zero by
construction.  This runner uses station-hour samples instead:

* labels are computed per station from train-year P95 and >=3h above-threshold
  runs;
* features are station lagged anomalies plus hour/day-of-year seasonality;
* evaluation pools station-hours in the held-out year.

Splits:
* Rome: train JJA 2019, evaluate JJA 2020.
* Temuco: train 2017, evaluate 2018.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_1b_classify import run_iforest, run_ocsvm, run_rf, run_xgb  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import STATION_BASE  # noqa: E402


LAGS = [1, 6, 12, 24]
STATION_DATA = {
    "rome": {
        "root": STATION_BASE / "rome_uhi" / "processed",
        "years": [2019, 2020],
        "train_year": 2019,
        "test_year": 2020,
        "time_fix": "unix_seconds",
        "note": "ASTI Rome JJA station anomaly; train=2019, eval=2020.",
    },
    "temuco": {
        "root": STATION_BASE / "uhi_temuco" / "processed",
        "years": [2017, 2018],
        "train_year": 2017,
        "test_year": 2018,
        "time_fix": "stored_seconds_divided_by_1000_round_to_hour",
        "note": "Temuco station anomaly; train=2017, eval=2018.",
    },
}


def clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    miss = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    far = fp / (fp + tp) if (fp + tp) > 0 else float("nan")
    return {
        "F1": float(f1),
        "MissRate": float(miss),
        "FAR": float(far),
        "n_pos": int(y_true.sum()),
        "n": int(len(y_true)),
    }


def split_metrics(y_true: np.ndarray, y_pred: np.ndarray, day: np.ndarray) -> dict:
    return {
        "overall": clf_metrics(y_true, y_pred),
        "day": clf_metrics(y_true[day], y_pred[day]),
        "night": clf_metrics(y_true[~day], y_pred[~day]),
    }


def _load_times(path: Path, city: str) -> pd.DatetimeIndex:
    raw = np.asarray(np.load(path), dtype=np.int64)
    if city == "temuco" and raw.max(initial=0) < 100_000_000:
        return pd.to_datetime(raw * 1000, unit="s").round("h")
    return pd.to_datetime(raw, unit="s")


def load_station_matrix(city: str) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    spec = STATION_DATA[city]
    root = spec["root"]
    values, times = [], []
    meta = {
        "city": city,
        "source_root": str(root),
        "years": spec["years"],
        "train_year": spec["train_year"],
        "test_year": spec["test_year"],
        "time_fix": spec["time_fix"],
        "note": spec["note"],
    }
    station_file = root / "station_ids.txt"
    if station_file.exists():
        meta["n_stations"] = len(station_file.read_text().splitlines())
    for year in spec["years"]:
        arr = np.asarray(np.load(root / f"anomaly_{year}.npy"), dtype=np.float32)
        idx = _load_times(root / f"timestamps_{year}.npy", city)
        if len(idx) != arr.shape[0]:
            raise RuntimeError(f"{city} {year}: timestamps {len(idx)} != anomaly rows {arr.shape[0]}")
        values.append(arr)
        times.append(idx)
        meta[f"{year}_rows"] = int(arr.shape[0])
        meta[f"{year}_nan_frac"] = float(np.isnan(arr).mean())
    v = np.concatenate(values, axis=0)
    t = pd.DatetimeIndex(np.concatenate([x.to_numpy() for x in times]))
    order = np.argsort(t.values)
    v = v[order]
    t = pd.DatetimeIndex(t.values[order])
    # If rounded Temuco times collide, average duplicate station rows.
    if t.has_duplicates:
        frames = []
        for j in range(v.shape[1]):
            frames.append(pd.Series(v[:, j], index=t).groupby(level=0).mean())
        df = pd.concat(frames, axis=1).sort_index()
        v = df.to_numpy(np.float32)
        t = pd.DatetimeIndex(df.index)
    meta["n_hours"] = int(v.shape[0])
    meta["start"] = str(t.min())
    meta["end"] = str(t.max())
    return v, t, meta


def station_extreme_labels(values: np.ndarray, train_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_t, n_s = values.shape
    labels = np.zeros((n_t, n_s), dtype=bool)
    thresholds = np.full(n_s, np.nan, dtype=np.float32)
    for j in range(n_s):
        series = values[:, j]
        train_vals = series[train_time & np.isfinite(series)]
        if train_vals.size < 10:
            continue
        thr = float(np.nanpercentile(train_vals, 95))
        thresholds[j] = thr
        above = np.isfinite(series) & (series > thr)
        i = 0
        while i < n_t:
            if above[i]:
                k = i
                while k < n_t and above[k]:
                    k += 1
                if k - i >= 3:
                    labels[i:k, j] = True
                i = k
            else:
                i += 1
    return labels, thresholds


def build_station_samples(values: np.ndarray, times: pd.DatetimeIndex, labels: np.ndarray,
                          train_year: int, test_year: int):
    n_t, n_s = values.shape
    train_time = times.year == train_year
    test_time = times.year == test_year
    med = np.nanmedian(np.where(train_time[:, None], values, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    lag_blocks = []
    valid_lag = np.ones((n_t, n_s), dtype=bool)
    for lag in LAGS:
        block = np.full((n_t, n_s), np.nan, dtype=np.float32)
        block[lag:] = values[:-lag]
        valid_lag &= np.isfinite(block)
        block = np.where(np.isfinite(block), block, med[None, :])
        lag_blocks.append(block)

    hour = np.asarray(times.hour)
    doy = np.asarray(times.dayofyear)
    seasonal = np.stack([
        hour / 23.0,
        np.sin(2 * np.pi * doy / 365.0),
        np.cos(2 * np.pi * doy / 365.0),
    ], axis=1).astype(np.float32)
    seas3 = np.repeat(seasonal[:, None, :], n_s, axis=1)
    X = np.concatenate([b[:, :, None] for b in lag_blocks] + [seas3], axis=2)
    y = labels.astype(np.int8)
    sample_ok = np.isfinite(values) & valid_lag
    train_mask = sample_ok & train_time[:, None]
    test_mask = sample_ok & test_time[:, None]
    day = ((hour >= 7) & (hour <= 17))[:, None]

    return (
        X[train_mask],
        y[train_mask],
        X[test_mask],
        y[test_mask],
        np.broadcast_to(day, sample_ok.shape)[test_mask],
        values,
        train_mask,
        test_mask,
    )


def stat_baselines(values: np.ndarray, labels: np.ndarray, thresholds: np.ndarray,
                   times: pd.DatetimeIndex, train_mask: np.ndarray, test_mask: np.ndarray):
    lag1 = np.full_like(values, np.nan, dtype=np.float32)
    lag1[1:] = values[:-1]
    pred_pct = lag1 > thresholds[None, :]

    hourly_thr = np.full((24, values.shape[1]), np.nan, dtype=np.float32)
    hour = np.asarray(times.hour)
    for h in range(24):
        hmask = (hour == h)[:, None] & train_mask & np.isfinite(lag1)
        for j in range(values.shape[1]):
            vals = lag1[:, j][hmask[:, j]]
            hourly_thr[h, j] = np.nanpercentile(vals, 95) if vals.size else thresholds[j]
    pred_seas = np.zeros_like(pred_pct)
    for i, h in enumerate(hour):
        pred_seas[i] = lag1[i] > hourly_thr[h]

    yte = labels[test_mask].astype(int)
    day_by_time = ((hour >= 7) & (hour <= 17))[:, None]
    day = np.broadcast_to(day_by_time, values.shape)[test_mask]
    return {
        "Percentile(L1:no-met)": split_metrics(yte, pred_pct[test_mask].astype(int), day),
        "SeasonalNaive(L1:no-met)": split_metrics(yte, pred_seas[test_mask].astype(int), day),
    }


def run_city(city: str) -> dict:
    values, times, meta = load_station_matrix(city)
    train_year = int(meta["train_year"])
    test_year = int(meta["test_year"])
    train_time = times.year == train_year
    labels, thresholds = station_extreme_labels(values, train_time)
    Xtr, ytr, Xte, yte, day_te, values, train_mask, test_mask = build_station_samples(
        values, times, labels, train_year, test_year
    )
    row = stat_baselines(values, labels, thresholds, times, train_mask, test_mask)
    row["RandomForest(L1:no-met)"] = run_rf(Xtr, ytr, Xte, yte, day_te)
    row["XGBoost(L1:no-met)"] = run_xgb(Xtr, ytr, Xte, yte, day_te)
    row["IsolationForest(L1:no-met)"] = run_iforest(Xtr, ytr, Xte, yte, day_te)
    row["OneClassSVM(L1:no-met)"] = run_ocsvm(Xtr, ytr, Xte, yte, day_te)
    meta.update({
        "label_scope": "per-station train-year P95 with >=3h runs",
        "feature_scope": "station lag anomalies [1,6,12,24] + seasonality",
        "train_samples": int(len(ytr)),
        "test_samples": int(len(yte)),
        "train_pos": int(ytr.sum()),
        "test_pos": int(yte.sum()),
        "valid_station_thresholds": int(np.isfinite(thresholds).sum()),
    })
    return {"meta": meta, "Ta_station": row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["rome", "temuco"],
                    choices=sorted(STATION_DATA))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "1c_station_air.json"))
    args = ap.parse_args()

    out = {}
    for city in args.cities:
        print(f"[1c/station-air] {city}", flush=True)
        out[city] = run_city(city)
        xgb_f1 = out[city]["Ta_station"]["XGBoost(L1:no-met)"]["overall"]["F1"]
        pct_f1 = out[city]["Ta_station"]["Percentile(L1:no-met)"]["overall"]["F1"]
        pos = out[city]["meta"]["test_pos"]
        n = out[city]["meta"]["test_samples"]
        print(f"  XGB F1={xgb_f1:.3f} Percentile F1={pct_f1:.3f} pos={pos}/{n}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
