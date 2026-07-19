"""Task 2c station-format Air-T anomaly forecasting for Rome and Temuco.

Rome and Temuco are station-anomaly matrices rather than gridded UHI cubes.
This supplement therefore forecasts station-hour anomalies directly and pools
held-out station-hours for MAE/RMSE.

Splits match the Task 2b station supplement:
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
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - handled at runtime
    xgb = None

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[0]
sys.path.insert(0, str(BENCH / "1b"))
from run_1b_station_air import STATION_DATA, load_station_matrix  # noqa: E402

HORIZONS = [1, 6, 12, 24, 48, 96]
LAGS = [0, 1, 3, 6, 12, 24, 48, 96, 168]


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    ok = np.isfinite(pred) & np.isfinite(true)
    if not np.any(ok):
        return {"MAE": None, "RMSE": None, "n": 0}
    err = pred[ok] - true[ok]
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "n": int(ok.sum()),
    }


def _compact_mae(row: dict[int, dict[str, float | int]]) -> dict[str, float | None]:
    return {f"{h}h": row[h]["MAE"] for h in HORIZONS}


def build_samples(values: np.ndarray, times: pd.DatetimeIndex, train_year: int, test_year: int):
    n_t, n_s = values.shape
    max_lag = max(LAGS)
    max_h = max(HORIZONS)
    base_idx = np.arange(max_lag, n_t - max_h)

    hour = times.hour.to_numpy()
    doy = times.dayofyear.to_numpy()
    seasonal = np.stack([
        hour / 23.0,
        np.sin(2 * np.pi * doy / 366.0),
        np.cos(2 * np.pi * doy / 366.0),
    ], axis=1).astype(np.float32)

    station_id = (np.arange(n_s, dtype=np.float32) / max(1, n_s - 1))[None, :, None]
    station_block = np.broadcast_to(station_id, (len(base_idx), n_s, 1))
    seas_block = np.broadcast_to(seasonal[base_idx, None, :], (len(base_idx), n_s, 3))
    lag_blocks = [values[base_idx - lag, :, None] for lag in LAGS]
    X3 = np.concatenate(lag_blocks + [seas_block, station_block], axis=2).astype(np.float32)

    train_time = times.year.to_numpy()[base_idx] == train_year
    test_time = times.year.to_numpy()[base_idx] == test_year
    lag_ok = np.isfinite(X3[:, :, :len(LAGS)]).all(axis=2)
    feat_ok = np.isfinite(X3).all(axis=2)
    train_base = train_time[:, None] & lag_ok & feat_ok
    test_base = test_time[:, None] & lag_ok & feat_ok

    out = {"train": {}, "test": {}, "meta": {}}
    for split, base_mask in [("train", train_base), ("test", test_base)]:
        X = X3[base_mask]
        out[split]["X"] = X
        out[split]["base_values"] = values[base_idx][base_mask].astype(np.float32)
        out[split]["base_indices"] = np.broadcast_to(base_idx[:, None], lag_ok.shape)[base_mask].astype(np.int32)
        out[split]["station_indices"] = np.broadcast_to(np.arange(n_s)[None, :], lag_ok.shape)[base_mask].astype(np.int32)
        for h in HORIZONS:
            y3 = values[base_idx + h]
            y = y3[base_mask].astype(np.float32)
            out[split][h] = y
    out["meta"].update({
        "n_stations": int(n_s),
        "n_hours": int(n_t),
        "lags": LAGS,
        "horizons": HORIZONS,
        "feature_scope": "station anomaly lags [0,1,3,6,12,24,48,96,168] + hour/day seasonality + station index",
    })
    return out


def fit_climatologies(values: np.ndarray, times: pd.DatetimeIndex, train_year: int):
    train = times.year.to_numpy() == train_year
    station_mean = np.nanmean(np.where(train[:, None], values, np.nan), axis=0).astype(np.float32)
    station_mean = np.where(np.isfinite(station_mean), station_mean, 0.0)
    hourly = np.full((24, values.shape[1]), np.nan, dtype=np.float32)
    hour = times.hour.to_numpy()
    for h in range(24):
        block = values[train & (hour == h)]
        if block.size:
            hourly[h] = np.nanmean(block, axis=0)
    hourly = np.where(np.isfinite(hourly), hourly, station_mean[None, :])
    return station_mean, hourly


def run_stat_baselines(samples: dict, values: np.ndarray, times: pd.DatetimeIndex, train_year: int):
    station_mean, hourly = fit_climatologies(values, times, train_year)
    test = samples["test"]
    base_idx = test["base_indices"]
    station_idx = test["station_indices"]
    hour = times.hour.to_numpy()
    rows: dict[str, dict[int, dict[str, float | int]]] = {
        "Persistence": {},
        "StationMean": {},
        "HourlyStationClimatology": {},
        "DailyPersistence": {},
    }
    for h in HORIZONS:
        y = test[h]
        rows["Persistence"][h] = _metrics(test["base_values"], y)
        rows["StationMean"][h] = _metrics(station_mean[station_idx], y)
        rows["HourlyStationClimatology"][h] = _metrics(hourly[hour[base_idx + h], station_idx], y)
        daily_idx = base_idx + h - 24
        valid_daily = daily_idx >= 0
        pred_daily = np.full_like(y, np.nan, dtype=np.float32)
        pred_daily[valid_daily] = values[daily_idx[valid_daily], station_idx[valid_daily]]
        rows["DailyPersistence"][h] = _metrics(pred_daily, y)
    return rows


def _fit_predict_sklearn(model, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray):
    ok_tr = np.isfinite(ytr) & np.isfinite(Xtr).all(axis=1)
    ok_te = np.isfinite(Xte).all(axis=1)
    pred = np.full(Xte.shape[0], np.nan, dtype=np.float32)
    if ok_tr.sum() < 50 or ok_te.sum() == 0:
        return pred
    model.fit(Xtr[ok_tr], ytr[ok_tr])
    pred[ok_te] = model.predict(Xte[ok_te]).astype(np.float32)
    return pred


def run_ml(samples: dict, seed: int, include_rf: bool, include_xgb: bool):
    Xtr = samples["train"]["X"]
    Xte = samples["test"]["X"]
    rows: dict[str, dict[int, dict[str, float | int]]] = {}
    if include_rf:
        rows["RandomForest"] = {}
    if include_xgb:
        rows["XGBoost"] = {}
    for h in HORIZONS:
        ytr = samples["train"][h]
        yte = samples["test"][h]
        if include_rf:
            rf = RandomForestRegressor(
                n_estimators=180,
                max_depth=18,
                min_samples_leaf=3,
                n_jobs=8,
                random_state=seed + h,
            )
            rows["RandomForest"][h] = _metrics(_fit_predict_sklearn(rf, Xtr, ytr, Xte), yte)
        if include_xgb:
            if xgb is None:
                rows["XGBoost"][h] = {"MAE": None, "RMSE": None, "n": 0}
            else:
                reg = xgb.XGBRegressor(
                    n_estimators=220,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    n_jobs=8,
                    random_state=seed + 1000 + h,
                    verbosity=0,
                )
                rows["XGBoost"][h] = _metrics(_fit_predict_sklearn(reg, Xtr, ytr, Xte), yte)
        print(f"    horizon {h}h done", flush=True)
    return rows


def run_city(city: str, seed: int, include_rf: bool, include_xgb: bool):
    values, times, meta = load_station_matrix(city)
    train_year = int(STATION_DATA[city]["train_year"])
    test_year = int(STATION_DATA[city]["test_year"])
    samples = build_samples(values, times, train_year, test_year)
    rows = run_stat_baselines(samples, values, times, train_year)
    rows.update(run_ml(samples, seed, include_rf, include_xgb))

    meta.update(samples["meta"])
    meta.update({
        "task": "1d station-format Air-T anomaly forecasting",
        "train_year": train_year,
        "test_year": test_year,
        "train_samples": int(samples["train"]["X"].shape[0]),
        "test_samples": int(samples["test"]["X"].shape[0]),
        "metric": "MAE/RMSE over held-out station-hours with finite target",
    })
    return {
        "meta": meta,
        "methods": {
            name: {
                "MAE": _compact_mae(hrow),
                "RMSE": {f"{h}h": hrow[h]["RMSE"] for h in HORIZONS},
                "n": {f"{h}h": hrow[h]["n"] for h in HORIZONS},
            }
            for name, hrow in rows.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["rome", "temuco"], choices=sorted(STATION_DATA))
    ap.add_argument("--out", default=str(HERE / "results" / "1d_station_air_forecast.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rf", action="store_true")
    ap.add_argument("--no-xgb", action="store_true")
    args = ap.parse_args()

    out = {
        "source": "benchmark/2c/run_1d_station_air.py",
        "horizons": [f"{h}h" for h in HORIZONS],
        "note": "Station-format Rome/Temuco supplement; not directly comparable to gridded 1d rows.",
        "Ta_station": {},
    }
    out_path = Path(args.out)
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if isinstance(old, dict):
            out.update(old)
            out.setdefault("Ta_station", {})
    for city in args.cities:
        print(f"[1d/station-air] {city}", flush=True)
        out["Ta_station"][city] = run_city(city, args.seed, not args.no_rf, not args.no_xgb)
        for name, row in out["Ta_station"][city]["methods"].items():
            mae = row["MAE"]
            print("  " + name + " " + " ".join(f"{h}={mae[h]:.4f}" if mae[h] is not None else f"{h}=NA"
                                                for h in out["horizons"]), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
