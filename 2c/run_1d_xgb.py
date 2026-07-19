"""Task 2c — XGBoost forecast baseline (self-contained, CPU).

Global channel-independent XGBoost: pooled across pixels, features =
[lagged UHI (t, t-1, t-3, t-6, t-12, t-24), hour, doy sin/cos, static feats],
target = UHI at t+h. One model per horizon. Train on 2015-2022, eval 2023-2025.

Output merged into 2c/results/1d_forecast.json.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_ta_field
from common.baselines import standardize_feats

HORIZONS = [1, 6, 12, 24, 48, 96]
LAGS = [0, 1, 3, 6, 12, 24]


def build_samples(values, static, hours, doy_arr, H_max, pixel_subsample, time_stride):
    """Vectorised: for each (pixel, t) → feat + targets. t in [t_min, T-H_max)."""
    T, N = values.shape
    t_min = max(max(LAGS), 24)
    t_max = T - H_max
    ts = np.arange(t_min, t_max, time_stride)
    seas = np.stack([hours[ts], np.sin(2*np.pi*doy_arr[ts]), np.cos(2*np.pi*doy_arr[ts])], axis=1)
    feats, ys = [], [[] for _ in HORIZONS]
    for p in pixel_subsample:
        series = values[:, p]
        if np.isfinite(series).sum() < T * 0.5:
            continue
        s = pd.Series(series).ffill().bfill().to_numpy()
        # lag features at each ts
        lag_cols = []
        for lg in LAGS:
            col = np.empty(T); col[:lg] = np.nan; col[lg:] = s[:T-lg]
            lag_cols.append(col[ts])
        lag_block = np.stack(lag_cols, axis=1)                 # [len(ts), len(LAGS)]
        st_block = np.broadcast_to(static[p], (len(ts), static.shape[1]))
        cur_valid = np.isfinite(series[ts])
        feat = np.concatenate([lag_block, seas, st_block], axis=1)  # [len(ts), nfeat]
        feats.append(feat[cur_valid])
        for hi, h in enumerate(HORIZONS):
            ys[hi].append((series[ts + h])[cur_valid])
    feats = np.concatenate(feats, axis=0)
    ys = [np.concatenate(y) for y in ys]
    return feats.astype(np.float32), [y.astype(np.float32) for y in ys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="munich")
    ap.add_argument("--train_years", type=int, nargs="+", default=[2015,2016,2017,2018,2019,2020,2021,2022])
    ap.add_argument("--test_years", type=int, nargs="+", default=[2023,2024,2025])
    ap.add_argument("--n_pixels", type=int, default=400)
    ap.add_argument("--train_stride", type=int, default=6)
    ap.add_argument("--test_stride", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "1d_forecast.json"))
    a = ap.parse_args()

    res = {}
    for split, yrs, stride in [("train", a.train_years, a.train_stride),
                               ("test", a.test_years, a.test_stride)]:
        f = load_ta_field(a.city, years=yrs)
        static = standardize_feats(f.feats)
        times = pd.to_datetime(f.times)
        rng = np.random.default_rng(42)
        psub = rng.choice(f.values.shape[1], min(a.n_pixels, f.values.shape[1]), replace=False)
        res[split] = build_samples(f.values, static, times.hour.to_numpy(),
                                   times.dayofyear.to_numpy(), max(HORIZONS), psub, stride)
        print(f"  {split}: feats {res[split][0].shape}")

    Xtr, ytr_list = res["train"]
    Xte, yte_list = res["test"]
    # drop rows with any NaN feat
    ok_tr = np.isfinite(Xtr).all(1)
    Xtr = Xtr[ok_tr]
    row = {}
    for hi, h in enumerate(HORIZONS):
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             n_jobs=8, verbosity=0)
        y = ytr_list[hi][ok_tr]
        oky = np.isfinite(y)
        m.fit(Xtr[oky], y[oky])
        ok_te = np.isfinite(Xte).all(1)
        pred = m.predict(Xte[ok_te])
        yt = yte_list[hi][ok_te]
        okt = np.isfinite(yt)
        row[f"{h}h"] = float(np.abs(pred[okt] - yt[okt]).mean())
        print(f"  +{h}h: MAE={row[f'{h}h']:.4f}")

    data = json.loads(Path(a.out).read_text())
    data.setdefault("Ta", {}).setdefault(a.city.capitalize(), {})["XGBoost(supervised)"] = row
    Path(a.out).write_text(json.dumps(data, indent=2))
    print(f"[merged] XGBoost {a.city} -> {a.out}")


if __name__ == "__main__":
    main()
