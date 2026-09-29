"""Task 1b-B — extreme-event DETECTION classifiers (binary).

Distinct from 1b-A label-agreement and the IF/OCSVM *labelers*: here we TRAIN
classifiers to PREDICT whether an hour is extreme from precursor features
(historical UHI lags [+ ERA5/static]), without peeking at the current value.

Input fairness: base = UHI lags + seasonal; +meteo = +ERA5; +static =
city-mean Tier-1 morphology broadcast over time; +meteo+static = both.
Models: XGBoost classifier (+ LSTM classifier). Real-truth cities Berlin/Munich/
Hamburg x {Ta, LST}. Metrics: F1 / Miss Rate / False Alarm Rate, day/night.
"""
from __future__ import annotations
import argparse, json, sys, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_1b import (
    DE_CITIES,
    label_extreme_by_split,
    lst_city_series,
    ta_city_series,
)
from common.paths import CACHE_ROOT, ERA5_BASE, STATIC_BASE
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler

ERA5_CITY_CACHE = CACHE_ROOT / "1c_era5_city_mean_classify"
LAGS = [1, 6, 12, 24]
METEO_COLS = ["t2m", "u10", "v10", "tcc", "d2m", "blh", "ssrd"]
CONFIG_LABEL = {
    "base": "L1:no-met",
    "meteo": "L2:+ERA5",
    "static": "+static",
    "meteo_static": "+meteo+static",
}
METHODS = ["rf", "xgb", "iforest", "ocsvm", "lstm"]
CITIES = {
    "berlin": "Cfb",
    "munich": "Cfb",
    "hamburg": "Cfb",
    "cologne": "Cfb",
    "cairo": "BWh",
    "riyadh": "BWh",
    "lagos": "Aw",
    "johannesburg": "Cwb",
}


def model_key(model: str, config: str) -> str:
    return f"{model}({CONFIG_LABEL[config]})"


def static_city_mean(city: str, n_static: int = 10):
    """City-mean Tier-1 static morphology vector for city-level detection."""
    p = STATIC_BASE / city / "static_features.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    feats = np.asarray(d["features"], np.float32)[:, :n_static]
    if feats.size == 0:
        return None
    finite = np.isfinite(feats)
    sums = np.where(finite, feats, 0.0).sum(axis=0, dtype=np.float64)
    counts = finite.sum(axis=0)
    vec = np.divide(
        sums,
        counts,
        out=np.zeros(n_static, dtype=np.float64),
        where=counts > 0,
    ).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def static_block(static_vec, n_rows: int):
    if static_vec is None:
        return None
    sv = np.asarray(static_vec, np.float32)
    sv = (sv - sv.mean()) / (sv.std() + 1e-6)
    return np.tile(sv[None, :], (n_rows, 1)).astype(np.float32)


def era5_city_series(city, years):
    import pyarrow.parquet as pq

    years = [int(y) for y in years]
    ERA5_CITY_CACHE.mkdir(parents=True, exist_ok=True)
    cache = ERA5_CITY_CACHE / f"{city}_{'-'.join(map(str, years))}_{'-'.join(METEO_COLS)}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        time_col = "datetime" if "datetime" in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        return df.set_index(time_col).sort_index()[METEO_COLS]

    fs = []
    for y in years:
        fs += [(y, f) for f in glob.glob(str(ERA5_BASE / city / f"era5_hourly_{y}.parquet"))]
    if not fs:
        return None
    parts = []
    for y, f in fs:
        pf = pq.ParquetFile(f)
        n_rows = pf.metadata.num_rows
        n_hours = 8784 if pd.Timestamp(y, 12, 31).dayofyear == 366 else 8760
        if n_rows % n_hours == 0:
            n_pix = n_rows // n_hours
            cols = []
            for col in METEO_COLS:
                arr = pq.read_table(f, columns=[col])[col].to_numpy(
                    zero_copy_only=False
                ).astype(np.float32, copy=False)
                cols.append(arr.reshape(n_hours, n_pix).mean(axis=1))
            idx = pd.date_range(f"{y}-01-01", periods=n_hours, freq="h")
            parts.append(pd.DataFrame(np.stack(cols, axis=1), index=idx, columns=METEO_COLS))
        else:
            df = pd.read_parquet(f, columns=["datetime", *METEO_COLS])
            df["datetime"] = pd.to_datetime(df["datetime"])
            parts.append(df.groupby("datetime")[METEO_COLS].mean())
    out = pd.concat(parts).groupby(level=0).mean().sort_index()
    out.rename_axis("datetime").reset_index().to_parquet(cache, index=False)
    return out


def build_feats(uhi, era5, label, static_vec=None, train_years_end=2022):
    """Return config -> (Xtr,ytr,Xte,yte,day_te)."""
    s = uhi.sort_index()
    yrs = s.index.year.values
    tr = yrs <= train_years_end
    train_values = s.values[tr]
    fill = float(np.nanmedian(train_values)) if np.isfinite(train_values).any() else 0.0
    ok = np.isfinite(s.values)
    s = pd.Series(np.where(ok, s.values, fill), index=s.index)
    hours = s.index.hour.values
    doy = s.index.dayofyear.values
    # lag features
    lag_feats = np.stack([s.shift(L).fillna(fill).values for L in LAGS], axis=1)
    seas = np.stack([hours, np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)], axis=1)
    X1 = np.concatenate([lag_feats, seas], axis=1)              # Layer1
    if era5 is not None:
        e = era5.reindex(s.index)
        train_medians = e.iloc[np.flatnonzero(tr)].median()
        e = e.fillna(train_medians).fillna(0.0)
        X2 = np.concatenate([X1, e.values], axis=1)             # +meteo
    else:
        X2 = X1
    sta = static_block(static_vec, len(s))
    matrices = {"base": X1, "meteo": X2}
    if sta is not None:
        matrices["static"] = np.concatenate([X1, sta], axis=1)
        matrices["meteo_static"] = np.concatenate([X2, sta], axis=1)
    y = label.values.astype(int)
    te = yrs >= train_years_end + 1
    day_all = (hours >= 7) & (hours <= 17)
    day_te = day_all[te]              # restrict to test subset (len = sum(te))
    return {name: (X[tr], y[tr], X[te], y[te], day_te) for name, X in matrices.items()}


def clf_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) > 0 else float("nan")
    miss = fn/(fn+tp) if (fn+tp) > 0 else float("nan")     # extreme missed
    far = fp/(fp+tp) if (fp+tp) > 0 else float("nan")      # predicted-extreme not real
    return {"F1": float(f1), "MissRate": float(miss), "FAR": float(far),
            "n_pos": int(y_true.sum()), "n": int(len(y_true))}


def run_xgb(Xtr, ytr, Xte, yte, day_te):
    pos = max(int((ytr == 0).sum() / max((ytr == 1).sum(), 1)), 1)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=pos, n_jobs=8, verbosity=0,
                          eval_metric="logloss")
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    out = {"overall": clf_metrics(yte, pred),
           "day": clf_metrics(yte[day_te], pred[day_te]),
           "night": clf_metrics(yte[~day_te], pred[~day_te])}
    return out


def _split_metrics(yte, pred, day_te):
    return {"overall": clf_metrics(yte, pred),
            "day": clf_metrics(yte[day_te], pred[day_te]),
            "night": clf_metrics(yte[~day_te], pred[~day_te])}


def run_stat_baselines(uhi, label, train_years_end=2022):
    """No-model statistical baselines (Layer1 only, no ERA5 interface).

    Predict extreme(t) from the lag-1 precursor against a threshold learned on
    train only: (i) global historical P95 (Percentile), (ii) per-hour-of-day P95
    (Seasonal Naive). Both are persistence-style -> conservative, high Miss / low FAR.
    Returns {name: {overall,day,night}}.
    """
    s = uhi.sort_index()
    yrs = s.index.year.values
    tr = yrs <= train_years_end
    train_values = s.values[tr]
    fill = float(np.nanmedian(train_values)) if np.isfinite(train_values).any() else 0.0
    vals = np.where(np.isfinite(s.values), s.values, fill)
    s = pd.Series(vals, index=s.index)
    hrs = s.index.hour.values
    y = label.values.astype(int)
    lag1 = s.shift(1).fillna(fill).values
    te = yrs >= train_years_end + 1
    day_all = (hrs >= 7) & (hrs <= 17)
    day_te = day_all[te]
    out = {}
    # (i) Percentile threshold: global train P95 of lag-1
    thr = np.nanpercentile(lag1[tr], 95)
    pred = (lag1 > thr).astype(int)
    out["Percentile(L1:no-met)"] = _split_metrics(y[te], pred[te], day_te)
    # (ii) Seasonal naive: per-hour-of-day P95 of lag-1 from train
    lag1_tr, hrs_tr = lag1[tr], hrs[tr]
    hourly = {h: np.nanpercentile(lag1_tr[hrs_tr == h], 95) for h in range(24)}
    seas_thr = np.array([hourly[h] for h in hrs])
    pred2 = (lag1 > seas_thr).astype(int)
    out["SeasonalNaive(L1:no-met)"] = _split_metrics(y[te], pred2[te], day_te)
    return out


def run_rf(Xtr, ytr, Xte, yte, day_te):
    m = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=8,
                               class_weight="balanced", random_state=0)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    return _split_metrics(yte, pred, day_te)


def _unsup_predict(model, Xte, yte, day_te):
    raw = model.predict(Xte)          # -1 = anomaly (extreme), 1 = normal
    pred = (raw == -1).astype(int)
    return _split_metrics(yte, pred, day_te)


def run_iforest(Xtr, ytr, Xte, yte, day_te):
    contam = float(max(ytr.mean(), 0.01))
    m = IsolationForest(n_estimators=300, contamination=contam,
                        random_state=0, n_jobs=8)
    m.fit(Xtr)
    return _unsup_predict(m, Xte, yte, day_te)


def run_ocsvm(Xtr, ytr, Xte, yte, day_te):
    """Scalable linear One-Class SVM (SGD-based, O(n))."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    contam = float(max(ytr.mean(), 0.01))
    m = SGDOneClassSVM(nu=contam, random_state=0)
    m.fit(Xtr_s)
    return _unsup_predict(m, Xte_s, yte, day_te)


def build_sequences(uhi, era5, label, static_vec=None, L=24, train_years_end=2022):
    """Sliding windows ending at t-1, with the extreme label at t."""
    import torch
    s = uhi.sort_index()
    hrs = s.index.hour.values; doy = s.index.dayofyear.values; yrs = s.index.year.values
    tr_hours = yrs <= train_years_end
    train_values = s.values[tr_hours]
    fill = float(np.nanmedian(train_values)) if np.isfinite(train_values).any() else 0.0
    s = pd.Series(np.where(np.isfinite(s.values), s.values, fill), index=s.index)
    seas = np.stack([hrs/23, np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)], axis=1)
    uhi_v = s.values[:, None]
    feat1 = np.concatenate([uhi_v, seas], axis=1)         # [T, 4]
    if era5 is not None:
        e = era5.reindex(s.index)
        train_medians = e.iloc[np.flatnonzero(tr_hours)].median()
        e = e.fillna(train_medians).fillna(0.0).values
        feat2 = np.concatenate([feat1, e], axis=1)
    else:
        feat2 = feat1
    sta = static_block(static_vec, len(s))
    features = {"base": feat1, "meteo": feat2}
    if sta is not None:
        features["static"] = np.concatenate([feat1, sta], axis=1)
        features["meteo_static"] = np.concatenate([feat2, sta], axis=1)
    y = label.values.astype(int)
    starts = np.arange(0, len(s) - L, 3)
    day_all = (hrs >= 7) & (hrs <= 17)
    out = {}
    for name, feat in features.items():
        Xw = np.stack([feat[i:i+L] for i in starts])          # [N,L,F]
        target_pos = starts + L
        yw = y[target_pos]                                      # predict next hour
        yr = yrs[target_pos]
        day_win = day_all[target_pos]
        tr = yr <= train_years_end; te = yr >= train_years_end + 1
        out[name] = (torch.from_numpy(Xw[tr]).float(), yw[tr],
                     torch.from_numpy(Xw[te]).float(), yw[te], day_win[te])
    return out


def run_lstm(Xtr, ytr, Xte, yte, day_te, device="cpu", epochs=12):
    import torch, torch.nn as nn
    pos = max(int(((1-ytr)==1).sum() / max(int(ytr.sum()),1)), 1)
    Fd = Xtr.shape[-1]
    class C(nn.Module):
        def __init__(s):
            super().__init__(); s.lstm = nn.LSTM(Fd, 32, batch_first=True)
            s.head = nn.Linear(32, 1)
        def forward(s, x):
            _, (h_n, _) = s.lstm(x); return s.head(h_n[-1]).squeeze(-1)
    mdl = C().to(device)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos), device=device))
    Xtr, ytr = Xtr.to(device), torch.from_numpy(ytr).float().to(device)
    for ep in range(epochs):
        mdl.train()
        idx = np.random.default_rng(0).permutation(len(Xtr))
        for i in range(0, len(Xtr), 512):
            b = idx[i:i+512]
            loss = bce(mdl(Xtr[b]), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
    mdl.eval()
    with torch.no_grad():
        prob = torch.sigmoid(mdl(Xte.to(device))).cpu().numpy()
    pred = (prob > 0.5).astype(int)
    return {"overall": clf_metrics(yte, pred),
            "day": clf_metrics(yte[day_te], pred[day_te]),
            "night": clf_metrics(yte[~day_te], pred[~day_te])}


def merge_rows(out_path: Path, city: str, mod: str, row: dict) -> None:
    if out_path.exists():
        try:
            latest = json.loads(out_path.read_text())
        except Exception:
            latest = {}
    else:
        latest = {}
    latest.setdefault(city, {}).setdefault(mod, {}).update(row)
    out_path.write_text(json.dumps(latest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2015, 2026)))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "1c_classify.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cities", nargs="+", default=list(CITIES),
                    help="subset of cities to run; existing output is preserved/merged")
    ap.add_argument("--mods", nargs="+", choices=["Ta", "LST"], default=["Ta", "LST"])
    ap.add_argument("--configs", nargs="+", choices=list(CONFIG_LABEL),
                    default=list(CONFIG_LABEL),
                    help="input configs to run; use 'static meteo_static' to only fill static rows")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS,
                    help="model families to run")
    ap.add_argument("--lstm-epochs", type=int, default=12)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    # robust device: fall back to CPU if CUDA unavailable
    try:
        import torch
        if a.device.startswith("cuda") and not torch.cuda.is_available():
            a.device = "cpu"
            print("[device] CUDA unavailable, falling back to CPU", flush=True)
    except Exception:
        a.device = "cpu"
    out_path = Path(a.out)
    if out_path.exists():
        try:
            res = json.loads(out_path.read_text())
        except Exception:
            res = {}
    else:
        res = {}
    for city in a.cities:
        clim = CITIES[city]
        res.setdefault(city, {})
        static_vec = static_city_mean(city)
        if static_vec is None and any(c in a.configs for c in ("static", "meteo_static")):
            print(f"[warn] {city}: no static_features.npz; static configs skipped", flush=True)
        for mod, loader in [("Ta", ta_city_series), ("LST", lst_city_series)]:
            if mod not in a.mods:
                continue
            uhi = loader(city, a.years)
            if uhi is None:
                continue
            era5 = era5_city_series(city, a.years)
            lab, _ = label_extreme_by_split(uhi, train_end=2022)
            feats = build_feats(uhi, era5, lab, static_vec=static_vec)
            row = {}
            existing = res.setdefault(city, {}).setdefault(mod, {})
            # --- statistical (Layer1 only, no ERA5 interface) ---
            if "base" in a.configs:
                stat_keys = ["Percentile(L1:no-met)", "SeasonalNaive(L1:no-met)"]
                if not (a.skip_existing and all(k in existing for k in stat_keys)):
                    stat = run_stat_baselines(uhi, lab)
                    for k in stat_keys:
                        if not (a.skip_existing and k in existing):
                            row[k] = stat[k]
            # --- ML and anomaly detection ---
            for cfg in a.configs:
                if cfg not in feats:
                    continue
                Xtr, ytr, Xte, yte, dte = feats[cfg]
                jobs = [
                    ("rf", "RandomForest", run_rf),
                    ("xgb", "XGBoost", run_xgb),
                    ("iforest", "IsolationForest", run_iforest),
                    ("ocsvm", "OneClassSVM", run_ocsvm),
                ]
                for method, label, fn in jobs:
                    key = model_key(label, cfg)
                    if method not in a.methods or (a.skip_existing and key in existing):
                        continue
                    row[key] = fn(Xtr, ytr, Xte, yte, dte)
            # --- DL (sequence) ---
            if "lstm" in a.methods:
                seq = build_sequences(uhi, era5, lab, static_vec=static_vec)
                for cfg in a.configs:
                    if cfg not in seq:
                        continue
                    key = model_key("LSTM", cfg)
                    if a.skip_existing and key in existing:
                        continue
                    Xtr, ytr, Xte, yte, dte = seq[cfg]
                    row[key] = run_lstm(
                        Xtr, ytr, Xte, yte, dte, device=a.device, epochs=a.lstm_epochs
                    )
            existing.update(row)
            if row:
                merge_rows(out_path, city, mod, row)
            parts = []
            for cfg in a.configs:
                key = model_key("XGBoost", cfg)
                if key in row:
                    parts.append(f"{key} F1={row[key]['overall']['F1']:.3f}")
            n_ref = next(iter(row.values()))["overall"]["n"] if row else len(uhi)
            n_pos = next(iter(row.values()))["overall"]["n_pos"] if row else 0
            status = " ".join(parts) if parts else f"{len(row)} rows"
            print(f"{city} {mod}: {status} (pos%={100*n_pos/max(n_ref,1):.1f})", flush=True)
    # Reload at save time so long-running classifier jobs do not clobber FM rows
    # appended by other runners while this process was computing.
    if out_path.exists():
        try:
            latest = json.loads(out_path.read_text())
        except Exception:
            latest = {}
    else:
        latest = {}
    for city, mods in res.items():
        if not isinstance(mods, dict):
            continue
        for mod, row in mods.items():
            if not isinstance(row, dict):
                continue
            latest.setdefault(city, {}).setdefault(mod, {}).update(row)
    out_path.write_text(json.dumps(latest, indent=2))
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
