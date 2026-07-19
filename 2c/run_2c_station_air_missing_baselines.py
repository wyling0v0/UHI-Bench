"""Task 2c Rome/Temuco station-format forecasting baseline completion.

Rome and Temuco are station-anomaly matrices, not gridded 64px/256px cubes.
This runner therefore adds station-format equivalents for the missing Task 2c
baseline rows and records that protocol in the output metadata.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH / "1b"))

from run_1b_station_air import STATION_DATA, load_station_matrix  # noqa: E402
from run_1d_station_air import HORIZONS, _metrics, fit_climatologies  # noqa: E402

LOOKBACK = 168
FM_H = max(HORIZONS)
BASE_KEYS = [
    "Persistence(base)",
    "Climatology(base)",
    "Persistence(64px)",
    "Climatology(64px)",
]
SUP_KEYS = [
    "SARIMA(city-mean)",
    "XGBoost(base,sup)",
    "LSTM(base,sup)",
    "PatchTST(base,sup)",
    "iTransformer(base,sup)",
    "DLinear(base,sup)",
    "TimesNet(256px,sup)",
    "DeepUHI(256px)",
]
FM_KEYS = [
    "Chronos(64px)",
    "TimesFM(64px)",
    "Chronos-2(base,64px)",
    "Chronos-2(+static,64px)",
]


def _compact(hrows: dict[int, dict[str, float | int]]) -> dict[str, dict[str, float | int | None]]:
    return {
        "MAE": {f"{h}h": hrows[h]["MAE"] for h in HORIZONS},
        "RMSE": {f"{h}h": hrows[h]["RMSE"] for h in HORIZONS},
        "n": {f"{h}h": hrows[h]["n"] for h in HORIZONS},
    }


def _complete(row: dict | None) -> bool:
    if not isinstance(row, dict):
        return False
    mae = row.get("MAE")
    if not isinstance(mae, dict):
        return False
    return all(isinstance(mae.get(f"{h}h"), (int, float)) and np.isfinite(mae[f"{h}h"]) for h in HORIZONS)


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "source": "benchmark/2c/run_1d_station_air.py",
        "horizons": [f"{h}h" for h in HORIZONS],
        "note": "Station-format Rome/Temuco supplement; not directly comparable to gridded 1d rows.",
        "Ta_station": {},
    }


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _merge_methods(path: Path, city: str, rows: dict[str, dict]) -> None:
    data = _load_json(path)
    data.setdefault("Ta_station", {}).setdefault(city, {}).setdefault("methods", {})
    data["Ta_station"][city]["methods"].update(rows)
    data.setdefault("protocol_notes", {})["rome_temuco_station_completion"] = {
        "script": "benchmark/2c/run_2c_station_air_missing_baselines.py",
        "note": (
            "Rome/Temuco are station-anomaly matrices. Rows named 64px/256px are "
            "station-format equivalents using all available stations because no "
            "gridded pixel cube exists for these two datasets."
        ),
        "history_context_hours": LOOKBACK,
        "horizons_hours": HORIZONS,
    }
    _save_json(path, data)


def _method_rows(data: dict, city: str) -> dict:
    return data.get("Ta_station", {}).get(city, {}).get("methods", {})


def add_base_aliases(out_path: Path, city: str, values: np.ndarray, times: pd.DatetimeIndex) -> dict[str, dict]:
    data = _load_json(out_path)
    existing = _method_rows(data, city)
    rows: dict[str, dict] = {}
    if "Persistence" in existing:
        rows["Persistence(base)"] = existing["Persistence"]
    if "HourlyStationClimatology" in existing:
        rows["Climatology(base)"] = existing["HourlyStationClimatology"]
    if "XGBoost" in existing:
        rows["XGBoost(base,sup)"] = existing["XGBoost"]

    train_year = int(STATION_DATA[city]["train_year"])
    test_year = int(STATION_DATA[city]["test_year"])
    station_mean, hourly = fit_climatologies(values, times, train_year)
    starts = weekly_starts(times, test_year)
    station_idx = np.arange(values.shape[1], dtype=np.int64)
    p_rows = {h: {"MAE": None, "RMSE": None, "n": 0} for h in HORIZONS}
    c_rows = {h: {"MAE": None, "RMSE": None, "n": 0} for h in HORIZONS}
    if starts.size:
        hour = times.hour.to_numpy()
        for h in HORIZONS:
            pred_p, pred_c, true = [], [], []
            for s in starts:
                t = int(s + LOOKBACK + h - 1)
                y = values[t, station_idx]
                pred_p.append(values[s + LOOKBACK - 1, station_idx])
                pred_c.append(hourly[hour[t], station_idx])
                true.append(y)
            p_rows[h] = _metrics(np.concatenate(pred_p), np.concatenate(true))
            c_rows[h] = _metrics(np.concatenate(pred_c), np.concatenate(true))
    rows["Persistence(64px)"] = _compact(p_rows)
    rows["Climatology(64px)"] = _compact(c_rows)
    return rows


def weekly_starts(times: pd.DatetimeIndex, test_year: int, stride: int = 168) -> np.ndarray:
    years = times.year.to_numpy()
    candidates = []
    for s in range(0, len(times) - LOOKBACK - FM_H):
        target_slice = years[s + LOOKBACK : s + LOOKBACK + FM_H]
        if target_slice.size == FM_H and np.all(target_slice == test_year):
            candidates.append(s)
    if not candidates:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(candidates, dtype=np.int64)
    first = int(arr[0])
    return arr[(arr - first) % stride == 0]


def fill_context(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=np.float32))
    if s.notna().any():
        s = s.ffill().bfill().fillna(float(s.mean()))
    else:
        s = s.fillna(0.0)
    return s.to_numpy(np.float32)


def station_static(city: str, n_stations: int) -> np.ndarray:
    root = STATION_DATA[city]["root"]
    path = root / "static_features.npy"
    if not path.exists():
        return np.zeros((n_stations, 1), np.float32)
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr[:n_stations]
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    out = (np.where(np.isfinite(arr), arr, mu) - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def city_mean_sarima(values: np.ndarray, times: pd.DatetimeIndex, train_year: int, test_year: int) -> dict:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    s = pd.Series(np.nanmean(values, axis=1), index=times).ffill().bfill().fillna(0.0)
    train = s[s.index.year == train_year]
    if len(train) < 100:
        return _compact({h: {"MAE": None, "RMSE": None, "n": 0} for h in HORIZONS})
    fit_train = train.iloc[-4096:] if len(train) > 4096 else train
    model = SARIMAX(
        fit_train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=40)
    years = times.year.to_numpy()
    train_pos = np.flatnonzero(years == train_year)
    test_base = np.flatnonzero(years == test_year)
    train_end = int(train_pos[-1])
    fc = np.asarray(model.forecast(steps=len(s) - train_end - 1), dtype=np.float32)
    fallback = float(np.nanmean(fit_train))
    lo, hi = np.nanpercentile(fit_train, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = fallback - 10.0, fallback + 10.0
    fc = np.nan_to_num(fc, nan=fallback, posinf=hi, neginf=lo)
    fc = np.clip(fc, lo, hi).astype(np.float32)
    base = test_base[(test_base >= LOOKBACK) & (test_base < len(s) - max(HORIZONS))]
    rows: dict[int, dict[str, float | int | None]] = {}
    for h in HORIZONS:
        preds, true = [], []
        for idx in base:
            fc_idx = int(idx + h - train_end - 1)
            if fc_idx < 0 or fc_idx >= len(fc):
                continue
            pred = fc[fc_idx]
            y = values[int(idx + h)]
            preds.append(np.full(values.shape[1], float(pred), dtype=np.float32))
            true.append(y.astype(np.float32))
        rows[h] = _metrics(np.concatenate(preds), np.concatenate(true)) if preds else {"MAE": None, "RMSE": None, "n": 0}
    return _compact(rows)


def build_sequences(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    train_year: int,
    test_year: int,
    min_valid_ratio: float,
):
    n_t, n_s = values.shape
    max_h = max(HORIZONS)
    station_fill = np.nanmean(np.where((times.year.to_numpy() == train_year)[:, None], values, np.nan), axis=0)
    station_fill = np.where(np.isfinite(station_fill), station_fill, 0.0).astype(np.float32)

    hour = times.hour.to_numpy()
    doy = times.dayofyear.to_numpy()
    seas = np.stack(
        [
            hour / 23.0,
            np.sin(2 * np.pi * doy / 366.0),
            np.cos(2 * np.pi * doy / 366.0),
        ],
        axis=1,
    ).astype(np.float32)

    X_parts = {"train": [], "test": []}
    stat_parts = {"train": [], "test": []}
    y_parts = {split: {h: [] for h in HORIZONS} for split in ["train", "test"]}

    for base in range(LOOKBACK - 1, n_t - max_h):
        year = int(times.year[base])
        split = "train" if year == train_year else "test" if year == test_year else None
        if split is None:
            continue
        window = values[base - LOOKBACK + 1 : base + 1]
        finite = np.isfinite(window)
        valid_ratio = finite.mean(axis=0)
        good_stations = np.flatnonzero(valid_ratio >= min_valid_ratio)
        if good_stations.size == 0:
            continue
        seas_win = seas[base - LOOKBACK + 1 : base + 1]
        for st in good_stations:
            seg = window[:, st].astype(np.float32)
            fill = station_fill[st]
            hist = np.where(np.isfinite(seg), seg, fill)
            mask = np.isfinite(seg).astype(np.float32)
            X_parts[split].append(
                np.stack(
                    [hist, mask, seas_win[:, 0], seas_win[:, 1], seas_win[:, 2]],
                    axis=0,
                ).astype(np.float32)
            )
            stat_parts[split].append(st)
            for h in HORIZONS:
                y_parts[split][h].append(values[base + h, st])

    out = {"train": {}, "test": {}}
    for split in ["train", "test"]:
        if not X_parts[split]:
            out[split]["X"] = np.empty((0, 5, LOOKBACK), np.float32)
            out[split]["station"] = np.empty(0, np.int64)
            for h in HORIZONS:
                out[split][h] = np.empty(0, np.float32)
            continue
        out[split]["X"] = np.stack(X_parts[split]).astype(np.float32)
        out[split]["station"] = np.asarray(stat_parts[split], dtype=np.int64)
        for h in HORIZONS:
            out[split][h] = np.asarray(y_parts[split][h], dtype=np.float32)
    return out


def sample_sequences(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    train_year: int,
    test_year: int,
    split: str,
    max_samples: int,
    min_valid_ratio: float,
    seed: int,
) -> dict:
    n_t, n_s = values.shape
    year = train_year if split == "train" else test_year
    years = times.year.to_numpy()
    base_all = np.arange(LOOKBACK - 1, n_t - max(HORIZONS), dtype=np.int64)
    base_all = base_all[years[base_all] == year]
    if base_all.size == 0:
        return {"X": np.empty((0, 5, LOOKBACK), np.float32), "station": np.empty(0, np.int64), **{h: np.empty(0, np.float32) for h in HORIZONS}}

    finite = np.isfinite(values).astype(np.int16)
    cs = np.concatenate([np.zeros((1, n_s), dtype=np.int32), np.cumsum(finite, axis=0, dtype=np.int32)], axis=0)
    counts = cs[base_all + 1] - cs[base_all + 1 - LOOKBACK]
    valid = counts >= int(math.ceil(LOOKBACK * min_valid_ratio))
    bi, st = np.nonzero(valid)
    if bi.size == 0:
        return {"X": np.empty((0, 5, LOOKBACK), np.float32), "station": np.empty(0, np.int64), **{h: np.empty(0, np.float32) for h in HORIZONS}}
    bases = base_all[bi]

    has_target = np.zeros_like(bases, dtype=bool)
    for h in HORIZONS:
        has_target |= np.isfinite(values[bases + h, st])
    bases = bases[has_target]
    st = st[has_target]
    if bases.size > max_samples:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(bases.size, max_samples, replace=False))
        bases = bases[keep]
        st = st[keep]

    train_mask = years == train_year
    station_fill = np.nanmean(np.where(train_mask[:, None], values, np.nan), axis=0)
    station_fill = np.where(np.isfinite(station_fill), station_fill, 0.0).astype(np.float32)
    hour = times.hour.to_numpy()
    doy = times.dayofyear.to_numpy()
    seas = np.stack(
        [
            hour / 23.0,
            np.sin(2 * np.pi * doy / 366.0),
            np.cos(2 * np.pi * doy / 366.0),
        ],
        axis=1,
    ).astype(np.float32)

    X = np.empty((len(bases), 5, LOOKBACK), np.float32)
    y = {h: np.empty(len(bases), np.float32) for h in HORIZONS}
    for i, (base, station) in enumerate(zip(bases, st)):
        win = values[base - LOOKBACK + 1 : base + 1, station].astype(np.float32)
        ok = np.isfinite(win)
        hist = np.where(ok, win, station_fill[station])
        seas_win = seas[base - LOOKBACK + 1 : base + 1]
        X[i] = np.stack([hist, ok.astype(np.float32), seas_win[:, 0], seas_win[:, 1], seas_win[:, 2]], axis=0)
        for h in HORIZONS:
            y[h][i] = values[base + h, station]
    return {"X": X, "station": st.astype(np.int64), **y}


class LSTMReg(nn.Module):
    def __init__(self, cin: int, hidden: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(cin, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = x[..., ::6].transpose(1, 2)
        _, (h_n, _) = self.lstm(z)
        return self.head(h_n[-1]).squeeze(-1)


class PatchTSTReg(nn.Module):
    def __init__(self, cin: int, hidden: int = 32, patch: int = 24, nhead: int = 2):
        super().__init__()
        self.patch = patch
        self.tok = nn.Linear(patch * cin, hidden)
        self.pos = nn.Parameter(torch.zeros(LOOKBACK // patch, hidden))
        enc = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        bsz = x.shape[0]
        xp = x.unfold(-1, self.patch, self.patch)
        xp = xp.permute(0, 2, 1, 3).reshape(bsz, xp.shape[2], -1)
        z = self.tok(xp) + self.pos[None]
        return self.head(self.tr(z).mean(1)).squeeze(-1)


class ITransformerReg(nn.Module):
    def __init__(self, cin: int, hidden: int = 32, nhead: int = 2):
        super().__init__()
        self.embed = nn.Linear(LOOKBACK, hidden)
        enc = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.tr(self.embed(x)).mean(1)).squeeze(-1)


class DLinearReg(nn.Module):
    def __init__(self, cin: int, ks: int = 25):
        super().__init__()
        self.ks = ks
        self.mix_t = nn.Linear(cin, 1)
        self.mix_s = nn.Linear(cin, 1)
        self.lin_t = nn.Linear(LOOKBACK, 1)
        self.lin_s = nn.Linear(LOOKBACK, 1)

    def forward(self, x):
        pad = self.ks // 2
        trend = nn.functional.avg_pool1d(x, self.ks, stride=1, padding=pad)[..., :LOOKBACK]
        seasonal = x - trend
        tm = self.mix_t(trend.transpose(1, 2)).squeeze(-1)
        sm = self.mix_s(seasonal.transpose(1, 2)).squeeze(-1)
        return (self.lin_t(tm) + self.lin_s(sm)).squeeze(-1)


class TimesNetReg(nn.Module):
    def __init__(self, cin: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1)).squeeze(-1)


class DeepUHIReg(nn.Module):
    def __init__(self, cin: int, static_dim: int, hidden: int = 32):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(cin, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.static = nn.Sequential(nn.Linear(static_dim, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x, s):
        zt = self.temporal(x).squeeze(-1)
        zs = self.static(s)
        return self.head(torch.cat([zt, zs], dim=1)).squeeze(-1)


MODEL_TYPES = {
    "LSTM(base,sup)": LSTMReg,
    "PatchTST(base,sup)": PatchTSTReg,
    "iTransformer(base,sup)": ITransformerReg,
    "DLinear(base,sup)": DLinearReg,
    "TimesNet(256px,sup)": TimesNetReg,
}


def zfit(X: np.ndarray):
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def zapply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    out = ((X - mu) / sd).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _subset(X, y, station, max_samples: int, seed: int):
    ok = np.isfinite(y)
    idx = np.flatnonzero(ok)
    if idx.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, max_samples, replace=False))
    return X[idx], y[idx].astype(np.float32), station[idx]


def train_torch(
    name: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    strn: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    ste: np.ndarray,
    static: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    mu, sd = zfit(Xtr)
    Xtr = zapply(Xtr, mu, sd)
    Xte = zapply(Xte, mu, sd)
    y_mu = float(np.mean(ytr))
    y_sd = float(np.std(ytr))
    if not np.isfinite(y_sd) or y_sd < 1e-6:
        y_sd = 1.0
    ytr_z = ((ytr - y_mu) / y_sd).astype(np.float32)

    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr_z).to(dev)
    Xte_t = torch.from_numpy(Xte).to(dev)
    if name == "DeepUHI(256px)":
        model = DeepUHIReg(Xtr.shape[1], static.shape[1]).to(dev)
        Str_t = torch.from_numpy(static[strn]).float().to(dev)
        Ste_t = torch.from_numpy(static[ste]).float().to(dev)
    else:
        model = MODEL_TYPES[name](Xtr.shape[1]).to(dev)
        Str_t = Ste_t = None

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            b = order[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(Xtr_t[b], Str_t[b]) if Str_t is not None else model(Xtr_t[b])
            loss = lossf(pred, ytr_t[b])
            loss.backward()
            opt.step()
    model.eval()
    out = np.full(len(yte), np.nan, np.float32)
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte), batch_size):
            xb = Xte_t[i : i + batch_size]
            if Ste_t is not None:
                pred = model(xb, Ste_t[i : i + batch_size])
            else:
                pred = model(xb)
            preds.append(pred.detach().cpu().numpy())
        out[:] = np.concatenate(preds) * y_sd + y_mu
    return out


def run_supervised(
    city: str,
    values: np.ndarray,
    times: pd.DatetimeIndex,
    methods: set[str],
    epochs: int,
    max_train_samples: int,
    max_eval_samples: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    min_valid_ratio: float,
) -> dict[str, dict]:
    train_year = int(STATION_DATA[city]["train_year"])
    test_year = int(STATION_DATA[city]["test_year"])
    seq = {
        "train": sample_sequences(values, times, train_year, test_year, "train", max_train_samples, min_valid_ratio, seed),
        "test": sample_sequences(values, times, train_year, test_year, "test", max_eval_samples, min_valid_ratio, seed + 999),
    }
    print(
        f"    sampled train={seq['train']['X'].shape[0]} test={seq['test']['X'].shape[0]}",
        flush=True,
    )
    static = station_static(city, values.shape[1])
    rows = {m: {} for m in methods}
    for h in HORIZONS:
        ytr_all = seq["train"][h]
        yte = seq["test"][h]
        Xtr, ytr, strn = _subset(seq["train"]["X"], ytr_all, seq["train"]["station"], max_train_samples, seed + h)
        Xte = seq["test"]["X"]
        ste = seq["test"]["station"]
        if len(ytr) < 50 or len(yte) == 0:
            for m in methods:
                rows[m][h] = {"MAE": None, "RMSE": None, "n": 0}
            continue
        for mi, name in enumerate(sorted(methods)):
            pred = train_torch(
                name,
                Xtr,
                ytr,
                strn,
                Xte,
                yte,
                ste,
                static,
                epochs,
                batch_size,
                lr,
                seed + h * 31 + mi,
                device,
            )
            rows[name][h] = _metrics(pred, yte)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        print(
            f"    supervised h={h}h "
            + " ".join(f"{m}={rows[m][h]['MAE']:.4f}" if rows[m][h]["MAE"] is not None else f"{m}=NA" for m in sorted(methods)),
            flush=True,
        )
    return {m: _compact(hrow) for m, hrow in rows.items()}


def add_errors(errs: dict[int, list[np.ndarray]], pred: np.ndarray, true: np.ndarray) -> None:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    for h in HORIZONS:
        k = h - 1
        ok = np.isfinite(pred[:, k]) & np.isfinite(true[:, k])
        if ok.any():
            errs[h].append(pred[ok, k] - true[ok, k])


def errs_to_row(errs: dict[int, list[np.ndarray]]) -> dict:
    hrows = {}
    for h in HORIZONS:
        if not errs[h]:
            hrows[h] = {"MAE": None, "RMSE": None, "n": 0}
            continue
        err = np.concatenate(errs[h])
        hrows[h] = {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err * err))),
            "n": int(err.size),
        }
    return _compact(hrows)


def run_chronos(values: np.ndarray, starts: np.ndarray, device: str, batch_items: int) -> dict:
    from chronos import ChronosBoltPipeline

    device_map = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device_map, local_files_only=True)
    errs = {h: [] for h in HORIZONS}
    series, true = [], []
    for wi, s in enumerate(starts):
        for st in range(values.shape[1]):
            series.append(fill_context(values[s : s + LOOKBACK, st]))
            true.append(values[s + LOOKBACK : s + LOOKBACK + FM_H, st])
        if len(series) >= batch_items or wi == len(starts) - 1:
            ctx = torch.from_numpy(np.stack(series)).float()
            with torch.no_grad():
                out = pipe.predict(ctx, prediction_length=FM_H)
            pred = out[:, out.shape[1] // 2, :].detach().cpu().numpy().astype(np.float32)
            add_errors(errs, pred, np.stack(true).astype(np.float32))
            series, true = [], []
            print(f"    chronos windows {wi + 1}/{len(starts)}", flush=True)
    return errs_to_row(errs)


def run_timesfm(values: np.ndarray, starts: np.ndarray, batch_items: int) -> dict:
    import timesfm
    from timesfm import configs

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        local_files_only=True,
        torch_compile=False,
    )
    model.compile(
        configs.ForecastConfig(
            max_context=LOOKBACK,
            max_horizon=FM_H,
            normalize_inputs=True,
            per_core_batch_size=batch_items,
        )
    )
    errs = {h: [] for h in HORIZONS}
    series, true = [], []
    for wi, s in enumerate(starts):
        for st in range(values.shape[1]):
            series.append(fill_context(values[s : s + LOOKBACK, st]))
            true.append(values[s + LOOKBACK : s + LOOKBACK + FM_H, st])
        if len(series) >= batch_items or wi == len(starts) - 1:
            pred, _ = model.forecast(horizon=FM_H, inputs=series)
            add_errors(errs, np.asarray(pred, np.float32), np.stack(true).astype(np.float32))
            series, true = [], []
            print(f"    timesfm windows {wi + 1}/{len(starts)}", flush=True)
    return errs_to_row(errs)


def run_chronos2(values: np.ndarray, times: pd.DatetimeIndex, starts: np.ndarray, city: str, config: str, device: str, window_batch: int) -> dict:
    from chronos import Chronos2Pipeline

    device_map = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device_map, local_files_only=True)
    static = station_static(city, values.shape[1])
    cov_names = [f"static_{i}" for i in range(static.shape[1])] if config == "static" else []
    errs = {h: [] for h in HORIZONS}
    for b0 in range(0, len(starts), window_batch):
        batch_starts = starts[b0 : b0 + window_batch]
        ctx_frames, fut_frames, meta = [], [], []
        for wi, s in enumerate(batch_starts):
            ctx_t = times[s : s + LOOKBACK]
            fut_t = times[s + LOOKBACK : s + LOOKBACK + FM_H]
            for st in range(values.shape[1]):
                sid = f"w{b0 + wi}_s{st}"
                ctx = pd.DataFrame({"id": sid, "timestamp": ctx_t, "target": fill_context(values[s : s + LOOKBACK, st])})
                fut = pd.DataFrame({"id": sid, "timestamp": fut_t})
                for ci, cn in enumerate(cov_names):
                    ctx[cn] = static[st, ci]
                    fut[cn] = static[st, ci]
                ctx_frames.append(ctx)
                fut_frames.append(fut)
                meta.append((sid, s, st))
        pred_df = pipe.predict_df(
            pd.concat(ctx_frames, ignore_index=True),
            future_df=pd.concat(fut_frames, ignore_index=True),
            prediction_length=FM_H,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            batch_size=256,
            context_length=LOOKBACK,
            validate_inputs=False,
        )
        pred_col = "0.5" if "0.5" in pred_df.columns else "mean"
        pred_map = {sid: g.sort_values("timestamp")[pred_col].to_numpy(np.float32) for sid, g in pred_df.groupby("id", sort=False)}
        preds, true = [], []
        for sid, s, st in meta:
            yhat = pred_map.get(sid)
            if yhat is None or len(yhat) < FM_H:
                continue
            preds.append(yhat[:FM_H])
            true.append(values[s + LOOKBACK : s + LOOKBACK + FM_H, st])
        if preds:
            add_errors(errs, np.stack(preds), np.stack(true).astype(np.float32))
        print(f"    chronos2-{config} windows {min(b0 + window_batch, len(starts))}/{len(starts)}", flush=True)
    return errs_to_row(errs)


def run_fm(
    city: str,
    values: np.ndarray,
    times: pd.DatetimeIndex,
    methods: set[str],
    device: str,
    batch_items: int,
    window_batch: int,
    max_windows: int,
) -> tuple[dict[str, dict], dict[str, str]]:
    test_year = int(STATION_DATA[city]["test_year"])
    starts = weekly_starts(times, test_year)
    if starts.size > max_windows:
        starts = starts[np.linspace(0, starts.size - 1, max_windows).astype(np.int64)]
    rows: dict[str, dict] = {}
    status: dict[str, str] = {}
    if starts.size == 0:
        return rows, {m: "no weekly evaluation starts" for m in methods}
    if "Chronos(64px)" in methods:
        try:
            rows["Chronos(64px)"] = run_chronos(values, starts, device, batch_items)
            status["Chronos(64px)"] = "ok"
        except Exception as exc:
            status["Chronos(64px)"] = f"failed: {type(exc).__name__}: {exc}"
    if "TimesFM(64px)" in methods:
        try:
            rows["TimesFM(64px)"] = run_timesfm(values, starts, batch_items)
            status["TimesFM(64px)"] = "ok"
        except Exception as exc:
            status["TimesFM(64px)"] = f"failed: {type(exc).__name__}: {exc}"
    for key, cfg in [("Chronos-2(base,64px)", "base"), ("Chronos-2(+static,64px)", "static")]:
        if key in methods:
            try:
                rows[key] = run_chronos2(values, times, starts, city, cfg, device, window_batch)
                status[key] = "ok"
            except Exception as exc:
                status[key] = f"failed: {type(exc).__name__}: {exc}"
    for unavailable in ["Chronos-2(+meteo,64px)", "Chronos-2(+meteo+static,64px)", "MOIRAI-2(+meteo,64px)"]:
        if unavailable in methods:
            status[unavailable] = "not run: no Rome/Temuco ERA5/meteo covariate tensor in station supplement"
    for unavailable in ["MOIRAI-2(base,64px)", "MOIRAI-2(+static,64px)", "MOIRAI-2(+meteo+static,64px)"]:
        if unavailable in methods:
            status[unavailable] = "not run: uni2ts package is not installed in the active Python environment"
    return rows, status


def export_csv(out_path: Path, csv_path: Path) -> None:
    data = _load_json(out_path)
    rows = []
    wanted = BASE_KEYS + SUP_KEYS + FM_KEYS + [
        "Chronos-2(+meteo,64px)",
        "Chronos-2(+meteo+static,64px)",
        "MOIRAI-2(+meteo,64px)",
        "MOIRAI-2(base,64px)",
        "MOIRAI-2(+static,64px)",
        "MOIRAI-2(+meteo+static,64px)",
    ]
    for city, obj in data.get("Ta_station", {}).items():
        methods = obj.get("methods", {})
        status = obj.get("fm_status", {})
        for name in wanted:
            row = methods.get(name)
            rec = {"city": city, "method": name, "status": status.get(name, "ok" if isinstance(row, dict) else "missing")}
            if isinstance(row, dict):
                for metric in ["MAE", "RMSE", "n"]:
                    vals = row.get(metric, {})
                    for h in HORIZONS:
                        rec[f"{metric}_{h}h"] = vals.get(f"{h}h")
            rows.append(rec)
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["rome", "temuco"], choices=sorted(STATION_DATA))
    ap.add_argument("--out", default=str(HERE / "results" / "1d_station_air_forecast.json"))
    ap.add_argument("--csv", default=str(HERE / "results" / "2c_rome_temuco_station_missing_baselines.csv"))
    ap.add_argument("--methods", nargs="+", default=["base", "sarima", "supervised", "fm"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-train-samples", type=int, default=30000)
    ap.add_argument("--max-eval-samples", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-valid-ratio", type=float, default=0.70)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fm-batch-items", type=int, default=256)
    ap.add_argument("--fm-window-batch", type=int, default=2)
    ap.add_argument("--fm-max-windows", type=int, default=16)
    args = ap.parse_args()

    out_path = Path(args.out)
    selected = set(args.methods)
    selected_methods = {
        m for m in args.methods if m not in {"base", "sarima", "supervised", "fm", "all"}
    }
    if "all" in selected:
        selected.update(["base", "sarima", "supervised", "fm"])

    for city in args.cities:
        print(f"[2c/station completion] {city}", flush=True)
        values, times, meta = load_station_matrix(city)
        existing = _method_rows(_load_json(out_path), city)
        rows: dict[str, dict] = {}

        if "base" in selected:
            base_rows = add_base_aliases(out_path, city, values, times)
            rows.update({k: v for k, v in base_rows.items() if not args.skip_existing or not _complete(existing.get(k))})

        if "sarima" in selected or "SARIMA(city-mean)" in selected_methods:
            if not args.skip_existing or not _complete(existing.get("SARIMA(city-mean)")):
                print("  SARIMA(city-mean)", flush=True)
                rows["SARIMA(city-mean)"] = city_mean_sarima(
                    values,
                    times,
                    int(STATION_DATA[city]["train_year"]),
                    int(STATION_DATA[city]["test_year"]),
                )

        sup = set(SUP_KEYS[2:])
        sup.update(m for m in selected_methods if m in set(SUP_KEYS[2:]))
        if "supervised" not in selected:
            sup = {m for m in sup if m in selected_methods}
        sup = {m for m in sup if not args.skip_existing or not _complete(existing.get(m))}
        if sup:
            print("  supervised " + ", ".join(sorted(sup)), flush=True)
            rows.update(
                run_supervised(
                    city,
                    values,
                    times,
                    sup,
                    args.epochs,
                    args.max_train_samples,
                    args.max_eval_samples,
                    args.batch_size,
                    args.lr,
                    args.seed,
                    args.device,
                    args.min_valid_ratio,
                )
            )

        fm_methods = set(FM_KEYS)
        fm_methods.update([
            "Chronos-2(+meteo,64px)",
            "Chronos-2(+meteo+static,64px)",
            "MOIRAI-2(base,64px)",
            "MOIRAI-2(+meteo,64px)",
            "MOIRAI-2(+static,64px)",
            "MOIRAI-2(+meteo+static,64px)",
        ])
        if "fm" not in selected:
            fm_methods = {m for m in fm_methods if m in selected_methods}
        fm_methods = {m for m in fm_methods if not args.skip_existing or not _complete(existing.get(m))}
        fm_status = {}
        if fm_methods:
            print("  FM " + ", ".join(sorted(fm_methods)), flush=True)
            fm_rows, fm_status = run_fm(
                city,
                values,
                times,
                fm_methods,
                args.device,
                args.fm_batch_items,
                args.fm_window_batch,
                args.fm_max_windows,
            )
            rows.update(fm_rows)

        if rows or fm_status:
            data = _load_json(out_path)
            data.setdefault("Ta_station", {}).setdefault(city, {}).setdefault("meta", {}).update(meta)
            if fm_status:
                data["Ta_station"][city].setdefault("fm_status", {}).update(fm_status)
            _save_json(out_path, data)
            if rows:
                _merge_methods(out_path, city, rows)
            for name, row in rows.items():
                mae = row["MAE"]
                print("  " + name + " " + " ".join(f"{h}h={mae[f'{h}h']:.4f}" if mae[f"{h}h"] is not None else f"{h}h=NA" for h in HORIZONS), flush=True)
        else:
            print("  no missing rows for selected methods", flush=True)

    export_csv(out_path, Path(args.csv))
    print(f"[saved] {out_path}", flush=True)
    print(f"[csv] {args.csv}", flush=True)


if __name__ == "__main__":
    main()
