"""Task 2c requested supervised sequence baselines.

Runs only the compact set requested for Table 21 completion:
  - Persistence
  - LSTM on base, +ERA5, +static, +ERA5+static inputs
  - XGBoost on base, +ERA5, +static, +ERA5+static engineered inputs
  - DLinear on base, +ERA5, +static, +ERA5+static inputs

Targets can be Ta (Air-T UHI, HOSTRADA or corrected pseudo Air-T) and/or LST.
Results are merged into 2c/results/1d_forecast.json using the same method names
that draft/generate_fm_result_tables.py reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3"))
import run_3_ood_transfer as T3  # noqa: E402


LOOKBACK = T3.LOOKBACK
HORIZONS = [1, 6, 12, 24, 48, 96]
CONFIGS = ["L1", "+ERA5", "+static", "+ERA5+static"]
TARGET_KIND = {"Ta": "airt", "LST": "lst"}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
XGB_CONFIG_TO_LAYER = {
    "L1": "L1",
    "+ERA5": "L2",
    "+static": "L1S",
    "+ERA5+static": "L3",
}


class LSTMReg(nn.Module):
    def __init__(self, cin: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(cin, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # [B, C, T]
        z = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(z)
        return self.head(h_n[-1]).squeeze(-1)


class DLinearReg(nn.Module):
    def __init__(self, cin: int, ks: int = 25):
        super().__init__()
        self.ks = ks
        self.mix_t = nn.Linear(cin, 1)
        self.mix_s = nn.Linear(cin, 1)
        self.lin_t = nn.Linear(LOOKBACK, 1)
        self.lin_s = nn.Linear(LOOKBACK, 1)

    def forward(self, x):  # [B, C, T]
        pad = self.ks // 2
        trend = nn.functional.avg_pool1d(x, self.ks, stride=1, padding=pad)[..., :LOOKBACK]
        seasonal = x - trend
        tm = self.mix_t(trend.transpose(1, 2)).squeeze(-1)
        sm = self.mix_s(seasonal.transpose(1, 2)).squeeze(-1)
        return (self.lin_t(tm) + self.lin_s(sm)).squeeze(-1)


def _last_finite_or_mean(seg: np.ndarray) -> float:
    finite = np.isfinite(seg)
    if not finite.any():
        return 0.0
    vals = seg[finite]
    return float(vals[-1])


def build_sequences(data, horizon: int, max_samples: int, min_valid_ratio: float, seed: int):
    t_idx, p_idx, _valid_ratio = T3.candidate_rows(
        data.values, horizon, min_valid_ratio, max_samples, seed
    )
    n = len(t_idx)
    if n == 0:
        return None

    hist = np.empty((n, LOOKBACK), np.float32)
    mask = np.empty_like(hist)
    era = np.empty((n, LOOKBACK, len(T3.DRIVERS)), np.float32)
    static = np.empty((n, data.static.shape[1]), np.float32)
    y = np.empty(n, np.float32)
    persistence = np.empty(n, np.float32)

    for i, (tt, pp) in enumerate(zip(t_idx, p_idx)):
        input_end = int(tt - horizon)
        start = input_end - LOOKBACK + 1
        seg = np.asarray(data.values[start : input_end + 1, pp], dtype=np.float32)
        finite = np.isfinite(seg)
        fill_value = float(np.nanmean(seg)) if finite.any() else 0.0
        hist[i] = np.where(finite, seg, fill_value)
        mask[i] = finite.astype(np.float32)
        era[i] = data.era5[start : input_end + 1, pp, :]
        static[i] = data.static[pp]
        y[i] = data.values[tt, pp]
        persistence[i] = _last_finite_or_mean(seg)

    base = np.concatenate([hist[:, None, :], mask[:, None, :]], axis=1)
    era_t = np.transpose(era, (0, 2, 1))
    static_t = np.broadcast_to(static[:, :, None], (n, static.shape[1], LOOKBACK))
    cfgs = {
        "L1": base,
        "+ERA5": np.concatenate([base, era_t], axis=1),
        "+static": np.concatenate([base, static_t], axis=1),
        "+ERA5+static": np.concatenate([base, era_t, static_t], axis=1),
    }
    target_times = data.times[t_idx].to_numpy()
    return {k: v.astype(np.float32) for k, v in cfgs.items()}, y, persistence, target_times


def zfit(X: np.ndarray):
    if np.isfinite(X).all():
        mu = X.mean(axis=(0, 2), keepdims=True)
        sd = X.std(axis=(0, 2), keepdims=True)
        sd = np.where(sd > 1e-6, sd, 1.0)
        return mu.astype(np.float32), sd.astype(np.float32)
    finite = np.isfinite(X)
    counts = finite.sum(axis=(0, 2), keepdims=True)
    filled = np.where(finite, X, 0.0)
    mu = filled.sum(axis=(0, 2), keepdims=True) / np.maximum(counts, 1)
    centered = np.where(finite, X - mu, 0.0)
    sd = np.sqrt((centered * centered).sum(axis=(0, 2), keepdims=True) / np.maximum(counts, 1))
    mu = np.where(counts > 0, mu, 0.0)
    sd = np.where((counts > 1) & (sd > 1e-6), sd, 1.0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def zapply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    out = np.empty_like(X, dtype=np.float32)
    np.subtract(X, mu, out=out, casting="unsafe")
    np.divide(out, sd, out=out, casting="unsafe")
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def train_model(model, Xtr, ytr, Xev, yev, epochs: int, batch_size: int, lr: float):
    y_mu = float(np.mean(ytr))
    y_sd = float(np.std(ytr))
    if not np.isfinite(y_sd) or y_sd < 1e-6:
        y_sd = 1.0
    ytr_z = ((ytr - y_mu) / y_sd).astype(np.float32)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr_z).to(DEVICE)
    Xev_t = torch.from_numpy(Xev).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xev_t).cpu().numpy() * y_sd + y_mu
    return float(np.mean(np.abs(pred - yev)))


def pool_sequences(data_list, horizon: int, max_samples: int, min_valid_ratio: float, seed: int):
    X_parts = {cfg: [] for cfg in CONFIGS}
    y_parts = []
    p_parts = []
    time_parts = []
    for i, data in enumerate(data_list):
        built = build_sequences(data, horizon, max_samples, min_valid_ratio, seed + i)
        if built is None:
            continue
        cfgs, y, pers, target_times = built
        for cfg in CONFIGS:
            X_parts[cfg].append(cfgs[cfg])
        y_parts.append(y)
        p_parts.append(pers)
        time_parts.append(target_times)
    if not y_parts:
        return None
    return (
        {cfg: np.concatenate(X_parts[cfg], axis=0) for cfg in CONFIGS},
        np.concatenate(y_parts, axis=0),
        np.concatenate(p_parts, axis=0),
        np.concatenate(time_parts, axis=0),
    )


def method_name(target: str, model: str, cfg: str) -> str:
    if model in {"LSTM", "DLinear", "XGBoost"}:
        return f"{model}({cfg})(sup)"
    raise ValueError(f"unknown model={model!r}")


def has_complete_row(out_path: Path, target: str, city: str, key: str) -> bool:
    if not out_path.exists():
        return False
    data = json.loads(out_path.read_text())
    city_key = city.replace("_", " ").title().replace(" ", "_")
    row = data.get(target, {}).get(city_key, {}).get(key)
    if not isinstance(row, dict):
        return False
    return all(isinstance(row.get(f"{h}h"), (int, float)) and np.isfinite(row[f"{h}h"]) for h in HORIZONS)


def build_xgb_configs(batch) -> dict[str, np.ndarray]:
    x_l1 = T3.layer_matrix(batch, "L1")
    x_l1s = T3.layer_matrix(batch, "L1S")
    x_l2 = T3.layer_matrix(batch, "L2")
    x_l3 = T3.layer_matrix(batch, "L3")
    return {
        "L1": x_l1,
        "+ERA5": x_l2,
        "+static": x_l1s,
        "+ERA5+static": x_l3,
    }


def pool_xgb_samples(data_list, horizon: int, max_samples: int, min_valid_ratio: float, seed: int):
    parts = []
    for i, data in enumerate(data_list):
        parts.append(
            T3.build_samples(
                data,
                horizon,
                max_samples,
                min_valid_ratio,
                seed + i,
            )
        )
    batch = T3.concat_batches(parts)
    if len(batch.y) == 0:
        return None
    return build_xgb_configs(batch), batch.y, batch.persistence, batch.target_times


def train_xgb_mae(Xtr, ytr, Xev, yev, seed: int, n_estimators: int, max_depth: int) -> float:
    mu, sd = T3.fit_feature_scaler(Xtr)
    Xtr_z = T3.apply_feature_scaler(Xtr, mu, sd)
    Xev_z = T3.apply_feature_scaler(Xev, mu, sd)
    model = T3.train_xgb(
        Xtr_z,
        ytr,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    pred = model.predict(Xev_z)
    return float(np.mean(np.abs(pred - yev)))
    if model == "LSTM":
        return "LSTM(L1)(sup)"
    return f"DLinear({cfg})(sup)"


def merge_results(out_path: Path, target: str, city: str, rows: dict[str, dict[str, float]]):
    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    city_key = city.replace("_", " ").title().replace(" ", "_")
    if city_key == "Johannesburg":
        pass
    data.setdefault(target, {}).setdefault(city_key, {})
    if target == "Ta":
        # Preserve older full/supervised rows when they already exist, but create
        # compatible fallback keys for cities that only have the requested rerun.
        if "DLinear(L1)(sup)" in rows and "dlinear(supervised)" not in data[target][city_key]:
            data[target][city_key]["dlinear(supervised)"] = rows["DLinear(L1)(sup)"]
        if "LSTM(L1)(sup)" in rows and "lstm(supervised)" not in data[target][city_key]:
            data[target][city_key]["lstm(supervised)"] = rows["LSTM(L1)(sup)"]
    for key, row in rows.items():
        old_row = data[target][city_key].get(key)
        old_has_value = isinstance(old_row, dict) and any(
            isinstance(v, (int, float)) and np.isfinite(v) for v in old_row.values()
        )
        if key in {"Persistence", "Persistence(sup)"} and old_has_value:
            continue
        data[target][city_key][key] = row
    out_path.write_text(json.dumps(data, indent=2))


def run_city_target(args, city: str, target: str):
    kind = TARGET_KIND[target]
    print(f"\n{'=' * 64}\n[Task2c requested] target={target} city={city} device={DEVICE}", flush=True)
    train_data = []
    for y in args.train_years:
        print(f"  load train {city} {y} target={target} n_pixels={args.n_pixels}", flush=True)
        train_data.append(T3.load_city_year(city, y, args.n_pixels, args.seed, target_kind=kind))
    eval_data = []
    for y in args.eval_years:
        print(f"  load eval  {city} {y} target={target} n_pixels={args.n_pixels}", flush=True)
        eval_data.append(T3.load_city_year(city, y, args.n_pixels, args.seed, target_kind=kind))

    out_path = Path(args.out)
    rows: dict[str, dict[str, float]] = {}
    if args.run_persistence:
        p_key = "Persistence" if target == "Ta" else "Persistence(sup)"
        if not args.skip_existing or not has_complete_row(out_path, target, city, p_key):
            rows[p_key] = {}
    if args.run_stat_sup:
        for key in ["Persistence(sup)", "Climatology(sup)"]:
            if not args.skip_existing or not has_complete_row(out_path, target, city, key):
                rows[key] = {}
    if args.run_lstm:
        for cfg in args.lstm_configs:
            key = method_name(target, "LSTM", cfg)
            if not args.skip_existing or not has_complete_row(out_path, target, city, key):
                rows[key] = {}
    if args.run_xgb:
        for cfg in args.xgb_configs:
            key = method_name(target, "XGBoost", cfg)
            if not args.skip_existing or not has_complete_row(out_path, target, city, key):
                rows[key] = {}
    if args.run_dlinear:
        for cfg in args.dlinear_configs:
            rows[method_name(target, "DLinear", cfg)] = {}
    if not rows:
        print("  all requested rows already complete; skipping", flush=True)
        return

    for horizon in HORIZONS:
        need_stat = any(key.startswith("Persistence") or key.startswith("Climatology") for key in rows)
        need_sequence = any(key.startswith("LSTM(") or key.startswith("DLinear(") for key in rows)
        need_xgb = any(key.startswith("XGBoost(") for key in rows)
        train = eval_ = None
        if need_sequence:
            train = pool_sequences(train_data, horizon, args.max_samples, args.min_valid_ratio, args.seed)
            eval_ = pool_sequences(eval_data, horizon, args.max_samples, args.min_valid_ratio, args.seed + 999)
        train_xgb = eval_xgb = None
        if need_xgb or (need_stat and not need_sequence):
            train_xgb = pool_xgb_samples(train_data, horizon, args.max_samples, args.min_valid_ratio, args.seed)
            eval_xgb = pool_xgb_samples(eval_data, horizon, args.max_samples, args.min_valid_ratio, args.seed + 999)
        if (need_sequence and (train is None or eval_ is None)) or ((need_xgb or (need_stat and not need_sequence)) and (train_xgb is None or eval_xgb is None)):
            print(f"  h={horizon}h empty", flush=True)
            continue
        hkey = f"{horizon}h"
        if need_stat:
            if need_sequence:
                _Xtr_cfg, ytr_stat, _ptr, tr_times_stat = train
                _Xev_cfg, yev_stat, pev_stat, ev_times_stat = eval_
            else:
                _Xtr_xgb_cfg, ytr_stat, _ptr, tr_times_stat = train_xgb
                _Xev_xgb_cfg, yev_stat, pev_stat, ev_times_stat = eval_xgb
            p_key = "Persistence" if target == "Ta" else "Persistence(sup)"
            if p_key in rows:
                rows[p_key][hkey] = float(np.mean(np.abs(pev_stat - yev_stat)))
            if "Persistence(sup)" in rows:
                rows["Persistence(sup)"][hkey] = float(np.mean(np.abs(pev_stat - yev_stat)))
            if "Climatology(sup)" in rows:
                clim = T3.fit_source_climatology(tr_times_stat, ytr_stat)
                pred = T3.predict_source_climatology(clim, ev_times_stat)
                rows["Climatology(sup)"][hkey] = float(np.mean(np.abs(pred - yev_stat)))

        if need_sequence:
            Xtr_cfg, ytr, _ptr, tr_times = train
            Xev_cfg, yev, pev, ev_times = eval_

        if args.run_lstm:
            for cfg in args.lstm_configs:
                key = method_name(target, "LSTM", cfg)
                if key not in rows:
                    continue
                mu, sd = zfit(Xtr_cfg[cfg])
                Xtr = zapply(Xtr_cfg[cfg], mu, sd)
                Xev = zapply(Xev_cfg[cfg], mu, sd)
                torch.manual_seed(args.seed)
                model = LSTMReg(Xtr.shape[1]).to(DEVICE)
                rows[key][hkey] = train_model(
                    model, Xtr, ytr, Xev, yev, args.lstm_epochs, args.batch_size, args.lr
                )
                del model
                torch.cuda.empty_cache()

        if args.run_dlinear:
            for cfg in args.dlinear_configs:
                key = method_name(target, "DLinear", cfg)
                if key not in rows:
                    continue
                mu, sd = zfit(Xtr_cfg[cfg])
                Xtr = zapply(Xtr_cfg[cfg], mu, sd)
                Xev = zapply(Xev_cfg[cfg], mu, sd)
                torch.manual_seed(args.seed)
                model = DLinearReg(Xtr.shape[1]).to(DEVICE)
                rows[key][hkey] = train_model(
                    model, Xtr, ytr, Xev, yev, args.dlinear_epochs, args.batch_size, args.lr
                )
                del model
                torch.cuda.empty_cache()

        if args.run_xgb:
            Xtr_xgb_cfg, ytr_xgb, _ptr_xgb, _tr_times_xgb = train_xgb
            Xev_xgb_cfg, yev_xgb, _pev_xgb, _ev_times_xgb = eval_xgb
            for cfg in args.xgb_configs:
                key = method_name(target, "XGBoost", cfg)
                if key not in rows:
                    continue
                rows[key][hkey] = train_xgb_mae(
                    Xtr_xgb_cfg[cfg],
                    ytr_xgb,
                    Xev_xgb_cfg[cfg],
                    yev_xgb,
                    seed=args.seed + horizon + CONFIGS.index(cfg),
                    n_estimators=args.xgb_estimators,
                    max_depth=args.xgb_max_depth,
                )

        summary = "  ".join(
            f"{k}={v[hkey]:.3f}" for k, v in rows.items() if hkey in v
        )
        n_train = len(train[1]) if train is not None else len(train_xgb[1])
        n_eval = len(eval_[1]) if eval_ is not None else len(eval_xgb[1])
        print(f"  h={horizon}h train={n_train} eval={n_eval}  {summary}", flush=True)

    merge_results(out_path, target, city, rows)
    print(f"[merged] {target}/{city} -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["munich", "berlin", "cairo", "lagos", "johannesburg"])
    ap.add_argument("--targets", nargs="+", choices=["Ta", "LST"], default=["Ta", "LST"])
    ap.add_argument("--train_years", type=int, nargs="+", default=list(range(2015, 2023)))
    ap.add_argument("--eval_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_pixels", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=2500)
    ap.add_argument("--min_valid_ratio", type=float, default=0.70)
    ap.add_argument("--dlinear_epochs", type=int, default=10)
    ap.add_argument("--lstm_epochs", type=int, default=8)
    ap.add_argument("--xgb_estimators", type=int, default=250)
    ap.add_argument("--xgb_max_depth", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "1d_forecast.json"))
    ap.add_argument("--lstm_configs", nargs="+", choices=CONFIGS, default=["L1"])
    ap.add_argument("--dlinear_configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    ap.add_argument("--xgb_configs", nargs="+", choices=CONFIGS, default=["+ERA5", "+static", "+ERA5+static"])
    ap.add_argument("--no_persistence", dest="run_persistence", action="store_false")
    ap.add_argument("--run_stat_sup", action="store_true")
    ap.add_argument("--run_xgb", action="store_true")
    ap.add_argument("--no_lstm", dest="run_lstm", action="store_false")
    ap.add_argument("--no_dlinear", dest="run_dlinear", action="store_false")
    ap.add_argument("--skip_existing", action="store_true")
    ap.set_defaults(run_lstm=True, run_dlinear=True, run_persistence=True)
    args = ap.parse_args()

    for target in args.targets:
        for city in args.cities:
            run_city_target(args, city, target)


if __name__ == "__main__":
    main()
