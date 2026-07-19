"""Task 2c DeepUHI-style graph-temporal forecasting baseline.

This runner fills the missing multi-city DeepUHI row with a reproducible
protocol matching the sampled supervised Task 2c reruns:
  - train 2015-2022, eval 2023-2025
  - 168h history -> horizons 1/6/12/24/48/96h
  - sampled pixels and sampled valid target rows per city-year
  - target-normalized supervised training

The original Munich-only DeepUHI metric was a standalone 256px result without a
checked-in runner. This file therefore adds a consistent DeepUHI-style baseline
instead of copying that row to other cities. The graph part is a kNN pixel graph
used to aggregate neighboring UHI histories; the temporal part is a compact
Conv-GRU forecaster.
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
from common.data import _load_static  # noqa: E402
import run_3_ood_transfer as T3  # noqa: E402


LOOKBACK = T3.LOOKBACK
HORIZONS = [1, 6, 12, 24, 48, 96]
TARGET_KIND = {"Ta": "airt", "LST": "lst"}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
METHOD_KEY = "DeepUHI(128px supervised)"


class DeepUHIReg(nn.Module):
    def __init__(self, cin: int, hidden: int = 64):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(cin, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):  # [B, C, T]
        z = self.temporal(x).transpose(1, 2)
        _, h = self.gru(z)
        return self.head(h[-1]).squeeze(-1)


def city_key(city: str) -> str:
    return city.replace("_", " ").title().replace(" ", "_")


def pixel_coords(city: str, pixel_ids: np.ndarray) -> np.ndarray:
    xy_m, _feats, _names, pids = _load_static(city, n_static=10)
    pixel_ids = np.asarray(pixel_ids, dtype=np.int64)
    if pixel_ids.max(initial=0) < len(xy_m):
        coords = xy_m[pixel_ids]
    else:
        row = {int(pid): i for i, pid in enumerate(pids)}
        coords = xy_m[np.array([row[int(pid)] for pid in pixel_ids], dtype=np.int64)]
    return np.asarray(coords, dtype=np.float32)


def build_knn_weights(city: str, pixel_ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    coords = pixel_coords(city, pixel_ids)
    n = len(coords)
    kk = min(k, max(1, n - 1))
    dist2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dist2, np.inf)
    idx = np.argpartition(dist2, kk, axis=1)[:, :kk]
    local_d = np.take_along_axis(dist2, idx, axis=1)
    scale = np.nanmedian(local_d[np.isfinite(local_d)])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    w = np.exp(-local_d / scale).astype(np.float32)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    return idx.astype(np.int64), w.astype(np.float32)


def weighted_neighbor_history(values: np.ndarray, neigh_idx: np.ndarray, weights: np.ndarray, pp: int):
    vals = values[:, neigh_idx[pp]].astype(np.float32)
    finite = np.isfinite(vals)
    w = weights[pp][None, :]
    denom = (finite * w).sum(axis=1)
    numer = np.where(finite, vals, 0.0) * w
    agg = numer.sum(axis=1) / np.maximum(denom, 1e-6)
    valid_frac = denom / np.maximum(w.sum(), 1e-6)
    return agg.astype(np.float32), valid_frac.astype(np.float32)


def build_sequences(data, horizon: int, max_samples: int, min_valid_ratio: float, seed: int, k: int):
    t_idx, p_idx, _vr = T3.candidate_rows(
        data.values, horizon, min_valid_ratio, max_samples, seed
    )
    n = len(t_idx)
    if n == 0:
        return None
    neigh_idx, weights = build_knn_weights(data.city, data.pixel_ids, k)

    x = np.empty((n, 5, LOOKBACK), np.float32)
    y = np.empty(n, np.float32)
    for i, (tt, pp) in enumerate(zip(t_idx, p_idx)):
        input_end = int(tt - horizon)
        start = input_end - LOOKBACK + 1
        seg = np.asarray(data.values[start : input_end + 1, pp], dtype=np.float32)
        finite = np.isfinite(seg)
        fill_value = float(np.nanmean(seg)) if finite.any() else 0.0
        self_hist = np.where(finite, seg, fill_value).astype(np.float32)
        self_mask = finite.astype(np.float32)

        neigh_hist, neigh_mask = weighted_neighbor_history(
            data.values[start : input_end + 1], neigh_idx, weights, int(pp)
        )
        if not np.isfinite(neigh_hist).any():
            neigh_hist = np.full(LOOKBACK, fill_value, np.float32)
        else:
            nf = np.isfinite(neigh_hist)
            nfill = float(np.nanmean(neigh_hist[nf])) if nf.any() else fill_value
            neigh_hist = np.where(nf, neigh_hist, nfill).astype(np.float32)

        x[i, 0] = self_hist
        x[i, 1] = self_mask
        x[i, 2] = neigh_hist
        x[i, 3] = neigh_mask
        x[i, 4] = self_hist - neigh_hist
        y[i] = data.values[tt, pp]
    return x, y


def pool_sequences(data_list, horizon: int, max_samples: int, min_valid_ratio: float, seed: int, k: int):
    xs, ys = [], []
    for i, data in enumerate(data_list):
        built = build_sequences(data, horizon, max_samples, min_valid_ratio, seed + i, k)
        if built is None:
            continue
        x, y = built
        xs.append(x)
        ys.append(y)
    if not ys:
        return None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def zfit(X: np.ndarray):
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
    return np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def train_model(model, Xtr, ytr, Xev, yev, epochs: int, batch_size: int, lr: float):
    y_mu = float(np.mean(ytr))
    y_sd = float(np.std(ytr))
    if not np.isfinite(y_sd) or y_sd < 1e-6:
        y_sd = 1.0
    ytr_z = ((ytr - y_mu) / y_sd).astype(np.float32)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr_z).to(DEVICE)
    Xev_t = torch.from_numpy(Xev).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xev_t).cpu().numpy() * y_sd + y_mu
    return float(np.mean(np.abs(pred - yev)))


def merge_results(out_path: Path, target: str, city: str, row: dict[str, float]):
    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    data.setdefault(target, {}).setdefault(city_key(city), {})
    data[target][city_key(city)][METHOD_KEY] = row
    out_path.write_text(json.dumps(data, indent=2))


def run_city_target(args, city: str, target: str):
    kind = TARGET_KIND[target]
    print(f"\n{'=' * 64}\n[Task2c DeepUHI] target={target} city={city} device={DEVICE}", flush=True)
    train_data = [
        T3.load_city_year(city, y, args.n_pixels, args.seed, target_kind=kind)
        for y in args.train_years
    ]
    eval_data = [
        T3.load_city_year(city, y, args.n_pixels, args.seed, target_kind=kind)
        for y in args.eval_years
    ]

    row: dict[str, float] = {}
    for horizon in HORIZONS:
        train = pool_sequences(train_data, horizon, args.max_samples, args.min_valid_ratio, args.seed, args.knn)
        eval_ = pool_sequences(eval_data, horizon, args.max_samples, args.min_valid_ratio, args.seed + 999, args.knn)
        hkey = f"{horizon}h"
        if train is None or eval_ is None:
            print(f"  h={hkey} empty", flush=True)
            continue
        Xtr_raw, ytr = train
        Xev_raw, yev = eval_
        mu, sd = zfit(Xtr_raw)
        Xtr = zapply(Xtr_raw, mu, sd)
        Xev = zapply(Xev_raw, mu, sd)
        torch.manual_seed(args.seed)
        model = DeepUHIReg(Xtr.shape[1], hidden=args.hidden).to(DEVICE)
        mae = train_model(model, Xtr, ytr, Xev, yev, args.epochs, args.batch_size, args.lr)
        row[hkey] = mae
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  h={hkey} train={len(ytr)} eval={len(yev)} DeepUHI={mae:.3f}", flush=True)

    merge_results(Path(args.out), target, city, row)
    print(f"[merged] {target}/{city} -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["munich", "berlin", "cairo", "lagos", "johannesburg"])
    ap.add_argument("--targets", nargs="+", choices=["Ta", "LST"], default=["Ta", "LST"])
    ap.add_argument("--train_years", type=int, nargs="+", default=list(range(2015, 2023)))
    ap.add_argument("--eval_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_pixels", type=int, default=128)
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--min_valid_ratio", type=float, default=0.70)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "1d_forecast.json"))
    args = ap.parse_args()

    for target in args.targets:
        for city in args.cities:
            run_city_target(args, city, target)


if __name__ == "__main__":
    main()
