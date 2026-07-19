"""Task 3 — Foundation Model zero-shot OOD transfer (Chronos / TimesFM / MOIRAI).

FM baselines do NOT train on source cities — they directly forecast each OOD
target city's LST-UHI from its own history. Results are therefore source-
independent (same for Cfb-4/Diverse-4/etc.) and serve as a horizontal benchmark
line in the source-diversity plot.

Mirrors run_3_ood_transfer.py's sampling protocol (128px, min_valid_ratio=0.70,
lookback 168h, horizons {1,6,24}h) for fair comparison with XGBoost/Persistence.

Usage:
  python benchmark/3/run_3_fm.py --fm chronos --device cuda:0
  python benchmark/3/run_3_fm.py --fm timesfm --device cuda:0
  python benchmark/3/run_3_fm.py --fm moirai --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common.data import _load_static
from common.paths import ERA5_BASE, LST_BASE

DRIVERS = ["u10", "v10", "tcc", "d2m", "blh", "ssrd"]
LOOKBACK = 168
OOD_CITIES = ["hamburg", "warsaw", "buenos_aires", "casablanca", "tehran", "khartoum"]


def stable_seed(seed, city, year=0, extra=0):
    return int(hashlib.md5(f"{seed}:{city}:{year}:{extra}".encode()).hexdigest()[:8], 16)


def choose_pixels(city, n_pixels, seed):
    _, _, _, pids = _load_static(city, n_static=10)
    pids = np.asarray(pids, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(seed, city))
    return np.sort(rng.choice(pids, min(n_pixels, len(pids)), replace=False))


def _read_pixel_parquet(path, columns, pixel_ids):
    filters = [("pixel_id", "in", [int(p) for p in pixel_ids])]
    return pd.read_parquet(path, columns=columns, filters=filters)


def load_pixel_series(city, year, pixel_ids):
    """Load [T, N] LST-UHI for selected pixels, one year."""
    path = LST_BASE / city / f"lst_uhi_1km_hourly_{year}.parquet"
    df = _read_pixel_parquet(path, ["datetime", "pixel_id", "lst_uhi_K"], pixel_ids)
    df["datetime"] = pd.to_datetime(df["datetime"])
    piv = df.pivot_table(index="datetime", columns="pixel_id", values="lst_uhi_K",
                         aggfunc="first").sort_index()
    piv = piv.reindex(columns=pixel_ids)
    times = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="h")[:-1]
    return piv.reindex(times).to_numpy(dtype=np.float32), times


def load_era5_pixel(city, year, pixel_ids):
    path = ERA5_BASE / city / f"era5_hourly_{year}.parquet"
    df = _read_pixel_parquet(path, ["datetime", "pixel_id"] + DRIVERS, pixel_ids)
    df["datetime"] = pd.to_datetime(df["datetime"])
    times = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="h")[:-1]
    arrs = []
    for d in DRIVERS:
        piv = df.pivot_table(index="datetime", columns="pixel_id", values=d,
                             aggfunc="first").reindex(index=times, columns=pixel_ids)
        arrs.append(piv.to_numpy(dtype=np.float32))
    return np.stack(arrs, axis=-1)  # [T, N, 6]


def candidate_windows(values, horizon, min_valid_ratio, max_samples, seed):
    """Return (target_t, pixel, valid_ratio) for qualifying samples."""
    t_min = LOOKBACK - 1 + horizon
    if t_min >= values.shape[0]:
        return np.array([]), np.array([]), np.array([])
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
            rows_p.append(np.full(int(ok.sum()), p))
            ratios.append((counts[ok] / float(LOOKBACK)).astype(np.float32))
    if not rows_t:
        return np.array([]), np.array([]), np.array([])
    t_idx = np.concatenate(rows_t); p_idx = np.concatenate(rows_p)
    vr = np.concatenate(ratios)
    rng = np.random.default_rng(seed)
    if len(t_idx) > max_samples:
        sel = rng.choice(len(t_idx), max_samples, replace=False)
        t_idx, p_idx, vr = t_idx[sel], p_idx[sel], vr[sel]
    order = np.lexsort((p_idx, t_idx))
    return t_idx[order], p_idx[order], vr[order]


def regression_metrics(y_true, y_pred):
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"MAE": None, "RMSE": None, "N": 0}
    err = y_pred[ok] - y_true[ok]
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err * err))),
            "N": int(ok.sum())}


def run_chronos(city_series_list, device):
    """Chronos zero-shot: per-pixel series, predict +H directly."""
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                               device_map=device, torch_dtype="auto")
    results = {}
    for city, horizon, t_idx, p_idx, values, y_true in city_series_list:
        preds = np.full(len(t_idx), np.nan, np.float32)
        for i, (tt, pp) in enumerate(zip(t_idx, p_idx)):
            ctx_start = max(0, tt - horizon - LOOKBACK)
            ctx = values[ctx_start:tt - horizon, pp]
            ctx = ctx[np.isfinite(ctx)]
            if len(ctx) < 10:
                continue
            # chronos predicts distribution; use median
            fc = pipe.predict(torch.from_numpy(ctx.astype(np.float32)),
                              prediction_length=horizon)
            # v2.3 returns Forecast; .mean may be property or method
            if hasattr(fc, '__iter__') and not hasattr(fc, 'mean'):
                fc = fc[0]
            if callable(getattr(fc, 'mean', None)):
                pred = fc.mean()
            else:
                pred = getattr(fc, 'mean', fc)
            pred = np.atleast_1d(np.asarray(pred, dtype=np.float32))
            preds[i] = float(pred[-1])
            preds[i] = pred[-1] if len(pred) > 0 else np.nan
        results.setdefault(city, {}).setdefault(f"{horizon}h", {})
        results[city][f"{horizon}h"]["Chronos"] = {
            "overall": regression_metrics(y_true, preds),
            "by_city": {city: regression_metrics(y_true, preds)},
        }
    return results


def run_timesfm(city_series_list, device):
    """TimesFM zero-shot."""
    import torch
    from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
    hparams = TimesFmHparams(context_len=192, horizon_len=96)
    checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
    model = TimesFm(hparams=hparams, checkpoint=checkpoint)
    # move inner model to GPU (TimesFm wrapper has no .to)
    model._model.to(device)
    import torch as T
    results = {}
    for city, horizon, t_idx, p_idx, values, y_true in city_series_list:
        preds = np.full(len(t_idx), np.nan, np.float32)
        contexts = []
        valid_idx = []
        for i, (tt, pp) in enumerate(zip(t_idx, p_idx)):
            ctx_start = max(0, tt - horizon - 192)
            ctx = values[ctx_start:tt - horizon, pp]
            ctx = ctx[np.isfinite(ctx)]
            if len(ctx) < 10:
                continue
            # pad to 192
            if len(ctx) < 192:
                pad = np.full(192 - len(ctx), ctx[0] if len(ctx) else 0, np.float32)
                ctx = np.concatenate([pad, ctx])
            else:
                ctx = ctx[-192:]
            contexts.append(ctx)
            valid_idx.append(i)
        if not contexts:
            continue
        contexts = np.stack(contexts)
        _, fc = model.forecast(T.from_numpy(contexts.astype(np.float32)).to(device))
        for j, i in enumerate(valid_idx):
            preds[i] = float(np.asarray(fc[j, horizon - 1]).ravel()[0])
        results.setdefault(city, {}).setdefault(f"{horizon}h", {})
        results[city][f"{horizon}h"]["TimesFM"] = {
            "overall": regression_metrics(y_true, preds),
            "by_city": {city: regression_metrics(y_true, preds)},
        }
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm", choices=["chronos", "timesfm"], required=True)
    ap.add_argument("--ood-cities", nargs="+", default=OOD_CITIES)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    ap.add_argument("--eval-years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n-pixels", type=int, default=128)
    ap.add_argument("--eval-samples-per-city-year", type=int, default=300)
    ap.add_argument("--min-valid-ratio", type=float, default=0.70)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    import torch

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {"task": f"Task 3 FM zero-shot ({a.fm})", "protocol": {
        "fm": a.fm, "ood_cities": a.ood_cities, "horizons": a.horizons,
        "eval_years": a.eval_years, "n_pixels": a.n_pixels,
        "min_valid_ratio": a.min_valid_ratio, "seed": a.seed,
    }}

    for city in a.ood_cities:
        print(f"\n[fm={a.fm}] {city}", flush=True)
        pix = choose_pixels(city, a.n_pixels, a.seed)
        city_data = []
        for year in a.eval_years:
            try:
                values, times = load_pixel_series(city, year, pix)
            except Exception as e:
                print(f"  skip {city} {year}: {e}"); continue
            for horizon in a.horizons:
                t_idx, p_idx, vr = candidate_windows(
                    values, horizon, a.min_valid_ratio,
                    a.eval_samples_per_city_year,
                    stable_seed(a.seed, city, year, horizon))
                if len(t_idx) == 0:
                    continue
                y_true = values[t_idx, p_idx]
                city_data.append((city, horizon, t_idx, p_idx, values, y_true))
                print(f"  {city} {year} +{horizon}h: {len(t_idx)} samples", flush=True)

        # run FM
        if a.fm == "chronos":
            res = run_chronos(city_data, a.device)
        else:
            res = run_timesfm(city_data, a.device)
        all_res.setdefault("results", {}).update(res)
        # per-city output
        (out_dir / f"3_fm_{a.fm}_{city}.json").write_text(
            json.dumps(res, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))

    out_path = out_dir / f"3_fm_{a.fm}_all.json"
    out_path.write_text(json.dumps(all_res, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))
    print(f"\n[saved] {out_path}")

    # quick summary
    print(f"\n=== {a.fm} summary (OOD avg MAE) ===")
    for h in a.horizons:
        vals = []
        for city in a.ood_cities:
            try:
                mae = all_res["results"][city][f"{h}h"][a.fm.capitalize() if a.fm=="chronos" else "TimesFM"]["overall"]["MAE"]
                if mae: vals.append(mae)
            except: pass
        avg = np.mean(vals) if vals else float("nan")
        print(f"  +{h}h: avg={avg:.3f}")


if __name__ == "__main__":
    main()
