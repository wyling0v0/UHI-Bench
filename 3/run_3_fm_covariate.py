"""Task 3 — Chronos-2 / MOIRAI-2 zero-shot OOD transfer, four configs.

Complements run_3_fm.py, which already covers Chronos and TimesFM base
zero-shot. This runner keeps the same Task-3 evaluation protocol:
OOD cities, eval years 2023-2025, lookback 168h, horizons 1/6/24h, sampled
valid windows. FM weights are never trained. Covariate normalization uses the
target city's train-period covariates (2015-2022) only; no target UHI labels are
used for fitting.

Configs:
  base          LST-UHI history only
  static        history + static10 broadcast as future-known dynamic covariates
  meteo         history + ERA5 six drivers
  meteo_static  history + ERA5 + static10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "3"))

import run_3_ood_transfer as T3  # noqa: E402

LOOKBACK = T3.LOOKBACK
CONFIGS = ["base", "meteo", "static", "meteo_static"]
CFG_LABEL = {
    "base": "base",
    "meteo": "+meteo",
    "static": "+static",
    "meteo_static": "+meteo+static",
}
FM_LABEL = {"chronos2": "Chronos-2", "moirai2": "MOIRAI-2"}


def fill_context(ctx: np.ndarray) -> np.ndarray:
    out = np.asarray(ctx, dtype=np.float32).copy()
    finite = np.isfinite(out)
    if finite.all():
        return out
    fill = float(out[finite].mean()) if finite.any() else 0.0
    out[~finite] = fill
    return out


def fit_stats(x: np.ndarray, axes):
    mu = np.nanmean(x, axis=axes, keepdims=True)
    sd = np.nanstd(x, axis=axes, keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_cov_stats(city: str, train_years, n_pixels: int, seed: int, config: str):
    stats = {}
    if config in ("meteo", "meteo_static"):
        parts = []
        used_years = []
        for y in train_years:
            try:
                d = T3.load_city_year(city, int(y), n_pixels, seed)
            except FileNotFoundError as e:
                print(f"  [warn] {city}: skip missing train-year covariates {y}: {e}", flush=True)
                continue
            parts.append(d.era5[::24])
            used_years.append(int(y))
        if not parts:
            raise FileNotFoundError(f"{city}: no available train-year covariates in {list(train_years)}")
        if used_years != list(train_years):
            print(f"  [cov-stats] {city}: using available train years {used_years}", flush=True)
        sample = np.concatenate(parts, axis=0).astype(np.float32)
        stats["meteo"] = fit_stats(sample, axes=(0, 1))
    if config in ("static", "meteo_static"):
        pixel_ids = T3.choose_pixels(city, n_pixels, seed)
        static = T3.static_for_pixels(city, pixel_ids)
        stats["static"] = fit_stats(static.astype(np.float32), axes=0)
    return stats


def build_covariates(data: T3.CityYearData, stats: dict, config: str):
    blocks = []
    if config in ("meteo", "meteo_static"):
        mu, sd = stats["meteo"]
        blocks.append(zscore(data.era5, mu, sd))
    if config in ("static", "meteo_static"):
        mu, sd = stats["static"]
        st = zscore(data.static.astype(np.float32), mu, sd)
        blocks.append(np.broadcast_to(st[None], (len(data.times), st.shape[0], st.shape[1])).astype(np.float32))
    if not blocks:
        return None
    return np.concatenate(blocks, axis=2).astype(np.float32, copy=False)


def build_eval_items(args):
    method = f"{FM_LABEL[args.fm]}({CFG_LABEL[args.config]})"
    items = []
    for city in args.ood_cities:
        print(f"\n[{method}] {city}: fitting cov stats from {args.train_years}", flush=True)
        stats = fit_cov_stats(city, args.train_years, args.n_pixels, args.seed, args.config)
        for year in args.eval_years:
            try:
                data = T3.load_city_year(city, int(year), args.n_pixels, args.seed)
            except Exception as e:
                print(f"  skip {city} {year}: {e}", flush=True)
                continue
            covs = build_covariates(data, stats, args.config)
            for horizon in args.horizons:
                t_idx, p_idx, valid_ratio = T3.candidate_rows(
                    data.values,
                    horizon,
                    args.min_valid_ratio,
                    args.eval_samples_per_city_year,
                    T3.stable_city_seed(args.seed, city, int(year), horizon + 2000),
                )
                if len(t_idx) == 0:
                    continue
                y_true = data.values[t_idx, p_idx]
                items.append({
                    "city": city,
                    "year": int(year),
                    "horizon": int(horizon),
                    "t_idx": t_idx,
                    "p_idx": p_idx,
                    "values": data.values,
                    "covs": covs,
                    "times": data.times,
                    "y_true": y_true,
                })
                print(f"  {city} {year} +{horizon}h: {len(t_idx)} samples", flush=True)
    return items


def run_chronos2(items, args):
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)
    method = f"{FM_LABEL[args.fm]}({CFG_LABEL[args.config]})"
    out = {}
    for item in items:
        h = item["horizon"]
        t_idx = item["t_idx"]
        p_idx = item["p_idx"]
        values = item["values"]
        covs = item["covs"]
        times = pd.DatetimeIndex(item["times"])
        cov_names = [f"c{i}" for i in range(covs.shape[-1])] if covs is not None else []
        preds = np.full(len(t_idx), np.nan, np.float32)
        for b0 in range(0, len(t_idx), args.batch_size):
            ctx_frames, fut_frames, meta = [], [], []
            for j in range(b0, min(b0 + args.batch_size, len(t_idx))):
                tt = int(t_idx[j])
                pp = int(p_idx[j])
                input_end = tt - h
                start = input_end - LOOKBACK + 1
                sid = f"{item['city']}_{item['year']}_{h}_{j}"
                ctx = pd.DataFrame({
                    "id": sid,
                    "timestamp": times[start:input_end + 1],
                    "target": fill_context(values[start:input_end + 1, pp]),
                })
                fut = pd.DataFrame({
                    "id": sid,
                    "timestamp": times[input_end + 1:tt + 1],
                })
                for ci, cn in enumerate(cov_names):
                    ctx[cn] = covs[start:input_end + 1, pp, ci]
                    fut[cn] = covs[input_end + 1:tt + 1, pp, ci]
                ctx_frames.append(ctx)
                fut_frames.append(fut)
                meta.append((j, sid))
            pred_df = pipe.predict_df(
                pd.concat(ctx_frames, ignore_index=True),
                future_df=pd.concat(fut_frames, ignore_index=True),
                prediction_length=h,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            pred_map = {sid: g.sort_values("timestamp")["0.5"].to_numpy(np.float32)
                        for sid, g in pred_df.groupby("id", sort=False)}
            for j, sid in meta:
                yhat = pred_map.get(sid)
                if yhat is not None and len(yhat):
                    preds[j] = float(yhat[-1])
            print(f"  chronos2 {item['city']} {item['year']} +{h}h {min(b0 + args.batch_size, len(t_idx))}/{len(t_idx)}", flush=True)
        add_item_result(out, item, method, preds)
    return out


def run_moirai2(items, args):
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    method = f"{FM_LABEL[args.fm]}({CFG_LABEL[args.config]})"
    out = {}
    for h in args.horizons:
        h_items = [it for it in items if it["horizon"] == h]
        if not h_items:
            continue
        cov_dim = int(h_items[0]["covs"].shape[-1]) if h_items[0]["covs"] is not None else 0
        module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        model = Moirai2Forecast(
            module=module,
            prediction_length=int(h),
            context_length=LOOKBACK,
            target_dim=1,
            feat_dynamic_real_dim=cov_dim,
            past_feat_dynamic_real_dim=0,
        ).to(args.device).eval()
        q_levels = np.asarray(model.module.quantile_levels, dtype=np.float32)
        q50 = int(np.argmin(np.abs(q_levels - 0.5)))
        for item in h_items:
            t_idx = item["t_idx"]
            p_idx = item["p_idx"]
            values = item["values"]
            covs = item["covs"]
            preds = np.full(len(t_idx), np.nan, np.float32)
            for b0 in range(0, len(t_idx), args.batch_size):
                rows = range(b0, min(b0 + args.batch_size, len(t_idx)))
                past, obs, feats = [], [], []
                for j in rows:
                    tt = int(t_idx[j])
                    pp = int(p_idx[j])
                    input_end = tt - h
                    start = input_end - LOOKBACK + 1
                    ctx = values[start:input_end + 1, pp].astype(np.float32)
                    finite = np.isfinite(ctx)
                    past.append(np.nan_to_num(ctx, nan=0.0))
                    obs.append(finite.astype(np.float32))
                    if covs is not None:
                        feats.append(covs[start:tt + 1, pp, :])
                past_t = torch.from_numpy(np.asarray(past, np.float32)[:, :, None]).to(args.device)
                # Match the working 1c/1d MOIRAI-2 path: after filling NaNs in
                # the context, treat the model input as observed. Passing sparse
                # observed masks can route through a Moirai2 time-index path that
                # expects integer ids and fails on CPU.
                obs_t = torch.ones_like(past_t, dtype=torch.bool)
                pad_t = torch.zeros((past_t.shape[0], LOOKBACK), dtype=torch.bool, device=args.device)
                kw = {}
                if covs is not None:
                    feat_t = torch.from_numpy(np.asarray(feats, np.float32)).to(args.device)
                    kw["feat_dynamic_real"] = feat_t
                    kw["observed_feat_dynamic_real"] = torch.ones_like(feat_t, dtype=torch.bool)
                with torch.no_grad():
                    pred = model(past_t, obs_t, pad_t, **kw)
                q = pred[:, q50, :].detach().cpu().numpy().astype(np.float32)
                for off, j in enumerate(rows):
                    preds[j] = float(q[off, -1])
                print(f"  moirai2 {item['city']} {item['year']} +{h}h {min(b0 + args.batch_size, len(t_idx))}/{len(t_idx)}", flush=True)
            add_item_result(out, item, method, preds)
    return out


def add_item_result(out, item, method, preds):
    city = item["city"]
    h_key = f"{item['horizon']}h"
    m = T3.regression_metrics(item["y_true"], preds)
    out.setdefault(city, {}).setdefault(h_key, {}).setdefault(method, {"parts": []})
    out[city][h_key][method]["parts"].append(m)


def finalize_results(out):
    for city, horizons in out.items():
        for h_key, models in horizons.items():
            for method, payload in list(models.items()):
                parts = payload.pop("parts")
                n = sum(p["N"] for p in parts)
                if n == 0:
                    mae = rmse = None
                else:
                    mae = sum((p["MAE"] or 0.0) * p["N"] for p in parts) / n
                    rmse = np.sqrt(sum(((p["RMSE"] or 0.0) ** 2) * p["N"] for p in parts) / n)
                models[method] = {
                    "overall": {"MAE": None if mae is None else float(mae),
                                "RMSE": None if rmse is None else float(rmse),
                                "N": int(n)},
                    "by_city": {city: {"MAE": None if mae is None else float(mae),
                                       "RMSE": None if rmse is None else float(rmse),
                                       "N": int(n)}},
                }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm", choices=["chronos2", "moirai2"], required=True)
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--ood-cities", nargs="+", default=T3.OOD_CITIES)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    ap.add_argument("--train-years", type=int, nargs="+", default=list(range(2015, 2023)))
    ap.add_argument("--eval-years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n-pixels", type=int, default=64)
    ap.add_argument("--eval-samples-per-city-year", type=int, default=200)
    ap.add_argument("--min-valid-ratio", type=float, default=0.70)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_path = out_dir / f"3_fm_{args.fm}_{args.config}_all.json"
    expected_protocol = {
        "fm": args.fm,
        "config": args.config,
        "train_years": args.train_years,
        "eval_years": args.eval_years,
        "ood_cities": args.ood_cities,
        "horizons": args.horizons,
        "n_pixels": args.n_pixels,
        "eval_samples_per_city_year": args.eval_samples_per_city_year,
        "min_valid_ratio": args.min_valid_ratio,
        "seed": args.seed,
        "note": "Zero-shot FM weights; covariate scalers use train_years only.",
    }
    if args.config != "base":
        expected_protocol["covariate_train_year_policy"] = (
            "Use requested train_years when present; skip missing city-years and never use eval_years."
        )
    if out_path.exists():
        try:
            old = json.loads(out_path.read_text())
        except Exception:
            old = {}
        if old.get("protocol") == expected_protocol and old.get("results"):
            print(f"[skip-existing] {FM_LABEL[args.fm]} {CFG_LABEL[args.config]} -> {out_path}", flush=True)
            return

    items = build_eval_items(args)
    if args.fm == "chronos2":
        results = run_chronos2(items, args)
    else:
        results = run_moirai2(items, args)
    results = finalize_results(results)
    payload = {
        "task": f"Task 3 FM zero-shot ({FM_LABEL[args.fm]} {CFG_LABEL[args.config]})",
        "protocol": expected_protocol,
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
