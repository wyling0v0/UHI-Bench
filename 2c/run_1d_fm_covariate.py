"""Task 2c — matched-subset FM baselines.

Zero-shot over pixel UHI series. Target is HOSTRADA Ta-UHI from v7 cache
(`uhi[..., 1]`); dynamic real covariates are ERA5/met features from `met.npy`.
The fifth met feature is currently all-NaN in the cache, so only the first four
features are used. Results are merged into `2c/results/1d_forecast.json`.

For fair covariate-gain checks, the same script also runs no-covariate matched
baselines (Chronos, TimesFM, Persistence, Climatology) on the exact same pixel
subset and weekly windows as Chronos-2 / MOIRAI-2.

Run with the isolated venv:
  .venvs/uhi-fm/bin/python benchmark/2c/run_1d_fm_covariate.py --model chronos  --city munich
  .venvs/uhi-fm/bin/python benchmark/2c/run_1d_fm_covariate.py --model timesfm  --city munich
  .venvs/uhi-fm/bin/python benchmark/2c/run_1d_fm_covariate.py --model chronos2 --city munich
  .venvs/uhi-fm/bin/python benchmark/2c/run_1d_fm_covariate.py --model moirai2  --city munich
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "results" / "1d_forecast.json"

import sys
sys.path.insert(0, str(BENCH))
from common.data import _load_static  # noqa: E402
from common.paths import ERA5_BASE, PSEUDO_TA_BASE, TA_BASE  # noqa: E402

HORIZONS = [1, 6, 12, 24, 48, 96]
T_CTX = 168
H = max(HORIZONS)
MET_KEEP = [0, 1, 2, 3]
ERA5_DRIVERS = ["u10", "v10", "tcc", "d2m", "blh", "ssrd"]
CONFIGS = ["base", "meteo", "static", "meteo_static"]
CFG_LABEL = {
    "base": "base",
    "meteo": "+meteo",
    "static": "+static",
    "meteo_static": "+meteo+static",
}
FM_LABEL = {
    "chronos": "Chronos",
    "timesfm": "TimesFM",
    "chronos2": "Chronos-2",
    "moirai2": "MOIRAI-2",
}


def load_cache(city: str, years: list[int], pix: np.ndarray):
    if not _has_hostrada_cache(city, years):
        values, times = load_values_cache(city, years, pix)
        pixel_ids = pixel_ids_for_indices(city, years[0], pix)
        covs, _ = _load_pseudo_era5_selected(city, years, pixel_ids, pd.DatetimeIndex(times))
        return values, covs[:, :, MET_KEEP], times

    uhi_parts, met_parts, time_parts = [], [], []
    for y in years:
        d = _ta_v7_dir(city, int(y))
        if not (d / "done.json").exists():
            raise FileNotFoundError(f"missing v7 cache: {d}")
        uhi = np.load(d / "uhi.npy", mmap_mode="r")
        met = np.load(d / "met.npy", mmap_mode="r")
        times = np.load(d / "times.npy", mmap_mode="r")
        uhi_parts.append(np.asarray(uhi[:, pix, 1], dtype=np.float32))
        met_parts.append(np.asarray(met[:, pix][:, :, MET_KEEP], dtype=np.float32))
        time_parts.append(np.asarray(times, dtype=np.int64))
    values = np.ascontiguousarray(np.concatenate(uhi_parts, axis=0))
    covs = np.ascontiguousarray(np.concatenate(met_parts, axis=0))
    times = np.concatenate(time_parts, axis=0)
    return values, covs, pd.to_datetime(times, unit="s")


def load_values_cache(city: str, years: list[int], pix: np.ndarray):
    if not _has_hostrada_cache(city, years):
        pixel_ids = pixel_ids_for_indices(city, years[0], pix)
        return _load_pseudo_values_selected(city, years, pixel_ids)

    uhi_parts, time_parts = [], []
    for y in years:
        d = _ta_v7_dir(city, int(y))
        if not (d / "done.json").exists():
            raise FileNotFoundError(f"missing v7 cache: {d}")
        uhi = np.load(d / "uhi.npy", mmap_mode="r")
        times = np.load(d / "times.npy", mmap_mode="r")
        uhi_parts.append(np.asarray(uhi[:, pix, 1], dtype=np.float32))
        time_parts.append(np.asarray(times, dtype=np.int64))
    values = np.ascontiguousarray(np.concatenate(uhi_parts, axis=0))
    times = np.concatenate(time_parts, axis=0)
    return values, pd.to_datetime(times, unit="s")


def choose_pixels(city: str, year: int, n_pixels: int, seed: int):
    if not _has_hostrada_cache(city, [year]):
        _, _, _, pids = _load_static(city, n_static=10)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(len(pids), min(n_pixels, len(pids)), replace=False))
    uhi = np.load(_ta_v7_dir(city, int(year)) / "uhi.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(uhi.shape[1], min(n_pixels, uhi.shape[1]), replace=False))


def pixel_ids_for_indices(city: str, year: int, pix: np.ndarray):
    if not _has_hostrada_cache(city, [year]):
        _, _, _, pids = _load_static(city, n_static=10)
        return np.asarray(pids, dtype=np.int64)[pix].astype(np.int64)
    return np.load(_ta_v7_dir(city, int(year)) / "pixel_ids.npy", mmap_mode="r")[pix].astype(np.int64)


def load_static_for_indices(city: str, year: int, pix: np.ndarray):
    pixel_ids = pixel_ids_for_indices(city, year, pix)
    _, feats, _, pids = _load_static(city, n_static=10)
    row = {int(pid): i for i, pid in enumerate(pids)}
    order = np.array([row[int(pid)] for pid in pixel_ids], dtype=np.int64)
    return feats[order].astype(np.float32)


def fit_cov_stats_from_cache(city: str, years: list[int], pix: np.ndarray, sample_stride: int = 24):
    if not _has_hostrada_cache(city, years):
        pixel_ids = pixel_ids_for_indices(city, years[0], pix)
        era, _ = _load_pseudo_era5_selected(city, years, pixel_ids, None)
        return fit_cov_stats(era[::sample_stride, :, MET_KEEP])

    parts = []
    for y in years:
        d = _ta_v7_dir(city, int(y))
        met = np.load(d / "met.npy", mmap_mode="r")
        parts.append(np.asarray(met[::sample_stride, pix][:, :, MET_KEEP], dtype=np.float32))
    sample = np.concatenate(parts, axis=0)
    return fit_cov_stats(sample)


def _has_hostrada_cache(city: str, years) -> bool:
    return all(
        (_ta_v7_dir(city, int(y)) / "done.json").exists()
        and (_ta_v7_dir(city, int(y)) / "uhi.npy").exists()
        for y in years
    )


def _ta_v7_dir(city: str, year: int) -> Path:
    host = TA_BASE / city / "cache" / f"v7_{int(year)}"
    if (host / "done.json").exists():
        return host
    pseudo = PSEUDO_TA_BASE / city / "cache" / f"v7_{int(year)}"
    if (pseudo / "done.json").exists():
        return pseudo
    return host


def _selected_cache_dir(city: str, years, pixel_ids: np.ndarray, kind: str,
                        time_index: pd.DatetimeIndex | None = None) -> Path:
    years_key = "_".join(map(str, [int(y) for y in years]))
    h = hashlib.sha1(np.asarray(pixel_ids, dtype=np.int64).tobytes())
    if time_index is not None:
        ti = pd.DatetimeIndex(time_index).asi8
        h.update(np.asarray([len(ti), int(ti[0]), int(ti[-1])], dtype=np.int64).tobytes())
    key = h.hexdigest()[:12]
    return PSEUDO_TA_BASE / city / "cache" / f"bench_1d_{kind}_{years_key}_n{len(pixel_ids)}_{key}"


def _read_selected_parquet(path: Path, columns: list[str], pixel_ids: np.ndarray) -> pd.DataFrame:
    ids = [int(p) for p in np.asarray(pixel_ids, dtype=np.int64)]
    try:
        return pd.read_parquet(path, columns=columns, filters=[("pixel_id", "in", ids)])
    except Exception:
        df = pd.read_parquet(path, columns=columns)
        return df[df["pixel_id"].isin(ids)]


def _year_hours(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{int(year)}-01-01", f"{int(year) + 1}-01-01", freq="h")[:-1]


def _load_pseudo_values_selected(city: str, years, pixel_ids: np.ndarray):
    pixel_ids = np.asarray(pixel_ids, dtype=np.int64)
    cache = _selected_cache_dir(city, years, pixel_ids, "ta_values")
    if ((cache / "done.json").exists() and (cache / "values.npy").exists()
            and (cache / "times.npy").exists()):
        values = np.load(cache / "values.npy", mmap_mode="r")
        times = pd.to_datetime(np.load(cache / "times.npy", mmap_mode="r"), unit="s")
        return values, times

    cache.mkdir(parents=True, exist_ok=True)
    vals, times_parts = [], []
    for y in [int(v) for v in years]:
        path = PSEUDO_TA_BASE / city / f"atuhi_ood_1km_hourly_{y}.parquet"
        df = _read_selected_parquet(path, ["datetime", "pixel_id", "uhi"], pixel_ids)
        df["datetime"] = pd.to_datetime(df["datetime"])
        idx = _year_hours(y)
        piv = df.pivot_table(index="datetime", columns="pixel_id", values="uhi",
                             aggfunc="first").reindex(index=idx, columns=pixel_ids)
        vals.append(piv.to_numpy(dtype=np.float32))
        times_parts.append(idx)
    values = np.ascontiguousarray(np.concatenate(vals, axis=0))
    times = pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in times_parts]))
    np.save(cache / "values.npy", values)
    np.save(cache / "times.npy", times.asi8 // 1_000_000_000)
    (cache / "done.json").write_text(json.dumps({
        "city": city,
        "years": [int(y) for y in years],
        "pixel_ids": [int(p) for p in pixel_ids],
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "source": "atuhi_ood_1km_hourly",
    }, indent=2))
    return np.load(cache / "values.npy", mmap_mode="r"), times


def _load_pseudo_era5_selected(city: str, years, pixel_ids: np.ndarray,
                               time_index: pd.DatetimeIndex | None):
    pixel_ids = np.asarray(pixel_ids, dtype=np.int64)
    time_index = pd.DatetimeIndex(time_index) if time_index is not None else None
    cache = _selected_cache_dir(city, years, pixel_ids, "era5", time_index)
    if ((cache / "done.json").exists() and (cache / "era5.npy").exists()
            and (cache / "times.npy").exists()):
        arr = np.load(cache / "era5.npy", mmap_mode="r")
        times = pd.to_datetime(np.load(cache / "times.npy", mmap_mode="r"), unit="s")
        return arr, times

    cache.mkdir(parents=True, exist_ok=True)
    arr_parts, time_parts = [], []
    for y in [int(v) for v in years]:
        path = ERA5_BASE / city / f"era5_hourly_{y}.parquet"
        df = _read_selected_parquet(path, ["datetime", "pixel_id", *ERA5_DRIVERS], pixel_ids)
        df["datetime"] = pd.to_datetime(df["datetime"])
        if time_index is None:
            idx = _year_hours(y)
        else:
            idx = time_index[time_index.year == y]
        channels = []
        for col in ERA5_DRIVERS:
            piv = df.pivot_table(index="datetime", columns="pixel_id", values=col,
                                 aggfunc="first").reindex(index=idx, columns=pixel_ids)
            channels.append(piv.to_numpy(dtype=np.float32))
        arr_parts.append(np.stack(channels, axis=-1))
        time_parts.append(idx)
    arr = np.ascontiguousarray(np.concatenate(arr_parts, axis=0))
    times = pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in time_parts]))
    np.save(cache / "era5.npy", arr)
    np.save(cache / "times.npy", times.asi8 // 1_000_000_000)
    (cache / "done.json").write_text(json.dumps({
        "city": city,
        "years": [int(y) for y in years],
        "pixel_ids": [int(p) for p in pixel_ids],
        "shape": [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])],
        "drivers": ERA5_DRIVERS,
    }, indent=2))
    return np.load(cache / "era5.npy", mmap_mode="r"), times


def fit_cov_stats(covs: np.ndarray):
    mu = np.nanmean(covs, axis=(0, 1), keepdims=True)
    sd = np.nanstd(covs, axis=(0, 1), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_cov(covs: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    z = (covs - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_static_stats(static: np.ndarray):
    mu = np.nanmean(static, axis=0, keepdims=True)
    sd = np.nanstd(static, axis=0, keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def build_covariates(city: str, train_years: list[int], test_years: list[int],
                     pix: np.ndarray, config: str):
    blocks = []
    if config in ("meteo", "meteo_static"):
        cov_mu, cov_sd = fit_cov_stats_from_cache(city, train_years, pix)
        _, met_test, _ = load_cache(city, test_years, pix)
        blocks.append(standardize_cov(met_test, cov_mu, cov_sd))
    if config in ("static", "meteo_static"):
        static = load_static_for_indices(city, train_years[0], pix)
        st_mu, st_sd = fit_static_stats(static)
        st_z = standardize_cov(static, st_mu, st_sd)
        n_time = blocks[0].shape[0] if blocks else load_values_cache(city, test_years, pix)[0].shape[0]
        blocks.append(np.broadcast_to(st_z[None, :, :], (n_time, st_z.shape[0], st_z.shape[1])).astype(np.float32))
    if not blocks:
        return None
    return np.concatenate(blocks, axis=2).astype(np.float32, copy=False)


def window_starts(n_time: int, stride: int):
    return list(range(0, n_time - T_CTX - H + 1, stride))


def add_errors(errs: dict[int, list[np.ndarray]], pred: np.ndarray, true: np.ndarray):
    for h in HORIZONS:
        p = pred[:, h - 1]
        y = true[:, h - 1]
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.any():
            errs[h].append(np.abs(p[ok] - y[ok]))


def fill_context(ctx: np.ndarray):
    out = np.asarray(ctx, dtype=np.float32).copy()
    if np.isfinite(out).all():
        return out
    fin = np.isfinite(out)
    if fin.sum() == 0:
        return np.zeros_like(out, dtype=np.float32)
    out[~fin] = float(out[fin].mean())
    return out


def pad_left_to_multiple(ctx: np.ndarray, multiple: int):
    pad = (-len(ctx)) % multiple
    if pad == 0:
        return ctx
    fill = ctx[0] if np.isfinite(ctx[0]) else float(np.nanmean(ctx))
    if not np.isfinite(fill):
        fill = 0.0
    return np.concatenate([np.full(pad, fill, np.float32), ctx.astype(np.float32, copy=False)])


def run_persistence(values, starts, n_pixels):
    errs = {h: [] for h in HORIZONS}
    for s in starts:
        last = values[s + T_CTX - 1, :n_pixels]
        pred = np.broadcast_to(last[:, None], (n_pixels, H)).astype(np.float32)
        true = values[s + T_CTX:s + T_CTX + H, :n_pixels].T
        add_errors(errs, pred, true)
    return errs


def fit_climatology(train_values, train_times):
    keys = (pd.to_datetime(train_times).dayofyear.to_numpy() - 1) * 24 + pd.to_datetime(train_times).hour.to_numpy()
    n_keys = 366 * 24
    clim = np.full((n_keys, train_values.shape[1]), np.nan, np.float32)
    fallback = np.nanmean(train_values, axis=0).astype(np.float32)
    for k in np.unique(keys):
        clim[k] = np.nanmean(train_values[keys == k], axis=0)
    miss = ~np.isfinite(clim)
    if miss.any():
        clim = np.where(np.isfinite(clim), clim, fallback[None, :])
    return clim.astype(np.float32)


def run_climatology(values, times, starts, n_pixels, clim):
    errs = {h: [] for h in HORIZONS}
    keys = (pd.to_datetime(times).dayofyear.to_numpy() - 1) * 24 + pd.to_datetime(times).hour.to_numpy()
    for s in starts:
        fut_idx = np.arange(s + T_CTX, s + T_CTX + H)
        pred = clim[keys[fut_idx], :n_pixels].T
        true = values[fut_idx, :n_pixels].T
        add_errors(errs, pred, true)
    return errs


def run_chronos(values, starts, n_pixels, window_batch, device):
    import torch
    from chronos import ChronosBoltPipeline

    pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device)
    errs = {h: [] for h in HORIZONS}
    series, true = [], []
    for wi, s in enumerate(starts):
        for p in range(n_pixels):
            series.append(fill_context(values[s:s + T_CTX, p]))
            true.append(values[s + T_CTX:s + T_CTX + H, p])
        if len(series) >= window_batch * n_pixels or wi == len(starts) - 1:
            ctx = torch.from_numpy(np.stack(series)).float()
            with torch.no_grad():
                out = pipe.predict(ctx, prediction_length=H)  # [B,Q,H]
            pred = out[:, out.shape[1] // 2, :].detach().cpu().numpy().astype(np.float32)
            add_errors(errs, pred, np.stack(true).astype(np.float32))
            series, true = [], []
            print(f"  chronos windows {wi + 1}/{len(starts)}", flush=True)
    return errs


def run_timesfm(values, starts, n_pixels, window_batch):
    import timesfm

    patch_len = 32
    tfm_context = T_CTX + ((-T_CTX) % patch_len)
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            context_len=tfm_context,
            horizon_len=H,
            input_patch_len=patch_len,
            backend="gpu",
            per_core_batch_size=max(1, window_batch * n_pixels),
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    errs = {h: [] for h in HORIZONS}
    series, true = [], []
    for wi, s in enumerate(starts):
        for p in range(n_pixels):
            series.append(pad_left_to_multiple(fill_context(values[s:s + T_CTX, p]), patch_len))
            true.append(values[s + T_CTX:s + T_CTX + H, p])
        if len(series) >= window_batch * n_pixels or wi == len(starts) - 1:
            pred, _ = model.forecast(series, normalize=True)
            add_errors(errs, np.asarray(pred, np.float32), np.stack(true).astype(np.float32))
            series, true = [], []
            print(f"  timesfm windows {wi + 1}/{len(starts)}", flush=True)
    return errs


def run_chronos2(values, covs, times, starts, n_pixels, window_batch, device):
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    errs = {h: [] for h in HORIZONS}
    cov_names = [f"c{i}" for i in range(covs.shape[-1])] if covs is not None else []
    for b0 in range(0, len(starts), window_batch):
        batch_starts = starts[b0:b0 + window_batch]
        ctx_frames, fut_frames, id_meta = [], [], []
        for wi, s in enumerate(batch_starts):
            ctx_t = times[s:s + T_CTX]
            fut_t = times[s + T_CTX:s + T_CTX + H]
            for p in range(n_pixels):
                sid = f"w{b0 + wi}_p{p}"
                ctx = pd.DataFrame({
                    "id": sid,
                    "timestamp": ctx_t,
                    "target": values[s:s + T_CTX, p],
                })
                fut = pd.DataFrame({"id": sid, "timestamp": fut_t})
                for ci, cn in enumerate(cov_names):
                    ctx[cn] = covs[s:s + T_CTX, p, ci]
                    fut[cn] = covs[s + T_CTX:s + T_CTX + H, p, ci]
                ctx_frames.append(ctx)
                fut_frames.append(fut)
                id_meta.append((sid, s, p))

        pred_df = pipe.predict_df(
            pd.concat(ctx_frames, ignore_index=True),
            future_df=pd.concat(fut_frames, ignore_index=True),
            prediction_length=H,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        pred_map = {sid: g.sort_values("timestamp")["0.5"].to_numpy(np.float32)
                    for sid, g in pred_df.groupby("id", sort=False)}
        preds, true = [], []
        for sid, s, p in id_meta:
            yhat = pred_map.get(sid)
            if yhat is None or len(yhat) < H:
                continue
            preds.append(yhat[:H])
            true.append(values[s + T_CTX:s + T_CTX + H, p])
        if preds:
            add_errors(errs, np.stack(preds), np.stack(true))
        print(f"  chronos2 windows {min(b0 + window_batch, len(starts))}/{len(starts)}", flush=True)
    return errs


def run_moirai2(values, covs, starts, n_pixels, window_batch, device):
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    cov_dim = int(covs.shape[-1]) if covs is not None else 0
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    model = Moirai2Forecast(
        module=module,
        prediction_length=H,
        context_length=T_CTX,
        target_dim=1,
        feat_dynamic_real_dim=cov_dim,
        past_feat_dynamic_real_dim=0,
    ).to(device).eval()
    q_levels = np.asarray(model.module.quantile_levels, dtype=np.float32)
    q50 = int(np.argmin(np.abs(q_levels - 0.5)))
    errs = {h: [] for h in HORIZONS}

    for b0 in range(0, len(starts), window_batch):
        batch_starts = starts[b0:b0 + window_batch]
        past, feat, true = [], [], []
        for s in batch_starts:
            for p in range(n_pixels):
                past.append(values[s:s + T_CTX, p])
                if covs is not None:
                    feat.append(covs[s:s + T_CTX + H, p, :])
                true.append(values[s + T_CTX:s + T_CTX + H, p])
        past_np = np.nan_to_num(np.asarray(past, np.float32), nan=0.0)
        true_np = np.asarray(true, np.float32)

        past_t = torch.from_numpy(past_np[:, :, None]).to(device)
        obs_t = torch.isfinite(torch.from_numpy(np.asarray(past, np.float32)[:, :, None])).to(device)
        pad_t = torch.zeros((past_t.shape[0], T_CTX), dtype=torch.bool, device=device)
        kw = {}
        if covs is not None:
            feat_np = np.nan_to_num(np.asarray(feat, np.float32), nan=0.0)
            feat_t = torch.from_numpy(feat_np).to(device)
            kw["feat_dynamic_real"] = feat_t
            kw["observed_feat_dynamic_real"] = torch.ones_like(feat_t, dtype=torch.bool)
        with torch.no_grad():
            out = model(
                past_t,
                obs_t,
                pad_t,
                **kw,
            )
        pred = out[:, q50, :].detach().cpu().numpy().astype(np.float32)
        add_errors(errs, pred, true_np)
        print(f"  moirai2 windows {min(b0 + window_batch, len(starts))}/{len(starts)}", flush=True)
    return errs


def result_name(model: str, config: str, n_pixels: int) -> str:
    if model in {"persistence", "climatology"}:
        return {
            "persistence": f"Persistence({n_pixels}px weekly subset)",
            "climatology": f"Climatology({n_pixels}px weekly subset)",
        }[model]
    if model == "timesfm":
        return f"TimesFM(base,{n_pixels}px weekly subset,padded)"
    return f"{FM_LABEL[model]}({CFG_LABEL[config]},{n_pixels}px weekly subset)"


def row_exists(out_path: Path, city: str, name: str) -> bool:
    if not out_path.exists():
        return False
    try:
        data = json.loads(out_path.read_text())
    except Exception:
        return False
    return name in data.get("Ta", {}).get(city.capitalize(), {})


def merge_row(city: str, model: str, config: str, n_pixels: int, row: dict[str, float], out_path: Path):
    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    name = result_name(model, config, n_pixels)
    data.setdefault("Ta", {}).setdefault(city.capitalize(), {})[name] = row
    data.setdefault("protocol_notes", {})["fm_four_config"] = {
        "train_years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
        "test_years": [2023, 2024, 2025],
        "n_pixels": n_pixels,
        "stride": "weekly by default",
        "note": "FM weights are zero-shot; target/covariate normalization uses train_years only.",
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"[merged] {name} {city} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["persistence", "climatology", "chronos", "timesfm", "chronos2", "moirai2"], required=True)
    ap.add_argument("--config", choices=CONFIGS, default="base")
    ap.add_argument("--city", default="munich")
    ap.add_argument("--train_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--test_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_pixels", type=int, default=64)
    ap.add_argument("--stride", type=int, default=168)
    ap.add_argument("--window_batch", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.model in {"timesfm", "chronos", "persistence", "climatology"} and args.config != "base":
        raise SystemExit(f"{args.model} supports only --config base in this runner")
    out_path = Path(args.out)
    name = result_name(args.model, args.config, args.n_pixels)
    if row_exists(out_path, args.city, name):
        print(f"[skip-existing] Ta {args.city.capitalize()} {name} -> {out_path}", flush=True)
        return

    pix = choose_pixels(args.city, args.train_years[0], args.n_pixels, args.seed)
    print(f"[fm-cov] model={args.model} config={args.config} city={args.city} pixels={len(pix)}", flush=True)
    if args.model in {"chronos2", "moirai2"}:
        if args.config in ("meteo", "meteo_static"):
            values, _, times = load_cache(args.city, args.test_years, pix)
        else:
            values, times = load_values_cache(args.city, args.test_years, pix)
        covs = build_covariates(args.city, args.train_years, args.test_years, pix, args.config)
    else:
        values, times = load_values_cache(args.city, args.test_years, pix)
        covs = None
    starts = window_starts(values.shape[0], args.stride)
    cov_msg = f" cov={covs.shape}" if covs is not None else ""
    print(f"[fm-cov] test={values.shape}{cov_msg} windows={len(starts)} stride={args.stride}", flush=True)

    if args.model == "persistence":
        errs = run_persistence(values, starts, len(pix))
    elif args.model == "climatology":
        train_values, train_times = load_values_cache(args.city, args.train_years, pix)
        clim = fit_climatology(train_values, train_times)
        errs = run_climatology(values, times, starts, len(pix), clim)
    elif args.model == "chronos":
        errs = run_chronos(values, starts, len(pix), args.window_batch, args.device)
    elif args.model == "timesfm":
        errs = run_timesfm(values, starts, len(pix), args.window_batch)
    elif args.model == "chronos2":
        errs = run_chronos2(values, covs, times, starts, len(pix), args.window_batch, args.device)
    else:
        errs = run_moirai2(values, covs, starts, len(pix), args.window_batch, args.device)

    row = {}
    for h in HORIZONS:
        ae = np.concatenate(errs[h]) if errs[h] else np.array([], dtype=np.float32)
        row[f"{h}h"] = float(ae.mean()) if ae.size else None
        print(f"  +{h}h: MAE={row[f'{h}h']:.4f}" if ae.size else f"  +{h}h: MAE=None")
    merge_row(args.city, args.model, args.config, len(pix), row, out_path)


if __name__ == "__main__":
    main()
