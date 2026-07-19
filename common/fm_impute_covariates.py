"""Shared FM helpers for 1-step imputation tasks.

The 1a/1b reconstruction protocols hide the current timestep and ask the model
to reconstruct it from the preceding window. Chronos-2 and MOIRAI-2 are
covariate-capable forecasting FMs, so the hidden step is represented as a 1-step
forecast with optional dynamic real covariates. TimesFM is included as a base
history-only FM because the current API does not expose a covariate channel.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.covariates import (  # noqa: E402
    fit_era5_stats_stream,
    load_era5_aligned,
    load_static_raw,
    standardize_era5,
    standardize_static,
)
from common.paths import CACHE_ROOT  # noqa: E402

FM_CHOICES = ["chronos2", "moirai2", "timesfm"]
CONFIG_CHOICES = ["base", "meteo", "static", "meteo_static"]
CFG_LABEL = {
    "base": "base",
    "meteo": "+meteo",
    "static": "+static",
    "meteo_static": "+meteo+static",
}
FM_LABEL = {
    "chronos2": "Chronos-2",
    "moirai2": "MOIRAI-2",
    "timesfm": "TimesFM",
}


def resolve_device(device: str) -> str:
    try:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[device] CUDA unavailable -> CPU", flush=True)
            return "cpu"
    except Exception:
        return "cpu"
    return device


def method_key(fm: str, config: str) -> str:
    return f"{FM_LABEL[fm]}({CFG_LABEL[config]})"


def synthetic_hourly_index(n: int) -> pd.DatetimeIndex:
    # Position-based tasks can have irregular real timestamps. Chronos-2 needs a
    # fixed-frequency index, so use a synthetic hourly grid aligned to positions.
    return pd.date_range("2000-01-01", periods=int(n), freq="h")


def fill_context(ctx: np.ndarray) -> np.ndarray:
    out = np.asarray(ctx, dtype=np.float32).copy()
    if np.isfinite(out).all():
        return out
    fin = np.isfinite(out)
    if fin.sum() == 0:
        return np.zeros_like(out, dtype=np.float32)
    out[~fin] = float(out[fin].mean())
    return out


def standardize_target(values: np.ndarray):
    if np.isnan(values).any():
        mu = np.nanmean(values, axis=0).astype(np.float32)
        sd = (np.nanstd(values, axis=0) + 1e-6).astype(np.float32)
    else:
        mu = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = (values.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0).astype(np.float32)
    return ((values - mu) / sd).astype(np.float32), mu, sd


def _cache_key(city: str, years, pixel_ids: np.ndarray, time_index: pd.DatetimeIndex) -> str:
    h = hashlib.sha1()
    h.update(str(city).encode())
    h.update(np.asarray([int(y) for y in years], dtype=np.int16).tobytes())
    h.update(np.asarray(pixel_ids, dtype=np.int64).tobytes())
    ti64 = pd.DatetimeIndex(time_index).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    h.update(np.asarray([len(ti64), int(ti64[0]), int(ti64[-1])], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _stats_key(stats) -> str:
    h = hashlib.sha1()
    for arr in stats:
        a = np.asarray(arr, dtype=np.float32)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()[:16]


def _era5_stats_cache_paths(city: str, years, pixel_ids: np.ndarray,
                            time_index: pd.DatetimeIndex):
    key = _cache_key(city, years, pixel_ids, time_index)
    cdir = CACHE_ROOT / "fm_covariates" / city
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir / f"era5_stats_{key}.npz", cdir / f"era5_stats_{key}.json", key


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    tmp.replace(path)


def _atomic_save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _load_or_build_era5_z(city: str, years, pixel_ids: np.ndarray,
                          time_index: pd.DatetimeIndex) -> np.ndarray:
    key = _cache_key(city, years, pixel_ids, time_index)
    cdir = CACHE_ROOT / "fm_covariates" / city
    cdir.mkdir(parents=True, exist_ok=True)
    arr_path = cdir / f"era5_z_{key}.npy"
    meta_path = cdir / f"era5_z_{key}.json"
    if arr_path.exists() and meta_path.exists():
        return np.load(arr_path, mmap_mode="r")
    print(f"    +meteo: loading ERA5 aligned to {len(time_index)} timestamps ...", flush=True)
    era = load_era5_aligned(city, years, pixel_ids, time_index)
    era_z, _ = standardize_era5(era)
    np.save(arr_path, era_z.astype(np.float32, copy=False))
    meta_path.write_text(json.dumps({
        "city": city,
        "years": [int(y) for y in years],
        "shape": list(map(int, era_z.shape)),
        "key": key,
    }, indent=2))
    return np.load(arr_path, mmap_mode="r")


def fit_covariate_stats(city: str, years, pixel_ids: np.ndarray,
                        time_index: pd.DatetimeIndex, config: str) -> dict:
    """Fit covariate normalization stats on stat_years; no FM weights are trained."""
    stats = {}
    if config in ("meteo", "meteo_static"):
        stat_path, meta_path, key = _era5_stats_cache_paths(city, years, pixel_ids, time_index)
        if stat_path.exists() and meta_path.exists():
            z = np.load(stat_path)
            stats["era5"] = (z["med"].astype(np.float32), z["mu"].astype(np.float32), z["sd"].astype(np.float32))
            print(f"    +meteo: ERA5 stats cache hit {stat_path.name}", flush=True)
        else:
            print(f"    +meteo: fitting ERA5 stats on {len(time_index)} stat timestamps ...", flush=True)
            stats["era5"] = fit_era5_stats_stream(city, years, pixel_ids)
            med, mu, sd = stats["era5"]
            _atomic_save_npz(stat_path, med=med, mu=mu, sd=sd)
            meta_path.write_text(json.dumps({
                "city": city,
                "years": [int(y) for y in years],
                "n_pixels": int(len(pixel_ids)),
                "n_timestamps": int(len(time_index)),
                "key": key,
            }, indent=2))
            print(f"    +meteo: ERA5 stats cache saved {stat_path.name}", flush=True)
    if config in ("static", "meteo_static"):
        print("    +static: fitting static stats ...", flush=True)
        _, stats["static"] = standardize_static(load_static_raw(city, pixel_ids))
    return stats


def _load_era5_z_with_stats(city: str, years, pixel_ids: np.ndarray,
                            time_index: pd.DatetimeIndex, stats=None) -> np.ndarray:
    if stats is None:
        return _load_or_build_era5_z(city, years, pixel_ids, time_index)
    eval_key = _cache_key(city, years, pixel_ids, time_index)
    stat_key = _stats_key(stats)
    cdir = CACHE_ROOT / "fm_covariates" / city
    cdir.mkdir(parents=True, exist_ok=True)
    arr_path = cdir / f"era5_z_{eval_key}_using_{stat_key}.npy"
    meta_path = cdir / f"era5_z_{eval_key}_using_{stat_key}.json"
    if arr_path.exists() and meta_path.exists():
        print(f"    +meteo: eval ERA5 z cache hit {arr_path.name}", flush=True)
        return np.load(arr_path, mmap_mode="r")
    print(f"    +meteo: loading ERA5 aligned to {len(time_index)} eval timestamps ...", flush=True)
    era = load_era5_aligned(city, years, pixel_ids, time_index)
    _standardize_era5_to_npy(era, arr_path, stats)
    meta_path.write_text(json.dumps({
        "city": city,
        "years": [int(y) for y in years],
        "shape": list(map(int, era.shape)),
        "eval_key": eval_key,
        "stats_key": stat_key,
    }, indent=2))
    print(f"    +meteo: eval ERA5 z cache saved {arr_path.name}", flush=True)
    return np.load(arr_path, mmap_mode="r")


def _standardize_era5_to_npy(arr: np.ndarray, out_path: Path, stats, chunk: int = 2048) -> None:
    """Standardize ERA5 in time chunks and write an npy cache atomically."""
    med, mu, sd = stats
    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    z = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=arr.shape)
    n = int(arr.shape[0])
    for s in range(0, n, int(chunk)):
        e = min(s + int(chunk), n)
        x = np.asarray(arr[s:e], dtype=np.float32)
        if np.isnan(x).any():
            x = np.where(np.isnan(x), med, x)
        y = ((x - mu) / sd).astype(np.float32, copy=False)
        z[s:e] = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if s == 0 or e == n or (e // int(chunk)) % 8 == 0:
            print(f"    +meteo: standardized eval ERA5 {e}/{n}", flush=True)
    z.flush()
    tmp.replace(out_path)


def build_covariates(city: str, years, pixel_ids: np.ndarray,
                     time_index: pd.DatetimeIndex, config: str,
                     stats: dict | None = None):
    """Return lazy covariate blocks and metadata for a four-config FM ablation."""
    blocks = {}
    sources = []
    stats = stats or {}
    if config in ("meteo", "meteo_static"):
        era_z = _load_era5_z_with_stats(city, years, pixel_ids, time_index, stats.get("era5"))
        blocks["era5"] = era_z.astype(np.float32, copy=False)
        sources.append("era5")
        print(f"    +meteo: {era_z.shape}", flush=True)
    if config in ("static", "meteo_static"):
        st_raw = load_static_raw(city, pixel_ids)
        st_z, _ = standardize_static(st_raw, stats.get("static"))
        blocks["static"] = st_z.astype(np.float32, copy=False)
        sources.append("static")
        print(f"    +static: {st_z.shape} -> lazy broadcast", flush=True)
    if not blocks:
        return None, {"cov_dim": 0, "covariates": []}
    cov_dim = int(sum(blocks[name].shape[-1] for name in sources))
    blocks["cov_dim"] = cov_dim
    blocks["covariates"] = sources
    blocks["shape"] = (len(time_index), len(pixel_ids), cov_dim)
    return blocks, {"cov_dim": cov_dim, "covariates": sources}


def _cov_dim(covs) -> int:
    if covs is None:
        return 0
    if isinstance(covs, dict):
        return int(covs["cov_dim"])
    return int(covs.shape[2])


def _cov_series(covs, t0: int, t1: int, p: int, ci: int) -> np.ndarray:
    if not isinstance(covs, dict):
        return covs[t0:t1, p, ci]
    era = covs.get("era5")
    if era is not None:
        n_era = int(era.shape[2])
        if ci < n_era:
            return era[t0:t1, p, ci]
        ci -= n_era
    st = covs.get("static")
    if st is not None:
        return np.full((t1 - t0,), st[p, ci], dtype=np.float32)
    raise IndexError(ci)


def _cov_window(covs, t0: int, t1: int, p0: int, p1: int) -> np.ndarray:
    if not isinstance(covs, dict):
        return covs[t0:t1, p0:p1, :]
    parts = []
    era = covs.get("era5")
    if era is not None:
        parts.append(era[t0:t1, p0:p1, :])
    st = covs.get("static")
    if st is not None:
        st_win = np.broadcast_to(st[None, p0:p1, :], (t1 - t0, p1 - p0, st.shape[1]))
        parts.append(st_win)
    return np.concatenate(parts, axis=2).astype(np.float32, copy=False)


def _cov_shape(covs):
    if isinstance(covs, dict):
        return covs["shape"]
    return covs.shape



def predict_chronos2_steps(values_z: np.ndarray, covs: np.ndarray | None,
                           starts, device: str, context: int = 24,
                           pixel_batch: int = 256) -> np.ndarray:
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    n_time, n_pixels = values_z.shape
    idx = synthetic_hourly_index(n_time)
    cov_names = [f"c{i}" for i in range(_cov_dim(covs))] if covs is not None else []
    starts = [int(s) for s in starts if int(s) >= context and int(s) < n_time]
    preds = np.full((len(starts), n_pixels), np.nan, dtype=np.float32)

    for si, t in enumerate(starts):
        ctx_t = idx[t - context:t]
        fut_t = idx[t:t + 1]
        for p0 in range(0, n_pixels, pixel_batch):
            ps = range(p0, min(p0 + pixel_batch, n_pixels))
            ctx_frames, fut_frames, meta = [], [], []
            for p in ps:
                sid = f"s{si}_p{p}"
                ctx = pd.DataFrame({
                    "id": sid,
                    "timestamp": ctx_t,
                    "target": fill_context(values_z[t - context:t, p]),
                })
                fut = pd.DataFrame({"id": sid, "timestamp": fut_t})
                for ci, cn in enumerate(cov_names):
                    ctx[cn] = _cov_series(covs, t - context, t, p, ci)
                    fut[cn] = _cov_series(covs, t, t + 1, p, ci)
                ctx_frames.append(ctx)
                fut_frames.append(fut)
                meta.append((sid, p))
            pred_df = pipe.predict_df(
                pd.concat(ctx_frames, ignore_index=True),
                future_df=pd.concat(fut_frames, ignore_index=True),
                prediction_length=1,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            pred_map = {
                sid: g.sort_values("timestamp")["0.5"].to_numpy(np.float32)
                for sid, g in pred_df.groupby("id", sort=False)
            }
            for sid, p in meta:
                yhat = pred_map.get(sid)
                if yhat is not None and len(yhat):
                    preds[si, p] = float(yhat[0])
        print(f"    chronos2 steps {si + 1}/{len(starts)}", flush=True)
    return preds


def predict_moirai2_steps(values_z: np.ndarray, covs: np.ndarray | None,
                          starts, device: str, context: int = 24,
                          pixel_batch: int = 256) -> np.ndarray:
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    cov_dim = _cov_dim(covs)
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    model = Moirai2Forecast(
        module=module,
        prediction_length=1,
        context_length=context,
        target_dim=1,
        feat_dynamic_real_dim=cov_dim,
        past_feat_dynamic_real_dim=0,
    ).to(device).eval()
    q_levels = np.asarray(model.module.quantile_levels, dtype=np.float32)
    q50 = int(np.argmin(np.abs(q_levels - 0.5)))

    n_time, n_pixels = values_z.shape
    starts = [int(s) for s in starts if int(s) >= context and int(s) < n_time]
    preds = np.full((len(starts), n_pixels), np.nan, dtype=np.float32)

    for si, t in enumerate(starts):
        for p0 in range(0, n_pixels, pixel_batch):
            p1 = min(p0 + pixel_batch, n_pixels)
            past = values_z[t - context:t, p0:p1].T.astype(np.float32)  # [B,W]
            past_np = np.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
            past_t = torch.from_numpy(past_np[:, :, None]).to(device)
            obs_t = torch.from_numpy(np.isfinite(past)[:, :, None]).to(device)
            pad_t = torch.zeros((past_t.shape[0], context), dtype=torch.bool, device=device)
            kw = {}
            if covs is not None:
                feat = _cov_window(covs, t - context, t + 1, p0, p1).transpose(1, 0, 2)
                feat_np = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                feat_t = torch.from_numpy(feat_np).to(device)
                kw["feat_dynamic_real"] = feat_t
                kw["observed_feat_dynamic_real"] = torch.ones_like(feat_t, dtype=torch.bool)
            with torch.no_grad():
                out = model(past_t, obs_t, pad_t, **kw)  # [B,Q,1]
            preds[si, p0:p1] = out[:, q50, 0].detach().cpu().numpy().astype(np.float32)
        print(f"    moirai2 steps {si + 1}/{len(starts)}", flush=True)
    return preds


def _pad_left_to_multiple(ctx: np.ndarray, multiple: int) -> np.ndarray:
    pad = (-len(ctx)) % multiple
    if pad == 0:
        return ctx.astype(np.float32, copy=False)
    fill = ctx[0] if np.isfinite(ctx[0]) else float(np.nanmean(ctx))
    if not np.isfinite(fill):
        fill = 0.0
    return np.concatenate([np.full(pad, fill, np.float32), ctx.astype(np.float32, copy=False)])


def predict_timesfm_steps(values_z: np.ndarray, covs: np.ndarray | None,
                          starts, device: str, context: int = 24,
                          pixel_batch: int = 256) -> np.ndarray:
    if covs is not None:
        raise ValueError("TimesFM does not support covariates here; use config=base")
    import timesfm

    patch_len = 32
    tfm_context = context + ((-context) % patch_len)
    backend = "gpu" if str(device).startswith("cuda") else "cpu"
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            context_len=tfm_context,
            horizon_len=1,
            input_patch_len=patch_len,
            backend=backend,
            per_core_batch_size=max(1, pixel_batch),
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    n_time, n_pixels = values_z.shape
    starts = [int(s) for s in starts if int(s) >= context and int(s) < n_time]
    preds = np.full((len(starts), n_pixels), np.nan, dtype=np.float32)

    for si, t in enumerate(starts):
        for p0 in range(0, n_pixels, pixel_batch):
            p1 = min(p0 + pixel_batch, n_pixels)
            series = [
                _pad_left_to_multiple(fill_context(values_z[t - context:t, p]), patch_len)
                for p in range(p0, p1)
            ]
            pred, _ = model.forecast(series, normalize=True)
            pred = np.asarray(pred, dtype=np.float32)
            preds[si, p0:p1] = pred[:, 0]
        print(f"    timesfm steps {si + 1}/{len(starts)}", flush=True)
    return preds


def predict_steps(fm: str, values_z: np.ndarray, covs: np.ndarray | None,
                  starts, device: str, context: int = 24,
                  pixel_batch: int = 256) -> np.ndarray:
    if fm == "chronos2":
        return predict_chronos2_steps(values_z, covs, starts, device, context, pixel_batch)
    if fm == "moirai2":
        return predict_moirai2_steps(values_z, covs, starts, device, context, pixel_batch)
    if fm == "timesfm":
        return predict_timesfm_steps(values_z, covs, starts, device, context, pixel_batch)
    raise ValueError(f"unsupported fm: {fm}")
