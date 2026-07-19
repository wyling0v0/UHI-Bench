"""Task 1b — Foundation Model zero-shot extreme-event DETECTION baselines.

Adds pretrained time-series FMs to the 1b-B detection table. The 1b task is
binary: predict whether hour t is an extreme UHI hour from precursor information
only (no peeking at the t value). Existing baselines (Percentile = lag-1
persistence + threshold; XGBoost; LSTM; ...) are trained or rule-based; here we
add **zero-shot** FMs as the top of the "information-source ladder" (pretrained
prior).

Protocol (aligned with run_1b_classify.py):
  * series    : city-MEAN hourly UHI scalar (Ta via HOSTRADA, LST via lstuhi_1km).
  * label     : label_extreme(series) over 2015-2025 (P95 + >=3h run), identical
                to the other baselines' ground truth.
  * split     : train <=2022 / test >=2023 (same masks).
  * FM detect : forecast yhat(t) 1-step-ahead from context s[t-168:t]; predict
                "extreme" iff yhat(t) > thr, where thr = train-period (<=2022)
                P95 of the *actual* UHI values. This is the direct FM analog of
                the Percentile(L1) persistence detector (same threshold, FM
                forecaster instead of lag-1 persistence) and uses no test leakage.
  * metrics   : clf_metrics -> F1 / MissRate / FAR, overall / day(7-17h) / night.

Layer fairness matches the rest of 1b:
  L1 = UHI history only (Chronos, TimesFM);
  L2 = +ERA5 six dynamic drivers as covariates (Chronos-2, MOIRAI-2).

Run with the FM venv (has chronos / timesfm / uni2ts):
  .venvs/uhi-fm/bin/python benchmark/1b/run_1b_fm.py --fm chronos  --device cuda:0
  .venvs/uhi-fm/bin/python benchmark/1b/run_1b_fm.py --fm timesfm  --device cuda:0
  .venvs/uhi-fm/bin/python benchmark/1b/run_1b_fm.py --fm chronos2 --device cuda:0
  .venvs/uhi-fm/bin/python benchmark/1b/run_1b_fm.py --fm moirai2  --device cuda:0

Four-config ablation (SAME covariate-capable FM across base/+meteo/+static/
+meteo+static; answers "does the same FM improve with each input?"):
  --fm chronos2 --config base
  --fm chronos2 --config meteo
  --fm chronos2 --config static
  --fm chronos2 --config meteo_static
  (likewise --fm moirai2). Static = city-mean Tier-1 morphology, broadcast as a
  constant dynamic covariate. Stored under keys "Chronos-2 (base/+meteo/...)".

Results are merged into 1b/results/1c_classify.json under new model keys.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import CACHE_ROOT, ERA5_BASE, LST_BASE, PSEUDO_TA_BASE, STATIC_BASE, TA_BASE  # noqa: E402
from run_1b import lst_city_series, ta_city_series, label_extreme  # noqa: E402

ERA5_CITY_CACHE = CACHE_ROOT / "1c_era5_city_mean"


def ta_series_local(city, years):
    """City-mean Ta-UHI per hour; reuse run_1b's v7-cache-aware loader."""
    return ta_city_series(city, years)


def lst_series_local(city, years):
    """City-mean LST-UHI per hour; reuse run_1b's v7-cache-aware loader."""
    return lst_city_series(city, years)


def era5_city_series(city, years):
    """City-mean ERA5 (6 drivers, no t2m) per hour — mirrors run_1b_classify."""
    import glob
    import pyarrow.parquet as pq
    years = [int(y) for y in years]
    ERA5_CITY_CACHE.mkdir(parents=True, exist_ok=True)
    cache = ERA5_CITY_CACHE / f"{city}_{'-'.join(map(str, years))}_{'-'.join(DRIVERS)}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        time_col = "datetime" if "datetime" in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        return df.set_index(time_col).sort_index()[DRIVERS]
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
            tbl = pq.read_table(f, columns=DRIVERS)
            cols = []
            for driver in DRIVERS:
                vals = tbl[driver].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                cols.append(vals.reshape(n_hours, n_pix).mean(axis=1))
            idx = pd.date_range(f"{y}-01-01", periods=n_hours, freq="h")
            parts.append(pd.DataFrame(np.stack(cols, axis=1), index=idx, columns=DRIVERS))
        else:
            df = pd.read_parquet(f, columns=["datetime", *DRIVERS])
            df["datetime"] = pd.to_datetime(df["datetime"])
            parts.append(df.groupby("datetime")[DRIVERS].mean())
    out = pd.concat(parts).groupby(level=0).mean().sort_index()
    out = out.rename_axis("datetime")
    out.reset_index().to_parquet(cache, index=False)
    return out


def static_city_mean(city: str, n_static: int = 10):
    """City-mean of the Tier-1 static morphology features -> [n_static] vector.

    Broadcast as a CONSTANT dynamic covariate for the covariate-capable FMs
    (Chronos-2 / MOIRAI-2): a per-city, time-invariant descriptor of urban form.
    Returns None if the static file is absent.
    """
    p = STATIC_BASE / city / "static_features.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    feats = np.asarray(d["features"], np.float32)[:, :n_static]
    if feats.size == 0:
        return None
    # International static rasters have some Tier-1 columns that are entirely
    # unavailable (all NaN). Keep the fixed 10-channel layout and collapse those
    # columns to 0, matching the GraphWaveNet/static covariate convention.
    finite = np.isfinite(feats)
    sums = np.where(finite, feats, 0.0).sum(axis=0, dtype=np.float64)
    counts = finite.sum(axis=0)
    vec = np.divide(sums, counts, out=np.zeros(n_static, dtype=np.float64),
                    where=counts > 0).astype(np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not np.isfinite(vec).any():
        return None
    return vec


def clf_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    miss = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    far = fp / (fp + tp) if (fp + tp) > 0 else float("nan")
    return {"F1": float(f1), "MissRate": float(miss), "FAR": float(far),
            "n_pos": int(y_true.sum()), "n": int(len(y_true))}

DRIVERS = ["u10", "v10", "tcc", "d2m", "blh", "ssrd"]
CITIES = ["berlin", "munich", "hamburg", "cairo", "lagos", "johannesburg"]
T_CTX = 168
H = 1
TRAIN_END = 2022
OUT = Path(__file__).parent / "results" / "1c_classify.json"

MODEL_KEY = {
    "chronos":  "Chronos(L1:no-met)",
    "timesfm":  "TimesFM(L1:no-met)",
    "chronos2": "Chronos-2(L2:+ERA5)",
    "moirai2":  "MOIRAI-2(L2:+ERA5)",
}


# ---------- helpers ----------
def fill_context(ctx: np.ndarray) -> np.ndarray:
    out = np.asarray(ctx, dtype=np.float32).copy()
    if np.isfinite(out).all():
        return out
    fin = np.isfinite(out)
    if fin.sum() == 0:
        return np.zeros_like(out, dtype=np.float32)
    out[~fin] = float(out[fin].mean())
    return out


def pad_left_to_multiple(ctx: np.ndarray, multiple: int) -> np.ndarray:
    pad = (-len(ctx)) % multiple
    if pad == 0:
        return ctx.astype(np.float32, copy=False)
    fill = ctx[0] if np.isfinite(ctx[0]) else float(np.nanmean(ctx))
    if not np.isfinite(fill):
        fill = 0.0
    return np.concatenate([np.full(pad, fill, np.float32), ctx.astype(np.float32, copy=False)])


def build_case(uhi: pd.Series, era5: pd.DataFrame | None,
               static_vec=None, config: str | None = None):
    """Return dict with filled series, masks, label, threshold, (optional) cov.

    config (four-config ablation, covariate-capable FMs only):
      None / "meteo"        : dynamic ERA5 drivers only  (legacy L2 behaviour)
      "base"                : NO covariates
      "static"              : static morphology only (constant over time)
      "meteo_static"        : ERA5 + static
    The assembled covariate block is stored under the legacy key ``era5_std``
    (shape [T, C]; None for the base config) so existing forecasters need only
    learn to tolerate None.
    """
    s_raw = uhi.sort_index()
    idx = s_raw.index
    vals = s_raw.values.astype(np.float32)
    med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 0.0
    s = np.where(np.isfinite(vals), vals, med).astype(np.float32)         # filled
    yrs = idx.year.values
    hrs = idx.hour.values
    tr = yrs <= TRAIN_END
    te = yrs >= TRAIN_END + 1
    # ground-truth label (full-data P95 + run), identical convention to classify
    lab, _ = label_extreme(s_raw)
    y = lab.values.astype(int)
    # detection threshold: train-period P95 of actual UHI (NaN-aware), no leakage
    thr = float(np.nanpercentile(vals[tr], 95))
    day_all = (hrs >= 7) & (hrs <= 17)
    # synthetic regular-hour index aligned to array positions (0..N-1). LST city-mean
    # series is irregular (cloud gaps) which Chronos-2's predict_df cannot handle
    # (it needs a fixed freq); mapping positions to a regular hourly grid keeps our
    # position-based windowing convention intact and gives a valid 'h' frequency.
    syn_idx = pd.date_range("2000-01-01", periods=len(s), freq="h")
    out = {"s": s, "vals": vals, "tr": tr, "te": te, "y": y,
           "day_te": day_all[te], "hrs": hrs, "thr": thr, "idx": idx,
           "syn_idx": syn_idx, "cov_dim": 0, "era5_std": None}

    inc_meteo = (config in (None, "meteo", "meteo_static")) and era5 is not None
    inc_static = config in ("static", "meteo_static") and static_vec is not None
    blocks = []
    if inc_meteo:
        e = era5.reindex(idx).fillna(era5.median()).to_numpy(dtype=np.float32)
        mu = e[tr].mean(axis=0, keepdims=True)
        sd = e[tr].std(axis=0, keepdims=True) + 1e-6
        blocks.append(((e - mu) / sd).astype(np.float32))
    if inc_static:
        sv = np.asarray(static_vec, np.float32)
        sv = (sv - sv.mean()) / (sv.std() + 1e-6)            # standardize the 10-vector
        blocks.append(np.tile(sv[None, :], (len(s), 1)).astype(np.float32))
    if blocks:
        cov = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
        out["era5_std"] = cov.astype(np.float32)
        out["cov_dim"] = int(cov.shape[1])
    return out


def te_indices(te: np.ndarray) -> np.ndarray:
    """All test hour positions with a full T_CTX context available before them."""
    pos = np.where(te)[0]
    return pos[pos >= T_CTX]


def detect_metrics(yhat: np.ndarray, t_idx: np.ndarray, case: dict) -> dict:
    thr = case["thr"]
    y_te = case["y"][t_idx]
    # day mask aligned to t_idx directly from hours
    day_t = ((case["hrs"][t_idx] >= 7) & (case["hrs"][t_idx] <= 17))
    pred = (yhat > thr).astype(int)
    return {"overall": clf_metrics(y_te, pred),
            "day":     clf_metrics(y_te[day_t], pred[day_t]),
            "night":   clf_metrics(y_te[~day_t], pred[~day_t]),
            "_n_pred_pos": int(pred.sum()), "_n": int(len(pred))}


# ---------- FM forecasters (return yhat over t_idx, 1-step-ahead) ----------
def forecast_chronos(case: dict, t_idx: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    import torch
    from chronos import ChronosBoltPipeline
    pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device)
    s = case["s"]
    preds = np.full(len(t_idx), np.nan, np.float32)
    for b0 in range(0, len(t_idx), batch):
        chunk = t_idx[b0:b0 + batch]
        ctx = np.stack([fill_context(s[tt - T_CTX:tt]) for tt in chunk])
        with torch.no_grad():
            out = pipe.predict(torch.from_numpy(ctx).float(), prediction_length=H)  # [B,Q,1]
        q = out.shape[1] // 2
        preds[b0:b0 + len(chunk)] = out[:, q, 0].detach().cpu().numpy().astype(np.float32)
        print(f"    chronos {b0 + len(chunk)}/{len(t_idx)}", flush=True)
    return preds


def forecast_timesfm(case: dict, t_idx: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    import torch
    import timesfm
    patch_len = 32
    tfm_ctx = T_CTX + ((-T_CTX) % patch_len)  # 192
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(context_len=tfm_ctx, horizon_len=H,
                                       input_patch_len=patch_len, backend="gpu",
                                       per_core_batch_size=batch),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    s = case["s"]
    preds = np.full(len(t_idx), np.nan, np.float32)
    for b0 in range(0, len(t_idx), batch):
        chunk = t_idx[b0:b0 + batch]
        series = [pad_left_to_multiple(fill_context(s[tt - T_CTX:tt]), patch_len)
                  for tt in chunk]
        pred, _ = model.forecast(series, normalize=True)
        pred = np.asarray(pred, np.float32)
        preds[b0:b0 + len(chunk)] = pred[:, 0]
        print(f"    timesfm {b0 + len(chunk)}/{len(t_idx)}", flush=True)
    return preds


def forecast_chronos2(case: dict, t_idx: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    s = case["s"]; e = case["era5_std"]; idx = case["syn_idx"]
    cov_names = [f"c{i}" for i in range(e.shape[1])] if e is not None else []
    preds = np.full(len(t_idx), np.nan, np.float32)
    for b0 in range(0, len(t_idx), batch):
        chunk = t_idx[b0:b0 + batch]
        ctx_frames, fut_frames, meta = [], [], []
        for j, tt in enumerate(chunk):
            sid = f"w{j}"
            ctx_t = idx[tt - T_CTX:tt]; fut_t = idx[tt:tt + H]
            ctx = pd.DataFrame({"id": sid, "timestamp": ctx_t, "target": s[tt - T_CTX:tt]})
            fut = pd.DataFrame({"id": sid, "timestamp": fut_t})
            for ci, cn in enumerate(cov_names):
                ctx[cn] = e[tt - T_CTX:tt, ci]; fut[cn] = e[tt:tt + H, ci]
            ctx_frames.append(ctx); fut_frames.append(fut); meta.append(sid)
        pred_df = pipe.predict_df(
            pd.concat(ctx_frames, ignore_index=True),
            future_df=pd.concat(fut_frames, ignore_index=True),
            prediction_length=H, quantile_levels=[0.1, 0.5, 0.9],
            id_column="id", timestamp_column="timestamp", target="target")
        pred_map = {sid: g.sort_values("timestamp")["0.5"].to_numpy(np.float32)
                    for sid, g in pred_df.groupby("id", sort=False)}
        for j, sid in enumerate(meta):
            yhat = pred_map.get(sid)
            if yhat is not None and len(yhat) >= H:
                preds[b0 + j] = float(yhat[0])
        print(f"    chronos2 {b0 + len(chunk)}/{len(t_idx)}", flush=True)
    return preds


def forecast_moirai2(case: dict, t_idx: np.ndarray, device: str, batch: int = 128) -> np.ndarray:
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    s = case["s"]; e = case["era5_std"]; cov_dim = case.get("cov_dim", 0) or 0
    model = Moirai2Forecast(module=module, prediction_length=H, context_length=T_CTX,
                            target_dim=1, feat_dynamic_real_dim=cov_dim,
                            past_feat_dynamic_real_dim=0).to(device).eval()
    preds = np.full(len(t_idx), np.nan, np.float32)
    for b0 in range(0, len(t_idx), batch):
        chunk = t_idx[b0:b0 + batch]
        past = np.stack([s[tt - T_CTX:tt] for tt in chunk])              # [B,168]
        past_np = np.nan_to_num(past, nan=0.0).astype(np.float32)
        past_t = torch.from_numpy(past_np[:, :, None]).to(device)
        obs_t = torch.ones_like(past_t, dtype=torch.bool)
        pad_t = torch.zeros((past_t.shape[0], T_CTX), dtype=torch.bool, device=device)
        kw = {}
        if e is not None:
            feat = np.stack([e[tt - T_CTX:tt + H] for tt in chunk])     # [B,169,C]
            feat_np = np.nan_to_num(feat, nan=0.0).astype(np.float32)
            feat_t = torch.from_numpy(feat_np).to(device)
            kw["feat_dynamic_real"] = feat_t
            kw["observed_feat_dynamic_real"] = torch.ones_like(feat_t, dtype=torch.bool)
        with torch.no_grad():
            out = model(past_t, obs_t, pad_t, **kw)                     # [B,Q,1]
        q_levels = np.asarray(model.module.quantile_levels, dtype=np.float32)
        q50 = int(np.argmin(np.abs(q_levels - 0.5)))
        preds[b0:b0 + len(chunk)] = out[:, q50, 0].detach().cpu().numpy().astype(np.float32)
        print(f"    moirai2 {b0 + len(chunk)}/{len(t_idx)}", flush=True)
    return preds


FORECASTERS = {"chronos": forecast_chronos, "timesfm": forecast_timesfm,
               "chronos2": forecast_chronos2, "moirai2": forecast_moirai2}


def already_done(city: str, mod: str, key: str, min_cov_dim: int = 0) -> bool:
    if not OUT.exists():
        return False
    try:
        d = json.loads(OUT.read_text())
    except Exception:
        return False
    row = d.get(city, {}).get(mod, {}).get(key)
    if row is None:
        return False
    if min_cov_dim > 0:
        return int(row.get("_cov_dim", 0) or 0) >= min_cov_dim
    return True


def merge_into(key_path, city, mod, key, metrics):
    d = json.loads(OUT.read_text())
    d.setdefault(city, {}).setdefault(mod, {})[key] = metrics
    OUT.write_text(json.dumps(d, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm", choices=list(MODEL_KEY), required=True)
    ap.add_argument("--config", choices=["base", "meteo", "static", "meteo_static"],
                    default=None,
                    help="four-config ablation (covariate-capable FMs chronos2/moirai2 "
                         "only): base=UHI only, meteo=+ERA5, static=+static, "
                         "meteo_static=+ERA5+static. Default None = legacy behaviour.")
    ap.add_argument("--cities", nargs="+", default=CITIES)
    ap.add_argument("--mods", nargs="+", default=["Ta", "LST"])
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2015, 2026)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every Nth test hour (1=all; >1 subsamples for "
                         "expensive covariate FMs, documented in output)")
    ap.add_argument("--batch", type=int, default=None,
                    help="override FM batch size without changing the eval set")
    a = ap.parse_args()
    try:
        import torch
        if a.device.startswith("cuda") and not torch.cuda.is_available():
            a.device = "cpu"; print("[device] CUDA unavailable -> CPU", flush=True)
    except Exception:
        a.device = "cpu"

    if a.config is not None and a.fm not in {"chronos2", "moirai2"}:
        raise SystemExit(f"--config only applies to covariate-capable FMs "
                         f"(chronos2/moirai2); {a.fm} has no covariate channel.")

    _CFG_LABEL = {"base": "base", "meteo": "+meteo", "static": "+static",
                  "meteo_static": "+meteo+static"}
    if a.config is not None:
        base_name = MODEL_KEY[a.fm].split("(")[0].strip()      # "Chronos-2" / "MOIRAI-2"
        key = f"{base_name} ({_CFG_LABEL[a.config]})"
        need_era5 = a.config in ("meteo", "meteo_static")
        need_static = a.config in ("static", "meteo_static")
    else:
        key = MODEL_KEY[a.fm]
        need_era5 = a.fm in {"chronos2", "moirai2"}
        need_static = False
    fcst = FORECASTERS[a.fm]
    min_cov_dim = (6 if need_era5 else 0) + (10 if need_static else 0)
    print(f"[1b-fm] fm={a.fm} config={a.config} key='{key}' device={a.device} "
          f"era5={need_era5} static={need_static}", flush=True)

    for city in a.cities:
        static_vec = static_city_mean(city) if need_static else None
        if need_static and static_vec is None:
            print(f"  [skip] {city}: no usable static features for {key}", flush=True)
            continue
        for mod in a.mods:
            if already_done(city, mod, key, min_cov_dim=min_cov_dim):
                print(f"  [skip-existing] {city} {mod} {key}", flush=True)
                continue
            print(f"  [load] {city} {mod}", flush=True)
            loader = ta_series_local if mod == "Ta" else lst_series_local
            uhi = loader(city, a.years)
            if uhi is None:
                print(f"  [skip] {city} {mod}: no series", flush=True); continue
            era5 = era5_city_series(city, a.years) if need_era5 else None
            case = build_case(uhi, era5, static_vec=static_vec, config=a.config)
            if case["cov_dim"] < min_cov_dim:
                print(f"  [skip] {city} {mod} {key}: cov_dim={case['cov_dim']} "
                      f"< expected {min_cov_dim}", flush=True)
                continue
            t_idx = te_indices(case["te"])
            if a.stride > 1:
                t_idx = t_idx[::a.stride]
            print(f"  [forecast] {city} {mod} {key}: n={len(t_idx)} "
                  f"cov_dim={case['cov_dim']} batch={a.batch or 'default'}", flush=True)
            if a.batch is None:
                yhat = fcst(case, t_idx, a.device)
            else:
                yhat = fcst(case, t_idx, a.device, batch=a.batch)
            metrics = detect_metrics(yhat, t_idx, case)
            metrics["_cov_dim"] = int(case["cov_dim"])
            metrics["_config"] = a.config or "legacy"
            metrics["_fm"] = a.fm
            merge_into(OUT, city, mod, key, metrics)
            o = metrics["overall"]
            print(f"  {city} {mod} {key}: F1={o['F1']:.3f} Miss={o['MissRate']:.3f} "
                  f"FAR={o['FAR']:.3f} pred_pos={metrics['_n_pred_pos']}/{metrics['_n']} "
                  f"thr={case['thr']:.3f} cov_dim={case['cov_dim']}", flush=True)
    print(f"\n[merged] {key} -> {OUT}")


if __name__ == "__main__":
    main()
