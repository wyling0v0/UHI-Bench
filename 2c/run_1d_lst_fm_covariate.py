"""Task 2c — LST-UHI covariate foundation models (Chronos-2 / MOIRAI-2), +ERA5.

LST adaptation of run_1d_fm_covariate.py (which is Ta-cache-specific). Reuses the
Ta module's FM helpers (Chronos-2 predict_df, MOIRAI-2 feat_dynamic_real) but
loads LST-UHI + ERA5 via the Task-3 clear-window loader (load_city_year). LST
cloud NaN in the context is filled (Chronos-2) or masked (MOIRAI-2); targets with
NaN are skipped by add_errors. Also runs no-covariate Chronos for a fair delta.

Usage:
  .venvs/uhi-fm/bin/python benchmark/2c/run_1d_lst_fm_covariate.py --model chronos2 --city munich
  .venvs/uhi-fm/bin/python benchmark/2c/run_1d_lst_fm_covariate.py --model moirai2  --city munich
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                  # benchmark/2c (sibling Ta module)
sys.path.insert(0, str(HERE.parents[0]))       # benchmark (common/)
sys.path.insert(0, str(HERE.parents[0] / "3")) # benchmark/3 (Task3 loader)
import run_1d_fm_covariate as ta          # reuse FM helpers
import run_3_ood_transfer as T3           # LST + ERA5 loader

H = ta.H; T_CTX = ta.T_CTX; HORIZONS = ta.HORIZONS
CONFIGS = ta.CONFIGS
CFG_LABEL = ta.CFG_LABEL
FM_LABEL = ta.FM_LABEL
OUT = HERE / "results" / "1d_forecast.json"


def load_lst(city, years, n_pixels, seed):
    """Concatenate per-year LST + ERA5 (aligned) via Task3 loader."""
    parts_v, parts_c, parts_t = [], [], []
    static = None
    for y in years:
        d = T3.load_city_year(city, y, n_pixels, seed)
        parts_v.append(d.values); parts_c.append(d.era5); parts_t.append(np.asarray(d.times))
        if static is None:
            static = d.static.astype(np.float32)
    V = np.concatenate(parts_v, axis=0).astype(np.float32)
    C = np.concatenate(parts_c, axis=0).astype(np.float32)
    times = np.concatenate(parts_t, axis=0)
    assert V.shape[0] == len(times) == C.shape[0], (V.shape, C.shape, len(times))
    return V, C, static, times


def choose_pixels(city, n_pixels, seed):
    return T3.choose_pixels(city, n_pixels, seed)


def build_covariates(city, train_years, test_years, n_pixels, seed, config):
    blocks = []
    if config in ("meteo", "meteo_static"):
        _, cov_train, _, _ = load_lst(city, train_years, n_pixels, seed)
        cov_mu, cov_sd = ta.fit_cov_stats(cov_train)
        _, cov_test, _, _ = load_lst(city, test_years, n_pixels, seed)
        blocks.append(ta.standardize_cov(cov_test, cov_mu, cov_sd))
    if config in ("static", "meteo_static"):
        if blocks:
            n_time = blocks[0].shape[0]
            n_pix = blocks[0].shape[1]
            static = load_lst(city, train_years[:1], n_pixels, seed)[2]
        else:
            v_test, _, static, _ = load_lst(city, test_years, n_pixels, seed)
            n_time, n_pix = v_test.shape
        st_mu, st_sd = ta.fit_static_stats(static)
        st_z = ta.standardize_cov(static, st_mu, st_sd)
        blocks.append(np.broadcast_to(st_z[None, :, :], (n_time, n_pix, st_z.shape[1])).astype(np.float32))
    if not blocks:
        return None
    return np.concatenate(blocks, axis=2).astype(np.float32, copy=False)


def run_chronos2_lst(values, covs, times, starts, n_pixels, window_batch, device):
    """Chronos-2 with past+future ERA5 covariates; LST context NaN filled."""
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    errs = {h: [] for h in HORIZONS}
    cov_names = [f"c{i}" for i in range(covs.shape[-1])] if covs is not None else []
    for b0 in range(0, len(starts), window_batch):
        batch_starts = starts[b0:b0 + window_batch]
        ctx_frames, fut_frames, id_meta = [], [], []
        for wi, s in enumerate(batch_starts):
            ctx_t = times[s:s + T_CTX]; fut_t = times[s + T_CTX:s + T_CTX + H]
            for p in range(n_pixels):
                sid = f"w{b0 + wi}_p{p}"
                ctx = pd.DataFrame({"timestamp": ctx_t,
                                    "target": ta.fill_context(values[s:s + T_CTX, p])})
                ctx["id"] = sid
                fut = pd.DataFrame({"timestamp": fut_t}); fut["id"] = sid
                for ci, cn in enumerate(cov_names):
                    ctx[cn] = covs[s:s + T_CTX, p, ci]
                    fut[cn] = covs[s + T_CTX:s + T_CTX + H, p, ci]
                ctx_frames.append(ctx); fut_frames.append(fut); id_meta.append((sid, s, p))
        pred_df = pipe.predict_df(pd.concat(ctx_frames, ignore_index=True),
                                  future_df=pd.concat(fut_frames, ignore_index=True),
                                  prediction_length=H, quantile_levels=[0.1, 0.5, 0.9],
                                  id_column="id", timestamp_column="timestamp", target="target")
        pred_map = {sid: g.sort_values("timestamp")["0.5"].to_numpy(np.float32)
                    for sid, g in pred_df.groupby("id", sort=False)}
        preds, true = [], []
        for sid, s, p in id_meta:
            yhat = pred_map.get(sid)
            if yhat is None or len(yhat) < H:
                continue
            preds.append(yhat[:H]); true.append(values[s + T_CTX:s + T_CTX + H, p])
        if preds:
            ta.add_errors(errs, np.stack(preds), np.stack(true))
        print(f"  chronos2 windows {min(b0 + window_batch, len(starts))}/{len(starts)}", flush=True)
    return errs


def result_name(model, config, n_pixels):
    return {"chronos": f"Chronos(base,{n_pixels}px weekly subset)",
            "timesfm": f"TimesFM(base,{n_pixels}px weekly subset,padded)",
            "chronos2": f"Chronos-2({CFG_LABEL[config]},{n_pixels}px weekly subset)",
            "moirai2": f"MOIRAI-2({CFG_LABEL[config]},{n_pixels}px weekly subset)"}[model]


def row_exists(city, name):
    if not OUT.exists():
        return False
    try:
        data = json.loads(OUT.read_text())
    except Exception:
        return False
    return name in data.get("LST", {}).get(city.capitalize(), {})


def merge_row(city, model, n_pixels, row):
    name = result_name(model, ARGS.config, n_pixels)
    data = json.loads(OUT.read_text())
    data.setdefault("LST", {}).setdefault(city.capitalize(), {})[name] = row
    data.setdefault("protocol_notes", {})["lst_fm_four_config"] = {
        "train_years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
        "test_years": [2023, 2024, 2025],
        "n_pixels": n_pixels,
        "stride": "weekly by default",
        "note": "FM weights are zero-shot; covariate normalization uses train_years only.",
    }
    OUT.write_text(json.dumps(data, indent=2))
    print(f"[merged] {name} {city} -> {OUT}")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["chronos", "timesfm", "chronos2", "moirai2"], required=True)
    ap.add_argument("--config", choices=CONFIGS, default="base")
    ap.add_argument("--city", default="munich")
    ap.add_argument("--train_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--test_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_pixels", type=int, default=64)
    ap.add_argument("--stride", type=int, default=168)
    ap.add_argument("--window_batch", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ARGS = a
    if a.model in {"chronos", "timesfm"} and a.config != "base":
        raise SystemExit(f"{a.model} supports only --config base in this runner")
    name = result_name(a.model, a.config, a.n_pixels)
    if row_exists(a.city, name):
        print(f"[skip-existing] LST {a.city.capitalize()} {name} -> {OUT}", flush=True)
        return

    pix = choose_pixels(a.city, a.n_pixels, a.seed)
    print(f"[fm-lst] model={a.model} config={a.config} city={a.city} pixels={len(pix)}", flush=True)
    values, _, _, times = load_lst(a.city, a.test_years, a.n_pixels, a.seed)
    covs = build_covariates(a.city, a.train_years, a.test_years, a.n_pixels, a.seed, a.config) if a.model in {"chronos2", "moirai2"} else None
    starts = ta.window_starts(values.shape[0], a.stride)
    cov_msg = f" cov={covs.shape}" if covs is not None else ""
    print(f"[fm-lst] test={values.shape}{cov_msg} windows={len(starts)} stride={a.stride}", flush=True)

    if a.model == "chronos":
        errs = ta.run_chronos(values, starts, len(pix), a.window_batch, a.device)
    elif a.model == "timesfm":
        errs = ta.run_timesfm(values, starts, len(pix), a.window_batch)
    elif a.model == "chronos2":
        errs = run_chronos2_lst(values, covs, times, starts, len(pix), a.window_batch, a.device)
    else:
        errs = ta.run_moirai2(values, covs, starts, len(pix), a.window_batch, a.device)

    row = {}
    for h in HORIZONS:
        ae = np.concatenate(errs[h]) if errs[h] else np.array([], dtype=np.float32)
        row[f"{h}h"] = float(ae.mean()) if ae.size else None
        print(f"  +{h}h: MAE={row[f'{h}h']:.4f}" if ae.size else f"  +{h}h: MAE=None")
    merge_row(a.city, a.model, len(pix), row)


if __name__ == "__main__":
    main()
