"""Task 2b — FM reconstruction baselines.

This is the Air-T sparse-station counterpart of 1a/run_1a_fm_covariate.py. It
uses the current 1b page axis (four random missing% bins) rather than the legacy
Chronos keep-fraction axis. Chronos-2 / MOIRAI-2 support the four-config
ablation:

Configs:
  base         : Ta-UHI history only
  meteo        : + per-pixel ERA5 dynamic drivers
  static       : + Tier-1 static morphology
  meteo_static : + ERA5 + static

Results are merged into ``results/1b_{city}_fm.json``.

Run with the FM venv:
  .venvs/uhi-fm/bin/python benchmark/2b/run_1b_fm_covariate.py --fm timesfm  --config base
  .venvs/uhi-fm/bin/python benchmark/2b/run_1b_fm_covariate.py --fm chronos2 --config meteo_static
  .venvs/uhi-fm/bin/python benchmark/2b/run_1b_fm_covariate.py --fm moirai2  --config meteo_static
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import Field, _load_static, load_ta_field, ta_cache_root  # noqa: E402
from common.fm_impute_covariates import (  # noqa: E402
    CONFIG_CHOICES,
    FM_CHOICES,
    build_covariates,
    fit_covariate_stats,
    method_key,
    predict_steps,
    resolve_device,
)
from common.masks import BIN_LABELS, random_mask_for_bin  # noqa: E402

W = 24


def align_to_pixel_ids(anchor: Field, other: Field) -> Field:
    if np.array_equal(anchor.pixel_ids, other.pixel_ids):
        return other
    pos = {int(p): i for i, p in enumerate(other.pixel_ids)}
    order = np.asarray([pos[int(p)] for p in anchor.pixel_ids], dtype=np.int64)
    return other._replace(
        values=np.ascontiguousarray(other.values[:, order]),
        xy_km=other.xy_km[order],
        feats=other.feats[order],
        pixel_ids=other.pixel_ids[order],
    )


def load_ta_field_fast(city: str, years) -> Field:
    """HOSTRADA fast path for the FM venv.

    This mirrors the cache loader used by run_1b_meteo.py but avoids importing
    that module, because the FM venv intentionally does not carry XGBoost.
    """
    years = [int(y) for y in years]
    cache_root = ta_cache_root(city)
    derived = cache_root / ("bench_1b_field_" + "_".join(map(str, years)))
    if ((derived / "done.json").exists() and (derived / "values.npy").exists()
            and (derived / "times.npy").exists() and (derived / "xy.npy").exists()
            and (derived / "pixel_ids.npy").exists()):
        values = np.load(derived / "values.npy", mmap_mode="r")
        xy_m = np.asarray(np.load(derived / "xy.npy", mmap_mode="r"), dtype=np.float64)
        pixel_ids = np.asarray(np.load(derived / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64).ravel()
        dt = pd.to_datetime(np.load(derived / "times.npy", mmap_mode="r"), unit="s").to_numpy()
        static_xy, feats, names, _ = _load_static(city)
        srow = {tuple(map(float, xy)): i for i, xy in enumerate(static_xy)}
        order = np.asarray([srow[tuple(map(float, xy))] for xy in xy_m], dtype=np.int64)
        return Field(values, (xy_m / 1000.0).astype(np.float32), feats[order],
                     names, dt, pixel_ids)

    cache_dirs = [cache_root / f"v7_{y}" for y in years]
    if not all((d / "done.json").exists() and (d / "uhi.npy").exists()
               and (d / "times.npy").exists() and (d / "xy.npy").exists()
               and (d / "pixel_ids.npy").exists()
               for d in cache_dirs):
        return load_ta_field(city, years=years)

    derived.mkdir(parents=True, exist_ok=True)
    shapes = [np.load(d / "uhi.npy", mmap_mode="r").shape for d in cache_dirs]
    total_t = int(sum(s[0] for s in shapes))
    n_pix = int(shapes[0][1])
    values_mm = np.lib.format.open_memmap(
        derived / "values.npy", mode="w+", dtype=np.float32, shape=(total_t, n_pix)
    )
    times_parts = []
    xy_m = np.asarray(np.load(cache_dirs[0] / "xy.npy", mmap_mode="r"), dtype=np.float64)
    pixel_ids = np.asarray(np.load(cache_dirs[0] / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64).ravel()
    off = 0
    for d in cache_dirs:
        uhi = np.load(d / "uhi.npy", mmap_mode="r")
        nt = uhi.shape[0]
        print(f"    [cache] building {derived.name}: {d.name} ({nt}h)", flush=True)
        values_mm[off:off + nt] = uhi[:, :, 1]
        off += nt
        times_parts.append(np.asarray(np.load(d / "times.npy", mmap_mode="r"), dtype=np.int64))
    values_mm.flush()
    times_raw = np.concatenate(times_parts)
    np.save(derived / "times.npy", times_raw)
    np.save(derived / "xy.npy", xy_m.astype(np.float32))
    np.save(derived / "pixel_ids.npy", pixel_ids)
    (derived / "done.json").write_text(json.dumps(
        {"city": city, "years": years, "shape": [total_t, n_pix], "source": "hostrada_v7"},
        indent=2,
    ))
    values = np.load(derived / "values.npy", mmap_mode="r")
    dt = pd.to_datetime(times_raw, unit="s").to_numpy()

    static_xy, feats, names, _ = _load_static(city)
    srow = {tuple(map(float, xy)): i for i, xy in enumerate(static_xy)}
    order = np.asarray([srow[tuple(map(float, xy))] for xy in xy_m], dtype=np.int64)
    return Field(values, (xy_m / 1000.0).astype(np.float32), feats[order],
                 names, dt, pixel_ids)


def choose_eval_times(values: np.ndarray, n_times: int, rng: np.random.Generator) -> np.ndarray:
    cov = np.isnan(values).mean(axis=1)
    full_idx = np.where(cov < 0.01)[0]
    full_idx = full_idx[full_idx >= W]
    if len(full_idx) > n_times:
        full_idx = rng.choice(full_idx, n_times, replace=False)
    return np.asarray(full_idx, dtype=np.int64)


def subset_cache_dir(city: str, years, n_pix: int, seed: int) -> Path:
    years_key = "_".join(map(str, [int(y) for y in years]))
    return ta_cache_root(city) / f"bench_1b_fm_subset_{years_key}_n{n_pix}_seed{int(seed)}"


def select_values_cached(field, city: str, years, pix: np.ndarray, seed: int):
    cache_dir = subset_cache_dir(city, years, len(pix), seed)
    if ((cache_dir / "done.json").exists() and (cache_dir / "values.npy").exists()
            and (cache_dir / "pix.npy").exists() and (cache_dir / "pixel_ids.npy").exists()):
        old_pix = np.load(cache_dir / "pix.npy")
        if np.array_equal(old_pix, pix):
            values = np.load(cache_dir / "values.npy", mmap_mode="r")
            pixel_ids = np.asarray(np.load(cache_dir / "pixel_ids.npy"), dtype=np.int64).ravel()
            return values, pixel_ids
    cache_dir.mkdir(parents=True, exist_ok=True)
    values_mm = np.lib.format.open_memmap(
        cache_dir / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(field.values.shape[0], len(pix)),
    )
    chunk = 1024
    for s in range(0, field.values.shape[0], chunk):
        values_mm[s:s + chunk] = np.asarray(field.values[s:s + chunk, :][:, pix], dtype=np.float32)
    values_mm.flush()
    pixel_ids = np.asarray(field.pixel_ids[pix], dtype=np.int64)
    np.save(cache_dir / "pix.npy", pix.astype(np.int64))
    np.save(cache_dir / "pixel_ids.npy", pixel_ids)
    (cache_dir / "done.json").write_text(json.dumps({
        "city": city,
        "years": [int(y) for y in years],
        "shape": [int(field.values.shape[0]), int(len(pix))],
        "seed": int(seed),
    }, indent=2))
    return np.load(cache_dir / "values.npy", mmap_mode="r"), pixel_ids


def standardize_values_cached(city: str, years, values: np.ndarray, seed: int):
    cache_dir = subset_cache_dir(city, years, values.shape[1], seed)
    z_path = cache_dir / "values_z.npy"
    mu_path = cache_dir / "mu.npy"
    sd_path = cache_dir / "sd.npy"
    if z_path.exists() and mu_path.exists() and sd_path.exists():
        return (np.load(z_path, mmap_mode="r"),
                np.load(mu_path),
                np.load(sd_path))

    arr = np.asarray(values, dtype=np.float32)
    mu = arr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = (arr.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0).astype(np.float32)
    z = ((arr - mu) / sd).astype(np.float32)
    np.save(z_path, z)
    np.save(mu_path, mu)
    np.save(sd_path, sd)
    return np.load(z_path, mmap_mode="r"), mu, sd


def standardize_with_stats(values: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float32) - mu) / sd).astype(np.float32)


def run_city(args) -> dict:
    rng_pix = np.random.default_rng(args.seed)
    rng_eval = np.random.default_rng(args.seed)
    print(f"\n{'=' * 60}\n[1b-fm] city={args.city} fm={args.fm} config={args.config}", flush=True)
    stat_fld = load_ta_field_fast(args.city, years=args.stat_years)
    fld = align_to_pixel_ids(stat_fld, load_ta_field_fast(args.city, years=args.years))
    pix = rng_pix.choice(stat_fld.values.shape[1], min(args.n_pixels, stat_fld.values.shape[1]), replace=False)
    pix.sort()
    stat_values, pixel_ids = select_values_cached(stat_fld, args.city, args.stat_years, pix, args.seed)
    values, pixel_ids = select_values_cached(fld, args.city, args.years, pix, args.seed)
    n_pixels = len(pix)
    print(f"    stat={stat_values.shape} eval={values.shape} pixels={n_pixels}", flush=True)
    _, mu, sd = standardize_values_cached(args.city, args.stat_years, stat_values, args.seed)
    values_z = standardize_with_stats(values, mu, sd)

    cov_stats = fit_covariate_stats(
        args.city,
        args.stat_years,
        pixel_ids,
        pd.DatetimeIndex(stat_fld.times),
        args.config,
    )
    covs, cov_meta = build_covariates(
        args.city,
        args.years,
        pixel_ids,
        pd.DatetimeIndex(fld.times),
        args.config,
        stats=cov_stats,
    )
    times = choose_eval_times(values, args.n_times, rng_eval)
    print(
        f"    eval ts={len(times)} splits={args.n_splits} bins={BIN_LABELS} cov_dim={cov_meta['cov_dim']}",
        flush=True,
    )
    pred_z = predict_steps(
        args.fm,
        values_z,
        covs,
        times,
        args.device,
        context=W,
        pixel_batch=args.pixel_batch,
    )

    bin_ae = {b: [] for b in range(len(BIN_LABELS))}
    for si, ti in enumerate(times):
        pred = pred_z[si] * sd + mu
        gt = values[ti]
        if not np.isfinite(gt).all():
            gt = np.where(np.isfinite(gt), gt, np.nanmean(gt))
        for _ in range(args.n_splits):
            for b in range(len(BIN_LABELS)):
                hide = np.where(random_mask_for_bin(n_pixels, b, rng_eval))[0]
                if len(hide) > args.max_pred:
                    hide = rng_eval.choice(hide, args.max_pred, replace=False)
                valid = np.isfinite(pred[hide]) & np.isfinite(gt[hide])
                if valid.any():
                    bin_ae[b].append(np.abs(pred[hide][valid] - gt[hide][valid]))

    key = method_key(args.fm, args.config)
    out = {
        "city": args.city,
        "model": "Covariate FM",
        "n_pixels": n_pixels,
        "bins": BIN_LABELS,
        "axis": "missing%",
        "protocol": {
            "years": [int(y) for y in args.years],
            "eval_years": [int(y) for y in args.years],
            "stat_years": [int(y) for y in args.stat_years],
            "mode": "zero-shot, train-period normalization",
            "n_times": int(len(times)),
            "n_splits": int(args.n_splits),
            "max_pred": int(args.max_pred),
            "seed": int(args.seed),
            "context": W,
        },
        "methods": {key: {}},
        "configs": {key: {"fm": args.fm, "config": args.config, **cov_meta}},
    }
    for b in range(len(BIN_LABELS)):
        ae = np.concatenate(bin_ae[b]) if bin_ae[b] else np.array([], dtype=np.float32)
        out["methods"][key][str(b)] = {
            "MAE": float(ae.mean()) if ae.size else None,
            "RMSE": float(np.sqrt((ae ** 2).mean())) if ae.size else None,
            "N": int(ae.size),
        }
        print(f"    {key} {BIN_LABELS[b]}: MAE={out['methods'][key][str(b)]['MAE']}", flush=True)
    return out


def merge_result(out_path: Path, new: dict) -> dict:
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        prev_protocol = prev.get("protocol", {})
        new_protocol = new.get("protocol", {})
        method_protocols = prev.setdefault("method_protocols", {})
        for old_key in prev.get("methods", {}):
            method_protocols.setdefault(old_key, prev_protocol)
        if prev_protocol != new_protocol:
            prev.setdefault("protocol_conflicts", []).append({
                "kept_existing_methods": sorted(prev.get("methods", {}).keys()),
                "added_methods": sorted(new.get("methods", {}).keys()),
                "existing_top_level_protocol": prev_protocol,
                "new_protocol": new_protocol,
            })
        prev.setdefault("methods", {}).update(new.get("methods", {}))
        prev.setdefault("configs", {}).update(new.get("configs", {}))
        for new_key in new.get("methods", {}):
            method_protocols[new_key] = new_protocol
        if prev_protocol == new_protocol:
            prev["protocol"] = new_protocol
        else:
            prev.setdefault("protocol_notes", {})["top_level_protocol"] = (
                "Per-method protocols are stored in method_protocols; older "
                "methods are preserved instead of being overwritten."
            )
        prev["n_pixels"] = new.get("n_pixels", prev.get("n_pixels"))
        return prev
    return new


def method_exists(out_path: Path, key: str) -> bool:
    if not out_path.exists():
        return False
    try:
        data = json.loads(out_path.read_text())
    except Exception:
        return False
    return key in data.get("methods", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm", choices=FM_CHOICES, required=True)
    ap.add_argument("--config", choices=CONFIG_CHOICES, required=True)
    ap.add_argument("--city", default="munich")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--stat_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--n_pixels", type=int, default=512)
    ap.add_argument("--n_times", type=int, default=30)
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--max_pred", type=int, default=500)
    ap.add_argument("--pixel_batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    if args.fm == "timesfm" and args.config != "base":
        raise SystemExit("TimesFM has no covariate channel here; use --config base")
    args.device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"1b_{args.city}_fm.json"
    key = method_key(args.fm, args.config)
    if method_exists(out_path, key):
        print(f"[skip-existing] {args.city} {key} -> {out_path}", flush=True)
        return

    res = run_city(args)
    merged = merge_result(out_path, res)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
