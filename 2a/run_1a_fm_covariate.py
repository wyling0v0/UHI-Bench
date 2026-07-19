"""Task 2a — FM reconstruction baselines.

Runs the same cloud-gap reconstruction protocol as the 1a page, but only for
forecasting FMs. Chronos-2 / MOIRAI-2 support the four-config ablation:

  base         : LST-UHI history only
  meteo        : + per-pixel ERA5 dynamic drivers
  static       : + Tier-1 static morphology
  meteo_static : + ERA5 + static

Results are merged into ``results/1a_{city}_fm.json`` so they can sit beside the
base ML/DL rows without overwriting the legacy Chronos file.

Run with the FM venv:
  .venvs/uhi-fm/bin/python benchmark/2a/run_1a_fm_covariate.py --fm timesfm  --config base
  .venvs/uhi-fm/bin/python benchmark/2a/run_1a_fm_covariate.py --fm chronos2 --config meteo_static
  .venvs/uhi-fm/bin/python benchmark/2a/run_1a_fm_covariate.py --fm moirai2  --config meteo_static
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib import format as np_format

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import Field, LST_BASE, _load_static, load_lst_field  # noqa: E402
from common.fm_impute_covariates import (  # noqa: E402
    CONFIG_CHOICES,
    FM_CHOICES,
    build_covariates,
    fit_covariate_stats,
    method_key,
    predict_steps,
    resolve_device,
    standardize_target,
)
from common.masks import BIN_LABELS, build_cloud_eval_pairs  # noqa: E402

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


def _cache_times_to_datetime(times: np.ndarray) -> np.ndarray:
    if np.issubdtype(times.dtype, np.datetime64):
        return times.astype("datetime64[ns]")
    return pd.to_datetime(times, unit="s").to_numpy()


def lst_cache_path(city: str, year: int) -> Path:
    cache = LST_BASE / city / "cache"
    p = cache / f"v6_{int(year)}.npz"
    return p if p.exists() else cache / f"v6b_{int(year)}.npz"


def _npz_npy_shape(path: Path, member: str = "uhi_arr.npy") -> tuple[int, ...]:
    with zipfile.ZipFile(path) as z:
        with z.open(member) as f:
            version = np_format.read_magic(f)
            reader = np_format.read_array_header_1_0 if version == (1, 0) else np_format.read_array_header_2_0
            shape, _, _ = reader(f)
            return tuple(int(x) for x in shape)


def load_lst_field_fast(city: str, years) -> Field:
    """Prefer annual LST-UHI cache, fallback to the parquet pivot loader."""
    years = [int(y) for y in years]
    cache_root = LST_BASE / city / "cache"
    derived = cache_root / ("bench_1a_field_" + "_".join(map(str, years)))
    if ((derived / "done.json").exists() and (derived / "values.npy").exists()
            and (derived / "times.npy").exists() and (derived / "pixel_ids.npy").exists()
            and (derived / "xy_km.npy").exists()):
        values = np.load(derived / "values.npy", mmap_mode="r")
        times = np.load(derived / "times.npy", mmap_mode="r").astype("datetime64[ns]")
        pixel_ids = np.asarray(np.load(derived / "pixel_ids.npy", mmap_mode="r"), dtype=np.int64).ravel()
        xy_km = np.asarray(np.load(derived / "xy_km.npy", mmap_mode="r"), dtype=np.float32)
        xy_m, feats, names, spids = _load_static(city)
        pid2row = {int(pid): i for i, pid in enumerate(np.asarray(spids, dtype=np.int64).ravel())}
        order = np.asarray([pid2row[int(pid)] for pid in pixel_ids], dtype=np.int64)
        return Field(values, xy_km, feats[order], names, times, pixel_ids)

    if not all(lst_cache_path(city, y).exists() for y in years):
        return load_lst_field(city, years=years)

    shapes = []
    for y in years:
        shapes.append(_npz_npy_shape(lst_cache_path(city, y)))
    total_t = int(sum(s[0] for s in shapes))
    n_pix = int(shapes[0][1])
    derived.mkdir(parents=True, exist_ok=True)
    values_mm = np.lib.format.open_memmap(
        derived / "values.npy", mode="w+", dtype=np.float32, shape=(total_t, n_pix)
    )
    time_parts = []
    pixel_ids = xy_km = None
    off = 0
    for y in years:
        p = lst_cache_path(city, y)
        with np.load(p) as d:
            arr = np.asarray(d["uhi_arr"], dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            nt = arr.shape[0]
            values_mm[off:off + nt] = arr
            off += nt
            time_parts.append(_cache_times_to_datetime(np.asarray(d["times"])))
            if pixel_ids is None:
                pixel_ids = np.asarray(d["pixel_ids"], dtype=np.int64).ravel()
                xy_km = np.asarray(d["xy_km"], dtype=np.float32)
        print(f"    [cache] building {derived.name}: {p.name} ({nt} rows)", flush=True)
    values_mm.flush()

    times = np.concatenate(time_parts)
    np.save(derived / "times.npy", times.astype("datetime64[ns]"))
    np.save(derived / "pixel_ids.npy", pixel_ids)
    np.save(derived / "xy_km.npy", xy_km)
    (derived / "done.json").write_text(json.dumps(
        {"city": city, "years": years, "shape": [total_t, n_pix], "source": "lstuhi_v6"},
        indent=2,
    ))
    values = np.load(derived / "values.npy", mmap_mode="r")
    xy_m, feats, names, spids = _load_static(city)
    pid2row = {int(pid): i for i, pid in enumerate(np.asarray(spids, dtype=np.int64).ravel())}
    order = np.asarray([pid2row[int(pid)] for pid in pixel_ids], dtype=np.int64)
    return Field(values, xy_km, feats[order], names, times, pixel_ids)


def subset_cache_dir(city: str, years, n_pix: int, seed: int, cache_key: str | None = None) -> Path:
    years_key = "_".join(map(str, [int(y) for y in years]))
    suffix = "" if cache_key is None else f"_{cache_key}"
    return LST_BASE / city / "cache" / f"bench_1a_fm_subset_{years_key}_n{n_pix}_seed{int(seed)}{suffix}"


def select_values_cached(field: Field, city: str, years, pix: np.ndarray, seed: int,
                         cache_key: str | None = None):
    cache_dir = subset_cache_dir(city, years, len(pix), seed, cache_key)
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
        e = min(s + chunk, field.values.shape[0])
        values_mm[s:e] = np.asarray(field.values[s:e, :][:, pix], dtype=np.float32)
        if s == 0 or e == field.values.shape[0] or ((s // chunk) + 1) % 16 == 0:
            print(f"    [subset-cache] {cache_dir.name}: rows {e}/{field.values.shape[0]}", flush=True)
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


def standardize_values_cached(city: str, years, values: np.ndarray, seed: int,
                              cache_key: str | None = None):
    cache_dir = subset_cache_dir(city, years, values.shape[1], seed, cache_key)
    z_path = cache_dir / "values_z.npy"
    mu_path = cache_dir / "mu.npy"
    sd_path = cache_dir / "sd.npy"
    if z_path.exists() and mu_path.exists() and sd_path.exists():
        return np.load(z_path, mmap_mode="r"), np.load(mu_path), np.load(sd_path)

    z, mu, sd = standardize_target(np.asarray(values, dtype=np.float32))
    np.save(z_path, z)
    np.save(mu_path, mu)
    np.save(sd_path, sd)
    return np.load(z_path, mmap_mode="r"), mu, sd


def standardize_with_stats(values: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float32) - mu) / sd).astype(np.float32)


def choose_pixels(field: Field, city: str, args, rng: np.random.Generator) -> np.ndarray:
    n_all = field.values.shape[1]
    n = min(int(args.n_pixels), n_all)
    if args.pixel_mode == "random":
        pix = rng.choice(n_all, n, replace=False)
    elif args.pixel_mode == "clear_anchor":
        anchor = align_to_pixel_ids(field, load_lst_field_fast(city, years=[args.anchor_year]))
        if not (0 <= int(args.anchor_row) < anchor.values.shape[0]):
            raise ValueError(f"anchor_row={args.anchor_row} outside [0,{anchor.values.shape[0]})")
        candidates = np.flatnonzero(np.isfinite(np.asarray(anchor.values[int(args.anchor_row)])))
        if len(candidates) < n:
            raise RuntimeError(f"clear_anchor has only {len(candidates)} valid pixels, need {n}")
        pix = rng.choice(candidates, n, replace=False)
        print(f"    clear-anchor year={args.anchor_year} row={args.anchor_row} "
              f"valid_candidates={len(candidates)}", flush=True)
    else:
        raise ValueError(f"unknown pixel mode: {args.pixel_mode}")
    return np.sort(pix).astype(np.int64)


def run_city(city: str, args) -> dict:
    rng = np.random.default_rng(args.seed)
    print(f"\n{'=' * 60}\n[1a-fm] city={city} fm={args.fm} config={args.config}", flush=True)
    stat_fld = load_lst_field_fast(city, years=args.stat_years)
    fld = align_to_pixel_ids(stat_fld, load_lst_field_fast(city, years=args.years))
    pix = choose_pixels(stat_fld, city, args, rng)
    cache_key = None
    if args.pixel_mode != "random":
        cache_key = f"{args.pixel_mode}_y{int(args.anchor_year)}_r{int(args.anchor_row)}"
    stat_values, pixel_ids = select_values_cached(
        stat_fld, city, args.stat_years, pix, args.seed, cache_key=cache_key
    )
    values, eval_pixel_ids = select_values_cached(
        fld, city, args.years, pix, args.seed, cache_key=cache_key
    )
    if not np.array_equal(pixel_ids, eval_pixel_ids):
        raise RuntimeError(f"{city}: train/eval subset pixel IDs differ")
    n_pixels = len(pix)
    _, mu, sd = standardize_values_cached(
        city, args.stat_years, stat_values, args.seed, cache_key=cache_key
    )
    values_z = standardize_with_stats(values, mu, sd)
    print(f"    stat={stat_values.shape} eval={values.shape} pixels={n_pixels} "
          f"mode={args.pixel_mode}", flush=True)

    cov_stats = fit_covariate_stats(
        city,
        args.stat_years,
        pixel_ids,
        pd.DatetimeIndex(stat_fld.times),
        args.config,
    )
    covs, cov_meta = build_covariates(
        city,
        args.years,
        pixel_ids,
        pd.DatetimeIndex(fld.times),
        args.config,
        stats=cov_stats,
    )
    pair_values = values if args.pixel_mode == "clear_anchor" else fld.values
    pairs = build_cloud_eval_pairs(
        pair_values,
        fld.times,
        rng,
        clear_thr=args.clear_thr,
        n_clear=args.n_clear,
        masks_per_bin=args.masks_per_bin,
    )
    clear_to_bins = {}
    for pr in pairs:
        t = int(pr["clear_t"])
        if t < W:
            continue
        clear_to_bins.setdefault(t, {b: [] for b in range(len(BIN_LABELS))})
        mask = pr["mask"] if args.pixel_mode == "clear_anchor" else pr["mask"][pix]
        if mask.any():
            clear_to_bins[t][pr["bin_idx"]].append(mask)
    starts = list(clear_to_bins)
    print(f"    eval clear_ts={len(starts)} pairs={len(pairs)} cov_dim={cov_meta['cov_dim']}", flush=True)

    pred_z = predict_steps(
        args.fm,
        values_z,
        covs,
        starts,
        args.device,
        context=W,
        pixel_batch=args.pixel_batch,
    )
    bin_ae = {b: [] for b in range(len(BIN_LABELS))}
    for si, t in enumerate(starts):
        pred = pred_z[si] * sd + mu
        gt = values[t]
        for b, masks in clear_to_bins[t].items():
            for mask in masks:
                valid = np.isfinite(gt) & mask
                if valid.any():
                    bin_ae[b].append(np.abs(pred[valid] - gt[valid]))

    key = method_key(args.fm, args.config)
    out = {
        "city": city,
        "model": "Covariate FM",
        "n_pixels": n_pixels,
        "bins": BIN_LABELS,
        "axis": "cloud missing%",
        "protocol": {
            "years": [int(y) for y in args.years],
            "eval_years": [int(y) for y in args.years],
            "stat_years": [int(y) for y in args.stat_years],
            "mode": "zero-shot, train-period normalization",
            "n_clear": int(args.n_clear),
            "masks_per_bin": int(args.masks_per_bin),
            "clear_thr": float(args.clear_thr),
            "seed": int(args.seed),
            "context": W,
            "n_pixels": int(n_pixels),
            "pixel_mode": args.pixel_mode,
            "anchor_year": int(args.anchor_year) if args.pixel_mode == "clear_anchor" else None,
            "anchor_row": int(args.anchor_row) if args.pixel_mode == "clear_anchor" else None,
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
    ap.add_argument("--cities", nargs="+", default=["cairo", "bucharest", "lagos"])
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--stat_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--n_pixels", type=int, default=512)
    ap.add_argument("--n_clear", type=int, default=15)
    ap.add_argument("--masks_per_bin", type=int, default=6)
    ap.add_argument("--clear_thr", type=float, default=0.02)
    ap.add_argument("--pixel_mode", choices=["random", "clear_anchor"], default="random")
    ap.add_argument("--anchor_year", type=int, default=2023)
    ap.add_argument("--anchor_row", type=int, default=3)
    ap.add_argument("--pixel_batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--output_tag", default=None,
                    help="optional suffix for output filename, e.g. clear_anchor")
    args = ap.parse_args()
    if args.fm == "timesfm" and args.config != "base":
        raise SystemExit("TimesFM has no covariate channel here; use --config base")
    args.device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = method_key(args.fm, args.config)

    for city in args.cities:
        tag = args.output_tag
        if tag is None and args.pixel_mode != "random":
            tag = args.pixel_mode
        out_path = out_dir / (f"1a_{city}_fm_{tag}.json" if tag else f"1a_{city}_fm.json")
        if method_exists(out_path, key):
            print(f"[skip-existing] {city} {key} -> {out_path}", flush=True)
            continue
        res = run_city(city, args)
        merged = merge_result(out_path, res)
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
