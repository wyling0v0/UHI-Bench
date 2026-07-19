"""Task 2a — Chronos zero-shot imputation baseline for LST cloud-gap recon.

Same window protocol as run_1a_saits.py: eval window ends at a clear scene; apply
the transferred real cloud mask at the last step; reconstruct and score by bin.

Difference: Chronos is a zero-shot FORECASTING FM (no training, no missing-mask
input). We treat the gap as a 1-step-ahead forecast from the preceding W clear
steps (per-pixel, batched over pixels). Preceding-step cloud NaN is zero-filled
in standardized space (= pixel mean) -- the known zero-shot limitation shared
with TimesFM (fails on long continuous cloud where context is also masked).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_lst_field
from common.masks import build_cloud_eval_pairs, BIN_LABELS

W = 24


def align_to_pixel_ids(anchor, other):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["cairo", "bucharest", "lagos"])
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--stat_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--n_pixels", type=int, default=512)
    ap.add_argument("--n_clear", type=int, default=15)
    ap.add_argument("--masks_per_bin", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                               device_map=a.device, torch_dtype="auto")

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for city in a.cities:
        print(f"\n{'='*50}\n[1a-chronos] {city}")
        stat_fld = load_lst_field(city, years=a.stat_years)
        fld = align_to_pixel_ids(stat_fld, load_lst_field(city, years=a.years))
        rng = np.random.default_rng(42)
        pix = rng.choice(stat_fld.values.shape[1], min(a.n_pixels, stat_fld.values.shape[1]), replace=False)
        pix.sort()
        stat_values = stat_fld.values[:, pix].astype(np.float32)
        V = fld.values[:, pix].astype(np.float32)           # [T, Nsub]
        N = len(pix)
        mu = np.nanmean(stat_values, axis=0); sd = np.nanstd(stat_values, axis=0) + 1e-6
        Vz = (V - mu) / sd
        Vz0 = np.nan_to_num(Vz, nan=0.0).astype(np.float32)  # zero-fill context for Chronos

        pairs = build_cloud_eval_pairs(fld.values, fld.times, rng,
                                       n_clear=a.n_clear, masks_per_bin=a.masks_per_bin)
        cs_masks = {}
        for p in pairs:
            cs_masks.setdefault(p["clear_t"], {b: [] for b in range(4)})
            fm = p["mask"][pix]
            if fm.any():
                cs_masks[p["clear_t"]][p["bin_idx"]].append(fm)

        bin_ae = {b: [] for b in range(4)}
        for ct, bins in cs_masks.items():
            t = ct
            if t < W:
                continue
            ctx = Vz0[t-W:t].T.astype(np.float32)           # [N, W] per-pixel context
            with torch.no_grad():
                fc = pipe.predict(torch.from_numpy(ctx), prediction_length=1)  # [N,1,?q]
            fc = fc.cpu().numpy()
            if fc.ndim == 3:
                imp_step = np.median(fc[:, 0, :], axis=1)   # median over quantiles/samples
            else:
                imp_step = fc[:, 0]
            imp = imp_step * sd + mu                         # de-standardize
            gt = fld.values[ct, pix] * 1.0
            for b, masks in bins.items():
                for m in masks:
                    valid = np.isfinite(gt) & m
                    if valid.any():
                        bin_ae[b].append(np.abs(imp[valid] - gt[valid]))
        out = {"city": city, "model": "Chronos", "n_pixels": N, "bins": BIN_LABELS,
               "protocol": {"years": list(map(int, a.years)),
                            "eval_years": list(map(int, a.years)),
                            "stat_years": list(map(int, a.stat_years)),
                            "mode": "zero-shot, train-period normalization"},
               "methods": {"Chronos": {}}}
        for b in range(4):
            ae = np.concatenate(bin_ae[b]) if bin_ae[b] else np.array([])
            out["methods"]["Chronos"][str(b)] = {"MAE": float(ae.mean()) if ae.size else None,
                                                 "RMSE": float(np.sqrt((ae**2).mean())) if ae.size else None,
                                                 "N": int(ae.size)}
            print(f"  {BIN_LABELS[b]}: MAE={out['methods']['Chronos'][str(b)]['MAE']}")
        all_res[city] = out
        (out_dir / f"1a_{city}_chronos.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
