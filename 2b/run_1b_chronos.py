"""Task 2b — Chronos zero-shot imputation baseline for Air-T station-sparse recon.

Same idea as 1a/run_1a_chronos.py but on the Air-T UHI field with RANDOM spatial
station masks (keep fractions {10,25,50}%) instead of cloud masks. Air-T is dense
(no cloud gaps), so the preceding W-step context is fully observed -- a cleaner
test of whether a zero-shot forecasting FM can fill spatial station gaps from
temporal context alone (no spatial neighbours, no static).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_ta_field

W = 24
KEEP = [0.10, 0.25, 0.50]


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
    ap.add_argument("--city", default="munich")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--stat_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--n_pixels", type=int, default=512)
    ap.add_argument("--n_times", type=int, default=30)
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                               device_map=a.device, torch_dtype="auto")

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    print(f"\n{'='*50}\n[1b-chronos] {a.city}")
    stat_fld = load_ta_field(a.city, years=a.stat_years)
    fld = align_to_pixel_ids(stat_fld, load_ta_field(a.city, years=a.years))
    pix = rng.choice(stat_fld.values.shape[1], min(a.n_pixels, stat_fld.values.shape[1]), replace=False)
    pix.sort()
    stat_values = stat_fld.values[:, pix].astype(np.float32)
    V = fld.values[:, pix].astype(np.float32)
    N = len(pix)
    mu = np.nanmean(stat_values, axis=0); sd = np.nanstd(stat_values, axis=0) + 1e-6
    Vz = (V - mu) / sd
    Vz0 = np.nan_to_num(Vz, nan=0.0).astype(np.float32)

    times = np.arange(W, Vz0.shape[0])
    times = rng.choice(times, min(a.n_times, len(times)), replace=False)
    ae = {str(k): [] for k in KEEP}
    for ti in sorted(times):
        ctx = Vz0[ti-W:ti].T.astype(np.float32)        # [N, W]
        with torch.no_grad():
            fc = pipe.predict(torch.from_numpy(ctx), prediction_length=1)
        fc = fc.cpu().numpy()
        imp_step = np.median(fc[:, 0, :], axis=1) if fc.ndim == 3 else fc[:, 0]
        imp = imp_step * sd + mu
        gt = fld.values[ti, pix] * 1.0
        for kf in KEEP:
            for _ in range(a.n_splits):
                n_hide = int(round((1.0 - kf) * N))
                hide = rng.choice(N, n_hide, replace=False)
                valid = np.isfinite(gt[hide])
                if valid.any():
                    ae[str(kf)].append(np.abs(imp[hide][valid] - gt[hide][valid]))

    out = {"city": a.city, "model": "Chronos", "n_pixels": N, "keep_bins": KEEP,
           "protocol": {"years": list(map(int, a.years)),
                        "eval_years": list(map(int, a.years)),
                        "stat_years": list(map(int, a.stat_years)),
                        "mode": "zero-shot, train-period normalization"},
           "methods": {"Chronos": {}}}
    for kf in KEEP:
        arr = np.concatenate(ae[str(kf)])
        out["methods"]["Chronos"][str(kf)] = {"MAE": float(arr.mean()),
                                              "RMSE": float(np.sqrt((arr**2).mean())),
                                              "N": int(arr.size)}
        print(f"  keep{kf}: MAE={out['methods']['Chronos'][str(kf)]['MAE']:.4f}")
    (out_dir / f"1b_{a.city}_chronos.json").write_text(json.dumps(out, indent=2))
    print(f"[saved] {out_dir}/1b_{a.city}_chronos.json")


if __name__ == "__main__":
    main()
