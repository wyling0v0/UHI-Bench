"""Task 2b — SAITS (PyPOTS) masked-imputation baseline for Air-T station-sparse recon.

Mirrors 1a/run_1a_saits.py but on the Air-T field with RANDOM station masks
(keep {10,25,50}%) instead of transferred cloud masks. Air-T is dense, so SAITS
is trained on artificially masked windows (MIT needs synthetic masks anyway).
Eval: window ends at an eval timestamp; apply the random hide mask at the last
step; SAITS imputes; MAE on hidden pixels, stratified by keep fraction.
Subsamples pixels for tractability.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_ta_field
from common.masks import BIN_LABELS, random_mask_for_bin

W = 24
NBINS = len(BIN_LABELS)   # 4 missing% bins, aligned with Task 2a x-axis


def align_test_to_train_pixels(train_fld, test_fld):
    if np.array_equal(train_fld.pixel_ids, test_fld.pixel_ids):
        return test_fld
    pos = {int(p): i for i, p in enumerate(test_fld.pixel_ids)}
    order = np.asarray([pos[int(p)] for p in train_fld.pixel_ids], dtype=np.int64)
    return test_fld._replace(
        values=np.ascontiguousarray(test_fld.values[:, order]),
        xy_km=test_fld.xy_km[order],
        feats=test_fld.feats[order],
        pixel_ids=test_fld.pixel_ids[order],
    )


def to_pypots(X):
    mask = (~np.isnan(X)).astype(np.float32)
    Xf = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return {"X": Xf, "missing_mask": mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="munich")
    ap.add_argument("--train_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--test_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_pixels", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n_times", type=int, default=30)
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    from pypots.imputation import SAITS

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    print(f"\n{'='*50}\n[1b-saits] {a.city}")
    tr_fld = load_ta_field(a.city, years=a.train_years)
    te_fld = align_test_to_train_pixels(tr_fld, load_ta_field(a.city, years=a.test_years))
    pix = rng.choice(tr_fld.values.shape[1], min(a.n_pixels, tr_fld.values.shape[1]), replace=False)
    pix.sort()
    Vtr = tr_fld.values[:, pix].astype(np.float32)     # [Ttr, Nsub]
    Vte = te_fld.values[:, pix].astype(np.float32)     # [Tte, Nsub]
    N = len(pix)
    mu = np.nanmean(Vtr, axis=0); sd = np.nanstd(Vtr, axis=0) + 1e-6
    Vtr_z = (Vtr - mu) / sd
    Vte_z = (Vte - mu) / sd

    # TRAIN windows: dense Air-T, apply random MIT-style masks
    starts = np.arange(0, Vtr_z.shape[0] - W, 3)
    tr_windows = []
    for s in starts:
        win = Vtr_z[s:s+W].copy()
        rm = rng.random(win.shape) < rng.uniform(0.05, 0.30)   # artificial missingness
        win[rm] = np.nan
        tr_windows.append(win)
    tr_windows = np.stack(tr_windows[:4000]).astype(np.float32)
    print(f"  train windows {tr_windows.shape}", flush=True)
    train_set = to_pypots(tr_windows)

    model = SAITS(n_steps=W, n_features=N, n_layers=2, d_model=64, n_heads=4,
                  d_k=16, d_v=16, d_ffn=128, dropout=0.1,
                  batch_size=32, epochs=a.epochs,
                  device=a.device, saving_path=None, model_saving_strategy=None,
                  verbose=False)
    model.fit(train_set)

    # EVAL: window ends at eval timestamp; random-scatter hide mask at last step,
    # stratified by missing% bins (aligned with Task 2a x-axis).
    eval_t = rng.choice(np.arange(W, Vte_z.shape[0]), min(a.n_times, Vte_z.shape[0]-W), replace=False)
    ae = {str(b): [] for b in range(NBINS)}
    for ti in sorted(eval_t):
        win = Vte_z[ti-W+1:ti+1].copy()                   # [W, N] last step = eval scene
        for b in range(NBINS):
            for _ in range(a.n_splits):
                hide = np.where(random_mask_for_bin(N, b, rng))[0]
                win_eval = win.copy()
                win_eval[-1, hide] = np.nan
                res = model.predict(to_pypots(win_eval[None]))
                imp = res["imputation"][0, -1] * sd + mu
                gt = te_fld.values[ti, pix] * 1.0
                valid = np.isfinite(gt[hide])
                if valid.any():
                    ae[str(b)].append(np.abs(imp[hide][valid] - gt[hide][valid]))

    out = {"city": a.city, "model": "SAITS", "n_pixels": N,
           "train_years": a.train_years, "test_years": a.test_years,
           "protocol": {"train_years": list(map(int, a.train_years)),
                        "test_years": list(map(int, a.test_years)),
                        "eval_years": list(map(int, a.test_years))},
           "bins": BIN_LABELS, "axis": "missing%", "methods": {"SAITS": {}}}
    for b in range(NBINS):
        arr = np.concatenate(ae[str(b)]) if ae[str(b)] else np.array([])
        out["methods"]["SAITS"][str(b)] = {"MAE": float(arr.mean()) if arr.size else None,
                                           "RMSE": float(np.sqrt((arr**2).mean())) if arr.size else None,
                                           "N": int(arr.size)}
        print(f"  {BIN_LABELS[b]}: MAE={out['methods']['SAITS'][str(b)]['MAE']}")
    (out_dir / f"1b_{a.city}_saits.json").write_text(json.dumps(out, indent=2))
    print(f"[saved] {out_dir}/1b_{a.city}_saits.json")


if __name__ == "__main__":
    main()
