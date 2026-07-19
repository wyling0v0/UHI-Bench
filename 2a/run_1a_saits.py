"""Task 2a — SAITS (PyPOTS) masked-imputation baseline for LST cloud-gap recon.

SAITS is a self-attention imputer over [W, N] windows (W timesteps × N pixels).
Train: windows with natural cloud NaN (SAITS learns to impute via ORT/MIT).
Eval: windows ending at a clear scene; apply the transferred real cloud mask at
the last step; SAITS imputes; MAE on masked pixels, stratified by cloud coverage.
Subsamples pixels for tractability.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_lst_field
from common.masks import build_cloud_eval_pairs, BIN_LABELS

W = 24   # temporal window


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
    """X [n, W, N] with NaN for missing -> dict for PyPOTS."""
    mask = (~np.isnan(X)).astype(np.float32)
    Xf = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return {"X": Xf, "missing_mask": mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["cairo", "bucharest", "lagos"])
    ap.add_argument("--train_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--test_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_pixels", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_clear", type=int, default=15)
    ap.add_argument("--masks_per_bin", type=int, default=6)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    from pypots.imputation import SAITS   # import after env settle

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for city in a.cities:
        print(f"\n{'='*50}\n[1a-saits] {city}")
        tr_fld = load_lst_field(city, years=a.train_years)
        te_fld = align_test_to_train_pixels(tr_fld, load_lst_field(city, years=a.test_years))
        rng = np.random.default_rng(42)
        pix = rng.choice(tr_fld.values.shape[1], min(a.n_pixels, tr_fld.values.shape[1]), replace=False)
        pix.sort()
        Vtr = tr_fld.values[:, pix].astype(np.float32)     # [Ttr, Nsub]
        Vte = te_fld.values[:, pix].astype(np.float32)     # [Tte, Nsub]
        N = len(pix)
        # standardize with TRAIN stats only
        mu = np.nanmean(Vtr, axis=0); sd = np.nanstd(Vtr, axis=0) + 1e-6
        Vtr_z = (Vtr - mu) / sd
        Vte_z = (Vte - mu) / sd

        # build TRAIN windows (natural cloud NaN) — random starts
        cov = np.isnan(Vtr).mean(1)
        train_starts = np.where(cov < 0.8)[0]              # skip fully-cloudy
        tr_windows = []
        for s in train_starts[::6]:
            if s + W <= len(Vtr_z):
                tr_windows.append(Vtr_z[s:s+W])
        tr_windows = np.stack(tr_windows[:4000]).astype(np.float32)  # cap
        train_set = to_pypots(tr_windows)
        print(f"  train windows {tr_windows.shape}", flush=True)

        model = SAITS(n_steps=W, n_features=N, n_layers=2, d_model=64, n_heads=4,
                      d_k=16, d_v=16, d_ffn=128, dropout=0.1,
                      batch_size=32, epochs=a.epochs,
                      device=a.device, saving_path=None, model_saving_strategy=None,
                      verbose=False)
        model.fit(train_set)

        # EVAL: clear-scene + transferred cloud mask on held-out years
        pairs = build_cloud_eval_pairs(te_fld.values, te_fld.times, rng,
                                       n_clear=a.n_clear, masks_per_bin=a.masks_per_bin)
        bin_ae = {b: [] for b in range(4)}
        # group by (clear_t, bin) -> mask; eval window ends at clear_t
        cs_masks = {}
        for p in pairs:
            cs_masks.setdefault(p["clear_t"], {b: [] for b in range(4)})
            full_mask = p["mask"][pix]                    # restrict to subset (local bool)
            if full_mask.any():
                cs_masks[p["clear_t"]][p["bin_idx"]].append(full_mask)
        for ct, bins in cs_masks.items():
            t = ct
            if t < W - 1:
                continue
            win = Vte_z[t-W+1:t+1].copy()                    # [W, N] last step = clear scene
            # mask the clear scene (last step) at eval-mask pixels
            for b, masks in bins.items():
                for m in masks:
                    win_eval = win.copy()
                    win_eval[-1, m] = np.nan                # hide eval pixels at clear step
                    res = model.predict(to_pypots(win_eval[None]))
                    imp = res["imputation"][0, -1] * sd + mu   # de-standardize last step
                    gt = te_fld.values[ct, pix] * 1.0
                    valid = np.isfinite(gt) & m
                    bin_ae[b].append(np.abs(imp[valid] - gt[valid]))
        out = {"city": city, "model": "SAITS", "n_pixels": N,
               "train_years": a.train_years, "test_years": a.test_years,
               "protocol": {"train_years": list(map(int, a.train_years)),
                            "test_years": list(map(int, a.test_years)),
                            "eval_years": list(map(int, a.test_years))},
               "bins": BIN_LABELS,
               "methods": {"SAITS": {}}}
        for b in range(4):
            ae = np.concatenate(bin_ae[b]) if bin_ae[b] else np.array([])
            out["methods"]["SAITS"][str(b)] = {"MAE": float(ae.mean()) if ae.size else None,
                                               "RMSE": float(np.sqrt((ae**2).mean())) if ae.size else None,
                                               "N": int(ae.size)}
            print(f"  {BIN_LABELS[b]}: MAE={out['methods']['SAITS'][str(b)]['MAE']}")
        all_res[city] = out
        (out_dir / f"1a_{city}_saits.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
