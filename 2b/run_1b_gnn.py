"""Task 2b — GNN (IGNNK) baseline. Same random-keep protocol as run_1b.py.
Trains on Munich full-grid timestamps, evaluates at keep{10,25,50}%.
Usage: python run_1b_gnn.py --city munich
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_ta_field
from common.models_dl import train_gnn, eval_gnn_on_masks
from common.masks import BIN_LABELS, random_mask_for_bin

NBINS = len(BIN_LABELS)


def _align_test_to_train_pixels(train_fld, test_fld):
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


def run(city, train_years, test_years, device, epochs, n_times, n_splits, max_pred, seed):
    rng = np.random.default_rng(seed)
    print(f"\n{'='*60}\n[1b-gnn] {city}  train={train_years} eval={test_years} ...")
    t0 = time.time()
    tr_fld = load_ta_field(city, years=train_years)
    te_fld = _align_test_to_train_pixels(tr_fld, load_ta_field(city, years=test_years))
    print(f"    train {tr_fld.values.shape} eval {te_fld.values.shape}  ({time.time()-t0:.0f}s)")
    cov = np.isnan(tr_fld.values).mean(1)
    train_idx = np.where(cov < 0.01)[0]
    if len(train_idx) > 4000:
        train_idx = rng.choice(train_idx, 4000, replace=False)
    print(f"    train clear timestamps: {len(train_idx)}")

    n_static = tr_fld.feats.shape[1]
    t0 = time.time()
    model, An_t, static_t = train_gnn(tr_fld, train_idx, n_static, device, epochs=epochs, seed=seed)
    print(f"    trained in {time.time()-t0:.0f}s")

    te_cov = np.isnan(te_fld.values).mean(1)
    eval_idx = np.where(te_cov < 0.001)[0]
    if len(eval_idx) > n_times:
        eval_idx = rng.choice(eval_idx, n_times, replace=False)

    bin_ae = {str(b): [] for b in range(NBINS)}
    for ei, ti in enumerate(eval_idx):
        scene = te_fld.values[ti].astype(np.float64)
        scene = np.where(np.isfinite(scene), scene, np.nanmean(scene))
        N = len(scene)
        for b in range(NBINS):
            for _ in range(n_splits):
                m = random_mask_for_bin(N, b, rng)
                ae = eval_gnn_on_masks(model, An_t, static_t, scene, [m], device,
                                       max_pred=max_pred, seed=seed)
                if ae.size:
                    bin_ae[str(b)].append(ae)
        if (ei + 1) % 5 == 0:
            print(f"    eval {ei+1}/{len(eval_idx)}", flush=True)

    out = {"city": city, "model": "GNN_IGNNK", "n_features": n_static, "epochs": epochs,
           "train_years": train_years, "test_years": test_years,
           "protocol": {"train_years": list(map(int, train_years)),
                        "test_years": list(map(int, test_years)),
                        "eval_years": list(map(int, test_years))},
           "bins": BIN_LABELS, "axis": "missing%", "methods": {"GNN": {}}}
    for b in range(NBINS):
        ae = np.concatenate(bin_ae[str(b)]) if bin_ae[str(b)] else np.array([])
        out["methods"]["GNN"][str(b)] = {
            "MAE": float(ae.mean()) if ae.size else None,
            "RMSE": float(np.sqrt((ae ** 2).mean())) if ae.size else None,
            "N": int(ae.size),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="munich")
    ap.add_argument("--train_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--test_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n_times", type=int, default=30)
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--max_pred", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    r = run(a.city, a.train_years, a.test_years, device, a.epochs, a.n_times, a.n_splits, a.max_pred, a.seed)
    (out_dir / f"1b_{a.city}_gnn.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\n--- {a.city} GNN MAE by missing% bin ---")
    print("  " + "  ".join(f"{BIN_LABELS[b]}={r['methods']['GNN'][str(b)]['MAE']:.4f}" for b in range(NBINS)))
    print(f"[saved] {out_dir}/1b_{a.city}_gnn.json")


if __name__ == "__main__":
    main()
