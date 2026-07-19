"""Task 2a — GNN (IGNNK) baseline. Same cloud-mask protocol as run_1a.py
so results are directly comparable to IDW/Kriging/RF/XGBoost.

Trains on clear-scene timestamps, evaluates on the cloud-coverage-bin pairs.
Usage: python run_1a_gnn.py --cities cairo bucharest lagos
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_lst_field
from common.masks import build_cloud_eval_pairs, BIN_LABELS
from common.models_dl import train_gnn, eval_gnn_on_masks


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


def run(city, train_years, test_years, device, epochs, n_clear, masks_per_bin, max_pred, seed):
    rng = np.random.default_rng(seed)
    print(f"\n{'='*60}\n[1a-gnn] {city}  train={train_years} eval={test_years} ...")
    t0 = time.time()
    tr_fld = load_lst_field(city, years=train_years)
    te_fld = _align_test_to_train_pixels(tr_fld, load_lst_field(city, years=test_years))
    print(f"    train {tr_fld.values.shape} eval {te_fld.values.shape}  ({time.time()-t0:.0f}s)")

    cov = np.isnan(tr_fld.values).mean(1)
    clear_all = np.where(cov < 0.02)[0]
    train_idx = clear_all[:max(40, len(clear_all)//2)]      # train on clear scenes
    print(f"    clear scenes: {len(clear_all)}  train on {len(train_idx)}")

    n_static = tr_fld.feats.shape[1]
    t0 = time.time()
    model, An_t, static_t = train_gnn(tr_fld, train_idx, n_static, device,
                                      epochs=epochs, seed=seed)
    print(f"    trained in {time.time()-t0:.0f}s")

    # eval pairs (same protocol as run_1a), strictly on held-out years
    pairs = build_cloud_eval_pairs(te_fld.values, te_fld.times, rng,
                                   n_clear=n_clear, masks_per_bin=masks_per_bin)
    # group masks by bin (use first clear scene per pair's GT = its clear_t)
    by_bin = {b: [] for b in range(4)}
    # For comparability we eval each pair: GT = clear scene values, mask = pair mask
    # To bound cost, eval one representative clear scene per many masks.
    clear_scene_idx = {p["clear_t"] for p in pairs}
    # map: for each clear_t, collect masks per bin
    cs_masks_bin = {cs: {b: [] for b in range(4)} for cs in clear_scene_idx}
    for p in pairs:
        cs_masks_bin[p["clear_t"]][p["bin_idx"]].append(p["mask"])
    bin_ae = {b: [] for b in range(4)}
    cs_list = sorted(clear_scene_idx)
    for ci, cs in enumerate(cs_list):
        scene = te_fld.values[cs].astype(np.float64)
        for b in range(4):
            ms = cs_masks_bin[cs][b]
            if not ms:
                continue
            ae = eval_gnn_on_masks(model, An_t, static_t, scene, ms, device,
                                   max_pred=max_pred, seed=seed)
            if ae.size:
                bin_ae[b].append(ae)
        if (ci + 1) % 10 == 0:
            print(f"    eval clear scenes {ci+1}/{len(cs_list)}", flush=True)

    out = {"city": city, "model": "GNN_IGNNK", "n_features": n_static,
           "train_years": train_years, "test_years": test_years,
           "protocol": {"train_years": list(map(int, train_years)),
                        "test_years": list(map(int, test_years)),
                        "eval_years": list(map(int, test_years))},
           "epochs": epochs, "bins": BIN_LABELS, "methods": {"GNN": {}}}
    for b in range(4):
        ae = np.concatenate(bin_ae[b]) if bin_ae[b] else np.array([])
        out["methods"]["GNN"][str(b)] = {
            "MAE": float(ae.mean()) if ae.size else None,
            "RMSE": float(np.sqrt((ae ** 2).mean())) if ae.size else None,
            "N": int(ae.size),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["cairo", "bucharest", "lagos"])
    ap.add_argument("--train_years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ap.add_argument("--test_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n_clear", type=int, default=25)
    ap.add_argument("--masks_per_bin", type=int, default=6)
    ap.add_argument("--max_pred", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for c in a.cities:
        r = run(c, a.train_years, a.test_years, device, a.epochs, a.n_clear, a.masks_per_bin, a.max_pred, a.seed)
        all_res[c] = r
        (out_dir / f"1a_{c}_gnn.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(f"\n--- {c} GNN MAE by cloud bin ---")
        print("  " + "  ".join(f"{BIN_LABELS[b]}={r['methods']['GNN'][str(b)]['MAE']:.4f}"
                                for b in range(4) if r['methods']['GNN'][str(b)]['MAE']))
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
