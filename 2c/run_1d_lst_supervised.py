"""Task 2c — LST-UHI supervised forecasting baselines, 4 input configurations.

Fills the LST gap in 1d (which previously had only zero-shot/stat baselines).
Reuses the Task-3 LST clear-window pipeline (load_city_year / build_samples) but
trains and evaluates IN-DOMAIN (city 2015-2022 -> same city 2023-2025).

Input configurations (XGBoost, flattened features):
  L1           : historical LST lags + mask + window stats (UHI only)
  +ERA5        : L1 + 6 ERA5 drivers (current + lags + rolling)        [= Task3 L2]
  +static      : L1 + 10 static urban-form features
  +ERA5+static : L1 + ERA5 + static                                    [= Task3 L3]

Sampling: lookback 168h, min_valid_ratio=0.70 (clear-window protocol), target t+H
must be finite. Horizons: +1/+6/+24h. Persistence included as a stat baseline.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3"))
import run_3_ood_transfer as T3   # noqa: E402

HORIZONS = [1, 6, 12, 24, 48, 96]
CONFIGS = ["L1", "+ERA5", "+static", "+ERA5+static"]


def build_4configs(batch):
    """Return dict config -> feature matrix, reusing Task3 L1/L2/L3 and slicing static."""
    x_l1 = T3.layer_matrix(batch, "L1")
    x_l2 = T3.layer_matrix(batch, "L2")     # L1 + ERA5
    x_l3 = T3.layer_matrix(batch, "L3")     # L1 + ERA5 + static
    n_l2 = x_l2.shape[1]
    static_block = x_l3[:, n_l2:]            # static features (last block of L3)
    return {
        "L1": x_l1,
        "+ERA5": x_l2,
        "+static": np.hstack([x_l1, static_block]),
        "+ERA5+static": x_l3,
    }


def run_city(city, train_years, eval_years, n_pixels, max_samples, seed):
    rng = np.random.default_rng(seed)
    pix = T3.choose_pixels(city, n_pixels, seed)
    print(f"\n{'='*50}\n[1d-LST-sup] {city}  pixels={len(pix)}")
    res = {"city": city, "horizons": HORIZONS, "configs": CONFIGS, "methods": {}}
    for h in HORIZONS:
        # build train batch across train years
        tr_parts = []
        for y in train_years:
            d = T3.load_city_year(city, y, n_pixels, seed)
            tr_parts.append(T3.build_samples(d, h, max_samples, 0.70, seed + y))
        tr = T3.concat_batches(tr_parts)
        ev_parts = []
        for y in eval_years:
            d = T3.load_city_year(city, y, n_pixels, seed)
            ev_parts.append(T3.build_samples(d, h, max_samples, 0.70, seed + y + 999))
        ev = T3.concat_batches(ev_parts)
        if len(tr.y) == 0 or len(ev.y) == 0:
            print(f"  h={h}h: empty (train={len(tr.y)} eval={len(ev.y)})"); continue

        cfgs_tr = build_4configs(tr)
        cfgs_ev = build_4configs(ev)
        key_h = f"{h}h"
        # Persistence (config-independent)
        pers_mae = float(np.mean(np.abs(ev.persistence - ev.y)))
        res["methods"].setdefault("Persistence", {})[key_h] = pers_mae
        # Climatology (month x hour-of-day, fit on train) -- reuse Task3 helper
        clim = T3.fit_source_climatology(tr.target_times, tr.y)
        clim_pred = T3.predict_source_climatology(clim, ev.target_times)
        res["methods"].setdefault("Climatology", {})[key_h] = float(np.mean(np.abs(clim_pred - ev.y)))
        # XGBoost per config
        for cfg in CONFIGS:
            xtr, xev = cfgs_tr[cfg], cfgs_ev[cfg]
            mu, sd = T3.fit_feature_scaler(xtr)
            xtr_z = T3.apply_feature_scaler(xtr, mu, sd)
            xev_z = T3.apply_feature_scaler(xev, mu, sd)
            model = T3.train_xgb(xtr_z, tr.y, seed, n_estimators=250, max_depth=6)
            pred = model.predict(xev_z)
            mae = float(np.mean(np.abs(pred - ev.y)))
            res["methods"].setdefault(f"XGBoost({cfg})", {})[key_h] = mae
        print(f"  h={h}h  (train={len(tr.y)} eval={len(ev.y)})  "
              + "  ".join(f"{m}={res['methods'][m][key_h]:.3f}" for m in res["methods"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["munich", "berlin"])
    ap.add_argument("--train_years", type=int, nargs="+", default=list(range(2015, 2023)))
    ap.add_argument("--eval_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_pixels", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for city in a.cities:
        all_res[city] = run_city(city, a.train_years, a.eval_years, a.n_pixels, a.max_samples, a.seed)
        (out_dir / f"1d_lst_supervised_{city}.json").write_text(json.dumps(all_res[city], indent=2))
    print(f"\n[saved] {out_dir}/1d_lst_supervised_*.json")


if __name__ == "__main__":
    main()
