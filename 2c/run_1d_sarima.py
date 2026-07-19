"""Task 2c — SARIMA baseline on the CITY-MEAN series (CPU).

Per-pixel hourly SARIMA over 4000+ pixels is infeasible, so this fits one
SARIMAX(1,0,1)(1,1,0,24) on the city-mean Ta-UHI series and does rolling-origin
forecasts at horizons 1/6/12/24/48/96h. MAE is on the city-mean (NOT per-pixel)
→ not directly comparable to per-pixel baselines; reported as the statistical
ceiling on the aggregated signal. Clearly caveated in output.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_ta_field
from statsmodels.tsa.statespace.sarimax import SARIMAX

HORIZONS = [1, 6, 12, 24, 48, 96]


def city_mean_series(city, years):
    f = load_ta_field(city, years=years)
    s = pd.Series(np.nanmean(f.values, axis=1), index=pd.to_datetime(f.times))
    return s.ffill().bfill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="munich")
    ap.add_argument("--train_years", type=int, nargs="+", default=[2015,2016,2017,2018,2019,2020,2021,2022])
    ap.add_argument("--test_years", type=int, nargs="+", default=[2023,2024,2025])
    ap.add_argument("--origins", type=int, default=80, help="rolling-origin eval points")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "1d_forecast.json"))
    a = ap.parse_args()

    tr = city_mean_series(a.city, a.train_years)
    te = city_mean_series(a.city, a.test_years)
    full = pd.concat([tr, te])
    H_max = max(HORIZONS)
    te_idx = np.linspace(H_max, len(te) - H_max - 1, a.origins).astype(int)

    print(f"[sarima] fitting SARIMAX(1,0,1)(1,1,0,24) on {a.city} train ({len(tr)}h)...")
    mdl = SARIMAX(tr, order=(1, 0, 1), seasonal_order=(1, 1, 0, 24),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    # extend to full series via append (no refit), then forecast from each origin
    res_ext = mdl.apply(full.values)
    row = {}
    for h in HORIZONS:
        errs = []
        for oi in te_idx:
            # forecast h steps ahead from origin oi (index into `full`)
            fcast = res_ext.get_prediction(start=oi, end=oi + h, dynamic=0)
            pred_mean = fcast.predicted_mean
            actual = full.iloc[oi + h]
            errs.append(abs(pred_mean[-1] - actual))
        row[f"{h}h"] = float(np.mean(errs))
        print(f"  +{h}h: city-mean MAE={row[f'{h}h']:.4f}")

    data = json.loads(Path(a.out).read_text())
    data.setdefault("Ta", {}).setdefault(a.city.capitalize(), {})["SARIMA(city-mean, caveat)"] = row
    Path(a.out).write_text(json.dumps(data, indent=2))
    print(f"[merged] SARIMA {a.city} -> {a.out}  (CAVEAT: city-mean, not per-pixel)")


if __name__ == "__main__":
    main()
