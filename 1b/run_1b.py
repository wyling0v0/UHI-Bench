"""Task 1b (city-level) — cross-modal extreme-event consistency.

For each city: build city-mean UHI hourly series for LST and Ta (each modality's
own history → P95 → extreme iff > P95 AND part of a ≥3h run). On common hours,
compute cross-modal F1 / Miss Rate / False Alarm Rate, stratified day vs night.
Grouped by Köppen cell: Cfb (Berlin/Munich/Hamburg), Aw (Lagos), BWh (Cairo/Riyadh).

No model training — pure label-agreement analysis. Answers RQ1:
"do LST-UHI and Air-T UHI flag the same extreme heat events?"
"""
from __future__ import annotations
import argparse, json, glob, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CACHE_ROOT, LST_BASE, MODEL_DERIVED_TA_BASE, TA_BASE  # noqa: E402

SERIES_CACHE = CACHE_ROOT / "1c_city_mean_series"
SERIES_YEAR_CACHE = CACHE_ROOT / "1c_city_mean_series_year"

CITIES = {
    "Cfb": ["berlin", "munich", "hamburg"],
    "Aw":  ["lagos"],
    "BWh": ["cairo", "riyadh"],
    "Cwb": ["johannesburg"],
}


def lst_city_series(city, years):
    s = _v7_city_mean(LST_BASE, city, years, channel=None)
    if s is not None:
        return s
    fs = []
    for y in years:
        fs += glob.glob(str(LST_BASE / city / f"lst_uhi_1km_hourly_{y}.parquet"))
    if not fs:
        return None
    df = pd.concat([pd.read_parquet(f, columns=["datetime", "lst_uhi_k" if False else "lst_uhi_K"])
                    for f in fs], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    s = df.groupby("datetime")["lst_uhi_K"].mean()        # city-mean per hour (NaN-aware)
    return s


DE_CITIES = {"berlin", "munich", "hamburg", "cologne", "frankfurt",
             "stuttgart", "dortmund", "dusseldorf"}


def ta_city_series(city, years):
    # German cities use HOSTRADA; international cities use released ATUHI-OOD.
    if city in DE_CITIES:
        s = _v7_city_mean(TA_BASE, city, years, channel=1)
        if s is not None:
            return s
        fs = []
        for y in years:
            fs += glob.glob(str(TA_BASE / city / "monthly_uhi" / f"uhi_{y}*.parquet"))
        if not fs:
            return None
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.groupby("datetime")["uhi"].mean()
    s = _v7_city_mean(MODEL_DERIVED_TA_BASE, city, years, channel=1)
    if s is not None:
        return s
    fs = []
    for y in years:
        fs += glob.glob(str(MODEL_DERIVED_TA_BASE / city / f"*{y}*.parquet"))
    fs += glob.glob(str(MODEL_DERIVED_TA_BASE / city / "*.parquet"))   # catch non-yearly names
    fs = sorted(set(fs))
    if not fs:
        return None
    df = pd.concat([pd.read_parquet(f, columns=["datetime", "uhi"]) for f in fs], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.groupby("datetime")["uhi"].mean()


def _v7_city_mean(base: Path, city: str, years, channel: int | None):
    years = [int(y) for y in years]
    SERIES_CACHE.mkdir(parents=True, exist_ok=True)
    channel_key = "lst" if channel is None else f"ch{int(channel)}"
    cache = SERIES_CACHE / f"{base.name}_{city}_{'-'.join(map(str, years))}_{channel_key}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        time_col = "datetime" if "datetime" in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        return pd.Series(df["value"].to_numpy(np.float32), index=df[time_col]).sort_index()
    parts = []
    for y in years:
        s = _v7_city_mean_year(base, city, y, channel, channel_key)
        if s is None:
            return None
        parts.append(s)
    if not parts:
        return None
    out = pd.concat(parts).groupby(level=0).mean().sort_index()
    out.rename("value").rename_axis("datetime").reset_index().to_parquet(cache, index=False)
    return out


def _v7_city_mean_year(base: Path, city: str, year: int, channel: int | None,
                       channel_key: str):
    SERIES_YEAR_CACHE.mkdir(parents=True, exist_ok=True)
    cache_dir = SERIES_YEAR_CACHE / base.name / city
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{int(year)}_{channel_key}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        time_col = "datetime" if "datetime" in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        return pd.Series(df["value"].to_numpy(np.float32), index=df[time_col]).sort_index()

    d = base / city / "cache" / f"v7_{int(year)}"
    if not ((d / "done.json").exists() and (d / "uhi.npy").exists()
            and (d / "times.npy").exists()):
        return None
    arr = np.load(d / "uhi.npy", mmap_mode="r")
    if arr.ndim == 3:
        arr = arr[:, :, channel if channel is not None else 0]
    vals = _pixel_mean(arr, dense=(base.name in {"hostrada_uhi", "atuhi_ood_1km_hourly"}))
    times_raw = np.load(d / "times.npy", allow_pickle=True)
    times = (pd.to_datetime(times_raw, unit="s")
             if np.issubdtype(times_raw.dtype, np.integer)
             else pd.to_datetime(times_raw))
    s = pd.Series(vals, index=times).sort_index()
    s.rename("value").rename_axis("datetime").reset_index().to_parquet(cache, index=False)
    return s


def _pixel_mean(arr, dense: bool, chunk: int = 512):
    """Mean over pixels for v7 mmap arrays.

    Ta caches are dense in practice, so a regular mean is much faster than
    np.nanmean. LST is cloud-masked and sparse, so use a chunked sum/count
    nanmean without materializing an entire city-year copy.
    """
    if dense:
        block = np.asarray(arr, dtype=np.float32)
        vals = block.mean(axis=1).astype(np.float32)
        bad = ~np.isfinite(vals)
        if bad.any():
            vals[bad] = _pixel_mean(block[bad], dense=False, chunk=chunk)
        return vals

    n = arr.shape[0]
    vals = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = np.asarray(arr[start:stop], dtype=np.float32)
        finite = np.isfinite(block)
        counts = finite.sum(axis=1)
        sums = np.where(finite, block, 0.0).sum(axis=1, dtype=np.float64)
        vals[start:stop] = np.divide(
            sums,
            counts,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=counts > 0,
        ).astype(np.float32)
    return vals


def label_extreme(series, pct=95, min_run=3):
    """Boolean series: extreme iff value > P95 AND in a run of ≥ min_run above-threshold hours."""
    thr = np.nanpercentile(series.values, pct)
    above = (series > thr).fillna(False).values
    n = len(above); ext = np.zeros(n, bool)
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_run:
                ext[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(ext, index=series.index), thr


def metrics(lst_ext, ta_ext):
    """Contingency treating Ta-extreme as reference. Returns F1, MissRate, FAR."""
    tp = int((lst_ext & ta_ext).sum())
    fp = int((lst_ext & (~ta_ext)).sum())
    fn = int(((~lst_ext) & ta_ext).sum())
    tn = int(((~lst_ext) & (~ta_ext)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    miss = fn / (fn + tp) if (fn + tp) > 0 else float("nan")     # Ta-extreme missed by LST
    far = fp / (fp + tp) if (fp + tp) > 0 else float("nan")      # LST-extreme not in Ta
    return {"F1": f1, "MissRate": miss, "FAR": far,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "n_ext_ta": int(ta_ext.sum()), "n_ext_lst": int(lst_ext.sum())}


def detector_labels(series, hours, doy, kind="iforest", frac=0.05):
    """Anomaly-detector extreme labels (top `frac` anomaly score among high-UHI hours).
    Returns boolean array (extreme = high-UHI + anomalous)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    v = series.values.astype(float)
    ok = np.isfinite(v)
    feat = np.column_stack([np.nan_to_num(v), hours, doy])
    if kind == "iforest":
        m = IsolationForest(n_estimators=100, contamination=frac, random_state=0)
        m.fit(feat[ok])
        score = -m.score_samples(feat)            # higher = more anomalous
    else:  # ocsvm
        m = OneClassSVM(nu=frac, gamma="scale")
        m.fit(feat[ok])
        score = -m.score_samples(feat)
    # extreme = top-frac anomaly score AND value above median (high-heat anomaly)
    thr = np.nanpercentile(score, 100 * (1 - frac))
    med = np.nanmedian(v)
    ext = (score >= thr) & (v > med) & ok
    # enforce >=3h run like the percentile labels
    n = len(ext); out = np.zeros(n, bool); i = 0
    while i < n:
        if ext[i]:
            j = i
            while j < n and ext[j]:
                j += 1
            if j - i >= 3:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _corr_report(lst, ta, day_mask):
    """Continuous signal consistency (more robust than binary F1)."""
    l = lst.values.astype(float); t = ta.values.astype(float)
    ok = np.isfinite(l) & np.isfinite(t)
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-9))
    res = {}
    res["overall_pearson"] = pear(l[ok], t[ok])
    d = day_mask & ok; n = (~day_mask) & ok
    res["day_pearson"] = pear(l[d], t[d]) if d.sum() > 10 else float("nan")
    res["night_pearson"] = pear(l[n], t[n]) if n.sum() > 10 else float("nan")
    # daily correlation
    ld = lst.resample("1D").mean().reindex(ta.resample("1D").mean().index)
    td = ta.resample("1D").mean()
    okd = np.isfinite(ld.values) & np.isfinite(td.values)
    res["daily_pearson"] = pear(ld.values[okd], td.values[okd]) if okd.sum() > 10 else float("nan")
    return res


def run_city(city, climate, years):
    lst = lst_city_series(city, years)
    ta = ta_city_series(city, years)
    if lst is None or ta is None:
        print(f"  [skip] {city}: lst={'ok' if lst is not None else 'MISSING'} ta={'ok' if ta is not None else 'MISSING'}")
        return None
    common = lst.index.intersection(ta.index)
    if len(common) < 100:
        print(f"  [skip] {city}: only {len(common)} common hours")
        return None
    lst = lst.reindex(common); ta = ta.reindex(common)

    # ---- hourly-level (strict same-hour) ----
    lst_ext_h, lst_thr = label_extreme(lst)
    ta_ext_h, ta_thr = label_extreme(ta)
    hrs = pd.Series(common.hour, index=common)
    day = ((hrs >= 7) & (hrs <= 17)).values

    # ---- daily-level (removes diurnal phase lag) ----
    lst_d = lst.resample("1D").mean()
    ta_d = ta.resample("1D").mean()
    cd = lst_d.index.intersection(ta_d.index)
    lst_d = lst_d.reindex(cd); ta_d = ta_d.reindex(cd)
    lst_ext_d, _ = label_extreme(lst_d)
    ta_ext_d, _ = label_extreme(ta_d)

    # diurnal distribution of hourly extremes (key finding)
    le_hours = pd.Series(lst_ext_h.index).dt.hour[lst_ext_h.values]
    te_hours = pd.Series(ta_ext_h.index).dt.hour[ta_ext_h.values]
    ta_night_frac = float(((te_hours < 7) | (te_hours >= 19)).mean())

    out = {"city": city, "climate": climate, "n_common_hours": len(common),
           "lst_thr_P95": float(lst_thr), "ta_thr_P95": float(ta_thr),
           "hourly": {"overall": metrics(lst_ext_h.values, ta_ext_h.values),
                      "day":     metrics(lst_ext_h.values[day], ta_ext_h.values[day]),
                      "night":   metrics(lst_ext_h.values[~day], ta_ext_h.values[~day])},
           "daily":  metrics(lst_ext_d.values, ta_ext_d.values),
           "diurnal": {"n_lst_ext_h": int(lst_ext_h.sum()), "n_ta_ext_h": int(ta_ext_h.sum()),
                       "ta_extreme_night_fraction": ta_night_frac},
           "corr": _corr_report(lst, ta, day)}
    # anomaly-detector extremes (robustness check vs percentile)
    ho = pd.Series(common.hour).values
    do = pd.Series(common.dayofyear).values
    det = {}
    for dk in ["iforest", "ocsvm"]:
        le = detector_labels(lst, ho, do, kind=dk)
        te = detector_labels(ta, ho, do, kind=dk)
        det[dk] = metrics(le, te)
    out["detectors"] = det
    print(f"  {city} [{climate}]  n={len(common)}  "
          f"hourly F1={out['hourly']['overall']['F1']:.3f}  daily F1={out['daily']['F1']:.3f}  "
          f"Ta-nocturnal={ta_night_frac:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for climate, cities in CITIES.items():
        all_res[climate] = []
        for c in cities:
            r = run_city(c, climate, a.years)
            if r is not None:
                all_res[climate].append(r)
    (out_dir / "1c_consistency.json").write_text(json.dumps(all_res, indent=2, ensure_ascii=False, default=str))

    print("\n=== 1c consistency (hourly strict vs daily) ===")
    print(f"{'climate':>6} {'city':>10} {'hr F1':>7} {'day F1':>7} {'night F1':>9} {'DAILY F1':>9} {'Ta nocturnal':>13}")
    for climate, rows in all_res.items():
        for r in rows:
            h = r["hourly"]
            print(f"{climate:>6} {r['city']:>10} {h['overall']['F1']:>7.3f} {h['day']['F1']:>7.3f} "
                  f"{h['night']['F1']:>9.3f} {r['daily']['F1']:>9.3f} "
                  f"{r['diurnal']['ta_extreme_night_fraction']:>13.2f}")
    print("\n=== 1c signal correlation (LST vs Ta city-mean) ===")
    print(f"{'climate':>6} {'city':>10} {'pearson':>9} {'day':>7} {'night':>7} {'daily':>7}")
    for climate, rows in all_res.items():
        for r in rows:
            c = r["corr"]
            print(f"{climate:>6} {r['city']:>10} {c['overall_pearson']:>9.3f} {c['day_pearson']:>7.3f} "
                  f"{c['night_pearson']:>7.3f} {c['daily_pearson']:>7.3f}")
    print(f"\n[saved] {out_dir}/1c_consistency.json")


if __name__ == "__main__":
    main()
