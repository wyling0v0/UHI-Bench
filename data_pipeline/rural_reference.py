"""Rural reference-ring selection and UHI anomaly construction.

The default rural-reference set for a city is selected from the 15-25 km annulus
around the city center. If land-cover labels are available, cropland and
grassland are preferred. If land-cover labels are not available, candidates are
ranked by a static-feature score that favours low built-up intensity and high
vegetation.

Formula:

    T_rural,c(t) = mean_{q in R_c} T_c(q, t)
    UHI_c(p, t) = T_c(p, t) - T_rural,c(t)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from common.paths import DATA_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.paths import DATA_ROOT


CITY_CENTERS: dict[str, tuple[float, float]] = {
    "berlin": (52.52, 13.405),
    "hamburg": (53.55, 10.00),
    "munich": (48.137, 11.576),
    "frankfurt": (50.11, 8.68),
    "cologne": (50.94, 6.96),
    "stuttgart": (48.78, 9.18),
    "dortmund": (51.51, 7.46),
    "dusseldorf": (51.23, 6.77),
    "bucharest": (44.45, 26.10),
    "buenos_aires": (-34.60, -58.38),
    "cairo": (30.05, 31.25),
    "johannesburg": (-26.20, 28.05),
    "lagos": (6.55, 3.35),
    "riyadh": (24.70, 46.72),
    "sao_paulo": (-23.55, -46.63),
    "warsaw": (52.23, 21.01),
    "tehran": (35.70, 51.40),
    "khartoum": (15.60, 32.50),
    "casablanca": (33.60, -7.60),
    "istanbul": (41.00, 29.00),
}

WORLD_COVER_NAMES = {
    10: "Tree",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "BuiltUp",
    60: "BareSparse",
    70: "SnowIce",
    80: "Water",
    90: "Wetland",
    95: "Mangrove",
    100: "MossLichen",
}
PREFERRED_WORLD_COVER = {30, 40}
EXCLUDED_WORLD_COVER = {50, 80}


def haversine_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance in kilometres."""
    radius_km = 6371.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat0)) * np.cos(np.radians(lat)) * np.sin(dlon / 2.0) ** 2
    )
    return radius_km * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def worldcover_tile_url(lat: float, lon: float) -> str:
    """ESA WorldCover 2021 public S3 tile URL for a lon/lat point."""
    tile_lat = int(np.floor(lat / 3.0) * 3)
    tile_lon = int(np.floor(lon / 3.0) * 3)
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    return (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        f"ESA_WorldCover_10m_2021_v200_{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}_Map.tif"
    )


def query_worldcover(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Nearest-neighbour ESA WorldCover lookup for candidate points."""
    import rasterio
    from rasterio.windows import from_bounds

    lat_min, lat_max = float(lats.min() - 0.05), float(lats.max() + 0.05)
    lon_min, lon_max = float(lons.min() - 0.05), float(lons.max() + 0.05)

    urls: set[str] = set()
    for lat in np.arange(np.floor(lat_min / 3.0) * 3.0, np.ceil(lat_max / 3.0) * 3.0 + 0.1, 3.0):
        for lon in np.arange(np.floor(lon_min / 3.0) * 3.0, np.ceil(lon_max / 3.0) * 3.0 + 0.1, 3.0):
            urls.add(worldcover_tile_url(float(lat), float(lon)))

    cover_values: list[np.ndarray] = []
    cover_lats: list[np.ndarray] = []
    cover_lons: list[np.ndarray] = []
    for url in sorted(urls):
        try:
            with rasterio.open("/vsicurl/" + url) as src:
                window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
                if window.width < 1 or window.height < 1:
                    continue
                data = src.read(1, window=window)
                transform = src.window_transform(window)
                rows, cols = np.mgrid[0:data.shape[0], 0:data.shape[1]]
                cover_values.append(data.ravel())
                cover_lats.append((transform.f + rows * transform.e).ravel())
                cover_lons.append((transform.c + cols * transform.a).ravel())
        except Exception as exc:
            print(f"WorldCover lookup skipped for {url}: {exc}", flush=True)

    if not cover_values:
        return np.zeros(len(lats), dtype=np.int16)

    values = np.concatenate(cover_values)
    wc_lat = np.concatenate(cover_lats)
    wc_lon = np.concatenate(cover_lons)
    result = np.zeros(len(lats), dtype=np.int16)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        dist2 = (wc_lat - lat) ** 2 + (wc_lon - lon) ** 2
        result[i] = int(values[np.argmin(dist2)])
    return result


def robust01(values: pd.Series) -> pd.Series:
    """Robust 0-1 scaling for rural candidate ranking."""
    x = values.astype("float64")
    med = x.median(skipna=True)
    x = x.fillna(med if np.isfinite(med) else 0.0)
    lo = x.quantile(0.05)
    hi = x.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x), dtype=np.float64), index=values.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def static_feature_frame(city: str, static_root: Path) -> pd.DataFrame:
    """Load released static features as a pixel-indexed table."""
    path = static_root / city / "static_features.npz"
    if not path.exists():
        return pd.DataFrame(columns=["pixel_id"])
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z["feat_names"]]
    feats = z["features"].astype(np.float32)
    out = pd.DataFrame({"pixel_id": z["pixel_ids"].astype(np.int64)})
    for col in ("BCR", "road_density", "nightlight", "water_ratio", "ndvi", "mean_height"):
        if col in names:
            out[col] = feats[:, names.index(col)]
    return out


def add_rural_score(candidates: pd.DataFrame) -> pd.DataFrame:
    """Lower score means more rural-like."""
    out = candidates.copy()
    for col in ("BCR", "road_density", "nightlight", "water_ratio", "ndvi", "mean_height"):
        if col not in out:
            out[col] = np.nan
    bcr = robust01(out["BCR"])
    road = robust01(out["road_density"])
    night = robust01(out["nightlight"])
    height = robust01(out["mean_height"])
    water = out["water_ratio"].fillna(0.0).clip(lower=0.0)
    ndvi = robust01(out["ndvi"])
    out["rural_score"] = (
        2.0 * bcr
        + 1.0 * road
        + 0.7 * night
        + 0.6 * height
        + 2.0 * (water > 0.20).astype(float)
        - 0.4 * ndvi
    )
    return out


def select_rural_pixels(
    grid: pd.DataFrame,
    city: str,
    center: tuple[float, float] | None = None,
    ring_km: tuple[float, float] = (15.0, 25.0),
    min_pixels: int = 5,
    max_pixels: int = 400,
    landcover: np.ndarray | None = None,
    static_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select rural reference pixels from a fixed annulus around the city center."""
    center = center or CITY_CENTERS.get(city)
    if center is None:
        center = (float(grid["lat"].mean()), float(grid["lon"].mean()))
    lat0, lon0 = center

    candidates = grid[["pixel_id", "lat", "lon"]].copy()
    candidates["dist_km"] = haversine_km(lat0, lon0, candidates["lat"].to_numpy(), candidates["lon"].to_numpy())
    lo, hi = ring_km
    candidates = candidates[(candidates["dist_km"] >= lo) & (candidates["dist_km"] <= hi)].copy()
    if len(candidates) == 0:
        raise RuntimeError(f"{city}: no candidate pixels in {lo}-{hi} km annulus")

    if landcover is not None:
        if len(landcover) != len(candidates):
            raise ValueError("landcover length must match annulus candidate count")
        candidates["landcover"] = landcover.astype(np.int16)
        preferred = candidates[candidates["landcover"].isin(PREFERRED_WORLD_COVER)].copy()
        if len(preferred) >= min_pixels:
            candidates = preferred
        else:
            candidates = candidates[~candidates["landcover"].isin(EXCLUDED_WORLD_COVER)].copy()

    if static_features is not None and not static_features.empty:
        candidates = candidates.merge(static_features, on="pixel_id", how="left")
        candidates = add_rural_score(candidates).sort_values("rural_score")
    else:
        candidates["rural_score"] = np.nan

    if len(candidates) < min_pixels:
        raise RuntimeError(f"{city}: only {len(candidates)} rural candidates after filtering")

    selected = candidates.head(max_pixels).copy()
    selected["ring_min_km"] = float(lo)
    selected["ring_max_km"] = float(hi)
    if "landcover" in selected:
        selected["landcover_name"] = [WORLD_COVER_NAMES.get(int(v), str(int(v))) for v in selected["landcover"]]
    return selected.reset_index(drop=True)


def load_grid_from_csv(path: Path) -> pd.DataFrame:
    grid = pd.read_csv(path)
    required = {"pixel_id", "lat", "lon"}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return grid[["pixel_id", "lat", "lon"]].copy()


def build_rural_timeseries(
    observations: pd.DataFrame,
    rural_pixel_ids: list[int] | np.ndarray,
    value_col: str,
) -> pd.DataFrame:
    """Compute T_rural(t) as the mean over selected rural pixels."""
    rural_ids = set(int(x) for x in rural_pixel_ids)
    rural = observations[observations["pixel_id"].astype(int).isin(rural_ids)]
    return (
        rural.groupby("datetime", observed=True)[value_col]
        .agg(rural_mean="mean", n_rural_valid="count")
        .reset_index()
    )


def compute_uhi(
    observations: pd.DataFrame,
    rural_ts: pd.DataFrame,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Subtract the rural reference series from every pixel observation."""
    merged = observations.merge(rural_ts[["datetime", "rural_mean"]], on="datetime", how="left")
    merged[out_col] = (merged[value_col] - merged["rural_mean"]).astype(np.float32)
    return merged


def compute_lstuhi_year(
    input_parquet: Path,
    rural_pixels_csv: Path,
    output_parquet: Path,
    value_col: str = "lst_K",
    out_col: str = "lst_uhi_K",
) -> None:
    observations = pd.read_parquet(input_parquet)
    observations["datetime"] = pd.to_datetime(observations["datetime"])
    rural_ids = pd.read_csv(rural_pixels_csv)["pixel_id"].astype(int).to_numpy()
    rural_ts = build_rural_timeseries(observations, rural_ids, value_col)
    uhi = compute_uhi(observations, rural_ts, value_col, out_col)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    uhi.to_parquet(output_parquet, index=False)


def compute_era5_ta_uhi_year(
    input_parquet: Path,
    rural_pixels_csv: Path,
    output_parquet: Path,
    value_col: str = "t2m",
) -> None:
    observations = pd.read_parquet(input_parquet, columns=["datetime", "pixel_id", value_col])
    observations["datetime"] = pd.to_datetime(observations["datetime"])
    observations["pixel_id"] = observations["pixel_id"].astype(np.int32)
    observations[value_col] = observations[value_col].astype(np.float32)
    rural_ids = pd.read_csv(rural_pixels_csv)["pixel_id"].astype(int).to_numpy()
    rural_ts = build_rural_timeseries(observations, rural_ids, value_col)
    uhi = compute_uhi(observations, rural_ts, value_col, "uhi")
    uhi = uhi.rename(columns={"uhi": "uhi_era5_K"})
    uhi["uhi"] = uhi["uhi_era5_K"].astype(np.float32)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    uhi[["datetime", "pixel_id", "uhi_era5_K", "uhi"]].to_parquet(output_parquet, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select rural-ring pixels or compute UHI anomalies")
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select")
    select.add_argument("--city", required=True)
    select.add_argument("--grid", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    select.add_argument("--ring-min-km", type=float, default=15.0)
    select.add_argument("--ring-max-km", type=float, default=25.0)
    select.add_argument("--min-pixels", type=int, default=5)
    select.add_argument("--max-pixels", type=int, default=400)
    select.add_argument("--use-worldcover", action="store_true")
    select.add_argument("--static-root", type=Path, default=DATA_ROOT / "static_features")

    lst = sub.add_parser("lst-uhi")
    lst.add_argument("--input", type=Path, required=True)
    lst.add_argument("--rural-pixels", type=Path, required=True)
    lst.add_argument("--out", type=Path, required=True)
    lst.add_argument("--value-col", default="lst_K")

    era5 = sub.add_parser("era5-uhi")
    era5.add_argument("--input", type=Path, required=True)
    era5.add_argument("--rural-pixels", type=Path, required=True)
    era5.add_argument("--out", type=Path, required=True)
    era5.add_argument("--value-col", default="t2m")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        grid = load_grid_from_csv(args.grid)
        annulus = grid.copy()
        center = CITY_CENTERS.get(args.city)
        if center is None:
            center = (float(grid["lat"].mean()), float(grid["lon"].mean()))
        dist = haversine_km(center[0], center[1], annulus["lat"].to_numpy(), annulus["lon"].to_numpy())
        annulus = annulus[(dist >= args.ring_min_km) & (dist <= args.ring_max_km)].copy()
        landcover = query_worldcover(annulus["lat"].to_numpy(), annulus["lon"].to_numpy()) if args.use_worldcover else None
        static_features = static_feature_frame(args.city, args.static_root)
        selected = select_rural_pixels(
            grid,
            args.city,
            center=center,
            ring_km=(args.ring_min_km, args.ring_max_km),
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            landcover=landcover,
            static_features=static_features,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.out, index=False)
        manifest = {
            "city": args.city,
            "ring_km": [args.ring_min_km, args.ring_max_km],
            "n_rural_pixels": int(len(selected)),
            "pixel_ids": [int(x) for x in selected["pixel_id"].to_numpy()],
        }
        args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    elif args.command == "lst-uhi":
        compute_lstuhi_year(args.input, args.rural_pixels, args.out, value_col=args.value_col)
        print(f"wrote {args.out}")
    elif args.command == "era5-uhi":
        compute_era5_ta_uhi_year(args.input, args.rural_pixels, args.out, value_col=args.value_col)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
