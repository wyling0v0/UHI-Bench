"""Static environmental feature construction for UHI-Bench.

This public pipeline keeps the reproducible processing logic while relying on
user-configured authentication and output paths.

Outputs are compatible with the benchmark loaders:

    static_features/{city}/static_features.npz
    static_features/{city}/temporal_static/{year}.npz

Feature order:

    BCR, road_density, poi_density, nightlight, ndvi, water_ratio,
    distance_to_waterbody, mean_height, dem, wind_exposure_proxy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from common.paths import DATA_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.paths import DATA_ROOT


PIPELINE_ROOT = Path(os.environ.get("UHI_BENCH_PIPELINE_ROOT", DATA_ROOT)).expanduser()
STATIC_DIR = PIPELINE_ROOT / "static_features"
MSG_LST_DIR = PIPELINE_ROOT / "msg_lst"

GRID_DEG = 0.01
GSW_MAX_YEAR = 2021

SEASONS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

FEATURE_COLS = [
    "BCR",
    "road_density",
    "poi_density",
    "nightlight",
    "ndvi",
    "water_ratio",
    "distance_to_waterbody",
    "mean_height",
    "dem",
    "wind_exposure_proxy",
]


CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "berlin": (13.00, 52.30, 13.80, 52.80),
    "hamburg": (9.70, 53.25, 10.35, 53.75),
    "munich": (11.10, 47.83, 11.95, 48.47),
    "frankfurt": (8.40, 49.95, 8.95, 50.35),
    "cologne": (6.75, 50.75, 7.20, 51.10),
    "stuttgart": (8.95, 48.60, 9.45, 48.90),
    "dortmund": (7.30, 51.40, 7.70, 51.70),
    "dusseldorf": (6.65, 51.10, 6.95, 51.35),
    "warsaw": (20.65, 51.85, 21.35, 52.60),
    "bucharest": (25.75, 44.10, 26.45, 44.80),
    "cairo": (30.85, 29.70, 31.65, 30.45),
    "riyadh": (46.35, 24.35, 47.10, 25.05),
    "lagos": (2.95, 6.20, 3.75, 6.90),
    "johannesburg": (27.70, -26.60, 28.45, -25.85),
    "sao_paulo": (-47.05, -24.05, -46.30, -23.20),
    "buenos_aires": (-58.75, -34.95, -57.95, -34.25),
    "tehran": (51.05, 35.40, 51.80, 36.05),
    "khartoum": (32.15, 15.20, 32.90, 15.95),
    "casablanca": (-7.95, 33.20, -7.20, 33.95),
    "istanbul": (28.60, 40.80, 29.45, 41.35),
}


def city_dir(city: str) -> Path:
    return STATIC_DIR / city


def gee_dir(city: str) -> Path:
    return city_dir(city) / "gee"


def osm_dir(city: str) -> Path:
    return city_dir(city) / "osm"


def temporal_dir(city: str) -> Path:
    return city_dir(city) / "temporal_static"


def grid_csv(city: str) -> Path:
    preferred = city_dir(city) / "grid_centers_1km.csv"
    alternate = city_dir(city) / "grid_centers.csv"
    if preferred.exists():
        return preferred
    if alternate.exists():
        return alternate
    return preferred


def output_grid_csv(city: str) -> Path:
    return city_dir(city) / "grid_centers_1km.csv"


def grid_gpkg(city: str) -> Path:
    return city_dir(city) / "grid.gpkg"


def msg_grid_csv(city: str) -> Path:
    return MSG_LST_DIR / city / "grid_centers.csv"


def ensure_dirs(city: str) -> None:
    for path in (
        city_dir(city),
        gee_dir(city),
        osm_dir(city),
        temporal_dir(city),
        gee_dir(city) / "annual",
        gee_dir(city) / "ndvi",
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_grid(city: str) -> pd.DataFrame:
    path = grid_csv(city)
    if not path.exists():
        raise FileNotFoundError(f"Missing grid file for {city}: {path}")
    return pd.read_csv(path)


def earth_engine_init():
    """Initialize Earth Engine through user-configured authentication."""
    import ee

    project = os.environ.get("EE_PROJECT")
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine is not initialized. Run `earthengine authenticate` or "
            "configure application-default credentials, then set EE_PROJECT if "
            "your account requires an explicit project."
        ) from exc
    return ee


def download_ee_tif(ee_image, out_path: Path, bbox: tuple[float, float, float, float],
                    scale: int = 1000, retries: int = 3) -> bool:
    """Download an Earth Engine image as a WGS84 GeoTIFF."""
    import requests

    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    params = {
        "crs": "EPSG:4326",
        "scale": scale,
        "region": [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
        ],
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    for attempt in range(1, retries + 1):
        try:
            url = ee_image.getDownloadURL(params)
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            out_path.write_bytes(response.content)
            return True
        except Exception as exc:
            print(f"download failed ({attempt}/{retries}) for {out_path.name}: {exc}", flush=True)
    return False


def sample_tif_at_points(tif_path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample a WGS84 GeoTIFF at lon/lat grid centers."""
    import rasterio
    from rasterio.transform import rowcol

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        rows, cols = rowcol(src.transform, lons, lats)
    valid = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out = np.full(len(lons), np.nan, dtype=np.float32)
    out[valid] = arr[rows[valid], cols[valid]]
    return out


def step_grid(city: str, bbox: tuple[float, float, float, float] | None = None) -> None:
    """Build a 0.01 degree WGS84 reference grid and projected grid polygons."""
    import geopandas as gpd
    from shapely.geometry import box

    bbox = bbox or CITY_BBOXES[city]
    ensure_dirs(city)
    lon_min, lat_min, lon_max, lat_max = bbox
    lons = np.arange(lon_min, lon_max + GRID_DEG * 0.5, GRID_DEG)
    lats = np.arange(lat_min, lat_max + GRID_DEG * 0.5, GRID_DEG)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid = pd.DataFrame(
        {
            "pixel_id": np.arange(lon_grid.size, dtype=np.int32),
            "lon": lon_grid.ravel(),
            "lat": lat_grid.ravel(),
        }
    )
    grid.to_csv(output_grid_csv(city), index=False)

    half = GRID_DEG / 2.0
    gdf = gpd.GeoDataFrame(
        grid,
        geometry=[
            box(row.lon - half, row.lat - half, row.lon + half, row.lat + half)
            for row in grid.itertuples()
        ],
        crs="EPSG:4326",
    )
    gdf.to_crs(gdf.estimate_utm_crs()).to_file(grid_gpkg(city), driver="GPKG")
    print(f"[{city}] grid: {len(grid)} pixels")


def step_gee_static(city: str) -> None:
    """Download DEM, GHS built-up surface, and GHS building height."""
    ee = earth_engine_init()
    bbox = CITY_BBOXES[city]
    ensure_dirs(city)

    images = {
        "dem": ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().rename("dem"),
        "impervious": (
            ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
            .filter(ee.Filter.eq("system:index", "2020"))
            .first()
            .select("built_surface")
            .rename("impervious")
        ),
        "mean_height": (
            ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_H")
            .first()
            .select("built_height")
            .rename("mean_height")
        ),
    }
    for name, image in images.items():
        ok = download_ee_tif(image, gee_dir(city) / f"{name}.tif", bbox)
        print(f"[{city}] {name}: {'ok' if ok else 'failed'}")


def step_gee_annual(city: str, years: list[int]) -> None:
    """Download annual nightlight, surface-water fraction, and wind speed."""
    ee = earth_engine_init()
    bbox = CITY_BBOXES[city]
    ensure_dirs(city)

    for year in years:
        out_dir = gee_dir(city) / "annual" / str(year)
        nightlight = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select("avg_rad")
            .mean()
            .rename("nightlight")
        )
        download_ee_tif(nightlight, out_dir / "nightlight.tif", bbox)

        gsw_year = min(year, GSW_MAX_YEAR)
        water = (
            ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
            .filter(ee.Filter.eq("year", gsw_year))
            .first()
            .select("waterClass")
            .gte(2)
            .toFloat()
            .rename("water_ratio")
        )
        download_ee_tif(water, out_dir / "water_ratio.tif", bbox)

        era5 = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
            .mean()
        )
        wind = era5.expression(
            "sqrt(u*u + v*v)",
            {
                "u": era5.select("u_component_of_wind_10m"),
                "v": era5.select("v_component_of_wind_10m"),
            },
        ).rename("wind_speed")
        download_ee_tif(wind, out_dir / "wind_speed.tif", bbox)
        print(f"[{city}] annual rasters: {year}")


def step_gee_ndvi(city: str, years: list[int]) -> None:
    """Download Sentinel-2 seasonal NDVI with scene-classification masking."""
    ee = earth_engine_init()
    bbox = CITY_BBOXES[city]
    geom = ee.Geometry.Rectangle(list(bbox))
    ensure_dirs(city)

    ranges = {
        "DJF": lambda y: [(f"{y}-01-01", f"{y}-03-01"), (f"{y}-12-01", f"{y + 1}-01-01")],
        "MAM": lambda y: [(f"{y}-03-01", f"{y}-06-01")],
        "JJA": lambda y: [(f"{y}-06-01", f"{y}-09-01")],
        "SON": lambda y: [(f"{y}-09-01", f"{y}-12-01")],
    }

    def scl_mask(image):
        scl = image.select("SCL")
        keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return image.updateMask(keep)

    def seasonal_ndvi(parts):
        collections = [
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filterBounds(geom)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(scl_mask)
            for start, end in parts
        ]
        merged = collections[0]
        for collection in collections[1:]:
            merged = merged.merge(collection)
        return merged.map(lambda image: image.normalizedDifference(["B8", "B4"]).rename("ndvi")).median()

    for year in years:
        for season, get_ranges in ranges.items():
            out = gee_dir(city) / "ndvi" / f"{year}_{season}.tif"
            download_ee_tif(seasonal_ndvi(get_ranges(year)), out, bbox)
        print(f"[{city}] NDVI: {year}")


def step_osm(city: str) -> None:
    """Aggregate OSM road length, POI count, water fraction, and water distance."""
    import geopandas as gpd
    import osmnx as ox
    from shapely.ops import unary_union

    out_csv = osm_dir(city) / "osm_features.csv"
    if out_csv.exists():
        print(f"[{city}] OSM features already exist")
        return
    ensure_dirs(city)

    grid = gpd.read_file(grid_gpkg(city))
    utm_crs = grid.crs
    bbox = CITY_BBOXES[city]
    cell_area = float(grid.geometry.area.mean())
    grid_indexed = grid.set_index("pixel_id")

    graph = ox.graph_from_bbox(bbox=bbox, network_type="all")
    roads = ox.graph_to_gdfs(graph, nodes=False).to_crs(utm_crs)
    road_join = gpd.sjoin(roads[["geometry"]], grid[["pixel_id", "geometry"]], how="inner", predicate="intersects")
    road_rows = []
    for pixel_id, group in road_join.groupby("pixel_id"):
        length_km = group.geometry.intersection(grid_indexed.at[pixel_id, "geometry"]).length.sum() / 1000.0
        road_rows.append({"pixel_id": pixel_id, "road_density": length_km})
    road_df = grid[["pixel_id"]].merge(pd.DataFrame(road_rows), on="pixel_id", how="left").fillna(0.0)

    tags = {"amenity": True, "shop": True, "office": True, "leisure": True}
    pois = ox.features_from_bbox(bbox=bbox, tags=tags)
    pois = pois[pois.geometry.geom_type == "Point"].to_crs(utm_crs)
    poi_join = gpd.sjoin(pois[["geometry"]], grid[["pixel_id", "geometry"]], how="inner", predicate="within")
    poi_df = (
        grid[["pixel_id"]]
        .merge(poi_join.groupby("pixel_id").size().reset_index(name="poi_density"), on="pixel_id", how="left")
        .fillna(0.0)
    )

    water_tags = {"natural": "water", "waterway": "riverbank", "landuse": ["reservoir", "basin"]}
    water = ox.features_from_bbox(bbox=bbox, tags=water_tags)
    water = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(utm_crs)
    water_intersection = gpd.overlay(water[["geometry"]], grid[["pixel_id", "geometry"]], how="intersection")
    water_intersection["water_area"] = water_intersection.geometry.area
    water_df = (
        grid[["pixel_id"]]
        .merge(
            water_intersection.groupby("pixel_id")["water_area"].sum().reset_index(),
            on="pixel_id",
            how="left",
        )
        .fillna({"water_area": 0.0})
    )
    water_df["water_ratio"] = (water_df["water_area"] / cell_area).clip(upper=1.0)
    water_df = water_df[["pixel_id", "water_ratio"]]

    big_water = water[water.geometry.area >= 10_000]
    if len(big_water):
        water_union = unary_union(big_water.geometry.values)
        centroids = grid.set_index("pixel_id").geometry.centroid
        distance_df = pd.DataFrame(
            {
                "pixel_id": centroids.index.values,
                "distance_to_waterbody": centroids.distance(water_union).values / 1000.0,
            }
        )
    else:
        distance_df = pd.DataFrame({"pixel_id": grid["pixel_id"].values, "distance_to_waterbody": np.nan})

    out = road_df.merge(poi_df, on="pixel_id").merge(water_df, on="pixel_id").merge(distance_df, on="pixel_id")
    out.to_csv(out_csv, index=False)
    print(f"[{city}] OSM features: {len(out)} pixels")


def step_align(city: str) -> None:
    """Sample permanent rasters and merge OSM features into `raster_aligned.csv`."""
    grid = load_grid(city)
    lons = grid["lon"].to_numpy()
    lats = grid["lat"].to_numpy()
    out = grid[["pixel_id"]].copy()
    for col, tif in (
        ("dem", "dem.tif"),
        ("BCR_raw", "impervious.tif"),
        ("mean_height", "mean_height.tif"),
    ):
        path = gee_dir(city) / tif
        out[col] = sample_tif_at_points(path, lons, lats) if path.exists() else np.nan

    # GHS built_surface is an area quantity at 100 m support. Dividing by
    # 10,000 converts m2 per 100 m cell to a fraction.
    out["BCR"] = (out.pop("BCR_raw") / 10_000.0).clip(0.0, 1.0)

    osm_csv = osm_dir(city) / "osm_features.csv"
    if osm_csv.exists():
        out = out.merge(pd.read_csv(osm_csv), on="pixel_id", how="left")
    else:
        for col in ("road_density", "poi_density", "water_ratio", "distance_to_waterbody"):
            out[col] = np.nan
    out.to_csv(city_dir(city) / "raster_aligned.csv", index=False)
    print(f"[{city}] aligned rasters")


def nearest_ndvi_year(city: str, year: int, season: str) -> int | None:
    candidates = []
    for path in (gee_dir(city) / "ndvi").glob(f"*_{season}.tif"):
        first = path.stem.split("_")[0]
        if first.isdigit():
            candidates.append(int(first))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (abs(candidate - year), -candidate))


def step_align_temporal(city: str, years: list[int]) -> None:
    """Build `temporal_static/{year}.npz` from annual and seasonal rasters."""
    grid = load_grid(city)
    pixel_ids = grid["pixel_id"].to_numpy(dtype=np.int64)
    lons = grid["lon"].to_numpy()
    lats = grid["lat"].to_numpy()
    temporal_dir(city).mkdir(parents=True, exist_ok=True)

    for year in years:
        arrays: dict[str, np.ndarray] = {"pixel_ids": pixel_ids}
        annual_dir = gee_dir(city) / "annual" / str(year)
        for key in ("nightlight", "water_ratio", "wind_speed"):
            path = annual_dir / f"{key}.tif"
            arrays[key] = sample_tif_at_points(path, lons, lats) if path.exists() else np.full(len(pixel_ids), np.nan, dtype=np.float32)

        for season in SEASONS:
            path = gee_dir(city) / "ndvi" / f"{year}_{season}.tif"
            if not path.exists():
                donor = nearest_ndvi_year(city, year, season)
                path = gee_dir(city) / "ndvi" / f"{donor}_{season}.tif" if donor is not None else path
            arrays[f"ndvi_{season}"] = sample_tif_at_points(path, lons, lats) if path.exists() else np.full(len(pixel_ids), np.nan, dtype=np.float32)

        np.savez_compressed(temporal_dir(city) / f"{year}.npz", **arrays)
        print(f"[{city}] temporal features: {year}")


def wind_exposure_proxy(wind_speed: np.ndarray, bcr: np.ndarray) -> np.ndarray:
    """Compute wind exposure as wind_speed times one minus normalized BCR."""
    lo = np.nanmin(bcr)
    hi = np.nanmax(bcr)
    bcr_norm = (bcr - lo) / (hi - lo + 1e-6)
    return (wind_speed * (1.0 - bcr_norm)).astype(np.float32)


def step_assemble(city: str, snapshot_year: int = 2024, snapshot_season: str = "JJA") -> None:
    """Assemble `static_features.npz` with the benchmark feature order."""
    import geopandas as gpd

    grid = load_grid(city)
    aligned_path = city_dir(city) / "raster_aligned.csv"
    if not aligned_path.exists():
        raise FileNotFoundError(f"Missing aligned raster table: {aligned_path}")
    df = grid.merge(pd.read_csv(aligned_path), on="pixel_id", how="left")

    temporal_path = temporal_dir(city) / f"{snapshot_year}.npz"
    if temporal_path.exists():
        temporal = np.load(temporal_path, allow_pickle=True)
        temp = pd.DataFrame({"pixel_id": temporal["pixel_ids"].astype(np.int64)})
        for key in ("nightlight", "water_ratio", "wind_speed", f"ndvi_{snapshot_season}"):
            if key in temporal:
                temp[key] = temporal[key].astype(np.float32)
        df = df.merge(temp, on="pixel_id", how="left")
        if f"ndvi_{snapshot_season}" in df:
            df["ndvi"] = df[f"ndvi_{snapshot_season}"]
        if "wind_speed" in df:
            df["wind_exposure_proxy"] = wind_exposure_proxy(df["wind_speed"].to_numpy(np.float32), df["BCR"].to_numpy(np.float32))

    for col in FEATURE_COLS:
        if col not in df:
            df[col] = np.nan

    gdf = gpd.read_file(grid_gpkg(city)).set_index("pixel_id")
    ordered = gdf.loc[df["pixel_id"].to_numpy()]
    xy = np.column_stack([ordered.geometry.centroid.x, ordered.geometry.centroid.y]).astype(np.float32)
    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    np.savez_compressed(
        city_dir(city) / "static_features.npz",
        pixel_ids=df["pixel_id"].to_numpy(dtype=np.int64),
        xy=xy,
        feat_names=np.array(FEATURE_COLS),
        features=features,
    )
    print(f"[{city}] static_features.npz")


def step_aggregate3km(city: str, years: list[int]) -> None:
    """Aggregate 1 km static and temporal features to MSG-grid centers."""
    import geopandas as gpd
    import pyproj
    from scipy.spatial import KDTree

    msg_path = msg_grid_csv(city)
    static_path = city_dir(city) / "static_features.npz"
    if not msg_path.exists():
        raise FileNotFoundError(f"Missing MSG grid centers: {msg_path}")
    if not static_path.exists():
        raise FileNotFoundError(f"Missing static feature file: {static_path}")

    msg = pd.read_csv(msg_path)
    utm_crs = gpd.read_file(grid_gpkg(city)).crs
    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    msg_x, msg_y = transformer.transform(msg["lon"].to_numpy(), msg["lat"].to_numpy())
    xy_msg = np.column_stack([msg_x, msg_y])

    static = np.load(static_path, allow_pickle=True)
    xy_1km = static["xy"]
    features_1km = static["features"].astype(np.float32)
    assignment = KDTree(xy_msg).query(xy_1km, workers=-1)[1]
    n_msg = len(msg)
    counts = np.bincount(assignment, minlength=n_msg).astype(np.int32)

    features_3km = np.full((n_msg, features_1km.shape[1]), np.nan, dtype=np.float32)
    for idx in range(features_1km.shape[1]):
        col = features_1km[:, idx].astype(np.float64)
        valid = np.isfinite(col)
        sums = np.bincount(assignment[valid], weights=col[valid], minlength=n_msg)
        nums = np.bincount(assignment[valid], minlength=n_msg).astype(np.float64)
        ok = nums > 0
        features_3km[ok, idx] = (sums[ok] / nums[ok]).astype(np.float32)

    out_dir = MSG_LST_DIR / city
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "features_3km_static.npz",
        pixel_ids=msg["pixel_id"].to_numpy(dtype=np.int32),
        xy_utm=xy_msg.astype(np.float64),
        features=features_3km,
        feat_names=static["feat_names"],
        assign_counts=counts,
        assign_1km_to_3km=assignment.astype(np.int32),
    )

    temporal_keys = ("nightlight", "water_ratio", "wind_speed", "ndvi_DJF", "ndvi_MAM", "ndvi_JJA", "ndvi_SON")
    for year in years:
        src = temporal_dir(city) / f"{year}.npz"
        if not src.exists():
            continue
        z = np.load(src, allow_pickle=True)
        arrays = {"pixel_ids": msg["pixel_id"].to_numpy(dtype=np.int32)}
        for key in temporal_keys:
            col = z[key].astype(np.float64) if key in z else np.full(len(assignment), np.nan)
            valid = np.isfinite(col)
            sums = np.bincount(assignment[valid], weights=col[valid], minlength=n_msg)
            nums = np.bincount(assignment[valid], minlength=n_msg).astype(np.float64)
            agg = np.full(n_msg, np.nan, dtype=np.float32)
            ok = nums > 0
            agg[ok] = (sums[ok] / nums[ok]).astype(np.float32)
            arrays[key] = agg
        np.savez_compressed(out_dir / f"features_3km_{year}.npz", **arrays)
    print(f"[{city}] aggregated features to MSG grid")


STEPS = {
    "gee_static": lambda city, years: step_gee_static(city),
    "gee_annual": lambda city, years: step_gee_annual(city, years),
    "gee_ndvi": lambda city, years: step_gee_ndvi(city, years),
    "osm": lambda city, years: step_osm(city),
    "align": lambda city, years: step_align(city),
    "align_temporal": lambda city, years: step_align_temporal(city, years),
    "assemble": lambda city, years: step_assemble(city),
    "aggregate3km": lambda city, years: step_aggregate3km(city, years),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UHI-Bench static environmental features")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--city")
    group.add_argument("--cities", nargs="+")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Optional WGS84 bounding box for --city when building a new grid.",
    )
    parser.add_argument("--step", nargs="+", default=["all"], choices=["grid", *STEPS, "all"])
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2026)))
    parser.add_argument("--list-cities", action="store_true")
    args = parser.parse_args()
    if args.list_cities:
        return args
    if not args.city and not args.cities:
        parser.error("one of --city, --cities, or --list-cities is required")
    if args.bbox and not args.city:
        parser.error("--bbox can only be used with --city")
    return args


def main() -> None:
    args = parse_args()
    if args.list_cities:
        for name, bbox in CITY_BBOXES.items():
            print(name, bbox)
        return
    cities = [args.city] if args.city else (list(CITY_BBOXES) if "all" in args.cities else args.cities)
    steps = ["grid", *STEPS] if "all" in args.step else args.step
    for city in cities:
        if city not in CITY_BBOXES and not args.bbox:
            raise KeyError(f"Unknown city: {city}")
        print(f"\n[{city}] steps={steps}")
        for step in steps:
            if step == "grid":
                step_grid(city, bbox=tuple(args.bbox) if args.bbox else None)
            else:
                STEPS[step](city, args.years)


if __name__ == "__main__":
    main()
