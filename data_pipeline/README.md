# Data-Construction Pipeline

This directory contains public UHI-Bench data-construction utilities for
reproducibility checks.

## Static Environmental Features

`static_features_pipeline.py` builds or re-aligns the 1 km static and
temporal-static feature stack used by the benchmark:

- build a city grid from a WGS84 bounding box,
- reuse an existing released `grid_centers_1km.csv` or `grid_centers.csv` when
  present,
- retrieve or align raster layers for DEM, built-up coverage, building height,
  VIIRS nightlight, JRC surface water, ERA5-Land wind speed, and Sentinel-2 NDVI,
- aggregate OSM roads, POIs, and water bodies to the grid,
- assemble `static_features.npz` and `temporal_static/{year}.npz`,
- optionally aggregate 1 km features to a coarser MSG grid.

The optional geospatial and Earth Engine dependencies are listed in
`requirements-pipeline.txt`.

Earth Engine authentication is expected to be configured by the user through the
standard Earth Engine CLI or application-default credentials. Raw source exports
are generated under the configured data root.

## Rural Reference Selection

`rural_reference.py` implements the rural reference-ring selection and UHI
anomaly construction:

1. compute the great-circle distance from each candidate pixel to the city
   center,
2. keep pixels in the 15-25 km annulus by default,
3. when land-cover labels are available, prefer cropland and grassland pixels,
4. otherwise rank candidates by low built-up coverage, low road density, low
   nightlight, low building height, low water fraction, and higher NDVI,
5. compute the rural reference series and subtract it from each urban pixel.

The target formula is:

```text
T_rural,c(t) = mean_{q in R_c} T(q, t)
UHI_c(p, t) = T_c(p, t) - T_rural,c(t)
```

where `R_c` is the selected rural reference set for city `c`.
