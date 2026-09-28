# UHI-Bench

Public benchmark code for **UHI-Bench: Benchmarking Urban Heat Island Modeling
Across Data Sources, Cities, and Climate Regimes**.

This repository contains experiment runners, common data loaders, and baseline
implementations. The benchmark dataset is distributed separately on Hugging Face:
<https://huggingface.co/datasets/WyLing0v0/uhi-bench>.

## Repository Layout

- `common/`: shared path configuration, data loaders, masks, covariate loaders,
  and baseline models.
- `1a/`: cross-source LST-UHI / AirT-UHI consistency diagnostics.
- `1b/`: heat-risk timing and extreme-event detection diagnostics.
- `2a/`: LST-UHI cloud-gap reconstruction.
- `2b/`: AirT-UHI sparse reconstruction.
- `2c/`: dual-source UHI forecasting.
- `2d/`: driver attribution and mechanism-stability summaries.
- `3/`: source-set transfer and directed climate-pair transfer.
- `data_pipeline/`: public static-feature construction and rural reference-ring
  selection utilities.

Large data files, raw source exports, generated caches, figures, and experiment
outputs are intentionally excluded from this GitHub release.

## Data Setup

Download or clone the dataset from Hugging Face, then point the code to that
directory:

```bash
git lfs install
git clone https://huggingface.co/datasets/WyLing0v0/uhi-bench /path/to/uhi-bench-data
export UHI_BENCH_DATA=/path/to/uhi-bench-data
```

The expected dataset root contains:

- `lstuhi_1km_hourly/`
- `hostrada_uhi/`
- `atuhi_ood_1km_hourly/`
- `temporal_weather/`
- `static_features/`
- `station_uhi/`

Optional environment variables:

```bash
export UHI_BENCH_OUTPUT=/path/to/outputs
export UHI_BENCH_CACHE=/path/to/cache
```

If unset, outputs and caches are written under `./outputs/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional foundation-model baselines require additional packages:

```bash
pip install -r requirements-fm.txt
```

Some optional model packages can be version-sensitive; the non-FM benchmark
scripts do not require them.

Optional data-construction utilities require additional geospatial packages:

```bash
pip install -r requirements-pipeline.txt
```

## License

The code in this GitHub repository is released under the MIT License; see
`LICENSE`.

The benchmark dataset is distributed separately on Hugging Face and follows the
license and third-party data-source terms described in the dataset card. The
dataset license is not changed by the MIT license used for this code repository.

External model packages, pretrained model weights, and third-party data sources
retain their own licenses and terms of use.
