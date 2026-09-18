"""Path configuration for the public UHI-Bench code release.

Set ``UHI_BENCH_DATA`` to the root of the released dataset, for example a
local clone/download of https://anonymous-hf.com/a/ybz41u8gi970/.
If the variable is not set, the code tries common local locations next to this
repository.
"""
from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


def _default_data_root() -> Path:
    env = os.environ.get("UHI_BENCH_DATA")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in (
        CODE_ROOT / "release",
        CODE_ROOT.parent / "release",
        CODE_ROOT / "data",
        CODE_ROOT.parent / "data",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (CODE_ROOT.parent / "release").resolve()


DATA_ROOT = _default_data_root()
OUTPUT_ROOT = Path(os.environ.get("UHI_BENCH_OUTPUT", CODE_ROOT / "outputs")).expanduser()
CACHE_ROOT = Path(os.environ.get("UHI_BENCH_CACHE", OUTPUT_ROOT / "cache")).expanduser()

LST_BASE = DATA_ROOT / "lstuhi_1km_hourly"
TA_BASE = DATA_ROOT / "hostrada_uhi"
PSEUDO_TA_BASE = DATA_ROOT / "atuhi_ood_1km_hourly"
ATUHI_BASE = PSEUDO_TA_BASE
HOSTRADA_BASE = TA_BASE
ERA5_BASE = DATA_ROOT / "temporal_weather"
STATIC_BASE = DATA_ROOT / "static_features"
STATION_BASE = DATA_ROOT / "station_uhi"


def public_data_note() -> str:
    return (
        f"UHI_BENCH_DATA={DATA_ROOT}. Set UHI_BENCH_DATA to the root of the "
        "released dataset if files are not found."
    )
