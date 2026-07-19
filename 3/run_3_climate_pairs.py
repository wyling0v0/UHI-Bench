#!/usr/bin/env python3
"""Run Task 3-B directed climate-pair transfer experiments.

Each run trains on one source city over 2015-2022 and evaluates the other
held-out target cities over 2023-2025. The default matrix uses seven climate
representatives and excludes the diagonal, giving 42 directed source->target
pairs in seven source-group runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common.paths import OUTPUT_ROOT  # noqa: E402


PAIR_CITIES = [
    "hamburg",
    "warsaw",
    "buenos_aires",
    "johannesburg",
    "lagos",
    "riyadh",
    "cologne",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cities", nargs="+", default=PAIR_CITIES)
    ap.add_argument("--target-kinds", nargs="+", default=["lst", "airt"], choices=["lst", "airt"])
    ap.add_argument("--runners", nargs="+", default=["xgboost", "chronos_adapter"],
                    choices=["xgboost", "chronos_adapter"])
    ap.add_argument("--models", nargs="+", default=["persistence", "climatology", "xgboost"],
                    choices=["persistence", "climatology", "xgboost"])
    ap.add_argument("--layers", nargs="+", default=["L1"],
                    choices=["L1", "L1S", "L2", "L3"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    ap.add_argument("--train-years", type=int, nargs="+", default=list(range(2015, 2023)))
    ap.add_argument("--eval-years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n-pixels", type=int, default=128)
    ap.add_argument("--train-samples-per-city-year", type=int, default=1500)
    ap.add_argument("--eval-samples-per-city-year", type=int, default=800)
    ap.add_argument("--min-valid-ratio", type=float, default=0.70)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--fm-train-samples-per-city-year", type=int, default=1000)
    ap.add_argument("--fm-eval-samples-per-city-year", type=int, default=500)
    ap.add_argument("--fm-epochs", type=int, default=15)
    ap.add_argument("--fm-device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(OUTPUT_ROOT / "task3b_climate_pairs_7city"))
    ap.add_argument("--force", action="store_true", help="Re-run pairs even if output JSON exists.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xgb_runner = BENCH / "3" / "run_3_ood_transfer.py"
    fm_runner = BENCH / "3" / "run_3_fm_finetune.py"
    runs = [(src, [tgt for tgt in args.cities if tgt != src]) for src in args.cities]
    n_pairs = sum(len(targets) for _, targets in runs)
    print(
        f"[task3-b] {n_pairs} directed pairs; target_kinds={args.target_kinds}; "
        f"runners={args.runners}; layers={args.layers}",
        flush=True,
    )

    total_runs = len(args.target_kinds) * len(runs) * len(args.runners)
    run_idx = 0
    for target_kind in args.target_kinds:
        for src, targets in runs:
            target_label = "_".join(targets)
            layer_label = "-".join(args.layers)
            if "xgboost" in args.runners:
                run_idx += 1
                out = out_dir / "xgboost" / target_kind / f"3b_xgboost_{target_kind}_{src}_to_{target_label}_{layer_label}.json"
                if out.exists() and not args.force:
                    print(f"[{run_idx:02d}/{total_runs}] skip xgboost {target_kind} {src}->{targets}: {out}", flush=True)
                else:
                    cmd = [
                        sys.executable, str(xgb_runner),
                        "--source-cities", src,
                        "--ood-cities", *targets,
                        "--target-kind", target_kind,
                        "--models", *args.models,
                        "--layers", *args.layers,
                        "--horizons", *map(str, args.horizons),
                        "--train-years", *map(str, args.train_years),
                        "--eval-years", *map(str, args.eval_years),
                        "--n-pixels", str(args.n_pixels),
                        "--train-samples-per-city-year", str(args.train_samples_per_city_year),
                        "--eval-samples-per-city-year", str(args.eval_samples_per_city_year),
                        "--min-valid-ratio", str(args.min_valid_ratio),
                        "--n-estimators", str(args.n_estimators),
                        "--max-depth", str(args.max_depth),
                        "--seed", str(args.seed),
                        "--out", str(out),
                    ]
                    print(f"[{run_idx:02d}/{total_runs}] xgboost {target_kind} {src}->{targets}", flush=True)
                    subprocess.run(cmd, cwd=BENCH, check=True)

            if "chronos_adapter" in args.runners:
                run_idx += 1
                out = out_dir / "chronos_adapter" / target_kind / f"3b_chronos_adapter_{target_kind}_{src}_to_{target_label}_{layer_label}.json"
                if out.exists() and not args.force:
                    print(f"[{run_idx:02d}/{total_runs}] skip chronos_adapter {target_kind} {src}->{targets}: {out}", flush=True)
                else:
                    cmd = [
                        sys.executable, str(fm_runner),
                        "--source-cities", src,
                        "--ood-cities", *targets,
                        "--target-kind", target_kind,
                        "--layers", *args.layers,
                        "--horizons", *map(str, args.horizons),
                        "--train-years", *map(str, args.train_years),
                        "--eval-years", *map(str, args.eval_years),
                        "--n-pixels", str(args.n_pixels),
                        "--train-samples-per-city-year", str(args.fm_train_samples_per_city_year),
                        "--eval-samples-per-city-year", str(args.fm_eval_samples_per_city_year),
                        "--min-valid-ratio", str(args.min_valid_ratio),
                        "--epochs", str(args.fm_epochs),
                        "--seed", str(args.seed),
                        "--device", args.fm_device,
                        "--out", str(out),
                    ]
                    print(f"[{run_idx:02d}/{total_runs}] chronos_adapter {target_kind} {src}->{targets}", flush=True)
                    subprocess.run(cmd, cwd=BENCH, check=True)

    print(f"[task3-b] outputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
