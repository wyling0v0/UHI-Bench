"""Task 3 — FM adapter fine-tuning (Chronos-Bolt frozen backbone + trainable MLP head).

For each source config (Cfb-4/Diverse-4/Cfb-7/Diverse-7/Diverse-8), train a
lightweight MLP head on top of the frozen Chronos-Bolt encoder embeddings,
using source-city pixel-hour data. Then evaluate on the 6 OOD targets. This
makes FM participate in the source-set diversity comparison (source-dependent,
unlike zero-shot).

Architecture:
  1. Chronos-Bolt-Small encoder (frozen) -> embedding [D]
  2. MLP head (trainable): [D + extra features] -> scalar prediction

Training: source cities 2015-2022, L2 features (history + ERA5), per-pixel-hour.
Eval: 6 OOD targets 2023-2025, same protocol as run_3_ood_transfer.py.

Usage:
  python benchmark/3/run_3_fm_finetune.py --source-set diverse4 --device cuda:0
  python benchmark/3/run_3_fm_finetune.py --source-set diverse8 --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common.data import _load_static
from run_3_ood_transfer import (
    SOURCE_SETS, OOD_CITIES, DRIVERS, HIST_LAGS, ERA5_LAGS, LOOKBACK,
    CityYearData, load_city_year, build_samples, concat_batches, SampleBatch,
    layer_matrix, fit_feature_scaler, apply_feature_scaler,
    regression_metrics, metrics_by_city, stable_city_seed,
)

OUT_DIR = Path(__file__).resolve().parent / "results"


class FMAdapter(nn.Module):
    """Frozen Chronos-Bolt encoder + trainable MLP head."""
    def __init__(self, pipe, extra_dim, hidden=128):
        super().__init__()
        self.pipe = pipe
        # freeze encoder
        for p in self.pipe.model.parameters():
            p.requires_grad = False
        # determine embedding dim by dry run
        self.embed_dim = self._probe_embed_dim()
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim + extra_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def _probe_embed_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 32, device=next(self.pipe.model.parameters()).device)
            emb = self.pipe.model.encode(dummy)
            if isinstance(emb, tuple):
                emb = emb[0]
            return emb.shape[-1]

    def forward(self, context, extra_features):
        """context: [B, T] float, extra_features: [B, F] float."""
        with torch.no_grad():
            emb = self.pipe.model.encode(context)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = emb.mean(dim=1)  # [B, D] pool over time
        x = torch.cat([emb, extra_features], dim=-1)
        return self.head(x).squeeze(-1)


def train_adapter(pipe, source_batch, layer, device, epochs=15, lr=1e-3, bs=256):
    """Train MLP head on source data with frozen FM encoder."""
    x_extra = layer_matrix(source_batch, layer)  # [N, F]
    mu, sd = fit_feature_scaler(x_extra)
    x_extra_z = apply_feature_scaler(x_extra, mu, sd)
    y = source_batch.y

    # build context sequences (168h lookback per sample, padded)
    # We need the raw LST series per sample — reconstruct from build_samples' internals
    # For efficiency, use the lag features as context proxy (shorter than full 168h)
    # Use last 32 valid values as context (Chronos-Bolt handles short context)
    contexts = []
    for i in range(len(y)):
        l1 = source_batch.x_l1[i]
        # reconstruct a short context from lag values (first 9 features are lag vals)
        lag_vals = l1[:9]
        contexts.append(lag_vals.astype(np.float32))
    contexts = np.array(contexts)

    model = FMAdapter(pipe, extra_dim=x_extra_z.shape[1]).to(device)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ctx_t = torch.from_numpy(contexts).to(device)
    ext_t = torch.from_numpy(x_extra_z).to(device)
    y_t = torch.from_numpy(y).to(device)

    n = len(y_t)
    rng = np.random.default_rng(42)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            pred = model(ctx_t[idx], ext_t[idx])
            loss = loss_fn(pred, y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0:
            print(f"    adapter ep {ep+1}/{epochs} loss={tot/n:.4f}", flush=True)

    model.eval()
    return model, mu, sd


def eval_adapter(model, ood_batch, layer, mu, sd, device):
    """Evaluate on OOD data."""
    x_extra = layer_matrix(ood_batch, layer)
    x_extra_z = apply_feature_scaler(x_extra, mu, sd)
    contexts = ood_batch.x_l1[:, :9].astype(np.float32)  # lag vals as context

    ctx_t = torch.from_numpy(contexts).to(device)
    ext_t = torch.from_numpy(x_extra_z).to(device)
    with torch.no_grad():
        pred = model(ctx_t, ext_t).cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-set", choices=list(SOURCE_SETS.keys()), default="diverse4")
    ap.add_argument("--source-cities", nargs="+", default=None)
    ap.add_argument("--ood-cities", nargs="+", default=OOD_CITIES)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    ap.add_argument("--layers", nargs="+", default=["L2"])
    ap.add_argument("--target-kind", choices=["lst", "airt"], default="lst")
    ap.add_argument("--train-years", type=int, nargs="+", default=list(range(2015, 2023)))
    ap.add_argument("--eval-years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n-pixels", type=int, default=128)
    ap.add_argument("--train-samples-per-city-year", type=int, default=1000)
    ap.add_argument("--eval-samples-per-city-year", type=int, default=500)
    ap.add_argument("--min-valid-ratio", type=float, default=0.70)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Defaults to results/3_fm_ft_<source-set>.json.")
    a = ap.parse_args()

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                               device_map=a.device, torch_dtype="auto")

    source_cities = a.source_cities if a.source_cities else SOURCE_SETS[a.source_set]
    sname = a.source_set if not a.source_cities else "custom"
    out_path = Path(a.out) if a.out else OUT_DIR / f"3_fm_ft_{sname}.json"
    print(f"[fm-finetune] target={a.target_kind} source={sname}: {source_cities}", flush=True)

    result = {
        "task": "Task 3 FM adapter fine-tune",
        "fm": "Chronos-Bolt-Small (frozen) + MLP head",
        "target_kind": a.target_kind,
        "source_set": sname, "source_cities": source_cities,
        "ood_cities": a.ood_cities,
        "protocol": {
            "train_years": a.train_years,
            "eval_years": a.eval_years,
            "horizons": a.horizons,
            "layers": a.layers,
            "n_pixels": a.n_pixels,
            "train_samples_per_city_year": a.train_samples_per_city_year,
            "eval_samples_per_city_year": a.eval_samples_per_city_year,
            "min_valid_ratio": a.min_valid_ratio,
            "epochs": a.epochs,
            "seed": a.seed,
            "note": "Adapter head is trained on source-city train_years only; OOD target labels are eval-only.",
        },
        "horizons": {},
    }

    for horizon in a.horizons:
        print(f"\n[horizon] +{horizon}h", flush=True)
        # build source train batch
        src_batches = []
        for city in source_cities:
            for year in a.train_years:
                data = load_city_year(city, year, a.n_pixels, a.seed, a.target_kind)
                b = build_samples(data, horizon, a.train_samples_per_city_year,
                                  a.min_valid_ratio, stable_city_seed(a.seed, city, year, horizon))
                src_batches.append(b)
        source_batch = concat_batches(src_batches)
        print(f"  source samples: {len(source_batch.y)}", flush=True)

        # build OOD eval batch
        ood_batches = []
        for city in a.ood_cities:
            for year in a.eval_years:
                data = load_city_year(city, year, a.n_pixels, a.seed, a.target_kind)
                b = build_samples(data, horizon, a.eval_samples_per_city_year,
                                  a.min_valid_ratio, stable_city_seed(a.seed, city, year, horizon + 1000))
                ood_batches.append(b)
        ood_batch = concat_batches(ood_batches)
        print(f"  OOD samples: {len(ood_batch.y)}", flush=True)

        if len(ood_batch.y) == 0:
            continue

        h_out = {"models": {}}
        for layer in a.layers:
            print(f"  training adapter ({layer})...", flush=True)
            model, mu, sd = train_adapter(pipe, source_batch, layer, a.device,
                                          epochs=a.epochs)
            pred = eval_adapter(model, ood_batch, layer, mu, sd, a.device)
            name = f"ChronosAdapter_{layer}"
            h_out["models"][name] = {
                "overall": regression_metrics(ood_batch.y, pred),
                "by_city": metrics_by_city(ood_batch.y, pred, ood_batch.city),
                "source_set": sname, "input_layer": layer,
            }
            mae = h_out["models"][name]["overall"]["MAE"]
            print(f"  {name}: OOD avg MAE={mae:.3f}" if mae else f"  {name}: NA", flush=True)

        result["horizons"][f"{horizon}h"] = h_out

    # also add Persistence for reference
    for horizon in a.horizons:
        h_key = f"{horizon}h"
        if h_key in result["horizons"]:
            ood_batches = []
            for city in a.ood_cities:
                for year in a.eval_years:
                    data = load_city_year(city, year, a.n_pixels, a.seed, a.target_kind)
                    b = build_samples(data, horizon, a.eval_samples_per_city_year,
                                      a.min_valid_ratio, stable_city_seed(a.seed, city, year, horizon + 1000))
                    ood_batches.append(b)
            ood_batch = concat_batches(ood_batches)
            if len(ood_batch.y):
                result["horizons"][h_key]["models"]["Persistence"] = {
                    "overall": regression_metrics(ood_batch.y, ood_batch.persistence),
                    "by_city": metrics_by_city(ood_batch.y, ood_batch.persistence, ood_batch.city),
                }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
