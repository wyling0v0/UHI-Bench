"""Task 2c — LST-UHI supervised DL forecasting (LSTM + PatchTST), 4 input configs.

Self-contained sequence forecaster on LST clear windows (reuses Task-3 data load).
Configs (channel sets):
  L1           : [LST(filled), mask]                      (2 ch)
  +ERA5        : L1 + 6 ERA5 drivers                      (8 ch)
  +static      : L1 + 10 static (broadcast over time)     (12 ch)
  +ERA5+static : L1 + ERA5 + static                       (18 ch)

Target: lst_uhi_K at t+H. lookback=168h, min_valid_ratio=0.70. train 2015-2022,
eval 2023-2025. Per-config standardization fit on train. Subsamples pixels/samples.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3"))
import run_3_ood_transfer as T3   # noqa: E402

LOOKBACK = T3.LOOKBACK
HORIZONS = [1, 6, 12, 24, 48, 96]
CONFIGS = ["L1", "+ERA5", "+static", "+ERA5+static"]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def build_seq(data, horizon, max_samples, min_valid_ratio, seed):
    """Return dict config -> X[n,C,168] (channel-first for conv-friendly), y[n], plus masks.
    Uses T3.candidate_rows for clear-window selection."""
    t_idx, p_idx, vr = T3.candidate_rows(data.values, horizon, min_valid_ratio, max_samples, seed)
    n = len(t_idx)
    if n == 0:
        return None
    lst = np.empty((n, LOOKBACK), np.float32); msk = np.empty_like(lst)
    era = np.empty((n, LOOKBACK, len(T3.DRIVERS)), np.float32)
    sta = np.empty((n, data.static.shape[1]), np.float32)
    y = np.empty(n, np.float32)
    for i, (tt, pp) in enumerate(zip(t_idx, p_idx)):
        ie = int(tt - horizon); s = ie - LOOKBACK + 1
        seg = data.values[s:ie+1, pp]
        finite = np.isfinite(seg)
        fill = np.where(finite, seg, np.nanmean(seg) if finite.any() else 0.0)
        lst[i] = fill; msk[i] = finite.astype(np.float32)
        era[i] = data.era5[s:ie+1, pp, :]
        sta[i] = data.static[pp]
        y[i] = data.values[tt, pp]
    # assemble channel-first [n, C, 168]
    cfgX = {}
    lst_t = lst[:, None, :]              # [n,1,168]
    msk_t = msk[:, None, :]
    l1 = np.concatenate([lst_t, msk_t], axis=1)
    cfgX["L1"] = l1
    era_t = np.transpose(era, (0, 2, 1))  # [n,6,168]
    cfgX["+ERA5"] = np.concatenate([l1, era_t], axis=1)
    sta_t = np.broadcast_to(sta[:, :, None], (n, sta.shape[1], LOOKBACK))
    cfgX["+static"] = np.concatenate([l1, sta_t], axis=1).astype(np.float32)
    cfgX["+ERA5+static"] = np.concatenate([l1, era_t, sta_t], axis=1).astype(np.float32)
    return cfgX, y


class LSTMReg(nn.Module):
    def __init__(self, cin, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(cin, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):  # x [B, C, T] -> [B]
        z = x.transpose(1, 2)  # [B, T, C]
        _, (h_n, _) = self.lstm(z)
        return self.head(h_n[-1]).squeeze(-1)


class PatchTSTReg(nn.Module):
    def __init__(self, cin, hidden=64, patch=24, nhead=2):
        super().__init__()
        self.patch = patch; self.tok = nn.Linear(patch * cin, hidden)
        self.pos = nn.Parameter(torch.zeros(LOOKBACK // patch, hidden))
        enc = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):  # x [B, C, T]
        B = x.shape[0]
        xp = x.unfold(-1, self.patch, self.patch)           # [B, C, npatch, patch]
        xp = xp.permute(0, 2, 1, 3).reshape(B, xp.shape[2], -1)  # [B, npatch, C*patch]
        z = self.tok(xp) + self.pos[None]
        z = self.tr(z)
        return self.head(z.mean(1)).squeeze(-1)


class DLinearReg(nn.Module):
    """DLinear: moving-avg trend + residual seasonal, each channel-mixed then linear over time."""
    def __init__(self, cin, ks=25):
        super().__init__()
        self.ks = ks
        self.mix_t = nn.Linear(cin, 1)
        self.mix_s = nn.Linear(cin, 1)
        self.lin_t = nn.Linear(LOOKBACK, 1)
        self.lin_s = nn.Linear(LOOKBACK, 1)
    def forward(self, x):  # [B,C,T]
        pad = self.ks // 2
        t = nn.functional.avg_pool1d(x, self.ks, stride=1, padding=pad)[..., :LOOKBACK]
        s = x - t
        # mix channels: [B,C,T] -> [B,T,C] -> Linear(C,1) -> [B,T]
        tm = self.mix_t(t.transpose(1, 2)).squeeze(-1)
        sm = self.mix_s(s.transpose(1, 2)).squeeze(-1)
        return (self.lin_t(tm) + self.lin_s(sm)).squeeze(-1)


class iTransformerReg(nn.Module):
    """iTransformer: attention over variates (each channel's time series = one token)."""
    def __init__(self, cin, hidden=64, nhead=2):
        super().__init__()
        self.embed = nn.Linear(LOOKBACK, hidden)
        enc = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):  # [B,C,T]
        tok = self.embed(x)              # [B, C, hidden]
        z = self.tr(tok)
        return self.head(z.mean(1)).squeeze(-1)


def zfit(X):
    mu = X.mean((0, 2), keepdims=True); sd = X.std((0, 2), keepdims=True) + 1e-6
    return mu, sd


def train_model(model, Xtr, ytr, Xev, yev, epochs=20, bs=256, lr=1e-3):
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE); ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xev_t = torch.from_numpy(Xev).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(Xtr)
    for ep in range(epochs):
        model.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            l = lossf(model(Xtr_t[b]), ytr_t[b]); l.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xev_t).cpu().numpy()
    return float(np.mean(np.abs(pred - yev)))


def run_city(city, train_years, eval_years, n_pixels, max_samples, epochs, seed):
    print(f"\n{'='*50}\n[1d-LST-DL] {city}  pixels={n_pixels}")
    res = {"city": city, "horizons": HORIZONS, "configs": CONFIGS, "methods": {}}
    for h in HORIZONS:
        tr = T3.concat_batches([T3.build_samples(T3.load_city_year(city, y, n_pixels, seed), h, max_samples, 0.70, seed+y) for y in train_years])
        ev = T3.concat_batches([T3.build_samples(T3.load_city_year(city, y, n_pixels, seed), h, max_samples, 0.70, seed+y+999) for y in eval_years])
        # build seq from data is heavy; instead reconstruct windows from raw data per sample
        # (use build_seq on the full CityYearData to keep windows aligned)
        tr_data = [T3.load_city_year(city, y, n_pixels, seed) for y in train_years]
        ev_data = [T3.load_city_year(city, y, n_pixels, seed) for y in eval_years]
        # concat data across years for window building
        from run_3_ood_transfer import CityYearData
        # simplest: build per year then pool
        def pool_seq(data_list, sd):
            Xc={c:[] for c in CONFIGS}; Y=[]
            for k,d in enumerate(data_list):
                r = build_seq(d, h, max_samples, 0.70, sd+k)
                if r is None: continue
                cfgX,y = r
                for c in CONFIGS: Xc[c].append(cfgX[c])
                Y.append(y)
            Y=np.concatenate(Y)
            return {c: np.concatenate(Xc[c]) for c in CONFIGS}, Y
        Xtr_c, ytr = pool_seq(tr_data, seed)
        Xev_c, yev = pool_seq(ev_data, seed+999)
        for cfg in CONFIGS:
            mu, sd = zfit(Xtr_c[cfg])
            Xtr_z = ((Xtr_c[cfg]-mu)/sd).astype(np.float32)
            Xev_z = ((Xev_c[cfg]-mu)/sd).astype(np.float32)
            cin = Xtr_z.shape[1]
            for mname, Mk in [("LSTM", LSTMReg), ("PatchTST", PatchTSTReg),
                              ("DLinear", DLinearReg), ("iTransformer", iTransformerReg)]:
                torch.manual_seed(seed)
                mdl = Mk(cin).to(DEVICE)
                mae = train_model(mdl, Xtr_z, ytr, Xev_z, yev, epochs=epochs)
                res["methods"].setdefault(f"{mname}({cfg})", {})[f"{h}h"] = round(mae, 4)
        dones = [m for m in res["methods"] if f"{h}h" in res["methods"][m]]
        print(f"  h={h}h  " + "  ".join(f"{m.split('(')[0]}({m.split('(')[1][:-1]})={res['methods'][m][f'{h}h']}" for m in dones))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["munich", "berlin"])
    ap.add_argument("--train_years", type=int, nargs="+", default=list(range(2015, 2023)))
    ap.add_argument("--eval_years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_pixels", type=int, default=128)
    ap.add_argument("--max_samples", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    for city in a.cities:
        r = run_city(city, a.train_years, a.eval_years, a.n_pixels, a.max_samples, a.epochs, a.seed)
        (out_dir / f"1d_lst_dl_{city}.json").write_text(json.dumps(r, indent=2))
    print(f"\n[saved] {out_dir}/1d_lst_dl_*.json")


if __name__ == "__main__":
    main()
