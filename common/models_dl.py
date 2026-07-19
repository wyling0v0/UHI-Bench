"""DL baselines for interpolation (GPU). IGNNK-style inductive GNN imputer.

Works on irregular point sets (no 2D raster needed) → reusable for both
Task 1a (LST cloud-gap) and Task 1b (AirT sparse stations).

Node features per timestamp: [value(0 if masked), obs_mask(1/0), *static_feats].
GCN aggregates neighbour info → predicts UHI at every node.
Training: take a full (clear) grid, randomly mask a subset, predict masked, MSE.
Inductive: same graph + static, eval on arbitrary masks (real cloud / random keep).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, eye
from scipy.sparse.csgraph import laplacian


def build_knn_adj(xy, k=12, self_loop=True):
    """Symmetric kNN adjacency [N,N] sparse. Returns normalised D^-1/2 (A+I) D^-1/2."""
    N = len(xy)
    k = min(k, N - 1)
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k + 1)            # incl. self
    rows = np.repeat(np.arange(N), k + 1)
    cols = idx.ravel()
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A = A.maximum(A.T)                            # symmetrise
    if self_loop:
        A = A + eye(N)
    # D^-1/2 A D^-1/2
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.power(deg, -0.5, where=deg > 0)
    dinv[~np.isfinite(dinv)] = 0.0
    D = csr_matrix((dinv, (np.arange(N), np.arange(N))), shape=(N, N))
    An = D @ A @ D
    return An


def sparse_to_torch(An, device):
    An = An.tocoo()
    idx = torch.tensor(np.vstack([An.row, An.col]), dtype=torch.long, device=device)
    val = torch.tensor(An.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, An.shape).coalesce()


class GCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out)
    def forward(self, x, An):
        # x: [B,N,F] → batched sparse mm via reshape to [N, B*F]
        B, N, Fd = x.shape
        x2 = x.permute(1, 0, 2).reshape(N, B * Fd)
        out = torch.sparse.mm(An, x2).reshape(N, B, Fd).permute(1, 0, 2)  # [B,N,F]
        return F.gelu(self.lin(out))


class GNNImputer(nn.Module):
    """IGNNK-style: value + mask + static → GCN stack → predicted value."""
    def __init__(self, n_static, hidden=64, n_layers=3, drop=0.1):
        super().__init__()
        din = 2 + n_static                      # [value, obs_mask, *static]
        layers = [GCNLayer(din, hidden)]
        for _ in range(n_layers - 1):
            layers.append(GCNLayer(hidden, hidden))
        self.gcn = nn.ModuleList(layers)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden, 1)

    def forward(self, value, mask, static, An):
        # value: [B,N] (masked→0), mask:[B,N] (1 obs / 0 masked), static:[N,F] (shared)
        B, N = value.shape
        static_exp = static.unsqueeze(0).expand(B, -1, -1)           # [B,N,F]
        x = torch.cat([value.unsqueeze(-1), mask.unsqueeze(-1), static_exp], dim=-1)
        for g in self.gcn:
            x = self.drop(g(x, An))
        return self.head(x).squeeze(-1)                              # [B,N]


def train_gnn(field, train_idx, n_static, device, epochs=30, bs=8, lr=5e-4,
              hidden=64, k=12, seed=0):
    """Train GNN on clear-scene timestamps (full grid). Returns trained model + An."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xy = field.xy_km.astype(np.float32)
    An = build_knn_adj(xy, k=k)
    An_t = sparse_to_torch(An, device)
    # standardise static
    st = field.feats.astype(np.float32)
    mu = np.nanmean(st, 0); sd = np.nanstd(st, 0); sd[sd < 1e-8] = 1
    st = np.nan_to_num((st - mu) / sd)
    static_t = torch.tensor(st, device=device)

    # standardise values per-city (global mean/std over training clear grids)
    V = field.values[train_idx].astype(np.float32)            # [T,N]
    valid = np.isfinite(V)                                     # [T,N] bool
    vmu = float(np.nanmean(V)); vsd = float(np.nanstd(V)) + 1e-6
    Vz = np.nan_to_num((V - vmu) / vsd, nan=0.0)              # NaN→0 (excluded from loss via valid)

    model = GNNImputer(n_static, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Vz_t = torch.tensor(Vz, device=device)                    # [T,N]
    valid_t = torch.tensor(valid.astype(np.float32), device=device)
    T = len(train_idx)
    losses = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(T)
        ep_loss = 0.0; nb = 0
        for i in range(0, T, bs):
            bidx = perm[i:i+bs]
            x = Vz_t[bidx]                                     # [b,N]
            vd = valid_t[bidx]                                 # [b,N]
            m = ((torch.rand_like(x) > 0.3).float()) * vd      # observe only valid pixels
            inp = x * m                                        # masked→0
            pred = model(inp, m, static_t, An_t)
            tgt = (1 - m) * vd                                  # hidden & valid → loss target
            num = (pred * tgt - x * tgt) ** 2
            loss = num.sum() / (tgt.sum() + 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / max(nb, 1))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    gnn ep {ep+1}/{epochs}  loss={losses[-1]:.4f}", flush=True)
    model.eval()
    model.vmu, model.vsd = vmu, vsd
    return model, An_t, static_t


def eval_gnn_on_masks(model, An_t, static_t, scene_vals, masks_list, device, max_pred=500, seed=0):
    """For each boolean mask [N] (True=hidden): predict hidden pixels, return abs errs.
    scene_vals: [N] ground-truth (clear). masks_list: list of np bool arrays."""
    rng = np.random.default_rng(seed)
    vmu, vsd = model.vmu, model.vsd
    vz = ((scene_vals.astype(np.float64) - vmu) / vsd).astype(np.float32)
    vz_t = torch.tensor(vz, device=device).unsqueeze(0)       # [1,N]
    model.eval()
    all_ae = []
    with torch.no_grad():
        for m in masks_list:
            mnb = torch.tensor(m.astype(np.float32), device=device).unsqueeze(0)
            inp = vz_t * (1 - mnb)                            # hidden→0, observed kept
            obs_mask = (1 - mnb)                             # 1=observed
            pred = model(inp, obs_mask, static_t, An_t)[0].cpu().numpy()
            pred_phys = pred * vsd + vmu
            pred_phys = np.where(np.isfinite(pred_phys), pred_phys, vmu)  # NaN→city mean (safeguard)
            hide = np.where(m & np.isfinite(scene_vals))[0]
            if len(hide) > max_pred:
                hide = rng.choice(hide, max_pred, replace=False)
            all_ae.append(np.abs(pred_phys[hide] - scene_vals[hide]))
    return np.concatenate(all_ae) if all_ae else np.array([])
