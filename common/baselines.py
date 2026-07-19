"""Standalone field-interpolation baselines (Layer1 minimal / Layer2 +static).

Unified interface: each baseline is callable
    pred = f(coords_obs, vals_obs, coords_pred, feat_obs=None, feat_pred=None)
All coords in km.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ── IDW ──────────────────────────────────────────────────────────────────────
class IDW:
    def __init__(self, k=8, p=2.0):
        self.k, self.p = k, p

    def __call__(self, co, vo, cp, feat_obs=None, feat_pred=None):
        tree = cKDTree(co)
        k = min(self.k, len(co))
        d, idx = tree.query(cp, k=k)                       # [Np,k]
        d = np.maximum(d, 1e-3)
        w = 1.0 / d ** self.p
        w /= w.sum(axis=1, keepdims=True)
        nbr = vo[idx]                                      # [Np,k]
        return (nbr * w).sum(axis=1)


# ── Local Ordinary / Regression Kriging ──────────────────────────────────────
def _fit_variogram_gaussian(co, vo, n_pairs=20000, n_lags=14):
    """Robust variogram for STANDARDIZED values (variance ≈ 1 → sill fixed = 1).
    Returns (sill=1.0, range_km, nugget). Range = lag where empirical γ first
    reaches 0.63*sill (Gaussian 1-e^-1 point), by interpolation."""
    n = len(co)
    rng = np.random.default_rng(0)
    m = min(n_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=m); j = rng.integers(0, n, size=m)
    ok = i != j; i, j = i[ok], j[ok]
    h = np.linalg.norm(co[i] - co[j], axis=1)
    g = 0.5 * (vo[i] - vo[j]) ** 2
    hmax = max(float(np.percentile(h, 90)), 1.0)
    edges = np.linspace(0, hmax, n_lags + 1)
    hc, gv = [], []
    for b in range(n_lags):
        sel = (h >= edges[b]) & (h < edges[b + 1])
        if sel.sum() > 3:
            hc.append(0.5 * (edges[b] + edges[b + 1])); gv.append(float(g[sel].mean()))
    if len(hc) < 2:
        return 1.0, hmax * 0.5, 0.1
    hc = np.array(hc); gv = np.array(gv)
    nugget = float(np.clip(gv[0] * 0.5, 0.0, 0.3))
    target = 0.63
    gv_c = np.maximum(gv - nugget, 0) / max(1 - nugget, 1e-3)   # rescale to [0,1]
    if gv_c.max() < target:
        range_ = hmax
    else:
        idx = int(np.where(gv_c >= target)[0][0])
        if idx == 0:
            range_ = hc[0]
        else:
            x0, x1 = hc[idx - 1], hc[idx]; y0, y1 = gv_c[idx - 1], gv_c[idx]
            range_ = x0 + (target - y0) / max(y1 - y0, 1e-6) * (x1 - x0)
    range_ = float(np.clip(range_, 0.5, hmax))
    return 1.0, range_, nugget


class LocalKriging:
    """Local Ordinary Kriging with a Gaussian variogram and fixed-k
    neighbourhoods, vectorised over target pixels. Ignores features (L1).
    Pass `vario=(sill,range,nugget)` to reuse a city-level fit (avoid refit/pair)."""
    def __init__(self, k=10):
        self.k = k
        self.vario = None

    def set_vario(self, vario):
        self.vario = vario

    def _solve(self, co, vo, cp):
        # standardize values → unit variance so kriging system is well-conditioned
        mu = float(np.mean(vo)); sd = float(np.std(vo)) + 1e-9
        vz = (vo - mu) / sd
        if self.vario is not None:
            sill, range_, nugget = self.vario
        else:
            sill, range_, nugget = _fit_variogram_gaussian(co, vz)
        nugget = max(nugget, 0.2)       # floor: prevents over-confident extreme extrapolation
        if sill < 1e-6:
            return None
        tree = cKDTree(co)
        k = min(self.k, len(co))
        d, idx = tree.query(cp, k=k)                        # [P,k]
        vz_nbr = vz[idx]                                    # [P,k]
        co_nbr = co[idx]                                    # [P,k,2]
        gamma_tn = nugget + sill * (1 - np.exp(-(d / range_) ** 2))
        h_nn = np.linalg.norm(co_nbr[:, :, None, :] - co_nbr[:, None, :, :], axis=3)
        gamma_nn = nugget + sill * (1 - np.exp(-(h_nn / range_) ** 2))
        di = np.arange(k); gamma_nn[:, di, di] = 0.0
        P = cp.shape[0]
        A = np.empty((P, k + 1, k + 1), np.float64)
        A[:, :k, :k] = gamma_nn
        A[:, :k, k] = 1.0; A[:, k, :k] = 1.0; A[:, k, k] = 0.0
        A += 1e-3 * np.eye(k + 1)                          # stronger ridge → stable weights
        rhs = np.empty((P, k + 1), np.float64)
        rhs[:, :k] = 1.0 - gamma_tn / max(sill, 1e-9)
        rhs[:, k] = 1.0
        sol = np.linalg.solve(A, rhs[..., None])[..., 0]
        pred = mu + sd * (sol[:, :k] * vz_nbr).sum(axis=1)
        # safeguard: clip to observed range (guards against ill-conditioned blow-up)
        return np.clip(pred, vo.min(), vo.max())

    def __call__(self, co, vo, cp, feat_obs=None, feat_pred=None):
        pred = self._solve(co, vo, cp)
        if pred is None:
            return IDW()(co, vo, cp)  # fallback
        return pred.astype(np.float32)


class RegressionKriging:
    """Regression Kriging with RF external drift (a.k.a. RF-Kriging hybrid):
    fit a Random Forest trend on [coords + static], ordinary-krige the residuals,
    add back. Residual kriging is SKIPPED when residuals show no spatial structure
    (range too short) — then returns the RF trend alone, so RK never underperforms
    its own trend. L2 variant (uses static)."""
    def __init__(self, k=10):
        self.krig = LocalKriging(k=k)

    def set_vario(self, vario):
        self.krig.set_vario(vario)

    def __call__(self, co, vo, cp, feat_obs=None, feat_pred=None):
        if feat_obs is None or (hasattr(feat_obs, "shape") and feat_obs.shape[1] == 0):
            return self.krig(co, vo, cp)
        from sklearn.ensemble import RandomForestRegressor
        Xtr = np.hstack([co, feat_obs])
        if len(co) > 2000:
            sel = np.random.default_rng(0).choice(len(co), 2000, replace=False)
            Xtr_, vo_ = Xtr[sel], vo[sel]
        else:
            Xtr_, vo_ = Xtr, vo
        rf = RandomForestRegressor(n_estimators=60, n_jobs=4, random_state=0,
                                   min_samples_leaf=5).fit(Xtr_, vo_)
        trend_tr = rf.predict(Xtr)
        trend_te = rf.predict(np.hstack([cp, feat_pred]))
        resid = vo - trend_tr
        # fit residual variogram; only krige if residuals have spatial structure
        mu = float(resid.mean()); sd = float(resid.std()) + 1e-9
        rz = (resid - mu) / sd
        sill, range_, nugget = _fit_variogram_gaussian(co, rz)
        kr = self.krig._solve(co, resid, cp)
        if kr is None or range_ < 1.0:        # no spatial structure → trend only
            return trend_te.astype(np.float32)
        return (trend_te + kr).astype(np.float32)


class _TreeReg:
    """RF/XGBoost regressor on [x,y,(+feat)] trained on observed pixels.
    Caps training samples (ML generalises; full obs not needed) for speed."""
    def __init__(self, kind="rf", max_train=2000):
        self.kind = kind
        self.max_train = max_train
    def _X(self, c, f):
        return c if (f is None or f.shape[1]==0) else np.hstack([c, f])
    def __call__(self, co, vo, cp, feat_obs=None, feat_pred=None):
        if len(co) > self.max_train:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(co), self.max_train, replace=False)
            co, vo = co[sel], vo[sel]
            if feat_obs is not None: feat_obs = feat_obs[sel]
        Xtr = self._X(co, feat_obs); ytr = vo
        Xte = self._X(cp, feat_pred)
        if self.kind == "rf":
            m = RandomForestRegressor(n_estimators=60, n_jobs=4, random_state=0,
                                      min_samples_leaf=5)
        else:
            m = xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.1,
                                 n_jobs=4, verbosity=0)
        m.fit(Xtr, ytr)
        return m.predict(Xte).astype(np.float32)


BASELINES = {
    "IDW": IDW(),
    "OrdinaryKriging": LocalKriging(),
    "RegressionKriging": RegressionKriging(),
    "RandomForest": _TreeReg("rf"),
    "XGBoost": _TreeReg("xgb"),
}


def standardize_feats(feats_all):
    feats = feats_all.copy()
    mu = np.nanmean(feats, axis=0)
    sd = np.nanstd(feats, axis=0); sd[sd < 1e-8] = 1.0
    feats = (feats - mu) / sd
    feats = np.nan_to_num(feats, nan=0.0)
    return feats.astype(np.float32)
