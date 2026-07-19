"""Mask construction + cloud-coverage binning for interpolation tasks."""
import numpy as np

CLOUD_BINS = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
BIN_LABELS = ["0-25%", "25-50%", "50-75%", ">75%"]


def cloud_cov_per_time(values: np.ndarray) -> np.ndarray:
    """values [T,N] with NaN=cloud → fraction NaN per timestamp [T]."""
    return np.isnan(values).mean(axis=1)


def cloud_bin_of(cov: float) -> int:
    for i, (lo, hi) in enumerate(CLOUD_BINS):
        if lo <= cov < hi:
            return i
    return len(CLOUD_BINS) - 1


def random_mask(N: int, keep_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask [N], True = held-out (to reconstruct). keep_frac observed."""
    n_hide = int(round((1.0 - keep_frac) * N))
    idx = rng.permutation(N)[:n_hide]
    m = np.zeros(N, bool)
    m[idx] = True
    return m


def random_mask_for_bin(N: int, bin_idx: int, rng: np.random.Generator,
                        max_missing: float = 0.97) -> np.ndarray:
    """Random-scatter mask whose missing fraction is uniform within cloud-bin `bin_idx`.

    Used by Task 1b so its x-axis (missing%) aligns with Task 1a's 4 bins, while
    keeping 1b's random-scatter mask shape (vs 1a's structured cloud). The top
    bin is capped at `max_missing` so >=3% of pixels remain observed.
    Returns boolean [N], True = held-out.
    """
    lo, hi = CLOUD_BINS[bin_idx]
    hi = min(hi, max_missing)
    frac = rng.uniform(lo, hi) if hi > lo else lo
    n_hide = max(1, int(round(frac * N)))
    n_hide = min(n_hide, N - 1)          # always leave >=1 observed pixel
    idx = rng.choice(N, n_hide, replace=False)
    m = np.zeros(N, bool)
    m[idx] = True
    return m


def build_cloud_eval_pairs(values, times, rng, clear_thr=0.02,
                           n_clear=80, masks_per_bin=25, seed=0):
    """1a protocol: clear-scene GT + transferred REAL cloud masks.

    Returns list of dicts: {clear_t, mask(boolean[N]), bin_idx, cov}.
    A 'clear scene' = timestamp with cloud coverage < clear_thr (full/near-full grid).
    Cloud masks sampled from real cloudy timestamps, grouped into 4 coverage bins.
    """
    T, N = values.shape
    cov = cloud_cov_per_time(values)
    clear_idx = np.where(cov < clear_thr)[0]
    cloudy_idx = np.where((cov >= clear_thr) & (cov < 1.0))[0]
    if len(clear_idx) == 0:
        raise RuntimeError(f"no clear scenes (cov<{clear_thr}); relax clear_thr")
    # sample clear scenes
    if len(clear_idx) > n_clear:
        clear_idx = rng.choice(clear_idx, size=n_clear, replace=False)
    # group cloudy masks by bin
    bin_masks = {b: [] for b in range(len(CLOUD_BINS))}
    for ti in cloudy_idx:
        m = np.isnan(values[ti])
        if m.any():
            bin_masks[cloud_bin_of(cov[ti])].append((ti, m, float(cov[ti])))
    pairs = []
    for b, lst in bin_masks.items():
        if not lst:
            continue
        sample = lst if len(lst) <= masks_per_bin else [lst[i] for i in rng.choice(len(lst), masks_per_bin, replace=False)]
        for _, m, c in sample:
            for ct in clear_idx:
                pairs.append({"clear_t": int(ct), "mask": m, "bin_idx": b, "cov": c})
    return pairs
