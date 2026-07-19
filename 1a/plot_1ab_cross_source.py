"""Generate Task 1a cross-source diagnostic figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys_path_root = CODE_ROOT
import sys
sys.path.insert(0, str(sys_path_root))
from common.paths import OUTPUT_ROOT  # noqa: E402


RESULT_DIR = OUTPUT_ROOT / "task1a_cross_source"
OUTDIR = OUTPUT_ROOT / "figures"
TWO_SOURCE_CITIES = [
    "berlin", "cologne", "dortmund", "dusseldorf", "frankfurt", "hamburg", "munich", "stuttgart",
    "bucharest", "buenos_aires", "cairo", "johannesburg", "lagos", "riyadh", "sao_paulo", "warsaw",
]


def _title_city(city: str) -> str:
    return city.replace("_", " ").title()


def plot_one(result: Path, outdir: Path, prefix: str, city: str, write_main_tex: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(result.read_text())
    corr = data["correlation"]
    imp = data["imputation"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {
        "all": "#4D4D4D",
        "day": "#D55E00",
        "night": "#0072B2",
        "temporal": "#6B7280",
        "spatial": "#009E73",
        "improve": "#009E73",
        "worse": "#CC3311",
        "missing": "#BDBDBD",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)

    # (a) Same-hour correlations.
    ax = axes[0]
    groups = ["all", "day", "night"]
    labels = ["All", "Day", "Night"]
    x = np.arange(len(groups))
    temporal = [corr["same_hour"][g]["temporal_mean_r"] for g in groups]
    spatial = [corr["same_hour"][g]["spatial_map_r_median"] for g in groups]
    width = 0.34
    ax.bar(x - width / 2, temporal, width, color=colors["temporal"], label="time-series corr.")
    ax.bar(x + width / 2, spatial, width, color=colors["spatial"], label="spatial-pattern corr.")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Pearson $r$")
    ax.set_title("(a) Same-hour association")
    ax.set_ylim(0, 0.48)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) Lagged city-mean correlations.
    ax = axes[1]
    for g in groups:
        scan = corr["lagged"][g]["scan"]
        lags = np.asarray([s["lag_h"] for s in scan], dtype=float)
        r = np.asarray([np.nan if s["r"] is None else s["r"] for s in scan], dtype=float)
        ax.plot(lags, r, lw=1.15, color=colors[g], label=labels[groups.index(g)])
        best_lag = corr["lagged"][g]["best_lag_h"]
        best_r = corr["lagged"][g]["best_r"]
        if best_lag is not None and best_r is not None:
            ax.scatter([best_lag], [best_r], s=12, color=colors[g], zorder=3)
    ax.axvline(0, color="0.35", lw=0.7, ls="--")
    ax.axhline(0, color="0.35", lw=0.7)
    ax.set_xlabel("Air-T lag relative to LST $\\ell$ (h)")
    ax.set_ylabel("Lagged Pearson $r$")
    ax.set_title("(b) Lag does not recover substitutability")
    ax.set_xlim(-48, 48)
    ax.set_xticks([-48, -24, 0, 24, 48])
    ax.legend(frameon=False, loc="lower left", ncol=1, handlelength=1.4)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) Cross-source imputation effect.
    ax = axes[2]
    names = ["LST gaps\n+Air-T", "Air-T sparse\n+LST"]
    changes_raw = [
        imp["lst_from_air_t"]["mae_improvement_pct"]["overall"],
        imp["air_t_from_lst"]["mae_improvement_pct"]["overall"],
    ]
    changes = [0.0 if v is None else float(v) for v in changes_raw]
    bar_colors = [
        colors["missing"] if v is None else (colors["improve"] if float(v) >= 0 else colors["worse"])
        for v in changes_raw
    ]
    ax.bar(np.arange(2), changes, color=bar_colors, width=0.55)
    ax.axhline(0, color="0.25", lw=0.8)
    ax.set_xticks(np.arange(2), names)
    ax.set_ylabel("Accuracy gain (%)")
    ax.set_title("(c) Auxiliary source is asymmetric")
    ax.set_ylim(-22, 8)
    for i, (raw, v) in enumerate(zip(changes_raw, changes)):
        if raw is None:
            ax.text(i, 0.8, "NA", ha="center", va="bottom", fontsize=7.5)
            continue
        ax.text(i, v + (0.8 if v >= 0 else -1.3), f"{v:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    for ax in axes:
        ax.grid(axis="y", color="0.90", lw=0.6)
        ax.set_axisbelow(True)

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = outdir / f"{prefix}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[wrote] {out}")

    title_city = _title_city(city)
    tex = r"""\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{%s.pdf}
  \caption{Cross-source consistency and complementarity diagnostic on %s
    (2023--2025). (a) Same-hour city-mean correlations are weak, while spatial
    map correlations are only moderate. (b) Lagging Air-T relative to LST does
    not recover a strong positive substitute relationship; the all/day absolute
    peaks may occur away from zero lag. (c) Cross-source imputation is
    asymmetric: the other source can help or hurt depending on direction and
    city. Accuracy gain is computed as relative MAE reduction against the base
    imputer.}\label{fig:1ab-cross-source-%s}
\end{figure*}
""" % (prefix, title_city, city.replace("_", "-"))
    tex_path = outdir / f"{prefix}.tex"
    tex_path.write_text(tex)
    print(f"[wrote] {tex_path}")

    if write_main_tex:
        main_tex = OUTDIR / "fig_1ab_cross_source.tex"
        main_tex.write_text(tex.replace(f"{prefix}.pdf", "fig_1ab_cross_source.pdf")
                           .replace(f" on {title_city}\n", " on Munich\n")
                           .replace(f"fig:1ab-cross-source-{city.replace('_', '-')}",
                                    "fig:1ab-cross-source"))
        print(f"[wrote] {main_tex}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="munich")
    ap.add_argument("--all_cities", action="store_true")
    ap.add_argument("--result_dir", default=str(RESULT_DIR))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--bycity_subdir", default="fig_1ab_cross_source_by_city")
    ap.add_argument("--paper_main", action="store_true",
                    help="also refresh fig_1ab_cross_source.{pdf,png,tex} for Munich")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    outdir = Path(args.outdir)

    if args.all_cities:
        bycity = outdir / args.bycity_subdir
        for city in TWO_SOURCE_CITIES:
            result = result_dir / f"1ab_{city}_cross_source.json"
            if not result.exists():
                print(f"[skip] missing {result}")
                continue
            plot_one(result, bycity, f"fig_1ab_cross_source_{city}", city)
        return

    city = args.city
    result = result_dir / f"1ab_{city}_cross_source.json"
    prefix = "fig_1ab_cross_source" if city == "munich" else f"fig_1ab_cross_source_{city}"
    plot_one(result, outdir, prefix, city, write_main_tex=args.paper_main or city == "munich")


if __name__ == "__main__":
    main()
