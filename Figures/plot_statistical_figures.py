"""Reproduce the manuscript's statistical figures (Figures 2--4).

This is a path-cleaned migration of the plotting cells in MLE-KS_0708.ipynb
and Geo_Plots.ipynb. It reads the released aggregate profiles and estimates;
it does not rerun model selection or bootstrap inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CITY_ORDER = ["London", "NYC", "Tokyo", "Taipei", "Tianjin"]
DISPLAY_NAME = {"NYC": "New York City"}
CITY_COLOR = {
    "London": "#1976B9",
    "NYC": "#D62728",
    "Tokyo": "#2CA02C",
    "Taipei": "#FF7F0E",
    "Tianjin": "#7E57C2",
}
REGIME_COLOR = {"short": "#0077B6", "mid": "#FF9F1C", "long": "#D90429"}


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = pd.read_csv(ROOT / "data" / "metro_distance_profiles.csv")
    estimates = pd.read_csv(ROOT / "results" / "metro_law_estimates.csv")
    estimates = estimates.set_index("city").loc[CITY_ORDER].reset_index()
    return profiles, estimates


def fit_prefactor(r: np.ndarray, pi: np.ndarray, alpha: float) -> float:
    valid = (r > 0) & (pi > 0) & np.isfinite(r) & np.isfinite(pi)
    return float(np.exp(np.mean(np.log(pi[valid]) + alpha * np.log(r[valid]))))


def nice_ticks(x_min: float, x_max: float) -> np.ndarray:
    span = x_max - x_min
    step = 2 if span <= 10 else 5 if span <= 20 else 10 if span <= 40 else 20
    ticks = np.arange(np.floor(x_min / step) * step, x_max + 0.5 * step, step)
    return ticks[(ticks >= x_min) & (ticks <= x_max)]


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def draw_figure_2(profiles: pd.DataFrame, estimates: pd.DataFrame, output: Path) -> None:
    """Metro distance-preference curves and selected scaling intervals."""
    plt.rcParams.update({
        "font.size": 8.7,
        "axes.titlesize": 9.8,
        "axes.labelsize": 9.8,
        "xtick.labelsize": 8.7,
        "ytick.labelsize": 8.7,
        "legend.fontsize": 8.7,
        "axes.linewidth": 0.8,
    })
    summary = estimates.set_index("city")
    fig = plt.figure(figsize=(7.2, 4.8))
    grid = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.38)
    axes = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(6)]

    for index, city in enumerate(CITY_ORDER):
        ax = axes[index]
        city_data = profiles.loc[profiles["city"] == city].sort_values("r_mid_km")
        r = city_data["r_mid_km"].to_numpy(float)
        pi = city_data["pi_r"].to_numpy(float)
        valid = (r > 0) & (pi > 0) & np.isfinite(r) & np.isfinite(pi)
        r, pi = r[valid], pi[valid]

        row = summary.loc[city]
        a, b, alpha = float(row.a_km), float(row.b_km), float(row.alpha_hat)
        x_left, x_right = max(0.0, a - 1.5), b + 4.0
        visible = (r >= x_left) & (r <= x_right)
        ax.plot(r[visible], pi[visible], linewidth=1.05)
        ax.axvspan(a, b, alpha=0.14)

        window = (r >= a) & (r < b)
        if window.sum() >= 3:
            coefficient = fit_prefactor(r[window], pi[window], alpha)
            r_line = np.linspace(a, b, 200)
            ax.plot(r_line, coefficient * r_line ** (-alpha), "--", linewidth=1.25)

        ax.set(xlim=(x_left, x_right), yscale="log")
        ax.set_xticks(nice_ticks(x_left, x_right))
        ax.set_title(DISPLAY_NAME.get(city, city), fontsize=10)
        if index % 3 == 0:
            ax.set_ylabel(r"$\pi(r)$", fontsize=10)
        if index // 3 == 1:
            ax.set_xlabel("Distance r (km)")
        ax.text(
            0.03,
            0.03,
            rf"$\hat{{\alpha}}$={alpha:.2f}" + "\n" + rf"[{a:.1f}, {b:.1f}) km",
            transform=ax.transAxes,
            fontsize=8.0,
            va="bottom",
        )
        clean_axes(ax)

    axes[5].axis("off")
    fig.savefig(output / "Figure_2.pdf", bbox_inches="tight")
    fig.savefig(output / "Figure_2.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def draw_figure_3(estimates: pd.DataFrame, output: Path) -> None:
    """Cross-city exponent estimates, intervals and bootstrap goodness of fit."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.labelsize": 13.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    data = estimates.copy()
    y = np.arange(len(data))
    lower = data["alpha_ci_lower"].to_numpy(float)
    upper = data["alpha_ci_upper"].to_numpy(float)
    x_min = min(0.58, float(np.nanmin(lower)) - 0.05)
    x_max = max(1.36, float(np.nanmax(upper)) + 0.06)

    fig = plt.figure(figsize=(9.15, 3.95), dpi=300)
    ax = fig.add_axes([0.245, 0.18, 0.47, 0.74])
    info = fig.add_axes([0.745, 0.18, 0.245, 0.74])
    info.set(xlim=(0, 1), ylim=(-0.7, len(data) - 0.3))
    info.axis("off")
    for position in y:
        ax.axhline(position, color="#E4E8ED", lw=0.9, zorder=0)
    ax.axvline(1.0, linestyle=":", linewidth=2.4, color="#4A4A4A", zorder=1)

    for index, row in data.iterrows():
        color = CITY_COLOR[row.city]
        ax.errorbar(
            row.alpha_hat,
            index,
            xerr=np.array([[row.alpha_hat - row.alpha_ci_lower],
                           [row.alpha_ci_upper - row.alpha_hat]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.35,
            capsize=5.2,
            capthick=2.35,
            markersize=8.4,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAME.get(c, c) for c in data.city], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"Scaling exponent $\hat{\alpha}$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.25)
    ax.spines["bottom"].set_linewidth(2.25)
    ax.tick_params(axis="x", direction="out", length=5.2, width=1.8)
    ax.tick_params(axis="y", direction="out", length=5.2, width=1.8, pad=7)

    header_y = len(data) - 0.05
    info.text(0.00, header_y, "Window", ha="left", va="bottom", fontsize=10.6, fontweight="bold")
    info.text(0.48, header_y, "Width", ha="left", va="bottom", fontsize=10.6, fontweight="bold")
    info.text(0.78, header_y, r"$P_{\mathrm{KS}}$", ha="left", va="bottom", fontsize=10.6, fontweight="bold")
    info.plot([0, 0.97], [len(data) - 0.20] * 2, color="#C8CDD2", lw=1.2)
    for index, row in data.iterrows():
        yy = len(data) - 1 - index
        info.text(0.00, yy, f"[{row.a_km:.1f}, {row.b_km:.1f})", va="center", fontsize=9.8)
        info.text(0.48, yy, f"{row.window_width_km:.1f} km", va="center", fontsize=9.8)
        info.text(0.78, yy, f"{row.bootstrap_p_value:.3f}", va="center", fontsize=9.8)

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555",
                   markeredgecolor="white", markersize=7.8, linestyle="None",
                   label=r"$\hat{\alpha}$ estimate"),
            Line2D([0], [0], color="#555555", lw=2.35, marker="|", markersize=9,
                   label="95% CI"),
            Line2D([0], [0], color="#4A4A4A", lw=2.4, linestyle=":", label=r"$\alpha=1$"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9.0,
        handlelength=2.1,
    )
    fig.savefig(output / "Figure_3.pdf", bbox_inches="tight")
    fig.savefig(output / "Figure_3.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def draw_figure_4(profiles: pd.DataFrame, estimates: pd.DataFrame, output: Path) -> None:
    """Observed flow, represented pairs and distance preference by regime."""
    order = ["Taipei", "Tianjin", "NYC", "London", "Tokyo"]
    summary = estimates.set_index("city")
    fig, axes = plt.subplots(3, 5, figsize=(25, 14), facecolor="white", constrained_layout=True)
    y_labels = [
        "Passenger Volume (%)",
        r"Available Station Pairs $p(r)$",
        r"Average Travel Intensity $\pi(r)$",
    ]

    for column, city in enumerate(order):
        data = profiles.loc[profiles.city == city].sort_values("r_mid_km").copy()
        a, b = float(summary.loc[city, "a_km"]), float(summary.loc[city, "b_km"])
        total_flow = data.f_r.sum()
        cumulative = data.f_r.cumsum() / total_flow
        cutoff = int(np.searchsorted(cumulative.to_numpy(), 0.995))
        x_max = float(data.iloc[min(cutoff, len(data) - 1)].r_mid_km) * 1.05
        rows = [
            data.f_r.to_numpy() / total_flow * 100,
            data.p_r.to_numpy(),
            data.pi_r.to_numpy(),
        ]

        for row_index, values in enumerate(rows):
            ax = axes[row_index, column]
            distances = data.r_mid_km.to_numpy()
            colors = [
                REGIME_COLOR["short"] if r < a else
                REGIME_COLOR["mid"] if r < b else
                REGIME_COLOR["long"]
                for r in distances
            ]
            ax.bar(distances, values, width=0.5, color=colors, edgecolor="none", alpha=0.9)
            ax.set(xlim=(0, x_max), ylim=(0, None))
            ax.set_title(city, fontsize=18, fontweight="bold", pad=12, color="#222222")
            if column == 0:
                ax.set_ylabel(y_labels[row_index], fontsize=14, color="#333333")
            if row_index == 2:
                ax.set_xlabel("OD Distance (km)", fontsize=14, color="#333333")
            if row_index in (1, 2):
                ax.ticklabel_format(style="plain", axis="y")
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.legend(
                handles=[
                    Patch(facecolor=REGIME_COLOR["short"], label=f"Short (<{a:g}km)"),
                    Patch(facecolor=REGIME_COLOR["mid"], label=f"Mid ({a:g}-{b:g}km)"),
                    Patch(facecolor=REGIME_COLOR["long"], label=f"Long (>{b:g}km)"),
                ],
                loc="upper right",
                frameon=True,
                fontsize=11,
                facecolor="white",
                edgecolor="#dddddd",
                framealpha=0.9,
                handlelength=1.0,
            )
            if column == 0:
                ax.text(-0.14, 1.06, "abc"[row_index], transform=ax.transAxes,
                        fontsize=22, fontweight="bold", ha="right", va="top", clip_on=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(colors="#555555", labelsize=12)

    fig.savefig(output / "Figure_4.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles, estimates = load_tables()
    draw_figure_2(profiles, estimates, args.output_dir)
    draw_figure_3(estimates, args.output_dir)
    draw_figure_4(profiles, estimates, args.output_dir)


if __name__ == "__main__":
    main()
