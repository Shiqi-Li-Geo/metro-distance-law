"""Reproduce the five city panels that make up manuscript Figure 5.

This is a path-cleaned migration of the final mapping cell in Geo_Plots.ipynb.
The released PNG files are included in this directory. Regeneration requires
the original city-level OD tables because coordinates cannot be recovered from
the aggregate distance profiles distributed with this repository.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import contextily as ctx
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


CITY_CONFIG = {
    "Taipei": {"a": 5.5, "b": 14.0, "top_n": 1500, "lw_base": 0.10, "lw_add": 1.0},
    "Tianjin": {"a": 5.5, "b": 11.5, "top_n": 2000, "lw_base": 0.15, "lw_add": 1.0},
    "NYC": {"a": 3.5, "b": 10.5, "top_n": 2500, "lw_base": 0.15, "lw_add": 1.0},
    "London": {"a": 3.5, "b": 11.5, "top_n": 2500, "lw_base": 0.15, "lw_add": 1.0},
    "Tokyo": {"a": 3.5, "b": 52.0, "top_n": 4000, "lw_base": 0.10, "lw_add": 0.9},
}
REGIME_COLOR = {"short": "#0077B6", "mid": "#FF9F1C", "long": "#D90429"}
ENCODINGS = ("utf-8", "utf-8-sig", "big5", "gbk", "cp950")


def read_od(path: Path) -> pd.DataFrame:
    error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as exc:
            error = exc
    raise RuntimeError(f"Could not decode {path}") from error


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    radius = 6_378_137.0
    x = radius * math.radians(lon)
    y = radius * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def draw_regime(
    ax: plt.Axes,
    subset: pd.DataFrame,
    full_data: pd.DataFrame,
    color: str,
    title: str,
    extent: tuple[float, float, float, float],
    config: dict[str, float],
) -> None:
    selected = subset.nlargest(int(config["top_n"]), "Taps").sort_values("Taps")
    if selected.empty:
        ax.set_title(f"{title}\n(No data)")
        ax.axis("off")
        return

    maximum = np.percentile(selected.Taps, 98)
    minimum = selected.Taps.min()
    rgb = colors.to_rgb(color)
    segments, widths, line_colors = [], [], []
    for row in selected.itertuples():
        segments.append([(row.x_o, row.y_o), (row.x_d, row.y_d)])
        value = min(row.Taps, maximum)
        weight = ((np.log(value) - np.log(minimum)) /
                  (np.log(maximum) - np.log(minimum))) if maximum > minimum else 0.5
        weight **= 2.0
        widths.append(config["lw_base"] + config["lw_add"] * weight)
        line_colors.append((*rgb, 0.25 + 0.7 * weight))
    ax.add_collection(LineCollection(segments, colors=line_colors, linewidths=widths, zorder=3))

    stations = full_data[["x_o", "y_o"]].drop_duplicates()
    ax.scatter(stations.x_o, stations.y_o, s=2, color="#999999", alpha=0.3,
               zorder=2, edgecolors="none")
    min_x, max_x, min_y, max_y = extent
    ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y), aspect="equal")

    mean_lat = full_data.lat_o.mean()
    scale_length = 10_000 / math.cos(math.radians(mean_lat))
    bar_x1 = max_x - (max_x - min_x) * 0.05
    bar_x0 = bar_x1 - scale_length
    bar_y = min_y + (max_y - min_y) * 0.05
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="#222222", linewidth=2.5, zorder=5)
    ax.text((bar_x0 + bar_x1) / 2, bar_y + (max_y - min_y) * 0.015, "10 km",
            fontsize=12, ha="center", va="bottom", fontweight="bold", zorder=5)
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, alpha=0.9, zorder=0)
    except Exception:
        ax.set_facecolor("#f4f4f4")
    ax.set_title(title, fontsize=18, color="#222222", pad=20, fontweight="bold")
    ax.axis("off")


def draw_city(city: str, source: Path, output: Path) -> None:
    config = CITY_CONFIG[city]
    data = read_od(source)
    if "Taps" not in data and "taps" in data:
        data = data.rename(columns={"taps": "Taps"})
    required = {"distance_km", "Taps", "lon_o", "lat_o", "lon_d", "lat_d"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{source.name} is missing columns: {sorted(missing)}")
    data = data.loc[data.distance_km > 0].copy()
    if city == "London":
        data = data.loc[(data.lon_o >= -0.5) & (data.lon_d >= -0.5)].copy()

    origin = [lonlat_to_mercator(lon, lat) for lon, lat in zip(data.lon_o, data.lat_o)]
    destination = [lonlat_to_mercator(lon, lat) for lon, lat in zip(data.lon_d, data.lat_d)]
    data["x_o"], data["y_o"] = zip(*origin)
    data["x_d"], data["y_d"] = zip(*destination)
    span_x, span_y = data.x_o.max() - data.x_o.min(), data.y_o.max() - data.y_o.min()
    extent = (
        data.x_o.min() - 0.05 * span_x,
        data.x_o.max() + 0.05 * span_x,
        data.y_o.min() - 0.05 * span_y,
        data.y_o.max() + 0.05 * span_y,
    )
    a, b = config["a"], config["b"]
    subsets = [data.loc[data.distance_km < a],
               data.loc[(data.distance_km >= a) & (data.distance_km < b)],
               data.loc[data.distance_km >= b]]
    titles = [f"{city}: Short (<{a:g} km)", f"{city}: Mid-range ({a:g}-{b:g} km)",
              f"{city}: Long Tail (>{b:g} km)"]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="white")
    for ax, subset, regime, title in zip(axes, subsets, REGIME_COLOR, titles):
        draw_regime(ax, subset, data, REGIME_COLOR[regime], title, extent, config)
    plt.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", type=Path, required=True,
                        help="Directory containing <City>_OD_with_geo_distance.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for city in CITY_CONFIG:
        source = args.raw_data_dir / f"{city}_OD_with_geo_distance.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        draw_city(city, source, args.output_dir / f"Figure_5_{city}.png")


if __name__ == "__main__":
    main()
