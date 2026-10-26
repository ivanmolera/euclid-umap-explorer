"""Scatter views of supported BIRCH-cluster counts over both parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont


CUTOFFS = (2, 5, 10, 15, 25, 30, 40, 50)
COLOR_STOPS = ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")
CURRENT = (6.2, 50)
SELECTED = (5.7, 49)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def branch_color(branch: int) -> tuple[int, int, int]:
    position = (branch - 25) / 30 * (len(COLOR_STOPS) - 1)
    low = min(int(position), len(COLOR_STOPS) - 2)
    fraction = position - low
    first = tuple(bytes.fromhex(COLOR_STOPS[low][1:]))
    second = tuple(bytes.fromhex(COLOR_STOPS[low + 1][1:]))
    return tuple(round(a + fraction * (b - a)) for a, b in zip(first, second))


def load_counts(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path).sort_values(["threshold", "branching_factor"])
    thresholds = sorted(data["threshold"].unique())
    branches = sorted(data["branching_factor"].unique())
    if (len(data) != len(thresholds) * len(branches)
            or data.duplicated(["threshold", "branching_factor"]).any()
            or any(f"ge_{cutoff}x" not in data for cutoff in CUTOFFS)):
        raise ValueError("The enriched-cluster parameter grid is incomplete")
    return data


def save_png(path: Path, data: pd.DataFrame, min_references: int) -> None:
    image = Image.new("RGBA", (2300, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((73, 31), "BIRCH: enriched-cluster counts across the parameter grid",
              font=font(40, True), fill="#14232c")
    draw.text((75, 89),
              f"One point per configuration. At least 100 objects and {min_references} "
              "labelled A/B/C candidates per cluster.",
              font=font(23), fill="#465964")
    for panel, cutoff in enumerate(CUTOFFS):
        px = 75 + (panel % 4) * 555
        py = 148 + (panel // 4) * 455
        left, top, right, bottom = px + 57, py + 54, px + 491, py + 337
        values = data[f"ge_{cutoff}x"]
        maximum = int(values.max())
        y_limit = max(maximum + 0.5, 1.5)
        draw.text((px + 55, py + 3), f">= {cutoff}x enrichment",
                  font=font(26, True), fill="#14232c")
        for tick in sorted(set([0, maximum, *range(0, maximum + 1, max(1, (maximum + 3) // 4))])):
            y = bottom - (tick / y_limit) * (bottom - top)
            draw.line((left, y, right, y), fill="#e2e8eb", width=2)
            draw.text((left - 37, y - 11), str(tick), font=font(17), fill="#566872")
        for threshold in (5.0, 5.5, 6.0, 6.5, 7.0):
            x = left + (threshold - 5) / 2 * (right - left)
            draw.line((x, top, x, bottom), fill="#edf1f3", width=1)
            draw.text((x - 13, bottom + 11), f"{threshold:.1f}",
                      font=font(17), fill="#566872")
        medians = data.groupby("threshold")[f"ge_{cutoff}x"].median()
        median_points = [
            (left + (threshold - 5) / 2 * (right - left),
             bottom - (value / y_limit) * (bottom - top))
            for threshold, value in medians.items()
        ]
        draw.line(median_points, fill="#172a34", width=3)

        points = Image.new("RGBA", image.size, (0, 0, 0, 0))
        dots = ImageDraw.Draw(points)
        for item in data.itertuples(index=False):
            value = getattr(item, f"ge_{cutoff}x")
            x = left + (item.threshold - 5) / 2 * (right - left)
            y = bottom - (value / y_limit) * (bottom - top)
            # Small deterministic offsets reveal overlapping integer-valued scores.
            x += (item.branching_factor - 40) * 0.42
            y += (item.branching_factor % 5 - 2) * 0.9
            color = branch_color(item.branching_factor)
            dots.ellipse((x - 3.5, y - 3.5, x + 3.5, y + 3.5), fill=(*color, 100))
        image = Image.alpha_composite(image, points)
        draw = ImageDraw.Draw(image)
        for (threshold, branch), color in ((CURRENT, "#dc3342"), (SELECTED, "#f29e17")):
            match = data.loc[data.threshold.eq(threshold) & data.branching_factor.eq(branch)]
            if not match.empty:
                value = int(match[f"ge_{cutoff}x"].iat[0])
                x = left + (threshold - 5) / 2 * (right - left)
                y = bottom - (value / y_limit) * (bottom - top)
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
        draw.line((left, top, left, bottom, right, bottom), fill="#647781", width=2)
        draw.text((left + 160, bottom + 39), "threshold",
                  font=font(18), fill="#465964")
        draw.text((left + 5, top + 6), "clusters", font=font(16), fill="#566872")

    draw.text((75, 1087), "branching factor", font=font(20, True), fill="#14232c")
    for index, branch in enumerate(range(25, 56)):
        x = 290 + index * 22
        draw.rectangle((x, 1090, x + 21, 1114), fill=branch_color(branch))
    for branch in (25, 30, 35, 40, 45, 50, 55):
        draw.text((290 + (branch - 25) * 22 - 9, 1120), str(branch),
                  font=font(16), fill="#566872")
    draw.ellipse((1122, 1089, 1140, 1107), outline="#dc3342", width=3)
    draw.text((1150, 1087), "Current 6.2 / 50", font=font(18), fill="#465964")
    draw.ellipse((1392, 1089, 1410, 1107), outline="#f29e17", width=3)
    draw.text((1420, 1087), "Validation-selected 5.7 / 49",
              font=font(18), fill="#465964")
    draw.text((75, 1160),
              "Black line: median across branching factors. Points are lightly jittered for visibility; "
              "each panel has its own Y-axis scale.",
              font=font(18), fill="#566872")
    image.convert("RGB").save(path)


def save_html(path: Path, data: pd.DataFrame, min_references: int) -> None:
    figure = go.Figure()
    for cutoff in CUTOFFS:
        column = f"ge_{cutoff}x"
        show = cutoff == 10
        figure.add_trace(go.Scatter3d(
            x=data.threshold, y=data.branching_factor, z=data[column],
            mode="markers", visible=show, name=f">= {cutoff}x",
            marker=dict(size=4, color=data[column], cmin=0, cmax=11,
                        colorscale=[[0, "#d8e1e5"], [0.09, "#87cebb"],
                                    [0.45, "#21918c"], [1, "#522a67"]],
                        opacity=0.78, colorbar=dict(title="Clusters")),
            customdata=data.n_clusters_total,
            hovertemplate=("threshold=%{x:.1f}<br>branching_factor=%{y}<br>"
                           "enriched clusters=%{z}<br>all clusters=%{customdata}<extra></extra>"),
        ))
        for (threshold, branch), color, name in (
            (CURRENT, "#dc3342", "Current 6.2 / 50"),
            (SELECTED, "#f29e17", "Validation-selected 5.7 / 49"),
        ):
            match = data.loc[data.threshold.eq(threshold) & data.branching_factor.eq(branch)].iloc[0]
            figure.add_trace(go.Scatter3d(
                x=[threshold], y=[branch], z=[int(match[column])],
                mode="markers+text", text=[name], textposition="top center",
                visible=show, showlegend=False,
                marker=dict(size=9, color=color, symbol="diamond", line=dict(color="#14232c", width=2)),
                hovertemplate=f"{name}<br>clusters=%{{z}}<extra></extra>",
            ))
    buttons = []
    for index, cutoff in enumerate(CUTOFFS):
        visible = [position // 3 == index for position in range(3 * len(CUTOFFS))]
        buttons.append(dict(label=f">= {cutoff}x", method="update", args=[
            {"visible": visible},
            {"title": f"BIRCH enriched clusters | >= {cutoff}x | "
                      f">= 100 objects and >= {min_references} labelled candidates"},
        ]))
    figure.update_layout(
        title=f"BIRCH enriched clusters | >= 10x | >= 100 objects and >= {min_references} labelled candidates",
        scene=dict(xaxis_title="threshold", yaxis_title="branching factor",
                   zaxis_title="qualifying clusters", aspectmode="cube"),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.02, y=0.95)],
        template="plotly_white", width=1200, height=900,
        margin=dict(l=10, r=10, b=10, t=70),
    )
    figure.write_html(path, include_plotlyjs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--min-references", type=int, default=50)
    args = parser.parse_args()
    data = load_counts(args.input_csv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_png(args.output_prefix.with_suffix(".png"), data, args.min_references)
    save_html(args.output_prefix.with_suffix(".html"), data, args.min_references)


if __name__ == "__main__":
    main()
