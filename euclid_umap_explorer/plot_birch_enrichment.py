"""Plot supported BIRCH-cluster counts across the parameter grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont


CUTOFFS = (2, 5, 10, 15, 25, 30, 40, 50)
COLORS = (
    "#f2f5f6", "#cce9dc", "#a5dac9", "#78c8b7",
    "#4cb2a7", "#309a9b", "#287f93", "#286b8b",
    "#365883", "#51487e", "#704073", "#913a65",
)
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


def grids(data: pd.DataFrame) -> tuple[list[int], list[float], dict[int, pd.DataFrame]]:
    branches = sorted(data["branching_factor"].unique().tolist())
    thresholds = sorted(data["threshold"].unique().tolist())
    if len(data) != len(branches) * len(thresholds) or data.duplicated(
        ["threshold", "branching_factor"]
    ).any():
        raise ValueError("The BIRCH parameter grid is incomplete or contains duplicates")
    matrices = {}
    for cutoff in CUTOFFS:
        column = f"ge_{cutoff}x"
        matrices[cutoff] = data.pivot(
            index="threshold", columns="branching_factor", values=column
        ).loc[thresholds, branches]
        if matrices[cutoff].isna().any().any():
            raise ValueError(f"Missing values in {column}")
    return branches, thresholds, matrices


def save_png(path: Path, branches: list[int], thresholds: list[float],
             matrices: dict[int, pd.DataFrame], min_references: int) -> None:
    image = Image.new("RGB", (2300, 1200), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text((75, 35), "BIRCH: supported enriched clusters across parameters",
              font=font(42, True), fill="#14232c")
    draw.text((76, 92),
              f"At least 100 objects and {min_references} labelled A/B/C candidates per cluster. "
              "Each panel uses a cumulative enrichment cutoff.",
              font=font(23), fill="#42545e")
    cell = 13
    max_count = len(COLORS) - 1
    if max(int(matrix.to_numpy().max()) for matrix in matrices.values()) > max_count:
        raise ValueError("The colour scale needs more levels for these counts")
    for panel, cutoff in enumerate(CUTOFFS):
        x = 75 + (panel % 4) * 555
        y = 145 + (panel // 4) * 455
        gx, gy = x + 75, y + 68
        matrix = matrices[cutoff]
        draw.text((x + 72, y + 7), f">= {cutoff}x enrichment",
                  font=font(26, True), fill="#14232c")
        draw.text((x + 75, y + 39), f"maximum: {int(matrix.to_numpy().max())} clusters",
                  font=font(17), fill="#596b75")
        for row, threshold in enumerate(reversed(thresholds)):
            for column, branch in enumerate(branches):
                value = int(matrix.loc[threshold, branch])
                cx, cy = gx + column * cell, gy + row * cell
                draw.rectangle((cx, cy, cx + cell - 1, cy + cell - 1),
                               fill=COLORS[value], outline="#ffffff", width=1)
        for threshold in (5.0, 5.5, 6.0, 6.5, 7.0):
            if threshold in thresholds:
                row = len(thresholds) - 1 - thresholds.index(threshold)
                draw.text((gx - 52, gy + row * cell - 1), f"{threshold:.1f}",
                          font=font(17), fill="#42545e")
        for branch in (25, 30, 35, 40, 45, 50, 55):
            if branch in branches:
                column = branches.index(branch)
                draw.text((gx + column * cell - 7, gy + len(thresholds) * cell + 8),
                          str(branch), font=font(16), fill="#42545e")
        draw.text((gx + 119, gy + len(thresholds) * cell + 36), "branching factor",
                  font=font(18), fill="#42545e")
        for (threshold, branch), color in ((CURRENT, "#dc3342"), (SELECTED, "#ffbd2e")):
            if threshold in thresholds and branch in branches:
                row = len(thresholds) - 1 - thresholds.index(threshold)
                column = branches.index(branch)
                cx, cy = gx + column * cell, gy + row * cell
                draw.rectangle((cx, cy, cx + cell - 1, cy + cell - 1),
                               outline=color, width=3)
    draw.text((76, 1074), "Qualifying clusters", font=font(20, True), fill="#14232c")
    for count, color in enumerate(COLORS):
        x = 285 + count * 55
        draw.rectangle((x, 1077, x + 30, 1107), fill=color, outline="#aab6ba")
        draw.text((x + 9, 1110), str(count), font=font(16), fill="#42545e")
    draw.rectangle((1040, 1080, 1068, 1108), outline="#dc3342", width=3)
    draw.text((1080, 1082), "Current 6.2 / 50", font=font(18), fill="#42545e")
    draw.rectangle((1315, 1080, 1343, 1108), outline="#ffbd2e", width=3)
    draw.text((1355, 1082), "Validation-selected 5.7 / 49",
              font=font(18), fill="#42545e")
    draw.text((76, 1152), "Rows: BIRCH threshold (5.0-7.0). Columns: branching factor (25-55). "
              "Counts are not independent across parameter settings.",
              font=font(18), fill="#596b75")
    image.save(path)


def save_html(path: Path, branches: list[int], thresholds: list[float],
              matrices: dict[int, pd.DataFrame], min_references: int) -> None:
    figure = make_subplots(rows=2, cols=4,
                           subplot_titles=[f">= {cutoff}x enrichment" for cutoff in CUTOFFS],
                           horizontal_spacing=0.045, vertical_spacing=0.13)
    scale = [[i / (len(COLORS) - 1), color] for i, color in enumerate(COLORS)]
    for panel, cutoff in enumerate(CUTOFFS):
        row, column = divmod(panel, 4)
        figure.add_trace(go.Heatmap(
            x=branches, y=thresholds, z=matrices[cutoff].to_numpy(),
            coloraxis="coloraxis", zsmooth=False,
            hovertemplate="threshold=%{y:.1f}<br>branching_factor=%{x}<br>clusters=%{z}<extra></extra>",
        ), row=row + 1, col=column + 1)
        for (threshold, branch), color, symbol in (
            (CURRENT, "#dc3342", "circle-open"),
            (SELECTED, "#ffbd2e", "diamond-open"),
        ):
            figure.add_trace(go.Scatter(
                x=[branch], y=[threshold], mode="markers", showlegend=False,
                marker=dict(symbol=symbol, size=17, color=color, line=dict(color=color, width=2)),
                hovertemplate=f"{threshold:.1f} / {branch}<extra></extra>",
            ), row=row + 1, col=column + 1)
        figure.update_xaxes(title_text="branching factor", dtick=5, row=row + 1, col=column + 1)
        figure.update_yaxes(title_text="threshold" if column == 0 else None,
                            dtick=0.5, row=row + 1, col=column + 1)
    figure.update_layout(
        title=("BIRCH enriched clusters | "
               f">= 100 objects and >= {min_references} labelled A/B/C candidates per cluster"),
        coloraxis=dict(colorscale=scale, cmin=0, cmax=len(COLORS) - 1,
                       colorbar=dict(title="Clusters", dtick=1)),
        width=1800, height=950, template="plotly_white", margin=dict(t=90, b=65, l=65, r=120),
    )
    figure.write_html(path, include_plotlyjs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--min-references", type=int, default=50)
    args = parser.parse_args()
    data = pd.read_csv(args.input_csv)
    branches, thresholds, matrices = grids(data)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_png(args.output_prefix.with_suffix(".png"), branches, thresholds,
             matrices, args.min_references)
    save_html(args.output_prefix.with_suffix(".html"), branches, thresholds,
              matrices, args.min_references)


if __name__ == "__main__":
    main()
