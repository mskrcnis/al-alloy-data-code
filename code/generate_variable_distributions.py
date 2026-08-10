#!/usr/bin/env python3
"""Generate the 4x5 variable-distribution figure for the released dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIABLES = [
    "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Ti", "Zr",
    "Li", "Ni", "Be", "Sc", "Tsol", "Tage", "tage", "YS", "UTS", "El",
]
TARGETS = ["YS", "UTS", "El"]
UNITS = {variable: "wt.%" for variable in VARIABLES[:14]}
UNITS.update({"Tsol": "°C", "Tage": "°C", "tage": "h", "YS": "MPa", "UTS": "MPa", "El": "%"})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/al_alloy_complete_target_records.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/variable_distributions.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input)
    missing = [column for column in VARIABLES if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data.dropna(subset=TARGETS).copy()
    if len(data) != 452:
        raise ValueError(f"Expected 452 complete-target records; found {len(data)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 5, figsize=(20, 14), dpi=600, constrained_layout=True)
    colour = "#4C78A8"
    edge_colour = "#1F3B5B"
    for axis, variable in zip(axes.ravel(), VARIABLES):
        values = pd.to_numeric(data[variable], errors="coerce").dropna().to_numpy(float)
        bins = np.histogram_bin_edges(values, bins="auto")
        if len(bins) < 8:
            bins = np.linspace(values.min(), values.max(), 9)
        axis.hist(values, bins=bins, color=colour, edgecolor=edge_colour, linewidth=0.55, alpha=0.9)
        xlabel = "Elongation (%)" if variable == "El" else f"{variable} ({UNITS[variable]})"
        axis.set_xlabel(xlabel, fontsize=9, labelpad=3)
        axis.set_ylabel("Count", fontsize=8.5, labelpad=2)
        axis.tick_params(axis="both", labelsize=7.5, length=3)
        axis.grid(axis="y", color="0.85", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(0.97, 0.92, f"n={len(values)}", transform=axis.transAxes, ha="right", va="top", fontsize=7.5, color="0.25")

    figure.suptitle("Variable distributions for 452 complete-target records", fontsize=15, y=1.01)
    figure.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
