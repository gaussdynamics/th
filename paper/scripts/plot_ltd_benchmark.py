#!/usr/bin/env python3
"""Figures for Chapter 11: the draft gear envelope and the benchmark comparison.

Writes three PDFs to paper/figures/:

* ``draft_gear_envelope.pdf`` -- the loading and unloading curves digitized
  from Spiryagin et al. (2017), Fig. 3 (simulator2/data/draft_gear_benchmark.csv).
* ``ltd_benchmark_strip.pdf`` -- every quantity of Wu et al. (2018), Tables 3-6,
  with the participants, the core-six range and ours, each row scaled to the
  core-six range so different units share an axis.
* ``ltd_benchmark_speed.pdf`` -- lead speed against location, ours against the
  paper's Fig. 2, Trains 1 and 3.

Input: results/ltd_benchmark/compare_s3k.json, written by

    python scripts/compare_ltd_benchmark.py --tag _s3k --json results/ltd_benchmark/compare_s3k.json
    python paper/scripts/plot_ltd_benchmark.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "paper" / "figures"
CORE = ("TABLDSS", "UM", "CRE-LTS", "TDEAS", "TsDyn", "CARS")
OURS, REF, BAND, PAPER = "#1f4e79", "#8a9096", "#d6e2ef", "#c25b2a"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def draft_gear() -> None:
    rows = list(csv.DictReader(open(REPO / "simulator2/data/draft_gear_benchmark.csv")))
    d = [float(r["deflection_mm"]) for r in rows]
    lo = [float(r["loading_kN"]) / 1e3 for r in rows]
    un = [float(r["unloading_kN"]) / 1e3 for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.fill_between(d, un, lo, color=BAND, lw=0, label="region where the gear sticks")
    ax.plot(d, lo, color=OURS, lw=1.8, label="loading")
    ax.plot(d, un, color=OURS, lw=1.8, ls="--", label="unloading")
    ax.set_xlabel("Gear deflection (mm)")
    ax.set_ylabel("Force (MN)")
    ax.set_xlim(0, max(d) * 1.02)
    ax.set_ylim(0, None)
    ax.grid(color="0.92")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "draft_gear_envelope.pdf")


def strip(data: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 4.0), sharey=True, constrained_layout=True)
    labels = None
    for ax, tr in zip(axes, ("1", "2", "3", "4")):
        rows = data["trains"][tr]["rows"]
        labels = [r["label"].replace(", 3-50 km", "") for r in rows]
        for i, r in enumerate(rows):
            core = [r["all"][k] for k in CORE if k in r["all"]]
            c0, c1 = min(core), max(core)
            span = (c1 - c0) or 1.0
            sc = lambda v: (v - c0) / span
            y = len(rows) - 1 - i
            ax.add_patch(plt.Rectangle((0, y - 0.3), 1, 0.6, color=BAND, lw=0))
            xs = [sc(v) for v in r["all"].values()]
            ax.scatter([min(max(x, -1.4), 2.4) for x in xs], [y] * len(xs), s=9, color=REF, zorder=2)
            xo = sc(r["ours"])
            ax.scatter([min(max(xo, -1.4), 2.4)], [y], s=26, color=OURS, zorder=3)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.axvline(1, color="0.75", lw=0.6)
        ax.set_xlim(-1.5, 2.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["min", "max"], fontsize=8)
        ax.set_xlabel("core six", fontsize=8)
        ax.set_title(f"Train {tr}", fontsize=10)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(list(reversed(labels)), fontsize=8)
    fig.savefig(FIG / "ltd_benchmark_strip.pdf")


def speed(data: dict) -> None:
    f2 = data["fig2"]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True, constrained_layout=True)
    for ax, tr, key in zip(axes, ("1", "3"), ("train1_kmh", "train3_4_kmh")):
        t = data["trains"][tr]["traces"]
        ax.plot(t["location_km"], t["speed_kmh"], color=OURS, lw=1.4, label="this simulator")
        ax.plot(f2["location_km"], f2[key], color=PAPER, lw=1.2, ls="--", label="Wu et al. (2018), Fig. 2")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title(f"Train {tr}", loc="left", fontsize=10)
        ax.grid(color="0.92")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].set_xlabel("Location of the lead locomotive (km)")
    axes[1].set_xlim(3, 50)
    fig.savefig(FIG / "ltd_benchmark_speed.pdf")


def main() -> None:
    data = json.loads((REPO / "results/ltd_benchmark/compare_s3k.json").read_text())
    draft_gear()
    strip(data)
    speed(data)
    print("wrote draft_gear_envelope.pdf, ltd_benchmark_strip.pdf, ltd_benchmark_speed.pdf")


if __name__ == "__main__":
    main()
