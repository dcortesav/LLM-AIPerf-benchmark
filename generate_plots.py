"""
Generate professional plots comparing all model/reasoning configurations
benchmarked with NVIDIA AIPerf against the Huawei MaaS endpoint.

Metrics plotted:
  - TTFT (p50, p95, p99)            — grouped vertical bar plot
  - Request Latency (p50, p95, p99) — grouped vertical bar plot
  - ISL (average)                   — horizontal bar plot
  - OSL (average)                   — horizontal bar plot
  - Total Token Consumption (ISL+OSL, average) — horizontal bar plot

Data is read dynamically from AIPerf profiling-phase JSON summaries under
results/<timestamp>/<variant>/phases/profiling/<variant>.json.
Plots are exported as PNG images into the 'results' folder.

Usage:
    .venv/bin/python generate_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="white", context="poster", palette="muted")

# Red shades — used for the three percentiles in grouped plots
PERCENTILES = ["p50", "p95", "p99"]
PERCENTILE_COLORS = ["#592525", "#AD2A2A", "#E61717"]

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR

# ──────────────────────────────────────────────────────────────────────
# Model display names (two-line labels read well on bar charts)
# ──────────────────────────────────────────────────────────────────────
DISPLAY_NAMES = {
    "glm-5.2-reasoning-on":           "GLM 5.2",
    "glm-5.2-reasoning-off":          "GLM 5.2\n(No Reasoning)",
    "glm-5.3-reasoning-low":          "GLM 5.3\n(Low Reasoning)",
    "glm-5.3-reasoning-default":      "GLM 5.3",
    "deepseek-v4.1-flash-reasoning-on":   "DS v4.1-Flash",
    "deepseek-v4.1-flash-reasoning-off":  "DS v4.1-Flash\n(No Reasoning)",
    "deepseek-v4.1-flash-1017-reasoning-on":  "DS v4.1-Flash-1017",
    "deepseek-v4.1-flash-1017-reasoning-off": "DS v4.1-Flash-1017\n(No Reasoning)",
}


def _generate_red_palette(n: int) -> list[str]:
    """Generate *n* shades of red from dark (#592525) to bright (#E61717)."""
    dark = np.array([0x59, 0x25, 0x25])
    bright = np.array([0xE6, 0x17, 0x17])
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        rgb = dark + t * (bright - dark)
        colors.append(f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}")
    return colors


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def find_run_dir() -> Path:
    """Find the most recent timestamped run directory under results/."""
    run_dirs = sorted(
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith((".", "smoke"))
    )
    if not run_dirs:
        raise RuntimeError(f"No benchmark run directories found under {RESULTS_DIR}")
    return run_dirs[-1]


def load_metrics(run_dir: Path) -> dict[str, dict]:
    """Load profiling-phase metrics for every model variant in *run_dir*.

    Returns ``{variant_label: {metric: value, ...}}`` where time metrics are
    converted from milliseconds to seconds and total tokens = ISL + OSL.
    """
    data: dict[str, dict] = {}
    for variant_dir in sorted(run_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        label = variant_dir.name
        json_path = variant_dir / "phases" / "profiling" / f"{label}.json"
        if not json_path.exists():
            print(f"  [SKIP] No profiling JSON for {label}")
            continue

        d = json.loads(json_path.read_text())
        ttft = d["time_to_first_token"]
        latency = d["request_latency"]
        isl = d["input_sequence_length"]["avg"]
        osl = d["output_sequence_length"]["avg"]

        data[label] = {
            "ttft_p50": ttft["p50"] / 1000,
            "ttft_p95": ttft["p95"] / 1000,
            "ttft_p99": ttft["p99"] / 1000,
            "latency_p50": latency["p50"] / 1000,
            "latency_p95": latency["p95"] / 1000,
            "latency_p99": latency["p99"] / 1000,
            "isl_avg": isl,
            "osl_avg": osl,
            "total_tokens_avg": isl + osl,
        }
    return data


# ──────────────────────────────────────────────────────────────────────
# Plot 1 & 2 — Grouped vertical bars (p50, p95, p99 per model)
# ──────────────────────────────────────────────────────────────────────
def _format_3_digits(val: float) -> str:
    """Format *val* to display exactly 3 digits (e.g. 0.74, 1.34, 12.3)."""
    if val < 10:
        return f"{val:.2f}"
    elif val < 100:
        return f"{val:.1f}"
    else:
        return f"{val:.0f}"


def plot_grouped_bars(
    data: dict[str, dict],
    metric_prefix: str,
    ylabel: str,
    filename: str,
) -> None:
    """Grouped vertical bar plot: p50 / p95 / p99 side-by-side per model."""
    models = list(data.keys())
    display = [DISPLAY_NAMES.get(m, m) for m in models]
    n = len(models)

    values = {p: [data[m][f"{metric_prefix}_{p}"] for m in models] for p in PERCENTILES}
    max_val = max(max(values[p]) for p in PERCENTILES)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    x = np.arange(n)
    width = 0.25

    for i, p in enumerate(PERCENTILES):
        bars = ax.bar(
            x + i * width,
            values[p],
            width,
            label=p,
            color=PERCENTILE_COLORS[i],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, val in zip(bars, values[p]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.015,
                _format_3_digits(val),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#333333",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(display, fontsize=11, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", labelpad=12)
    ax.set_ylim(0, max_val * 1.15)
    ax.legend(
        title="Percentile",
        fontsize=12,
        title_fontsize=13,
        loc="upper left",
        frameon=True,
        edgecolor="black",
    )

    sns.despine()
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {output_path.name}")


# ──────────────────────────────────────────────────────────────────────
# Plots 3-5 — Single horizontal bars (average per model)
# ──────────────────────────────────────────────────────────────────────
def plot_single_bars(
    data: dict[str, dict],
    metric_key: str,
    xlabel: str,
    filename: str,
) -> None:
    """Horizontal bar plot for one average metric, sorted descending.

    Mirrors the original generate_plots.py aesthetics: per-model red shades,
    value annotations, highlighted maximum, despine.
    """
    models = list(data.keys())
    raw_values = [data[m][metric_key] for m in models]
    display = [DISPLAY_NAMES.get(m, m) for m in models]

    # Sort by value descending so the largest bar is at the top.
    pairs = sorted(zip(display, raw_values), key=lambda p: p[1], reverse=True)
    sorted_models = [p[0] for p in pairs]
    sorted_values = [p[1] for p in pairs]

    # Per-model red shades (consistent regardless of plotting order).
    model_colors = _generate_red_palette(len(models))
    display_to_color = {DISPLAY_NAMES.get(m, m): model_colors[i] for i, m in enumerate(models)}
    palette = {m: display_to_color[m] for m in sorted_models}

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    bars = sns.barplot(
        x=sorted_values,
        y=sorted_models,
        hue=sorted_models,
        palette=palette,
        order=sorted_models,
        hue_order=sorted_models,
        legend=False,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    max_value = max(raw_values)
    for bar, value in zip(bars.patches, sorted_values):
        ax.text(
            bar.get_width() + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#333333",
        )

    # Highlight the model with the minimum value (least tokens consumed).
    extreme_display = sorted_models[-1]
    for label in ax.get_yticklabels():
        if label.get_text() == extreme_display:
            label.set_fontweight("bold")
            label.set_fontsize(16)
            label.set_color("white")
            label.set_bbox(
                dict(
                    facecolor="#750808",
                    edgecolor="black",
                    linewidth=1.2,
                    boxstyle="round,pad=0.35",
                )
            )

    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold", labelpad=12)
    ax.set_ylabel("Model", fontsize=14, fontweight="bold", labelpad=12)
    ax.set_xlim(0, max_value * 1.18)

    sns.despine()
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)

    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {output_path.name}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    run_dir = find_run_dir()
    print(f"Run directory: {run_dir}")

    data = load_metrics(run_dir)
    print(f"Loaded {len(data)} model variants: {', '.join(data.keys())}\n")

    print(f"Output directory: {OUTPUT_DIR}")
    print("Generating plots...\n")

    # Grouped bar plots — p50, p95, p99 per model
    plot_grouped_bars(data, "ttft", "Time to First Token (s)", "TTFT.png")
    plot_grouped_bars(data, "latency", "Duration (s)", "Duration.png")

    # Single bar plots — average per model
    plot_single_bars(data, "isl_avg", "Average Input Tokens (tokens)", "Input_Tokens.png")
    plot_single_bars(data, "osl_avg", "Average Output Tokens (tokens)", "Output_Tokens.png")
    plot_single_bars(data, "total_tokens_avg", "Total Tokens Consumption (tokens)", "Total_Token_Consumption.png")

    print(f"\nAll plots exported to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
