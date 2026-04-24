#!/usr/bin/env python3
"""
stats_analysis_plot_only.py

Plot/report-only version of stats_analysis.py.

This script does NOT retrain any models and does NOT import the model files.
It reads existing saved outputs from:

    stats_analysis_outputs/{elliptic,parabolic,hyperbolic}_runs/*/final_metrics.json
    stats_analysis_outputs/{elliptic,parabolic,hyperbolic}_runs/*/history.json

Then it regenerates:
    - per-PDE final metric CSV files
    - per-PDE summary statistic CSV files
    - combined summary statistic CSV
    - per-metric boxplots / stripplots
    - convergence mean ± std plots
    - markdown report

Usage:
    python stats_analysis_plot_only.py

Place this file in the same Code directory that contains stats_analysis_outputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PDES = ["elliptic", "parabolic", "hyperbolic"]
SEEDS = [0, 1, 2, 3, 4]
N_DATA = 0

# Existing root output directory from the original training/statistics run.
ANALYSIS_ROOT = BASE_DIR / "stats_analysis_outputs"

PRIMARY_METRICS = {
    "elliptic": [
        "rel_l2_test",
        "heldout_residual_mse",
        "boundary_rel_l2",
        "h1_semi_rel",
        "mean_barrier_violation",
        "max_lower_violation",
        "max_upper_violation",
        "min_pred",
        "max_pred",
    ],
    "parabolic": [
        "rel_l2_test",
        "final_time_rel_l2",
        "heldout_residual_mse",
        "mean_barrier_violation",
        "frac_negative",
        "neg_part_l2",
        "min_pred",
        "max_pred",
        "ic_max_abs",
        "bc_max_abs",
    ],
    "hyperbolic": [
        "rel_l1_test",
        "final_time_rel_l1",
        "final_time_rel_l2",
        "heldout_residual_mse",
        "heldout_entropy_violation",
        "shock_location_error_final",
        "shock_width_final",
        "overshoot",
        "undershoot",
        "mean_entropy_violation_eval",
        "max_entropy_violation_eval",
    ],
}

CONVERGENCE_METRICS = {
    "elliptic": [
        "total_loss",
        "loss_energy",
        "loss_qual",
        "heldout_residual_mse",
        "rel_l2_test",
        "boundary_rel_l2",
        "h1_semi_rel",
        "mean_barrier_violation",
    ],
    "parabolic": [
        "total_loss",
        "loss_res",
        "loss_qual",
        "heldout_residual_mse",
        "rel_l2_test",
        "final_time_rel_l2",
        "mean_barrier_violation",
        "frac_negative",
    ],
    "hyperbolic": [
        "total_loss",
        "loss_res",
        "loss_ent",
        "heldout_residual_mse",
        "heldout_entropy_violation",
        "rel_l1_test",
        "final_time_rel_l1",
        "shock_location_error_final",
    ],
}


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bootstrap_mean_ci(
    values: Iterable[float],
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    vals = np.asarray(list(values), dtype=float)
    if len(vals) == 0:
        return (math.nan, math.nan)
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(vals)
    for i in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        means[i] = np.mean(sample)

    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def coefficient_of_variation(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    if len(vals) == 0:
        return math.nan
    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if abs(mu) < 1e-14:
        return math.nan
    return sigma / abs(mu)


def maybe_use_log_scale(ax, arrays: List[np.ndarray]) -> None:
    """
    Match the original intent: use log scale for very wide positive spreads.

    Robust fix: collect positive entries elementwise instead of requiring an entire
    group to be strictly positive. This prevents crashes for metrics with valid
    zeros, e.g. max_lower_violation, max_upper_violation, or overshoot.
    """
    positive_groups = [np.asarray(v, dtype=float)[np.asarray(v, dtype=float) > 0] for v in arrays]
    positive_groups = [v for v in positive_groups if len(v) > 0]
    if not positive_groups:
        return

    positive_vals = np.concatenate(positive_groups)
    if len(positive_vals) == 0:
        return

    ymin = np.nanmin(positive_vals)
    ymax = np.nanmax(positive_vals)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0:
        spread = ymax / max(ymin, 1e-16)
        if spread > 50:
            ax.set_yscale("log")


# ============================================================
# Data collection
# ============================================================

@dataclass
class RunRecord:
    pde: str
    seed: int
    run_dir: Path
    final_metrics: Dict[str, Any]
    history: Dict[str, Any]


def find_run_dir_for_seed(outdir: Path, seed: int) -> Optional[Path]:
    """Find newest subdirectory containing seed{seed} and final_metrics.json."""
    if not outdir.exists():
        return None

    candidates = []
    for p in outdir.rglob("*"):
        if p.is_dir() and f"seed{seed}" in p.name and (p / "final_metrics.json").exists():
            candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_run_record(pde: str, seed: int, run_dir: Path) -> RunRecord:
    with open(run_dir / "final_metrics.json", "r", encoding="utf-8") as f:
        final_metrics = json.load(f)

    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {}

    return RunRecord(
        pde=pde,
        seed=seed,
        run_dir=run_dir,
        final_metrics=final_metrics,
        history=history,
    )


def collect_existing_all() -> Dict[str, List[RunRecord]]:
    if not ANALYSIS_ROOT.exists():
        raise FileNotFoundError(
            f"Could not find {ANALYSIS_ROOT}. Run this script from the same directory "
            "that contains stats_analysis_outputs."
        )

    all_records: Dict[str, List[RunRecord]] = {pde: [] for pde in PDES}

    for pde in PDES:
        model_outdir = ANALYSIS_ROOT / f"{pde}_runs"
        records: List[RunRecord] = []

        for seed in SEEDS:
            run_dir = find_run_dir_for_seed(model_outdir, seed)
            if run_dir is None:
                print(f"WARNING: No existing run directory found for {pde}, seed={seed}")
                continue
            print(f"[{pde}] seed={seed}: using {run_dir}")
            records.append(load_run_record(pde, seed, run_dir))

        all_records[pde] = records

    return all_records


# ============================================================
# Summary tables
# ============================================================

def final_metrics_dataframe(records: List[RunRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {"pde": rec.pde, "seed": rec.seed, "run_dir": str(rec.run_dir)}
        for k, v in rec.final_metrics.items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame, pde: str) -> pd.DataFrame:
    numeric_cols = [
        c for c in df.columns
        if c not in {"pde", "seed", "run_dir"} and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for col in numeric_cols:
        vals = df[col].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue
        ci_lo, ci_hi = bootstrap_mean_ci(vals, seed=12345)
        rows.append({
            "pde": pde,
            "metric": col,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "cv": float(coefficient_of_variation(vals)),
            "bootstrap_ci95_lo": ci_lo,
            "bootstrap_ci95_hi": ci_hi,
        })
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def save_metric_boxplots(df: pd.DataFrame, pde: str, outdir: Path) -> None:
    ensure_dir(outdir)
    available = [
        m for m in PRIMARY_METRICS.get(pde, [])
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m])
    ]
    if not available:
        return

    for metric in available:
        vals = df[metric].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(vals, widths=0.35, vert=True)
        jitter = 1.0 + 0.04 * np.random.default_rng(0).normal(size=len(vals))
        ax.scatter(jitter, vals, alpha=0.8, s=25)
        ax.set_xticks([1])
        ax.set_xticklabels([metric], rotation=0)
        ax.set_title(f"{pde}: distribution across seeds")
        ax.set_ylabel(metric)

        maybe_use_log_scale(ax, [vals])

        fig.tight_layout()
        fig.savefig(outdir / f"{pde}_{metric}_boxplot.png", dpi=160)
        plt.close(fig)


def save_combined_metric_plots(all_dfs: Dict[str, pd.DataFrame], outdir: Path) -> None:
    ensure_dir(outdir)
    candidate_metrics = [
        "rel_l2_test",
        "heldout_residual_mse",
        "min_pred",
        "max_pred",
    ]

    for metric in candidate_metrics:
        plot_data = []
        labels = []
        for pde, df in all_dfs.items():
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                vals = df[metric].dropna().astype(float).to_numpy()
                if len(vals) > 0:
                    plot_data.append(vals)
                    labels.append(pde)

        if len(plot_data) < 2:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))

        # Matplotlib >= 3.9 renamed labels -> tick_labels. Fall back for older versions.
        try:
            ax.boxplot(plot_data, tick_labels=labels)
        except TypeError:
            ax.boxplot(plot_data, labels=labels)

        for i, vals in enumerate(plot_data, start=1):
            jitter = i + 0.05 * np.random.default_rng(i).normal(size=len(vals))
            ax.scatter(jitter, vals, alpha=0.7, s=20)

        ax.set_title(f"Cross-PDE comparison: {metric}")
        ax.set_ylabel(metric)
        maybe_use_log_scale(ax, plot_data)

        fig.tight_layout()
        fig.savefig(outdir / f"combined_{metric}_boxplot.png", dpi=160)
        plt.close(fig)


def histories_to_dataframe(records: List[RunRecord], metric: str) -> Optional[pd.DataFrame]:
    frames = []
    for rec in records:
        hist = rec.history
        if "epoch" not in hist or metric not in hist:
            continue

        epochs = np.asarray(hist["epoch"], dtype=float)
        vals = np.asarray(hist[metric], dtype=float)
        if len(epochs) != len(vals):
            continue

        frames.append(pd.DataFrame({"epoch": epochs, "value": vals, "seed": rec.seed}))

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def save_convergence_bands(records: List[RunRecord], pde: str, outdir: Path) -> None:
    ensure_dir(outdir)

    for metric in CONVERGENCE_METRICS.get(pde, []):
        hist_df = histories_to_dataframe(records, metric)
        if hist_df is None or hist_df.empty:
            continue

        grouped = hist_df.groupby("epoch")["value"]
        summary = grouped.agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = summary["epoch"].to_numpy()
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].fillna(0.0).to_numpy(dtype=float)

        ax.plot(epochs, mean, label=f"{metric} mean")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.25, label="±1 std")

        finite_mean = mean[np.isfinite(mean)]
        if len(finite_mean) > 0 and np.all(finite_mean > 0):
            if np.nanmax(finite_mean) / max(np.nanmin(finite_mean), 1e-16) > 50:
                ax.set_yscale("log")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{pde}: mean ± std across seeds")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{pde}_{metric}_mean_std.png", dpi=160)
        plt.close(fig)


# ============================================================
# Markdown report
# ============================================================

def build_markdown_report(
    summaries: Dict[str, pd.DataFrame],
    all_dfs: Dict[str, pd.DataFrame],
    outpath: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Statistical uncertainty analysis")
    lines.append("")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- N_DATA: {N_DATA}")
    lines.append("- Baseline included: No")
    lines.append("- Generated from existing saved outputs only; no models were retrained.")
    lines.append("")

    lines.append("## Main interpretation")
    lines.append("")
    lines.append(
        "This analysis treats the trained informed PINN as a random outcome of initialization, "
        "collocation sampling, and optimization. Final metrics are therefore summarized empirically "
        "across repeated seeds rather than reported only as single-run point estimates."
    )
    lines.append("")

    for pde, summary_df in summaries.items():
        lines.append(f"## {pde.capitalize()}")
        lines.append("")
        if summary_df.empty:
            lines.append("_No numeric summary available._")
            lines.append("")
            continue

        priority = [m for m in PRIMARY_METRICS.get(pde, []) if m in set(summary_df["metric"])]
        shown = priority[:6] if priority else list(summary_df["metric"])[:6]

        for metric in shown:
            row = summary_df[summary_df["metric"] == metric]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"- **{metric}**: mean = {r['mean']:.6e}, std = {r['std']:.6e}, "
                f"median = {r['median']:.6e}, 95% bootstrap CI for mean = "
                f"[{r['bootstrap_ci95_lo']:.6e}, {r['bootstrap_ci95_hi']:.6e}]"
            )
        lines.append("")

        df = all_dfs[pde]
        if "rel_l2_test" in df.columns:
            vals = df["rel_l2_test"].dropna().astype(float).to_numpy()
            if len(vals) > 1:
                cv = coefficient_of_variation(vals)
                lines.append(f"- Relative L2 coefficient of variation: {cv:.3e}")
                lines.append("")
        if "rel_l1_test" in df.columns:
            vals = df["rel_l1_test"].dropna().astype(float).to_numpy()
            if len(vals) > 1:
                cv = coefficient_of_variation(vals)
                lines.append(f"- Relative L1 coefficient of variation: {cv:.3e}")
                lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Wider uncertainty bands or larger seed-to-seed spread suggest greater sensitivity to initialization "
        "and collocation randomness."
    )
    lines.append(
        "- This is an empirical repeated-training uncertainty analysis, not a full Bayesian posterior analysis."
    )
    lines.append(
        "- The summary statistics here can directly support the report's reliability / statistical perspectives section."
    )
    lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_dir(ANALYSIS_ROOT)

    print("Collecting existing run outputs only. No models will be retrained.\n")
    all_records = collect_existing_all()

    all_dfs: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, pd.DataFrame] = {}

    summary_dir = ANALYSIS_ROOT / "summaries"
    plot_dir = ANALYSIS_ROOT / "plots"
    ensure_dir(summary_dir)
    ensure_dir(plot_dir)

    combined_summary_frames = []

    for pde, records in all_records.items():
        print(f"\n=== Aggregating {pde} ===")
        if not records:
            print(f"WARNING: No records found for {pde}; skipping.")
            all_dfs[pde] = pd.DataFrame()
            summaries[pde] = pd.DataFrame()
            continue

        df = final_metrics_dataframe(records)
        all_dfs[pde] = df

        pde_summary_dir = summary_dir / pde
        pde_plot_dir = plot_dir / pde
        ensure_dir(pde_summary_dir)
        ensure_dir(pde_plot_dir)

        df.to_csv(pde_summary_dir / f"{pde}_final_metrics_by_seed.csv", index=False)

        summary_df = numeric_summary(df, pde)
        summaries[pde] = summary_df
        summary_df.to_csv(pde_summary_dir / f"{pde}_summary_statistics.csv", index=False)
        combined_summary_frames.append(summary_df)

        save_metric_boxplots(df, pde, pde_plot_dir)
        save_convergence_bands(records, pde, pde_plot_dir)

    combined_summary = (
        pd.concat(combined_summary_frames, ignore_index=True)
        if combined_summary_frames else pd.DataFrame()
    )
    combined_summary.to_csv(summary_dir / "combined_summary_statistics.csv", index=False)

    nonempty_dfs = {k: v for k, v in all_dfs.items() if not v.empty}
    save_combined_metric_plots(nonempty_dfs, plot_dir / "combined")

    build_markdown_report(
        summaries=summaries,
        all_dfs=all_dfs,
        outpath=ANALYSIS_ROOT / "stats_analysis_report.md",
    )

    print("\nFinished.")
    print(f"Outputs saved in: {ANALYSIS_ROOT}")


if __name__ == "__main__":
    main()
