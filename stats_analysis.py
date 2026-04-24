#!/usr/bin/env python3
"""
stats_analysis.py

Automated repeated-seed statistical analysis for the three informed PINNs:
    - elliptic_pinn.py
    - parabolic_pinn.py
    - hyperbolic_pinn.py

This script:
1. Runs each PINN over multiple random seeds at N_DATA = 0 by default.
2. Collects final_metrics.json and history.json from each run.
3. Computes statistical summaries:
   - mean, std, median, min, max
   - coefficient of variation
   - 95% bootstrap confidence intervals for the mean
4. Produces:
   - per-PDE summary CSV files
   - a combined summary CSV
   - boxplots / stripplots for important metrics
   - mean ± std convergence plots
   - a markdown report with key takeaways

Assumptions
-----------
Each model file lives in the same directory as this script and exposes:

    run_single(seed: int = 0, n_data: int = 0, outdir: str = "...") -> Dict[str, float]

Each run should save at least:
    - final_metrics.json
    - history.json

Usage
-----
python stats_analysis.py

or edit the CONFIG section below.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import random
import sys
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

MODEL_FILES = {
    "elliptic": BASE_DIR / "elliptic_pinn.py",
    "parabolic": BASE_DIR / "parabolic_pinn.py",
    "hyperbolic": BASE_DIR / "hyperbolic_pinn.py",
}

SEEDS = [0, 1, 2, 3, 4]
N_DATA = 0
RERUN_MODELS = True

# Root output directory for this analysis
ANALYSIS_ROOT = BASE_DIR / "stats_analysis_outputs"

# These are the main metrics the script will try to plot if present.
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

# Metrics to try for convergence plots if present in history.json
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


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool)


def bootstrap_mean_ci(
    values: Iterable[float],
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for the mean.
    """
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


def newest_subdirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)


# ============================================================
# Module loading and model execution
# ============================================================

def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_run_single(module, seed: int, n_data: int, outdir: str) -> Dict[str, Any]:
    if not hasattr(module, "run_single"):
        raise AttributeError(f"Module {module.__name__} has no run_single(...) function")

    fn = module.run_single
    sig = inspect.signature(fn)
    kwargs = {}

    if "seed" in sig.parameters:
        kwargs["seed"] = seed
    if "n_data" in sig.parameters:
        kwargs["n_data"] = n_data
    if "outdir" in sig.parameters:
        kwargs["outdir"] = outdir

    return fn(**kwargs)


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
    """
    Find a subdirectory whose name contains 'seed{seed}' and that has final_metrics.json.
    """
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


def run_or_collect_all() -> Dict[str, List[RunRecord]]:
    ensure_dir(ANALYSIS_ROOT)
    all_records: Dict[str, List[RunRecord]] = {k: [] for k in MODEL_FILES.keys()}

    for pde, file_path in MODEL_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Expected {file_path.name} in {BASE_DIR}, but it was not found."
            )

        model_outdir = ANALYSIS_ROOT / f"{pde}_runs"
        ensure_dir(model_outdir)

        if RERUN_MODELS:
            print(f"\n=== Running {pde} model from {file_path.name} ===")
            module = load_module_from_file(f"{pde}_module", file_path)
            for seed in SEEDS:
                print(f"\n[{pde}] seed={seed}, n_data={N_DATA}")
                call_run_single(module, seed=seed, n_data=N_DATA, outdir=str(model_outdir))

        # Collect outputs
        records = []
        for seed in SEEDS:
            run_dir = find_run_dir_for_seed(model_outdir, seed)
            if run_dir is None:
                print(f"WARNING: No run directory found for {pde}, seed={seed}")
                continue
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
    numeric_cols = [c for c in df.columns if c not in {"pde", "seed", "run_dir"} and pd.api.types.is_numeric_dtype(df[c])]
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
    available = [m for m in PRIMARY_METRICS.get(pde, []) if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]
    if not available:
        return

    # One metric per figure for readability
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
        if np.all(vals > 0):
            spread = np.max(vals) / max(np.min(vals), 1e-16)
            if spread > 50:
                ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(outdir / f"{pde}_{metric}_boxplot.png", dpi=160)
        plt.close(fig)


def save_combined_metric_plots(all_dfs: Dict[str, pd.DataFrame], outdir: Path) -> None:
    ensure_dir(outdir)
    # A few cross-PDE comparisons when names exist in >=2 PDEs
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
        ax.boxplot(plot_data, labels=labels)
        for i, vals in enumerate(plot_data, start=1):
            jitter = i + 0.05 * np.random.default_rng(i).normal(size=len(vals))
            ax.scatter(jitter, vals, alpha=0.7, s=20)
        ax.set_title(f"Cross-PDE comparison: {metric}")
        ax.set_ylabel(metric)
        positive_vals = np.concatenate([v for v in plot_data if np.all(v > 0)])
        if len(positive_vals) > 0:
            spread = np.max(positive_vals) / max(np.min(positive_vals), 1e-16)
            if spread > 50:
                ax.set_yscale("log")
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

        positive = np.all(mean[np.isfinite(mean)] > 0)
        if positive and np.nanmax(mean) / max(np.nanmin(mean), 1e-16) > 50:
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
    lines.append(f"- Baseline included: No")
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

        # Brief stability comment
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

    all_records = run_or_collect_all()

    all_dfs: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, pd.DataFrame] = {}

    summary_dir = ANALYSIS_ROOT / "summaries"
    plot_dir = ANALYSIS_ROOT / "plots"
    ensure_dir(summary_dir)
    ensure_dir(plot_dir)

    combined_summary_frames = []

    for pde, records in all_records.items():
        print(f"\n=== Aggregating {pde} ===")
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

    combined_summary = pd.concat(combined_summary_frames, ignore_index=True) if combined_summary_frames else pd.DataFrame()
    combined_summary.to_csv(summary_dir / "combined_summary_statistics.csv", index=False)

    save_combined_metric_plots(all_dfs, plot_dir / "combined")

    build_markdown_report(
        summaries=summaries,
        all_dfs=all_dfs,
        outpath=ANALYSIS_ROOT / "stats_analysis_report.md",
    )

    print("\nFinished.")
    print(f"Outputs saved in: {ANALYSIS_ROOT}")


if __name__ == "__main__":
    main()
