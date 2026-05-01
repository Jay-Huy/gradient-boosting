#!/usr/bin/env python3
"""
plot_training_metrics.py
------------------------
Visualize boosting training metrics from metrics.json files.

Usage:
    python plot_training_metrics.py
    python plot_training_metrics.py --input-dir results/runs
    python plot_training_metrics.py --input-dir path/to/single/run
    python plot_training_metrics.py --input-dir results/runs --score-label "Perplexity" --dpi 150

For each run folder containing a metrics.json, this script creates:
    <run_folder>/
    └── visualization/
        ├── training-overview.png   ← full (num_learners+1) × 4 grid
        ├── overall.png             ← overall row only
        ├── learner-1.png           ← learner 1 row only
        ├── learner-2.png           ← learner 2 row only
        └── ...
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────── Style constants ───────────────────────────────

LOSS_COLOR   = "#2563EB"   # blue
SCORE_COLOR  = "#C2410C"   # orange-red
FILL_ALPHA   = 0.07
LINE_ALPHA   = 0.88
LINE_WIDTH   = 1.7
STOP_COLOR   = "#94A3B8"   # slate for best-step marker
ANNOT_STYLE  = dict(
    boxstyle="round,pad=0.4",
    facecolor="#FAFAFA",
    edgecolor="#CBD5E1",
    alpha=0.95,
)
AX_BG        = "#FDFEFF"
GRID_COLOR   = "#E2E8F0"


# ─────────────────────────── Helpers ───────────────────────────────────────

def style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=10.5, fontweight="semibold", color="#1E293B", pad=5)
    ax.set_xlabel(xlabel, fontsize=8, color="#64748B")
    ax.set_ylabel(ylabel, fontsize=8, color="#64748B")
    ax.tick_params(labelsize=7.5, colors="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.set_facecolor(AX_BG)
    ax.grid(True, linestyle="--", linewidth=0.4, color=GRID_COLOR)


def plot_line(ax, x, y, color: str) -> None:
    ax.plot(x, y, color=color, linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
    ax.fill_between(x, y, min(y), alpha=FILL_ALPHA, color=color)


def annotate_val_improvement(ax, lp: dict, kind: str) -> None:
    """Add Before/After/Improvement text box for val metrics from learner_progress."""
    if kind == "loss":
        txt = (
            f"Before: {lp['before_loss']:.4f}\n"
            f"After:   {lp['after_loss']:.4f}\n"
            f"Imprv:  {lp['loss_improvement']:.4f}"
        )
    else:
        txt = (
            f"Before: {lp['before_score']:.2f}\n"
            f"After:   {lp['after_score']:.2f}\n"
            f"Imprv:  {lp['score_improvement']:.2f}"
        )
    ax.text(
        0.97, 0.97, txt,
        transform=ax.transAxes,
        fontsize=7.2, va="top", ha="right",
        bbox=ANNOT_STYLE, color="#334155",
        fontfamily="monospace",
    )


def mark_best_step(ax, best_step: int, y_values: list) -> None:
    ymin, ymax = min(y_values), max(y_values)
    ax.axvline(best_step, color=STOP_COLOR, linestyle=":", linewidth=1.1, alpha=0.9)
    ax.text(
        best_step + 12,
        ymin + (ymax - ymin) * 0.92,
        f"best@{best_step}",
        fontsize=6.2, color=STOP_COLOR, va="top",
    )


# ─────────────────────────── Row builders ──────────────────────────────────

def build_overall_row(
    fig, gs, row: int, col_axes: list,
    train_losses, train_scores, val_losses, val_scores,
    score_label: str,
) -> list:
    """Plot the Overall row. Returns list of 4 Axes."""
    configs = [
        (train_losses, LOSS_COLOR,  f"Train Loss",           "Step Index", "Loss"),
        (val_losses,   LOSS_COLOR,  f"Val Loss",             "Step Index", "Loss"),
        (train_scores, SCORE_COLOR, f"Train {score_label}",  "Step Index", score_label),
        (val_scores,   SCORE_COLOR, f"Val {score_label}",    "Step Index", score_label),
    ]
    row_axes = []
    for col, (arr, color, title, xl, yl) in enumerate(configs):
        sharey = col_axes[col] if col_axes[col] is not None else None
        ax = fig.add_subplot(gs[row, col], sharey=sharey)
        plot_line(ax, list(range(len(arr))), arr, color)
        style_ax(ax, title, xl, yl)
        row_axes.append(ax)
    return row_axes


def build_learner_row(
    fig, gs, row: int, col_axes: list,
    lid: int, train_recs: list, val_recs: list,
    learner_progress: dict, max_step: int,
    score_label: str,
) -> list:
    """Plot one learner row. Returns list of 4 Axes."""
    train_recs = sorted(train_recs, key=lambda r: r["step"])
    val_recs   = sorted(val_recs,   key=lambda r: r["step"])

    ts  = [r["step"]      for r in train_recs]
    tl  = [r["avg_loss"]  for r in train_recs]
    tsc = [r["avg_score"] for r in train_recs]
    vs  = [r["step"]      for r in val_recs]
    vl  = [r["avg_loss"]  for r in val_recs]
    vsc = [r["avg_score"] for r in val_recs]

    lp         = learner_progress.get(lid, {})
    best_step  = lp.get("best_step")

    configs = [
        (ts, tl,  LOSS_COLOR,  f"Learner {lid} — Train Loss",         "Step", "Loss",      None),
        (vs, vl,  LOSS_COLOR,  f"Learner {lid} — Val Loss",           "Step", "Loss",      "loss"),
        (ts, tsc, SCORE_COLOR, f"Learner {lid} — Train {score_label}","Step", score_label, None),
        (vs, vsc, SCORE_COLOR, f"Learner {lid} — Val {score_label}",  "Step", score_label, "score"),
    ]

    row_axes = []
    for col, (xv, yv, color, title, xl, yl, annot_kind) in enumerate(configs):
        sharey = col_axes[col] if col_axes[col] is not None else None
        ax = fig.add_subplot(gs[row, col], sharey=sharey)
        plot_line(ax, xv, yv, color)
        ax.set_xlim(0, max_step + 80)
        style_ax(ax, title, xl, yl)

        if annot_kind and lp:
            annotate_val_improvement(ax, lp, annot_kind)

        if best_step:
            mark_best_step(ax, best_step, yv)

        row_axes.append(ax)
    return row_axes


# ─────────────────────────── Single-row figure ─────────────────────────────

def save_single_row(
    row_data: dict,
    out_path: Path,
    score_label: str,
    dpi: int,
) -> None:
    """
    Re-render a single row as a standalone figure and save it.
    row_data keys: "title", "configs" → list of (x, y, color, title, xl, yl, annot_kind, lp, best_step, max_step)
    """
    n_cols = 4
    fig, axes = plt.subplots(1, n_cols, figsize=(22, 4.4), facecolor="white")
    fig.suptitle(row_data["title"], fontsize=14, fontweight="bold", color="#0F172A",
                 fontfamily="serif", y=1.02)

    for ax, (xv, yv, color, title, xl, yl, annot_kind, lp, best_step, max_step) in \
            zip(axes, row_data["configs"]):
        plot_line(ax, xv, yv, color)
        if max_step is not None:
            ax.set_xlim(0, max_step + 80)
        style_ax(ax, title, xl, yl)
        if annot_kind and lp:
            annotate_val_improvement(ax, lp, annot_kind)
        if best_step:
            mark_best_step(ax, best_step, yv)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────── Main per-run logic ────────────────────────────

def process_run(metrics_path: Path, score_label: str, dpi: int) -> None:
    print(f"\n→ Processing: {metrics_path}")

    with open(metrics_path) as f:
        data = json.load(f)

    # ── Config ──
    model_params  = data["metadata"]["config"]["model"]["params"]
    num_learners  = model_params["num_learners"]
    n_layer       = model_params["weak_learner"]["n_layer"]
    main_title    = f"{num_learners} Learners — {n_layer} Layers"

    learner_progress = {
        lp["learner_id"]: lp
        for lp in data["metadata"]["boosting"]["learner_progress"]
    }

    # ── Data ──
    train_losses_overall = data["train"]["losses"]
    train_scores_overall = data["train"]["scores"]["perplexity"]
    val_losses_overall   = data["val"]["losses"]
    val_scores_overall   = data["val"]["scores"]["perplexity"]

    train_by_learner: dict = collections.defaultdict(list)
    val_by_learner:   dict = collections.defaultdict(list)
    for r in data["train"]["records"]:
        train_by_learner[r["learner_id"]].append(r)
    for r in data["val"]["records"]:
        val_by_learner[r["learner_id"]].append(r)

    learner_ids = sorted(train_by_learner.keys())
    max_step    = max(
        max(r["step"] for r in train_by_learner[lid]) for lid in learner_ids
    )

    # ── Output dir ──
    vis_dir = metrics_path.parent / "visualization"
    vis_dir.mkdir(exist_ok=True)

    # ════════════════════════════════════════════════
    #  BIG DIAGRAM  (num_learners+1) × 4
    # ════════════════════════════════════════════════
    n_rows = 1 + num_learners
    n_cols = 4

    fig = plt.figure(figsize=(24, 5.2 * n_rows), facecolor="white")
    fig.suptitle(main_title, fontsize=22, fontweight="bold",
                 y=1.005, color="#0F172A", fontfamily="serif")

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.60, wspace=0.40)

    # col_axes[col] = reference ax for sharey (set from row 0)
    col_axes = [None] * n_cols

    # -- Row 0: Overall --
    overall_axes = build_overall_row(
        fig, gs, 0, col_axes,
        train_losses_overall, train_scores_overall,
        val_losses_overall, val_scores_overall,
        score_label,
    )
    for col, ax in enumerate(overall_axes):
        col_axes[col] = ax

    # -- Rows 1+: Per-Learner --
    for row_idx, lid in enumerate(learner_ids):
        build_learner_row(
            fig, gs, row_idx + 1, col_axes,
            lid,
            train_by_learner[lid],
            val_by_learner[lid],
            learner_progress,
            max_step,
            score_label,
        )

    plt.tight_layout(rect=[0, 0, 1, 1])
    overview_path = vis_dir / "training-overview.png"
    fig.savefig(overview_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: {overview_path}")

    # ════════════════════════════════════════════════
    #  INDIVIDUAL ROW IMAGES
    # ════════════════════════════════════════════════

    # -- Overall row --
    overall_row_data = {
        "title": f"{main_title}  |  Overall",
        "configs": [
            (list(range(len(train_losses_overall))), train_losses_overall,
             LOSS_COLOR,  "Train Loss",          "Step Index", "Loss",      None, None, None, None),
            (list(range(len(val_losses_overall))),   val_losses_overall,
             LOSS_COLOR,  "Val Loss",            "Step Index", "Loss",      None, None, None, None),
            (list(range(len(train_scores_overall))), train_scores_overall,
             SCORE_COLOR, f"Train {score_label}","Step Index", score_label, None, None, None, None),
            (list(range(len(val_scores_overall))),   val_scores_overall,
             SCORE_COLOR, f"Val {score_label}",  "Step Index", score_label, None, None, None, None),
        ],
    }
    save_single_row(overall_row_data, vis_dir / "overall.png", score_label, dpi)
    print(f"  ✓ Saved: {vis_dir / 'overall.png'}")

    # -- Per-Learner rows --
    for lid in learner_ids:
        train_recs = sorted(train_by_learner[lid], key=lambda r: r["step"])
        val_recs   = sorted(val_by_learner[lid],   key=lambda r: r["step"])

        ts  = [r["step"]      for r in train_recs]
        tl  = [r["avg_loss"]  for r in train_recs]
        tsc = [r["avg_score"] for r in train_recs]
        vs  = [r["step"]      for r in val_recs]
        vl  = [r["avg_loss"]  for r in val_recs]
        vsc = [r["avg_score"] for r in val_recs]

        lp        = learner_progress.get(lid, {})
        best_step = lp.get("best_step")

        learner_row_data = {
            "title": f"{main_title}  |  Learner {lid}",
            "configs": [
                (ts, tl,  LOSS_COLOR,  f"Learner {lid} — Train Loss",
                 "Step", "Loss",      None,    None,      None,      max_step),
                (vs, vl,  LOSS_COLOR,  f"Learner {lid} — Val Loss",
                 "Step", "Loss",      "loss",  lp or None, best_step, max_step),
                (ts, tsc, SCORE_COLOR, f"Learner {lid} — Train {score_label}",
                 "Step", score_label, None,    None,      None,      max_step),
                (vs, vsc, SCORE_COLOR, f"Learner {lid} — Val {score_label}",
                 "Step", score_label, "score", lp or None, best_step, max_step),
            ],
        }
        out = vis_dir / f"learner-{lid}.png"
        save_single_row(learner_row_data, out, score_label, dpi)
        print(f"  ✓ Saved: {out}")


# ─────────────────────────── CLI ───────────────────────────────────────────

def find_metrics_files(input_dir: Path) -> list[Path]:
    """
    Search strategy:
      1. If input_dir itself contains metrics.json → treat as single run.
      2. Otherwise scan one level of subfolders for metrics.json.
    """

    direct = input_dir / "metrics.json"
    if direct.exists():
        return [direct]

    found = sorted(input_dir.glob("*/metrics.json"))
    if not found:
        print(f"[warn] No metrics.json found under {input_dir}", file=sys.stderr)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot boosting training metrics from metrics.json files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        help="Root directory to search for run folders.",
    )
    parser.add_argument(
        "--score-label", "-s",
        type=str,
        default="Perplexity",
        help="Label for the score metric axis (default: Perplexity).",
    )
    parser.add_argument(
        "--dpi", "-d",
        type=int,
        default=150,
        help="DPI for saved images (default: 150).",
    )
    args = parser.parse_args()

    base_dir = "results/runs/"
    input_dir = Path(base_dir + args.input_dir)

    if not input_dir.exists():
        print(f"[error] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    metrics_files = find_metrics_files(input_dir)
    if not metrics_files:
        sys.exit(1)

    print(f"Found {len(metrics_files)} run(s) to process.")
    for mf in metrics_files:
        try:
            process_run(mf, score_label=args.score_label, dpi=args.dpi)
        except Exception as exc:
            print(f"  [error] Failed to process {mf}: {exc}", file=sys.stderr)

    print("\nAll done!")


if __name__ == "__main__":
    main()