"""
Exploratory analysis of the intention bias mechanism.

Compares six runs of the exploratory intention bias experiment with ArgRL-Full and PPO baseline
using the same loading, final-50 summary and bootstrap functions as the confirmatory analysis.

Usage:
  python exploratory/bandit_analysis.py
  python exploratory/bandit_analysis.py --verbose   # show run directories
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))

from confirmatory_analysis import (
    COND_LABEL,          # condition identifier -> thesis name
    LOG_ROOT,
    load_soups,          # soups_delivered scalars from a run directory
    seed_summary,        # mean over the final FINAL_WINDOW outcomes
    bca_ci,              # BCa 95% CI on a mean difference, 10k resamples
    _count_scalars,      # tie-break helper (most logged points wins)
    _find_run_dir,       # discovery for ArgRL-Full and PPO baseline
    FINAL_WINDOW,
)
from text_tables import fixed, print_table

SEEDS = list(range(6))            # 0-5, identical to confirmatory


def _find_bandit_dir(seed: int):
    """Run directory for cramped_room_bandit_seed{seed}_<timestamp>.

    Matches on the name prefix and, among several matches, keeps the run with
    the most logged points, as the confirmatory loader does.
    """
    pat = f"cramped_room_bandit_seed{seed}_"
    cands = [d for d in LOG_ROOT.iterdir() if d.is_dir() and d.name.startswith(pat)]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    return max(cands, key=lambda p: _count_scalars(p, "train/soups_delivered"))


def _summaries(finder) -> list[float]:
    """Return the available seed-level summaries for a condition."""
    values = []
    for seed in SEEDS:
        run_dir = finder(seed)
        if run_dir is None:
            continue
        try:
            values.append(seed_summary(load_soups(run_dir)))
        except Exception:
            continue
    return values


def bandit_seeds(verbose: bool) -> tuple[np.ndarray, list[dict]]:
    """Per-seed summaries for the exploratory intention bias experiment, and the rows to print.

    The directory name and the logged-point count are diagnostics rather than
    results, so they appear only under --verbose or for a seed that is missing.
    """
    values, rows = [], []
    for seed in SEEDS:
        run_dir = _find_bandit_dir(seed)
        if run_dir is None:
            rows.append({"Seed": seed, "Final mean": "missing", "Run": "-"})
            continue
        soups = load_soups(run_dir)
        summary = seed_summary(soups)
        values.append(summary)
        row = {"Seed": seed, "Final mean": summary}
        if verbose:
            row["Logged points"] = len(soups)
            row["Run"] = run_dir.name
        rows.append(row)

    if not verbose and not any(r.get("Run") == "-" for r in rows):
        for row in rows:
            row.pop("Run", None)
    return np.array(values), rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--verbose", action="store_true",
        help="Also show the run directory and logged-point count per seed",
    )
    args = parser.parse_args()

    print("Exploratory intention bias experiment")
    print(f"Cramped Room, 1M steps, six seeds. Performance is the mean over "
          f"the final {FINAL_WINDOW} logged episode outcomes.")
    print()

    bandit, seed_rows = bandit_seeds(args.verbose)
    print("Intention bias experiment runs")
    print_table(seed_rows, formatters={"Final mean": fixed()})

    if len(bandit) < 2:
        print("Fewer than two intention bias experiment seeds are available, so no comparison is made.")
        return 1

    argrl_full = np.asarray(_summaries(lambda seed: _find_run_dir("argrl_full", seed)))
    ppo_baseline = np.asarray(_summaries(lambda seed: _find_run_dir("ppo_baseline", seed)))

    print("Final performance")
    print_table(
        [
            {
                "Condition": COND_LABEL[cond],
                "Mean": values.mean(),
                "SD": values.std(ddof=1) if len(values) > 1 else 0.0,
            }
            for cond, values in [("bandit", bandit),
                                 ("argrl_full", argrl_full),
                                 ("ppo_baseline", ppo_baseline)]
            if len(values)
        ],
        formatters={"Mean": fixed(), "SD": fixed()},
    )

    rows = []
    for cond, values in [("argrl_full", argrl_full), ("ppo_baseline", ppo_baseline)]:
        if len(values) < 2:
            rows.append({"Comparison": f"{COND_LABEL['bandit']} minus {COND_LABEL[cond]}",
                         "Difference": "insufficient data", "BCa 95% CI": "-"})
            continue
        lo, hi = bca_ci(values, bandit)   # bca_ci(a, b) gives mean_b - mean_a
        rows.append({
            "Comparison": f"{COND_LABEL['bandit']} minus {COND_LABEL[cond]}",
            "Difference": float(bandit.mean() - values.mean()),
            "BCa 95% CI": f"[{lo:.3f}, {hi:.3f}]",
        })

    print("Mean differences")
    print("BCa intervals summarise the seed-level mean differences.")
    print_table(rows, formatters={"Difference": fixed(signed=True)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
