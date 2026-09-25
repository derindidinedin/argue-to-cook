"""
Exploratory analysis of the ArgRL-Shape condition.

ArgRL-Shape uses reward shaping but hides the intention profile from the
policy observation, so the reward-shaping channel is active and the observation
channel is disabled.

The script reports final performance and learning dynamics over six seeds
using the same loading, summary and bootstrap functions as the confirmatory
analysis.

Usage:
  python exploratory/shaping_only_analysis.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import confirmatory_analysis as ca
from text_tables import fixed, print_table


COND_NAME = ca.COND_LABEL


def seed_curves(condition: str):
    """Return list of (steps, soups) per seed, and the final-window mean array."""
    curves, means = [], []
    for seed in range(6):
        run_dir = ca.find_run_dir(condition, seed)
        if run_dir is None:
            raise FileNotFoundError(f"missing {condition} seed {seed}")
        steps, soups = ca.load_series(run_dir)
        curves.append((steps, soups))
        means.append(ca.seed_summary(soups))
    return curves, np.array(means)


def auc_and_reach(curves):
    aucs, reach = [], []
    for steps, soups in curves:
        aucs.append(float(np.mean(soups / ca.MAX_SOUPS_NORM)))
        reach.append(ca.steps_to_threshold(soups, steps))
    return np.array(aucs), np.array(reach)


def ci(arr):
    lo, hi = ca.bca_ci_mean(arr)
    return lo, hi


def main():
    conditions = ["ppo_baseline", "argrl_obs", "argrl_full", "argrl_shape"]
    curves = {c: seed_curves(c) for c in conditions}
    # argrl_frozen for the performance table only, not the dynamics.
    _, frozen_means = seed_curves("argrl_frozen")

    final_means = {c: curves[c][1] for c in conditions}
    final_means["argrl_frozen"] = frozen_means
    argrl_shape = final_means["argrl_shape"]

    print("ArgRL-Shape condition")
    print("ArgRL-Shape uses reward shaping but hides the intention profile "
          "from the policy observation, so the reward-shaping channel is "
          "active and the observation channel is disabled.")
    print(f"Cramped Room, 1M steps, six seeds. Performance is the mean over "
          f"the final {ca.FINAL_WINDOW} logged episode outcomes, with the seed "
          "as the unit of analysis.")
    print()

    print("Final performance")
    print_table(
        [
            {
                "Condition": COND_NAME[c],
                "Mean": final_means[c].mean(),
                "SD": final_means[c].std(ddof=1),
            }
            for c in ["ppo_baseline", "argrl_frozen", "argrl_obs",
                      "argrl_full", "argrl_shape"]
        ],
        formatters={"Mean": fixed(), "SD": fixed()},
    )

    print("ArgRL-Shape comparisons")
    print("BCa intervals summarise the seed-level mean differences.")
    print_table(
        [
            {
                "Reference": COND_NAME[c],
                "Difference": float(argrl_shape.mean() - final_means[c].mean()),
                "BCa 95% CI": "[{:.3f}, {:.3f}]".format(
                    *ca.bca_ci(final_means[c], argrl_shape)
                ),
            }
            for c in ["ppo_baseline", "argrl_obs", "argrl_full"]
        ],
        formatters={"Difference": fixed(signed=True)},
    )

    print("Learning dynamics")
    print("Normalised area under the training curve, higher is better, and the "
          "step at which the smoothed curve first reaches 0.8 of that seed's "
          "final mean, lower is faster. Bracketed values are BCa 95% "
          "intervals over six seeds.")
    rows = []
    for c in conditions:
        aucs, reach = auc_and_reach(curves[c][0])
        auc_lo, auc_hi = ca.bca_ci_mean(aucs)
        reached = reach[~np.isnan(reach)]
        if len(reached) >= 2:
            lo, hi = ca.bca_ci_mean(reached)
            steps = f"{reached.mean():,.0f} [{lo:,.0f}, {hi:,.0f}]"
            # Only worth saying when a seed never got there.
            if len(reached) < len(reach):
                steps += f" ({len(reached)}/{len(reach)} seeds)"
        else:
            steps = "not reached"
        rows.append({
            "Condition": COND_NAME[c],
            "Normalised AUC": f"{aucs.mean():.4f} [{auc_lo:.4f}, {auc_hi:.4f}]",
            "Steps to 80%": steps,
        })
    print_table(rows)


if __name__ == "__main__":
    main()
