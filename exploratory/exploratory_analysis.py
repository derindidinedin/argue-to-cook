"""
Exploratory analyses based on cached training runs.

Reports changes in the scoring weights, alternative profile-selection
strategies and potential-component ablations, performance on Coordination Ring,
the shaping strength comparison, and convergence during training.

They use the loading, summary, smoothing and bootstrap functions from
analysis/confirmatory_analysis.py.

Run from the repository root:

    python exploratory/exploratory_analysis.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))
import confirmatory_analysis as ca  # noqa: E402
from text_tables import fixed, print_table  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
SEEDS = range(6)

# Each entry lists directory prefixes in preference order: first the name the
# current analysis/train.py writes, then the name the cached run carries. The
# single-seed and shaping-strength runs were labelled before train.py gained the strategy
# and scale tags, so a retrained arm and the cached one sit under different
# prefixes. Matching both keeps the documented commands working without
# renaming any cached directory.
SINGLE_SEED_SUITE = [
    ("ArgRL-Full, aggressive", [
        "cramped_room_argrl_full_aggressive_seed42_",
        "cramped_room_argrl_full_aggressive_single_"]),
    ("ArgRL-Full, conservative", [
        "cramped_room_argrl_full_conservative_seed42_",
        "cramped_room_argrl_full_conservative_single_"]),
    ("Without coalition commitment", [
        "cramped_room_argrl_full_aggressive_ablate_coalition_seed42_",
        "cramped_room_argrl_full_ablate_coalition_single_"]),
    ("Without load-ready bonus", [
        "cramped_room_argrl_full_aggressive_ablate_load_ready_seed42_",
        "cramped_room_argrl_full_ablate_load_ready_single_"]),
    ("Without individual purpose", [
        "cramped_room_argrl_full_aggressive_ablate_individual_seed42_",
        "cramped_room_argrl_full_ablate_individual_single_"]),
]

RING_CONDITIONS = [
    ("PPO baseline",  ["coordination_ring_ppo_baseline_seed"]),
    ("ArgRL-Full", ["coordination_ring_argrl_full_aggressive_seed"]),
]

SHAPING_STRENGTH_ARMS = [
    ("Shaping 5x", [
        "cramped_room_argrl_full_aggressive_shaping5x_seed",
        "cramped_room_shaping5x_seed"]),
    ("Shaping 25x", [
        "cramped_room_argrl_full_aggressive_shaping25x_seed",
        "cramped_room_shaping25x_seed"]),
]

CONVERGENCE_CONDITIONS = [
    ("PPO baseline",             "ppo_baseline"),
    ("ArgRL-Full",            "argrl_full"),
    ("ArgRL-Obs", "argrl_obs"),
    ("ArgRL-Shape",        "argrl_shape"),
    ("ArgRL-Frozen",       "argrl_frozen"),
]


def best_run(prefixes: list[str]) -> pathlib.Path:
    """Run directory for the first of `prefixes` that matches anything.

    Prefixes are tried in order, so a retrained run under the current label
    wins over the cached one. Among several matches on the same prefix the one
    with the most logged points wins, matching the confirmatory loader.
    """
    for prefix in prefixes:
        matches = sorted(RESULTS.glob(prefix + "*"))
        if matches:
            return max(matches, key=lambda d: len(ca.load_soups(d)))
    raise RuntimeError(f"no run matching any of {prefixes}")


def seed_means(prefixes: list[str], seeds=SEEDS) -> np.ndarray:
    """Final-window mean per seed, for runs named `<prefix><seed>_<timestamp>`."""
    out = []
    for seed in seeds:
        run_dir = best_run([f"{p}{seed}_" for p in prefixes])
        out.append(ca.seed_summary(ca.load_soups(run_dir)))
    return np.array(out)


def confirmatory_seed_means(condition: str) -> np.ndarray:
    """Final-window mean per seed for a condition the confirmatory map covers."""
    out = []
    for seed in SEEDS:
        run_dir = ca.find_run_dir(condition, seed)
        if run_dir is None:
            raise RuntimeError(f"no cached run for {condition} seed {seed}")
        out.append(ca.seed_summary(ca.load_soups(run_dir)))
    return np.array(out)


def weight_behaviour() -> None:
    """
    Starting and final values of the three weights across the ArgRL-Full seeds.

    Also reports how many seeds end below where they started, and the largest
    rise between consecutive logged points. The weights receive no task-reward
    gradient, so the decline is L2 shrinkage rather than learning, and the
    per-point series is noisy rather than strictly decreasing.
    """
    labels = {"train/w1": "w1, proximity",
              "train/w2": "w2, downstream subtasks",
              "train/w3": "w3, congestion"}
    rows = []
    for tag, label in labels.items():
        starts, ends, declined, rises = [], [], 0, []
        for seed in SEEDS:
            run_dir = ca.find_run_dir("argrl_full", seed)
            _, values = ca.load_series(run_dir, tag)
            starts.append(values[0])
            ends.append(values[-1])
            declined += values[-1] < values[0]
            rises.append(float(np.diff(values).max()))
        rows.append({
            "Weight": label,
            "Mean start": float(np.mean(starts)),
            "Mean end": float(np.mean(ends)),
            "Seeds down": f"{declined} of {len(SEEDS)}",
            "Largest rise": max(rises),
        })

    print("Scoring weight behaviour, six ArgRL-Full seeds")
    print_table(rows, formatters={"Mean start": fixed(4),
                                  "Mean end": fixed(4),
                                  "Largest rise": fixed(4)})


def profile_and_ablations() -> None:
    print("Profile selection and potential ablations, one seed each")
    print_table(
        [
            {
                "Configuration": label,
                "Soups per episode": ca.seed_summary(
                    ca.load_soups(best_run(prefixes))
                ),
            }
            for label, prefixes in SINGLE_SEED_SUITE
        ],
        formatters={"Soups per episode": fixed()},
    )


def coordination_ring() -> None:
    print("Coordination Ring, per seed")
    rows = []
    for label, prefixes in RING_CONDITIONS:
        values = seed_means(prefixes, seeds=range(3))
        row = {"Condition": label}
        row.update({f"Seed {i}": v for i, v in enumerate(values)})
        rows.append(row)
    print_table(rows, formatters={f"Seed {i}": fixed() for i in range(3)})


def shaping_strength() -> None:
    ppo_baseline = confirmatory_seed_means("ppo_baseline")
    argrl_full = confirmatory_seed_means("argrl_full")

    performance = [
        {"Condition": "PPO baseline", "Mean": ppo_baseline.mean(),
         "SD": ppo_baseline.std(ddof=1)},
        {"Condition": "ArgRL-Full", "Mean": argrl_full.mean(),
         "SD": argrl_full.std(ddof=1)},
    ]
    comparisons = []

    for label, prefixes in SHAPING_STRENGTH_ARMS:
        arm = seed_means(prefixes)
        performance.append({"Condition": label, "Mean": arm.mean(),
                            "SD": arm.std(ddof=1)})
        for reference, base in (("PPO baseline", ppo_baseline), ("ArgRL-Full", argrl_full)):
            lo, hi = ca.bca_ci(base, arm)
            comparisons.append({
                "Condition": label,
                "Reference": reference,
                "Difference": float(arm.mean() - base.mean()),
                "BCa 95% CI": f"[{lo:.3f}, {hi:.3f}]",
            })

    print("Shaping strength, six seeds per condition")
    print_table(performance, formatters={"Mean": fixed(), "SD": fixed()})

    print("Mean differences")
    print_table(comparisons, formatters={"Difference": fixed(signed=True)})


def convergence() -> None:
    rows = []
    for label, condition in CONVERGENCE_CONDITIONS:
        aucs, reach = [], []
        for seed in SEEDS:
            run_dir = ca.find_run_dir(condition, seed)
            if run_dir is None:
                continue
            steps, soups = ca.load_series(run_dir)
            aucs.append(float(np.mean(soups / ca.MAX_SOUPS_NORM)))
            reach.append(ca.steps_to_threshold(soups, steps))
        aucs = np.array(aucs)
        reach = np.array(reach)
        if np.all(np.isnan(reach)):
            steps_cell = "never reaches"
        else:
            steps_cell = (f"{np.nanmean(reach) / 1e6:.3f} +/- "
                          f"{np.nanstd(reach, ddof=1) / 1e6:.3f}")
        rows.append({
            "Condition": label,
            "Normalised AUC": f"{aucs.mean():.3f} +/- {aucs.std(ddof=1):.3f}",
            "Steps to 80% (1e6)": steps_cell,
        })

    print("Convergence, six seeds per condition")
    print_table(rows)


def main() -> None:
    weight_behaviour()
    profile_and_ablations()
    coordination_ring()
    shaping_strength()
    convergence()


if __name__ == "__main__":
    main()
