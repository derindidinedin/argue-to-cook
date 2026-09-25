"""
Run the confirmatory analysis over the cached thesis runs.

The analysis summarises six seeds for each of four conditions using the mean
of the final 50 logged episode outcomes. It reports the four pairwise
comparisons, corrected Mann-Whitney U results, Welch tests, effect sizes and
BCa bootstrap intervals.

Usage:
    python analysis/confirmatory_analysis.py
    python analysis/confirmatory_analysis.py --dry-run
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from text_tables import fixed, print_table

LOG_ROOT = pathlib.Path(__file__).resolve().parent.parent / "results"

FINAL_WINDOW = 50   # logged episode outcomes in the seed summary statistic
N_BOOT = 10_000
BOOT_SEED = 42
ALPHA = 0.05

# ---------------------------------------------------------------------------
# Conditions and seeds
# ---------------------------------------------------------------------------

# Condition identifier -> the name used in the thesis and in every figure
# legend. The exploratory scripts import this so all three report the same
# names. See the conditions table in README.md.
COND_LABEL = {
    "ppo_baseline":             "PPO baseline",
    "argrl_frozen":       "ArgRL-Frozen",
    "argrl_full":            "ArgRL-Full",
    "argrl_obs": "ArgRL-Obs",
    "argrl_shape":        "ArgRL-Shape (exploratory)",
    "bandit":              "Intention bias (exploratory)",
}

# Directories use the layout-prefixed label that analysis/train.py writes:
#   cramped_room_{condition}[_{strategy}]_seed{seed}_{timestamp}
# The single-seed and shaping-strength runs behind the exploratory results were labelled
# under an older scheme and are matched in exploratory/exploratory_analysis.py.

# 4 confirmatory conditions, 6 seeds each
CONFIRMATORY_CONDITIONS = {
    "ppo_baseline": {
        "pattern": lambda seed: [f"cramped_room_ppo_baseline_seed{seed}_"],
    },
    "argrl_frozen": {
        "pattern": lambda seed: [f"cramped_room_argrl_frozen_seed{seed}_"],
    },
    "argrl_full": {
        "pattern": lambda seed: [f"cramped_room_argrl_full_aggressive_seed{seed}_"],
    },
    "argrl_obs": {
        "pattern": lambda seed: [f"cramped_room_argrl_obs_seed{seed}_"],
    },
}

CONFIRMATORY_SEEDS = list(range(6))   # 0-5

# 4 pre-specified pairwise tests (Holm-Bonferroni applied across all 4)
PAIRWISE_TESTS = [
    ("T1", "argrl_full",        "ppo_baseline",     "ArgRL-Full vs PPO baseline"),
    ("T2", "argrl_obs", "ppo_baseline",     "ArgRL-Obs vs PPO baseline"),
    ("T3", "argrl_full",        "argrl_obs", "ArgRL-Full vs ArgRL-Obs"),
    ("T4", "argrl_frozen",  "ppo_baseline",     "ArgRL-Frozen vs PPO baseline"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_run_dir(condition: str, seed: int) -> Optional[pathlib.Path]:
    """
    Find the best log directory for a given (condition, seed) pair.

    Only directories with the layout-prefixed label format are considered
    (e.g. cramped_room_ppo_baseline_seed0_<timestamp>).  Old-format directories
    from pre-batch experiments are intentionally excluded.

    "Best" among multiple matches = most soups_delivered scalars recorded.
    Returns None if no matching directory is found.
    """
    patterns = CONFIRMATORY_CONDITIONS[condition]["pattern"](seed)
    candidates = []
    for d in LOG_ROOT.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        for pat in patterns:
            if name.startswith(pat):
                candidates.append(d)
                break

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple candidates: pick the one with the most soups_delivered scalars
    best = None
    best_n = -1
    for cand in candidates:
        n = _count_scalars(cand, "train/soups_delivered")
        if n > best_n:
            best_n = n
            best = cand
    return best


def _count_scalars(run_dir: pathlib.Path, tag: str) -> int:
    """Return the number of scalar events for `tag` in a log directory."""
    try:
        subdirs = sorted(x for x in run_dir.iterdir() if x.is_dir())
        src = str(subdirs[0] if subdirs else run_dir)
        ea = EventAccumulator(src, size_guidance={"scalars": 0})
        ea.Reload()
        events = ea.Scalars(tag) if tag in ea.Tags().get("scalars", []) else []
        return len(events)
    except Exception:
        return 0


def load_soups(run_dir: pathlib.Path) -> np.ndarray:
    """Load all soups_delivered scalar values from a run directory."""
    subdirs = sorted(x for x in run_dir.iterdir() if x.is_dir())
    src = str(subdirs[0] if subdirs else run_dir)
    ea = EventAccumulator(src, size_guidance={"scalars": 0})
    ea.Reload()
    tag = "train/soups_delivered"
    if tag not in ea.Tags().get("scalars", []):
        raise RuntimeError(f"Tag '{tag}' not found in {run_dir}")
    return np.array([e.value for e in ea.Scalars(tag)])


def seed_summary(soups: np.ndarray, final_window: int = FINAL_WINDOW) -> float:
    """Mean soups over the final `final_window` logged episode outcomes."""
    return float(soups[-final_window:].mean())


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from MWU statistic."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """(mean_b - mean_a) / pooled std  (positive = b > a)."""
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled_std)


def bca_ci(
    a: np.ndarray, b: np.ndarray,
    n_resamples: int = N_BOOT, seed: int = BOOT_SEED,
) -> tuple[float, float]:
    """
    BCa bootstrap 95% CI on (mean_b - mean_a) using scipy.stats.bootstrap.

    Each input is resampled independently (paired=False by default).
    """
    def statistic(x, y, axis):
        return np.mean(y, axis=axis) - np.mean(x, axis=axis)

    result = scipy_bootstrap(
        (a, b),
        statistic=statistic,
        n_resamples=n_resamples,
        confidence_level=0.95,
        method="BCa",
        random_state=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def holm_bonferroni(raw_p: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down correction.

    Returns adjusted p-values in the original order.
    """
    k = len(raw_p)
    order = np.argsort(raw_p)
    adjusted = np.array(raw_p, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = raw_p[idx] * (k - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Shared helpers for the exploratory learning-dynamics analysis
#
# exploratory/shaping_only_analysis.py imports these so its numbers are computed
# with exactly the same loading, smoothing and bootstrap settings as everything
# else here. They are analysis helpers, so they live with the analysis code.
# ---------------------------------------------------------------------------

# Learning-dynamics constants.
MAX_SOUPS_NORM = 12.0    # AUC normaliser: the run captures this fraction of it
SMOOTH_WINDOW = 5        # rolling-mean window for the steps-to-threshold curve
THRESHOLD_FRACTION = 0.8  # a run has "reached" at this fraction of its final level
MEANINGFUL_FLOOR = 1.0   # below this final level, a relative threshold means nothing

# Directory name prefix per condition. This map also covers the exploratory
# argrl_shape runs, which are deliberately absent from CONFIRMATORY_CONDITIONS.
_RUN_PREFIX = {
    "ppo_baseline":             lambda seed: f"cramped_room_ppo_baseline_seed{seed}_",
    "argrl_frozen":       lambda seed: f"cramped_room_argrl_frozen_seed{seed}_",
    "argrl_full":            lambda seed: f"cramped_room_argrl_full_aggressive_seed{seed}_",
    "argrl_obs": lambda seed: f"cramped_room_argrl_obs_seed{seed}_",
    "argrl_shape":        lambda seed: f"cramped_room_argrl_shape_seed{seed}_",
}


def find_run_dir(condition: str, seed: int) -> Optional[pathlib.Path]:
    """Run directory for (condition, seed), matched by name prefix.

    Covers the confirmatory conditions and the exploratory argrl_shape runs.
    Among multiple matches, keeps the one with the most soups scalars.
    """
    prefix = _RUN_PREFIX[condition](seed)
    candidates = [d for d in LOG_ROOT.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda d: _count_scalars(d, "train/soups_delivered"))


def load_series(run_dir: pathlib.Path, tag: str = "train/soups_delivered"):
    """Return (steps, values) arrays for a scalar tag in a run directory."""
    subdirs = sorted(x for x in run_dir.iterdir() if x.is_dir())
    src = str(subdirs[0] if subdirs else run_dir)
    ea = EventAccumulator(src, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise RuntimeError(f"Tag '{tag}' not found in {run_dir}")
    events = ea.Scalars(tag)
    return (np.array([e.step for e in events]),
            np.array([e.value for e in events]))


def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    """Uniform rolling mean; output length = len(arr) - w + 1."""
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def first_reach(soups: np.ndarray, steps: np.ndarray,
                threshold: float, smooth_w: int) -> float:
    """First timestep at which the smoothed curve reaches `threshold`, else nan."""
    if len(soups) < smooth_w:
        return float("nan")
    smoothed = _smooth(soups, smooth_w)
    step_ends = steps[smooth_w - 1:]
    reached = np.where(smoothed >= threshold)[0]
    return float(step_ends[reached[0]]) if len(reached) else float("nan")


def steps_to_threshold(soups: np.ndarray, steps: np.ndarray,
                       smooth_w: int = SMOOTH_WINDOW,
                       fraction: float = THRESHOLD_FRACTION,
                       final_window: int = FINAL_WINDOW) -> float:
    """
    Training step at which a run first reaches `fraction` of its own final level.

    The threshold is per seed, tau = fraction * mean of the final
    `final_window` logged episode outcomes, rather than an absolute soup count,
    so a run is measured against where it ended up. A run finishing below
    MEANINGFUL_FLOOR returns nan, since a threshold that low says nothing about
    convergence.
    """
    final_mean = float(soups[-final_window:].mean())
    if final_mean < MEANINGFUL_FLOOR:
        # A run that delivers essentially nothing has a threshold near zero,
        # which its opening points already meet. ArgRL-Frozen is the only
        # condition this catches, and its convergence is not meaningful.
        return float("nan")
    return first_reach(soups, steps, fraction * final_mean, smooth_w)


def bca_ci_mean(arr: np.ndarray) -> tuple[float, float]:
    """BCa bootstrap 95% CI on the mean of one sample; (nan, nan) if n < 2."""
    import warnings

    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = scipy_bootstrap(
            (arr,),
            statistic=lambda x, axis: np.mean(x, axis=axis),
            n_resamples=N_BOOT,
            confidence_level=0.95,
            method="BCa",
            random_state=BOOT_SEED,
        )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Confirmatory statistical analysis")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print data availability only; do not run statistical tests.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Also show the run directory and logged-point count per seed",
    )
    args = parser.parse_args()

    print("Confirmatory analysis")
    print("Cramped Room, 1M steps, six seeds per condition, four parallel "
          "environments.")
    print(f"Performance is the mean over the final {FINAL_WINDOW} logged "
          "episode outcomes, with the seed as the unit of analysis.")
    print()

    # -- load data ----------------------------------------------------------

    data: dict[str, list[float]] = {c: [] for c in CONFIRMATORY_CONDITIONS}
    missing: dict[str, list[int]] = {c: [] for c in CONFIRMATORY_CONDITIONS}
    rows = []

    for cond in CONFIRMATORY_CONDITIONS:
        for seed in CONFIRMATORY_SEEDS:
            label = COND_LABEL[cond]
            run_dir = _find_run_dir(cond, seed)
            if run_dir is None:
                rows.append({"Condition": label, "Seed": seed,
                             "Final mean": "missing", "Run": "-"})
                missing[cond].append(seed)
                continue
            try:
                soups = load_soups(run_dir)
                summary = seed_summary(soups)
                data[cond].append(summary)
                row = {"Condition": label, "Seed": seed, "Final mean": summary}
                if args.verbose:
                    row["Logged points"] = len(soups)
                    row["Run"] = run_dir.name
                rows.append(row)
            except Exception as exc:
                rows.append({"Condition": label, "Seed": seed,
                             "Final mean": f"error: {exc}", "Run": run_dir.name})
                missing[cond].append(seed)

    print("Runs")
    print_table(rows, formatters={"Final mean": fixed()})

    n_missing = sum(len(v) for v in missing.values())
    if n_missing > 0:
        print(f"{n_missing} runs missing. Complete the batch before drawing "
              "conclusions.")
        print("Missing: " + ", ".join(
            f"{COND_LABEL[c]} seeds {v}" for c, v in missing.items() if v
        ))
        print()

    if args.dry_run:
        print("Dry run, so the statistical tests were skipped.")
        return

    insufficient = [COND_LABEL[c] for c, v in data.items() if len(v) < 2]
    if insufficient:
        print(f"Fewer than two seeds for {', '.join(insufficient)}, so the "
              "tests cannot run.")
        return

    # -- descriptive statistics ----------------------------------------------

    arrays = {c: np.array(data[c]) for c in CONFIRMATORY_CONDITIONS}

    print("Final performance")
    print_table(
        [
            {
                "Condition": COND_LABEL[cond],
                "Seeds": len(arr),
                "Mean": arr.mean(),
                "SD": arr.std(ddof=1) if len(arr) > 1 else 0.0,
            }
            for cond, arr in arrays.items() if len(arr)
        ],
        formatters={"Mean": fixed(), "SD": fixed()},
    )

    # -- pairwise tests -----------------------------------------------------

    results = []
    for tid, cond_b, cond_a, desc in PAIRWISE_TESTS:
        a = arrays.get(cond_a, np.array([]))
        b = arrays.get(cond_b, np.array([]))
        if len(a) < 2 or len(b) < 2:
            results.append({"id": tid, "desc": desc, "cond_a": cond_a,
                            "cond_b": cond_b, "skip": True})
            continue

        u_stat, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
        try:
            ci_lo, ci_hi = bca_ci(a, b)
        except Exception:
            ci_lo, ci_hi = float("nan"), float("nan")

        results.append({
            "id": tid, "desc": desc, "cond_a": cond_a, "cond_b": cond_b,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(b.mean() - a.mean()),
            "u_stat": u_stat, "p_mw": p_mw,
            "r_rb": rank_biserial_r(u_stat, len(a), len(b)),
            "t_stat": t_stat, "p_t": p_t, "d": cohens_d(a, b),
            "ci_lo": ci_lo, "ci_hi": ci_hi, "skip": False,
        })

    valid_idx = [i for i, r in enumerate(results) if not r.get("skip")]
    adj_p = holm_bonferroni([results[i]["p_mw"] for i in valid_idx])
    for i, vidx in enumerate(valid_idx):
        results[vidx]["p_mw_adj"] = adj_p[i]

    def comparison(r):
        return f"{COND_LABEL[r['cond_b']]} vs {COND_LABEL[r['cond_a']]}"

    print("Pairwise tests, Mann-Whitney U, two-sided")
    print("Holm-Bonferroni corrected across the four comparisons.")
    print_table(
        [
            {
                "Test": r["id"],
                "Comparison": comparison(r),
                "Difference": "-" if r["skip"] else r["diff"],
                "U": "-" if r["skip"] else f"{r['u_stat']:.0f}",
                "p": "-" if r["skip"] else f"{r['p_mw']:.4e}",
                "p (Holm)": "-" if r["skip"] else f"{r['p_mw_adj']:.4e}",
                "r": "-" if r["skip"] else f"{r['r_rb']:+.3f}",
                "BCa 95% CI": ("-" if r["skip"]
                               else f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"),
            }
            for r in results
        ],
        formatters={"Difference": fixed(signed=True)},
    )

    print("Secondary tests, Welch's t, two-sided")
    print_table(
        [
            {
                "Test": r["id"],
                "Comparison": comparison(r),
                "t": "-" if r["skip"] else f"{r['t_stat']:.3f}",
                "p": "-" if r["skip"] else f"{r['p_t']:.4e}",
                "Cohen's d": "-" if r["skip"] else f"{r['d']:+.3f}",
                "Magnitude": "-" if r["skip"] else _magnitude(r["d"]),
            }
            for r in results
        ],
    )

    # -- conclusions --------------------------------------------------------

    print("Conclusions")
    print(f"A comparison counts as positive when the Holm-corrected p is below "
          f"{ALPHA} and the BCa interval excludes zero.")
    print_table([
        {
            "Test": r["id"],
            "Comparison": r["desc"],
            "Verdict": _verdict(r),
        }
        for r in results
    ])


def _magnitude(d: float) -> str:
    """Conventional label for a Cohen's d value."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _verdict(r: dict) -> str:
    """One-phrase outcome for a pairwise comparison."""
    if r.get("skip"):
        return "insufficient data"
    ci_excludes_zero = r["ci_lo"] > 0 or r["ci_hi"] < 0
    if r.get("p_mw_adj", 1.0) < ALPHA and ci_excludes_zero:
        return "significant"
    if r["p_mw"] < ALPHA:
        return "nominal only, does not survive correction"
    return "not significant"


if __name__ == "__main__":
    main()
