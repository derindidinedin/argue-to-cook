#!/usr/bin/env python3
"""
Check that the reward stream is deterministic for the four confirmatory
conditions.

For PPO baseline, ArgRL-Full, ArgRL-Obs and ArgRL-Frozen, the probe builds
the same environment used by analysis/train.py, applies a fixed seeded
joint-action sequence and records the per-step reward received by PPO.

Each condition is run twice. The reward streams must match byte for byte. The
probe reports a SHA-256 digest for each condition and one argrl_full digest.

Because the actions are fixed, any mismatch comes from the environment,
argumentation or reward-shaping code rather than policy sampling. ArgRL-Full and
ArgRL-Obs use the default scoring weights because no policy is run.

The process exits with status 0 when all repeated runs match, and 1 otherwise.
"""

import hashlib
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from af.env_wrapper import OvercookedGymEnv, OvercookedAFEnv  # noqa: E402
from af.policy import WeightStore  # noqa: E402
from confirmatory_analysis import COND_LABEL  # noqa: E402
from text_tables import format_table  # noqa: E402

LAYOUT = "cramped_room"
HORIZON = 400
STEPS = 1200          # environment steps per run (spans several 400-step episodes)
ACTION_SEED = 12345   # fixed joint-action sequence
RESET_SEED = 777      # base episode reset seed
CONDITIONS = ["ppo_baseline", "argrl_full", "argrl_obs", "argrl_frozen"]



def build_env(condition):
    """Mirror the env construction in analysis/train.py for each condition."""
    if condition == "ppo_baseline":
        return OvercookedGymEnv(layout_name=LAYOUT, horizon=HORIZON)
    if condition == "argrl_frozen":
        return OvercookedAFEnv(
            layout_name=LAYOUT, horizon=HORIZON,
            strategy="aggressive",
            weights=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            weight_store=None,
        )
    if condition == "argrl_full":
        return OvercookedAFEnv(
            layout_name=LAYOUT, horizon=HORIZON,
            strategy="aggressive",
            weight_store=WeightStore(),
            ablate=None,
            include_profile_in_obs=True,
        )
    if condition == "argrl_obs":
        return OvercookedAFEnv(
            layout_name=LAYOUT, horizon=HORIZON,
            strategy="aggressive",
            weight_store=WeightStore(),
            ablate={"individual", "coalition", "load_ready"},
            include_profile_in_obs=True,
        )
    raise ValueError(f"unknown condition: {condition!r}")


def run_once(condition):
    """Drive one condition with the fixed action sequence; return the reward stream."""
    env = build_env(condition)
    action_rng = np.random.default_rng(ACTION_SEED)
    rewards = []
    episode = 0
    env.reset(seed=RESET_SEED)
    for _ in range(STEPS):
        action = action_rng.integers(0, 6, size=2)
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            episode += 1
            env.reset(seed=RESET_SEED + episode)
    return rewards


def hash_stream(rewards):
    """Stable SHA-256 over the reward stream packed as little-endian float64."""
    buf = struct.pack("<%dd" % len(rewards), *rewards)
    return hashlib.sha256(buf).hexdigest()


def git_revision() -> str:
    """Short git branch and commit if this is a checkout, otherwise "unknown"."""
    try:
        branch, commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(PROJECT_ROOT)
            ).decode().strip(),
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT)
            ).decode().strip(),
        )
        return f"{branch} {commit}"
    except Exception:
        return "unknown"


def main() -> int:
    # The environment build prints progress of its own, so the report is
    # assembled as text rather than captured from stdout.
    rows, digests, all_ok = [], [], True
    for cond in CONDITIONS:
        stream_a = run_once(cond)
        stream_b = run_once(cond)
        reproducible = stream_a == stream_b
        digest = hash_stream(stream_a)
        digests.append(digest)
        all_ok = all_ok and reproducible
        rows.append({
            "Condition": COND_LABEL[cond],
            "Steps": len(stream_a),
            "Identical across runs": "yes" if reproducible else "NO",
            "Reward stream SHA-256": digest,
        })

    argrl_full = hashlib.sha256("".join(digests).encode()).hexdigest()

    report = "\n".join([
        "Determinism probe",
        f"Cramped Room, horizon {HORIZON}, {STEPS} steps per run, two runs "
        "per condition.",
        f"Fixed action seed {ACTION_SEED} and reset seed {RESET_SEED}, so a "
        "mismatch would come from the environment, the framework or the "
        "shaping rather than from policy sampling.",
        f"Code under test: {git_revision()}.",
        "",
        "Reward streams",
        format_table(rows),
        "",
        f"Combined SHA-256 over the four digests: {argrl_full}",
        "PASS, every condition was byte-identical across its two runs"
        if all_ok else
        "FAIL, at least one condition was not reproducible",
        "",
    ])

    print(report, end="")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
