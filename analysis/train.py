"""
Train PPO under any supported experimental condition.

Examples:
    python analysis/train.py --condition ppo_baseline --n-envs 4
    python analysis/train.py --condition argrl_full --n-envs 4
    python analysis/train.py --condition argrl_full --strategy conservative
    python analysis/train.py --condition argrl_full --ablate coalition

Conditions:
    ppo_baseline
        PPO with Overcooked's sparse and built-in dense rewards.

    argrl_frozen
        The argumentation framework and reward shaping with the policy fixed at
        its initial parameters.

    argrl_full
        The argumentation framework supplies the intention profile through the
        observation and reward-shaping channels.

    argrl_obs
        The profile remains in the observation, but reward shaping is disabled.

    argrl_shape
        Reward shaping remains active, but the profile is hidden from the
        policy observation.

    bandit
        The argumentation framework supplies the intention profile through both
        channels, using fixed base weights and outcome-driven per-intention
        biases.

The strategy, ablation and shaping-scale options apply only to ArgRL-Full.
Training logs and checkpoints are written under logs/ and checkpoints/.

The main performance metric is soups delivered per episode. Training reward is
logged separately.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Ensure the project root is on sys.path regardless of where the script is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from af.env_wrapper import OvercookedGymEnv, OvercookedAFEnv
from af.policy import ArgumentWeightPolicy, WeightStore
from af.bandit import BanditWeights


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class SoupsDeliveredCallback(BaseCallback):
    """
    Logs per-episode statistics to TensorBoard.

    Reads from the step info dict populated by OvercookedGymEnv:
        soups_delivered   - total soups in the episode (primary metric)
        ep_sparse_reward  - cumulative sparse reward (= soups * 20 for ppo_baseline)
        ep_shaped_reward  - cumulative shaped reward (0 for ppo_baseline)
        ep_length         - episode length in steps

    Also logs a rolling mean over the last `window` episodes for
    soups_delivered so the TensorBoard chart is less noisy.

    When a WeightStore is supplied, logs train/w1, train/w2, train/w3 every
    `weight_log_freq` timesteps.
    """

    def __init__(
        self,
        window: int = 10,
        weight_store: WeightStore | None = None,
        weight_log_freq: int = 5000,
        bandit: BanditWeights | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._window = window
        self._weight_store = weight_store
        self._weight_log_freq = weight_log_freq
        self._bandit = bandit
        self._recent_soups: list[float] = []
        self._ep_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue
            self._ep_count += 1

            soups = info.get("soups_delivered", 0)
            self._recent_soups.append(soups)
            if len(self._recent_soups) > self._window:
                self._recent_soups.pop(0)

            self.logger.record("train/soups_delivered", soups)
            self.logger.record(
                f"train/soups_delivered_mean{self._window}",
                float(np.mean(self._recent_soups)),
            )
            self.logger.record(
                "train/ep_sparse_reward", info.get("ep_sparse_reward", 0.0)
            )
            self.logger.record(
                "train/ep_shaped_reward", info.get("ep_shaped_reward", 0.0)
            )
            self.logger.record("train/ep_length", info.get("ep_length", 0))
            self.logger.record("train/episode_count", self._ep_count)

            # AF metrics, present only in conditions using the argumentation framework.
            if "af_trigger_count" in info:
                triggers = info["af_trigger_count"]
                completions = info.get("intention_completion_count", 0)
                self.logger.record("train/af_trigger_count", triggers)
                self.logger.record("train/intention_completion_count", completions)
                if triggers > 0:
                    self.logger.record(
                        "train/intention_completion_rate", completions / triggers
                    )

        # Log the current argument weights every weight_log_freq timesteps
        if (
            self._weight_store is not None
            and self.num_timesteps % self._weight_log_freq == 0
            and self.num_timesteps > 0
        ):
            w = self._weight_store.weights
            self.logger.record("train/w1", float(w[0]))
            self.logger.record("train/w2", float(w[1]))
            self.logger.record("train/w3", float(w[2]))

        # Log bandit theta per intention arm every weight_log_freq timesteps.
        if (
            self._bandit is not None
            and self.num_timesteps % self._weight_log_freq == 0
            and self.num_timesteps > 0
        ):
            for intention, val in self._bandit.theta_dict().items():
                self.logger.record(f"train/theta_{intention.name.lower()}", val)

        return True


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_ppo_baseline_env(layout: str, horizon: int):
    """Returns a callable that creates a fresh OvercookedGymEnv (for make_vec_env)."""
    def _init():
        return OvercookedGymEnv(layout_name=layout, horizon=horizon)
    return _init


def make_argrl_frozen_env(layout: str, horizon: int, gamma: float):
    """
    Returns a callable that creates a fresh OvercookedAFEnv for the argrl_frozen
    condition: AF active with fixed weights [1,1,1], no weight store, aggressive
    strategy. No learning happens. Policy weights are frozen after the model is
    created.
    """
    def _init():
        return OvercookedAFEnv(
            layout_name=layout,
            horizon=horizon,
            strategy="aggressive",
            weights=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            weight_store=None,
            gamma=gamma,
        )
    return _init


def make_argrl_full_env(
    layout: str,
    horizon: int,
    gamma: float,
    strategy: str = "aggressive",
    weight_store: WeightStore | None = None,
    ablate: set | None = None,
    include_profile_in_obs: bool = True,
    bandit: BanditWeights | None = None,
    potential_scale: float = 1.0,
):
    """Returns a callable that creates a fresh OvercookedAFEnv (for make_vec_env)."""
    def _init():
        return OvercookedAFEnv(
            layout_name=layout,
            horizon=horizon,
            strategy=strategy,
            weight_store=weight_store,
            ablate=ablate,
            include_profile_in_obs=include_profile_in_obs,
            bandit=bandit,
            potential_scale=potential_scale,
            gamma=gamma,
        )
    return _init


# ---------------------------------------------------------------------------
# VecEnv construction
# ---------------------------------------------------------------------------

def _make_vec_env(env_fn, n_envs: int, seed: int, condition: str):
    """
    Build a VecEnv, using SubprocVecEnv for genuine parallelism where safe.

    Constraints
    -----------
    - n_envs == 1: DummyVecEnv (subprocess overhead is not worth it).
    - condition == 'argrl_full': DummyVecEnv regardless of n_envs.
      SubprocVecEnv forks/spawns each env into a subprocess, where the
      WeightStore is a separate copy.  The policy writes to the main-process
      store; subprocesses can't see those writes, so weight sharing silently
      breaks.  DummyVecEnv runs all envs in the same process, preserving the
      shared-reference semantics that WeightStore relies on.
    - all other conditions with n_envs > 1: SubprocVecEnv, with automatic
      fallback to DummyVecEnv if initialisation raises (e.g. MotionPlanner
      C-extension conflict under fork).
    """
    if n_envs == 1:
        return make_vec_env(env_fn, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)

    if condition in ("argrl_full", "argrl_obs", "argrl_shape", "bandit"):
        print(
            f"[VecEnv] Using DummyVecEnv({n_envs}) for {condition!r} condition - "
            "SubprocVecEnv would break shared-store (WeightStore / BanditWeights) "
            "sharing between the main process and the envs."
        )
        return make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)

    # ppo_baseline / argrl_frozen: envs are independent, SubprocVecEnv is safe.
    try:
        vec_env = make_vec_env(
            env_fn, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv
        )
        print(f"[VecEnv] Using SubprocVecEnv({n_envs} workers).")
        return vec_env
    except Exception as exc:
        print(
            f"[VecEnv] SubprocVecEnv failed ({exc}); "
            f"falling back to DummyVecEnv({n_envs})."
        )
        return make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def _build_run_label(args) -> str:
    """Construct a human-readable run identifier from the active flags."""
    parts = [args.layout, args.condition]
    if args.condition == "argrl_full":
        parts.append(args.strategy)
        if args.ablate:
            parts.append("ablate")
            parts.extend(sorted(args.ablate))
        if args.scale != 1.0:
            parts.append(f"shaping{args.scale:g}x")
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def train(args):
    # Set all random seeds for reproducibility across runs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    ablate_set = set(args.ablate) if args.ablate else set()
    run_label = _build_run_label(args)
    run_id = f"{run_label}_{int(time.time())}"
    log_dir = Path(args.log_dir) / run_id
    ckpt_dir = Path(args.checkpoint_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Condition : {args.condition}")
    if args.condition == "argrl_full":
        print(f"Strategy  : {args.strategy}")
        if ablate_set:
            print(f"Ablate    : {sorted(ablate_set)}")
    elif args.condition == "argrl_obs":
        print("Strategy  : aggressive (fixed)")
        print("Ablate    : individual coalition load_ready (all, shaping=0 by construction)")
    elif args.condition == "argrl_shape":
        print("Strategy  : aggressive (fixed)")
        print("Profile   : hidden from observation (shaping active, no obs augmentation)")
    print(f"Timesteps : {args.total_timesteps:,}")
    print(f"TensorBoard: {log_dir}")
    print(f"Checkpoints: {ckpt_dir}")

    # Shared stores (created per condition below).
    weight_store: WeightStore | None = None
    bandit: BanditWeights | None = None

    if args.condition == "ppo_baseline":
        env_fn = make_ppo_baseline_env(args.layout, args.horizon)
    elif args.condition == "argrl_frozen":
        env_fn = make_argrl_frozen_env(args.layout, args.horizon, args.gamma)
    elif args.condition == "argrl_full":
        weight_store = WeightStore()
        env_fn = make_argrl_full_env(
            args.layout, args.horizon, args.gamma,
            strategy=args.strategy,
            weight_store=weight_store,
            ablate=ablate_set if ablate_set else None,
            potential_scale=args.scale,
        )
    elif args.condition == "argrl_obs":
        # AF active, profile in obs, weights network-produced (no task-reward
        # gradient, L2 shrinkage only, not learned from the task), but
        # potential shaping = 0.
        # Achieved by ablating all three PotentialFunction components, which makes
        # phi() return exactly 0.0 for any state, so shaping = gamma*0 - 0 = 0.
        weight_store = WeightStore()
        env_fn = make_argrl_full_env(
            args.layout, args.horizon, args.gamma,
            strategy="aggressive",
            weight_store=weight_store,
            ablate={"individual", "coalition", "load_ready"},
        )
    elif args.condition == "argrl_shape":
        # Exploratory. Reward shaping stays active while the intention
        # profile is hidden from the policy observation, so the reward-shaping
        # channel is active and the observation channel is disabled.
        weight_store = WeightStore()
        env_fn = make_argrl_full_env(
            args.layout, args.horizon, args.gamma,
            strategy="aggressive",
            weight_store=weight_store,
            ablate=None,
            include_profile_in_obs=False,
        )
    elif args.condition == "bandit":
        # Exploratory. The condition fixes the base scoring weights at one and
        # adds an outcome-driven bias per intention, so there is no WeightStore
        # and the policy is a standard MlpPolicy. BanditWeights updates the
        # biases once per episode from argument outcomes (af/bandit.py). The
        # profile in the observation and the shaping both stay active.
        bandit = BanditWeights(lr=args.bandit_lr)
        print(f"Intention bias: lr={args.bandit_lr}, theta in "
              f"[{bandit.theta_min}, {bandit.theta_max}]")
        env_fn = make_argrl_full_env(
            args.layout, args.horizon, args.gamma,
            strategy="aggressive",
            weight_store=None,
            ablate=None,
            include_profile_in_obs=True,
            bandit=bandit,
        )
    else:
        raise ValueError(f"Unknown condition: {args.condition!r}")

    vec_env = _make_vec_env(env_fn, args.n_envs, args.seed, args.condition)

    # argrl_frozen and ppo_baseline use standard MlpPolicy.
    # argrl_full, argrl_obs and argrl_shape use ArgumentWeightPolicy so the argument
    # scoring reads weights from the shared store during rollout collection.
    if args.condition in ("argrl_full", "argrl_obs", "argrl_shape"):
        policy_cls = ArgumentWeightPolicy
        policy_kwargs = {"weight_store": weight_store}
    else:
        policy_cls = "MlpPolicy"
        policy_kwargs = {}

    model = PPO(
        policy=policy_cls,
        policy_kwargs=policy_kwargs,
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
    )

    if args.condition == "argrl_frozen":
        # Freeze all policy parameters. The argrl_frozen condition never learns.
        for param in model.policy.parameters():
            param.requires_grad_(False)
        # Replace the PPO train() method with a no-op so rollouts are collected
        # and logged (via the callback) but no gradient update loop runs.
        # n_epochs=0 is insufficient: SB3 references approx_kl_divs after the
        # loop body, causing UnboundLocalError when the loop never executes.
        # Replacing train() entirely avoids that and prevents backward() being
        # called on frozen params (which would raise RuntimeError).
        model.train = lambda: None

    callbacks = [
        SoupsDeliveredCallback(window=10, weight_store=weight_store, bandit=bandit),
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = ckpt_dir / "ppo_final"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}.zip")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Overcooked")

    # Core flags
    p.add_argument(
        "--condition",
        choices=["ppo_baseline", "argrl_full", "argrl_frozen", "argrl_obs", "argrl_shape", "bandit"],
        default="ppo_baseline",
        help="Experimental condition to run",
    )
    p.add_argument(
        "--bandit-lr", type=float, default=0.05, dest="bandit_lr",
        help="Learning rate on the bandit theta (bandit condition only)",
    )
    p.add_argument(
        "--strategy",
        choices=["aggressive", "conservative"],
        default="aggressive",
        help="AF profile-selection strategy (argrl_full condition only)",
    )
    p.add_argument(
        "--ablate",
        nargs="*",
        choices=["individual", "coalition", "load_ready"],
        default=[],
        metavar="COMPONENT",
        help=(
            "Potential-function components to remove (argrl_full condition only). "
            "Zero or more of: individual coalition load_ready"
        ),
    )
    p.add_argument(
        "--scale", type=float, default=1.0,
        help=(
            "Multiplier on the potential function, and so on the shaping term "
            "(argrl_full condition only). 1.0 is the confirmatory setting."
        ),
    )
    p.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        dest="total_timesteps",
        help="Total environment steps for training",
    )
    p.add_argument("--layout", default="cramped_room", help="Overcooked layout name")
    p.add_argument("--horizon", type=int, default=400, help="Episode horizon (steps)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-envs", type=int, default=1, dest="n_envs",
                   help="Number of parallel envs for SB3 VecEnv")

    # Logging / checkpointing
    p.add_argument("--log-dir", default="logs", dest="log_dir")
    p.add_argument("--checkpoint-dir", default="checkpoints", dest="checkpoint_dir")
    p.add_argument(
        "--checkpoint-freq", type=int, default=50_000, dest="checkpoint_freq",
        help="Save a checkpoint every N total timesteps",
    )

    # PPO hyperparameters (SB3 defaults are reasonable; exposed for ablations)
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--n-steps", type=int, default=2048, dest="n_steps",
                   help="Steps per rollout per env")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--n-epochs", type=int, default=10, dest="n_epochs")
    p.add_argument(
        "--gamma", type=float, default=0.99,
        help=(
            "Discount factor. Used by PPO and by the potential shaping term, "
            "which must share it for policy invariance to hold"
        ),
    )
    p.add_argument("--gae-lambda", type=float, default=0.95, dest="gae_lambda")
    p.add_argument("--ent-coef", type=float, default=0.01, dest="ent_coef")

    args = p.parse_args()

    # train() reads these only in the argrl_full branch, so accepting them
    # elsewhere would run a condition the caller did not ask for and label it
    # as though the flag had applied.
    if args.condition != "argrl_full":
        for flag, value, default in (
            ("--strategy", args.strategy, "aggressive"),
            ("--ablate", args.ablate, []),
            ("--scale", args.scale, 1.0),
        ):
            if value != default:
                p.error(f"{flag} applies only to --condition argrl_full")

    return args


if __name__ == "__main__":
    train(parse_args())
