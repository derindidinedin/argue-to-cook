"""
Tests for af/policy.py - ArgumentWeightPolicy and WeightStore.

No overcooked_ai environment interaction is required for most tests.
The env-integration test (TestEnvReadsWeightStore) does create an
OvercookedAFEnv, so it needs overcooked_ai installed.

Run with: pytest tests/test_policy.py -v
"""

import numpy as np
import pytest
import gymnasium
import torch as th
import torch.nn.functional as F

from af.policy import ArgumentWeightPolicy, WeightStore
from af.env_wrapper import OvercookedAFEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 1050   # matches OvercookedAFEnv observation space


def _make_policy(weight_store=None, obs_dim=OBS_DIM):
    """Instantiate ArgumentWeightPolicy against a synthetic observation space."""
    obs_space = gymnasium.spaces.Box(
        low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
    )
    act_space = gymnasium.spaces.MultiDiscrete([6, 6])
    policy = ArgumentWeightPolicy(
        obs_space,
        act_space,
        lr_schedule=lambda _: 3e-4,
        weight_store=weight_store,
    )
    return policy


def _rand_obs(batch: int = 4, obs_dim: int = OBS_DIM) -> th.Tensor:
    return th.rand(batch, obs_dim)


# ---------------------------------------------------------------------------
# Policy instantiation
# ---------------------------------------------------------------------------

class TestPolicyInstantiation:
    def test_build_order_produces_weight_head(self):
        """_build creates weight_head_mlp before calling super()._build() so the
        optimizer automatically picks it up via self.parameters()."""
        policy = _make_policy()
        assert hasattr(policy, "weight_head_mlp")

    def test_weight_head_output_is_three_dimensional(self):
        """Weight head must produce exactly three values (w1, w2, w3)."""
        policy = _make_policy()
        obs = _rand_obs(batch=1)
        with th.no_grad():
            features = policy.extract_features(obs)
        out = policy.weight_head_mlp(features)
        assert out.shape[-1] == 3

    def test_weight_head_output_shape(self):
        policy = _make_policy()
        obs = _rand_obs(batch=3)
        with th.no_grad():
            features = policy.extract_features(obs)
        out = policy.weight_head_mlp(features)
        assert out.shape == (3, 3), f"Expected (3, 3), got {out.shape}"

    def test_weight_head_params_in_optimizer(self):
        """All weight_head_mlp parameters must be tracked by the optimizer."""
        policy = _make_policy()
        optimizer_params = set()
        for group in policy.optimizer.param_groups:
            for p in group["params"]:
                optimizer_params.add(id(p))

        wh_params = set(id(p) for p in policy.weight_head_mlp.parameters())
        assert wh_params.issubset(optimizer_params), (
            "weight_head_mlp parameters not included in optimizer"
        )

    def test_instantiates_with_weight_store(self):
        store = WeightStore()
        policy = _make_policy(weight_store=store)
        assert policy._weight_store is store


# ---------------------------------------------------------------------------
# Weight head non-negativity
# ---------------------------------------------------------------------------

class TestWeightHeadNonNeg:
    def test_softplus_output_nonneg_typical(self):
        policy = _make_policy()
        obs = _rand_obs(batch=8)
        with th.no_grad():
            features = policy.extract_features(obs)
            weights = F.softplus(policy.weight_head_mlp(features))
        assert (weights >= 0).all(), "Some weights went negative"

    def test_softplus_output_nonneg_large_positive(self):
        policy = _make_policy()
        obs = th.full((1, OBS_DIM), 10.0)
        with th.no_grad():
            features = policy.extract_features(obs)
            weights = F.softplus(policy.weight_head_mlp(features))
        assert (weights >= 0).all()

    def test_softplus_output_nonneg_large_negative(self):
        policy = _make_policy()
        obs = th.full((1, OBS_DIM), -10.0)
        with th.no_grad():
            features = policy.extract_features(obs)
            weights = F.softplus(policy.weight_head_mlp(features))
        assert (weights >= 0).all()

    def test_weight_head_output_method_nonneg(self):
        """The internal helper _weight_head_output must always be >= 0."""
        policy = _make_policy()
        for _ in range(5):
            obs = th.randn(4, OBS_DIM)
            with th.no_grad():
                features = policy.extract_features(obs)
                weights = policy._weight_head_output(features)
            assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# WeightStore updated after forward pass
# ---------------------------------------------------------------------------

class TestWeightStoreUpdated:
    def test_store_weights_shape_after_forward(self):
        store = WeightStore()
        policy = _make_policy(weight_store=store)
        obs = _rand_obs(batch=1)
        policy.eval()
        with th.no_grad():
            policy.forward(obs)
        assert store.weights.shape == (3,)

    def test_store_weights_nonneg_after_forward(self):
        store = WeightStore()
        policy = _make_policy(weight_store=store)
        obs = _rand_obs(batch=1)
        policy.eval()
        with th.no_grad():
            policy.forward(obs)
        assert (store.weights >= 0).all(), (
            f"Store weights have negative values: {store.weights}"
        )

    def test_store_updated_after_forward(self):
        """Forward pass must write to the store (overwrite the default [1,1,1])."""
        store = WeightStore()
        # Force a non-unit default so any write is detectable
        store.weights = np.array([-99.0, -99.0, -99.0], dtype=np.float64)
        policy = _make_policy(weight_store=store)
        obs = _rand_obs(batch=1)
        policy.eval()
        with th.no_grad():
            policy.forward(obs)
        assert not np.allclose(store.weights, [-99.0, -99.0, -99.0]), (
            "forward() did not write weights to the store"
        )

    def test_store_updated_after_evaluate_actions(self):
        """evaluate_actions must also update the store."""
        store = WeightStore()
        store.weights = np.array([-99.0, -99.0, -99.0], dtype=np.float64)
        policy = _make_policy(weight_store=store)
        obs = _rand_obs(batch=4)
        # Need valid actions for evaluate_actions
        policy.eval()
        with th.no_grad():
            actions, _, _ = policy.forward(obs)
        # Reset sentinel
        store.weights = np.array([-99.0, -99.0, -99.0], dtype=np.float64)
        # evaluate_actions is called during PPO training
        policy.train()
        vals, lp, ent = policy.evaluate_actions(obs, actions)
        assert not np.allclose(store.weights, [-99.0, -99.0, -99.0]), (
            "evaluate_actions() did not write weights to the store"
        )

    def test_store_changes_with_different_observations(self):
        """Different observations should produce different weight values."""
        store = WeightStore()
        policy = _make_policy(weight_store=store)
        policy.eval()

        obs_a = th.zeros(1, OBS_DIM)
        with th.no_grad():
            policy.forward(obs_a)
        w_a = store.weights.copy()

        obs_b = th.ones(1, OBS_DIM)
        with th.no_grad():
            policy.forward(obs_b)
        w_b = store.weights.copy()

        assert not np.allclose(w_a, w_b), (
            "Weight store returned the same values for zero and one observations; "
            "weight head may not be applying the feature extractor"
        )

    def test_forward_succeeds_without_weight_store(self):
        """forward() must return valid actions when weight_store is None."""
        policy = _make_policy(weight_store=None)
        obs = _rand_obs(batch=1)
        policy.eval()
        with th.no_grad():
            actions, values, log_prob = policy.forward(obs)
        assert actions.shape[0] == 1


# ---------------------------------------------------------------------------
# Env reads from weight store at AF trigger
# ---------------------------------------------------------------------------

class TestEnvReadsWeightStore:
    def test_env_wired_to_weight_store(self):
        """Env constructed with a WeightStore must use it (not fall back to fixed weights)."""
        store = WeightStore()
        env = OvercookedAFEnv(weight_store=store)
        assert env._weight_store is store

    def test_env_uses_store_in_run_af(self):
        """
        After reset, manually overwrite the store and call _run_af().
        The env should produce a valid profile using the overwritten weights.
        Verifies the read path: store.weights → score_argument → AF → profile.
        """
        store = WeightStore()
        env = OvercookedAFEnv(weight_store=store)
        env.reset()

        # Overwrite with extreme weights: only distance matters, no downstream
        store.weights = np.array([100.0, 0.0, 0.0], dtype=np.float64)
        env._run_af()
        profile = env._profile

        # Profile must be a valid dict mapping each agent to an Intention
        from af.framework import Intention
        assert isinstance(profile, dict)
        assert set(profile.keys()) == {0, 1}
        assert all(isinstance(v, Intention) for v in profile.values())

    def test_env_fallback_to_fixed_weights_when_no_store(self):
        """When weight_store is None, env uses self._weights (fixed array)."""
        env = OvercookedAFEnv(weight_store=None, weights=np.array([2.0, 1.0, 0.5]))
        env.reset()
        env._run_af()
        profile = env._profile
        assert set(profile.keys()) == {0, 1}

    def test_env_reset_and_step_with_weight_store(self):
        """Full reset + 5 steps must not raise with a connected weight store."""
        store = WeightStore()
        env = OvercookedAFEnv(weight_store=store)
        obs, _ = env.reset()
        assert obs.shape == (1050,)

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (1050,)
            assert isinstance(reward, float)
