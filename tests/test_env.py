"""
Sanity tests for OvercookedGymEnv and OvercookedAFEnv.

Run with: pytest tests/test_env.py -v
"""

import numpy as np
import pytest
from af.env_wrapper import OvercookedGymEnv, OvercookedAFEnv
from af.framework import Intention


@pytest.fixture
def env():
    e = OvercookedGymEnv(layout_name="cramped_room", horizon=400)
    yield e


def test_observation_space_shape(env):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.shape == (1040,)


def test_action_space(env):
    assert hasattr(env.action_space, "nvec")
    assert list(env.action_space.nvec) == [6, 6]


def test_obs_in_bounds(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "Reset obs outside declared space"


def test_ten_random_steps(env):
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (1040,), "Obs shape changed after step"
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "soups_this_step" in info
        assert info["soups_this_step"] >= 0

        if terminated or truncated:
            break  # episode ended early; that's fine


def test_soups_delivered_tracking(env):
    """soups_delivered in terminal info should equal sum of soups_this_step."""
    env.reset()
    total_from_steps = 0
    terminal_info = None

    done = False
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        total_from_steps += info["soups_this_step"]
        done = terminated or truncated
        if done:
            terminal_info = info

    assert terminal_info is not None
    assert "soups_delivered" in terminal_info
    assert terminal_info["soups_delivered"] == total_from_steps
    assert terminal_info["soups_delivered"] == env._soups_delivered


def test_reset_clears_episode_state(env):
    env.reset()
    for _ in range(5):
        env.step(env.action_space.sample())

    env.reset()
    assert env._soups_delivered == 0
    assert env._ep_sparse_reward == 0.0
    assert env._ep_length == 0


def test_step_reward_is_scalar_float(env):
    """step() must return a Python float, not a numpy scalar or tensor."""
    env.reset()
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)
    assert type(reward) is float


# ---------------------------------------------------------------------------
# OvercookedAFEnv - A* planner step-count tests
# ---------------------------------------------------------------------------

@pytest.fixture
def af_env():
    e = OvercookedAFEnv(layout_name="cramped_room", horizon=400)
    e.reset()
    yield e


def test_af_step_counts_are_nonneg_integers(af_env):
    """All A* step counts must be non-negative integers."""
    af_state = af_env._extract_af_state()
    for agent_steps in af_state.agent_steps_to_target:
        for intention, steps in agent_steps.items():
            assert isinstance(steps, int), (
                f"{intention}: expected int, got {type(steps).__name__}"
            )
            assert steps >= 0, f"{intention}: step count is negative ({steps})"


def test_af_wait_step_count_is_zero(af_env):
    """Wait always returns 0 steps regardless of agent position."""
    af_state = af_env._extract_af_state()
    for agent_id, agent_steps in enumerate(af_state.agent_steps_to_target):
        assert agent_steps[Intention.WAIT] == 0, (
            f"Agent {agent_id}: Wait step count should be 0, "
            f"got {agent_steps[Intention.WAIT]}"
        )


def test_af_step_counts_differ_between_agents(af_env):
    """
    Agents start at different positions, so at least one intention must have
    different step counts between agent 0 and agent 1.
    """
    af_state = af_env._extract_af_state()
    steps0 = af_state.agent_steps_to_target[0]
    steps1 = af_state.agent_steps_to_target[1]

    non_wait = [i for i in Intention if i is not Intention.WAIT]
    differs = any(steps0[i] != steps1[i] for i in non_wait)
    assert differs, (
        "Agent 0 and agent 1 have identical step counts for all intentions - "
        "planner may be returning the same flat value for all agents.\n"
        f"Agent 0: {steps0}\nAgent 1: {steps1}"
    )


# ---------------------------------------------------------------------------
# obs_control: shaping must be exactly zero when all components ablated
# ---------------------------------------------------------------------------

def test_obs_control_shaping_is_zero():
    """
    OvercookedAFEnv with all three potential components ablated must return
    shaped_reward == 0.0 at every step throughout a full episode.

    This verifies the obs_control condition invariant: AF still fires and the
    profile is still appended to the observation, but potential shaping
    contributes nothing to the training reward.
    """
    env = OvercookedAFEnv(
        layout_name="cramped_room",
        horizon=400,
        ablate={"individual", "coalition", "load_ready"},
    )
    env.reset()
    done = False
    step_idx = 0
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        assert info["shaped_reward"] == 0.0, (
            f"Step {step_idx}: expected shaped_reward=0.0, got {info['shaped_reward']}"
        )
        step_idx += 1
