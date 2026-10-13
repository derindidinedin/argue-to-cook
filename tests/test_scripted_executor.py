"""
Unit tests for af/scripted_executor.py.

Tests are fast (no full episodes): they cover target resolution, navigation
to an onion dispenser, Wait no-op, and deadlock re-trigger behaviour.

All tests use cramped_room because that is the primary layout; the executor
has no layout-specific assumptions, but testing on the same layout as the
evaluation maintains consistency.
"""

import numpy as np
import pytest

from af.env_wrapper import OvercookedAFEnv
from af.framework import Intention
from af.scripted_executor import DEADLOCK_STEPS, INTERACT_IDX, STAY_IDX, ScriptedExecutor
from overcooked_ai_py.mdp.overcooked_mdp import Action

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def af_env():
    env = OvercookedAFEnv(layout_name="cramped_room", horizon=400)
    env.reset()
    return env


@pytest.fixture
def executor(af_env):
    ex = ScriptedExecutor(af_env)
    ex.reset()
    return ex


# ---------------------------------------------------------------------------
# (a) Each intention resolves to the correct target tile class
# ---------------------------------------------------------------------------


class TestTargetResolution:
    """_resolve_targets returns tiles of the right kind for each intention."""

    def test_fetch_onion_resolves_to_onion_dispensers(self, af_env, executor):
        state  = af_env.base_env.state
        tiles  = executor._resolve_targets(Intention.FETCH_ONION, state)
        expected = set(af_env.mdp.get_onion_dispenser_locations())
        assert set(tiles) == expected, f"got {tiles}, expected {expected}"

    def test_fetch_dish_resolves_to_dish_dispensers(self, af_env, executor):
        state = af_env.base_env.state
        tiles = executor._resolve_targets(Intention.FETCH_DISH, state)
        expected = set(af_env.mdp.get_dish_dispenser_locations())
        assert set(tiles) == expected

    def test_deliver_soup_resolves_to_serving_counters(self, af_env, executor):
        state = af_env.base_env.state
        tiles = executor._resolve_targets(Intention.DELIVER_SOUP, state)
        expected = set(af_env.mdp.get_serving_locations())
        assert set(tiles) == expected

    def test_pickup_soup_fallback_when_no_ready_pot(self, af_env, executor):
        """No soup ready at episode start → fall back to any pot."""
        state  = af_env.base_env.state
        tiles  = executor._resolve_targets(Intention.PICKUP_SOUP, state)
        all_pots = set(af_env.mdp.get_pot_locations())
        assert set(tiles).issubset(all_pots), (
            f"tiles {tiles} not a subset of all pots {all_pots}"
        )
        assert len(tiles) > 0

    def test_wait_resolves_to_empty(self, af_env, executor):
        state = af_env.base_env.state
        tiles = executor._resolve_targets(Intention.WAIT, state)
        assert tiles == []


# ---------------------------------------------------------------------------
# (b) Agent reaches onion dispenser and emits INTERACT within bounded steps
# ---------------------------------------------------------------------------


class TestNavigationToOnionDispenser:
    """
    Force profile = {0: FetchOnion, 1: Wait} and step until agent 0 emits
    INTERACT, verifying it happens within a generous step bound.
    """

    MAX_STEPS = 20  # cramped_room is tiny; 10 is theoretical max from any start

    def test_agent_reaches_onion_and_interacts(self, af_env):
        # Override the profile so agent 0 fetches onion, agent 1 waits.
        af_env._profile = {0: Intention.FETCH_ONION, 1: Intention.WAIT}
        executor = ScriptedExecutor(af_env)
        executor.reset()

        interacted = False
        for step in range(self.MAX_STEPS):
            actions = executor.act()
            a0 = int(actions[0])

            if a0 == INTERACT_IDX:
                # Check agent 0 is adjacent to an onion dispenser.
                state = af_env.base_env.state
                player_pos = state.players[0].position
                onion_locs = af_env.mdp.get_onion_dispenser_locations()
                mp = af_env._motion_planner
                goals = []
                for loc in onion_locs:
                    goals.extend(mp.motion_goals_for_pos.get(loc, []))
                goal_positions = {g[0] for g in goals}
                assert player_pos in goal_positions, (
                    f"INTERACT emitted at {player_pos} but not adjacent to "
                    f"any onion dispenser; goals = {goals}"
                )
                interacted = True
                break

            # Advance the environment (agent 1 stays).
            af_env.step(actions)

        assert interacted, (
            f"Agent 0 did not INTERACT with onion dispenser within "
            f"{self.MAX_STEPS} steps"
        )

    def test_agent1_stays_when_assigned_wait(self, af_env):
        """Agent assigned Wait emits STAY on the very first act() call."""
        af_env._profile = {0: Intention.WAIT, 1: Intention.WAIT}
        executor = ScriptedExecutor(af_env)
        executor.reset()

        actions = executor.act()
        assert int(actions[0]) == STAY_IDX
        assert int(actions[1]) == STAY_IDX


# ---------------------------------------------------------------------------
# (c) Wait produces STAY action
# ---------------------------------------------------------------------------


class TestWaitProducesStay:
    def test_wait_returns_stay_both_agents(self, af_env, executor):
        af_env._profile = {0: Intention.WAIT, 1: Intention.WAIT}
        actions = executor.act()
        assert int(actions[0]) == STAY_IDX
        assert int(actions[1]) == STAY_IDX

    def test_stay_index_matches_action_enum(self):
        assert STAY_IDX == Action.ACTION_TO_INDEX[Action.STAY]

    def test_interact_index_matches_action_enum(self):
        assert INTERACT_IDX == Action.ACTION_TO_INDEX[Action.INTERACT]


# ---------------------------------------------------------------------------
# (d) Deadlock rule triggers after DEADLOCK_STEPS blocked steps
# ---------------------------------------------------------------------------


class TestDeadlockDetection:
    """
    Simulate an agent that never moves by feeding the same position into
    _check_deadlock. After DEADLOCK_STEPS calls with the same position
    the AF should have been re-triggered at least once.
    """

    def test_af_retrigger_after_deadlock(self, af_env):
        executor = ScriptedExecutor(af_env)
        executor.reset()

        initial_trigger_count = af_env._af_trigger_count

        # Patch the state so both agents sit at a fixed position each call.
        # We call _check_deadlock directly rather than running full steps.
        state = af_env.base_env.state
        n_agents = af_env.mdp.num_players

        # Fill position history with the same location DEADLOCK_STEPS times.
        for _ in range(DEADLOCK_STEPS):
            executor._check_deadlock(state, n_agents)

        assert af_env._af_trigger_count > initial_trigger_count, (
            f"AF was not re-triggered after {DEADLOCK_STEPS} identical positions "
            f"(trigger_count stayed at {initial_trigger_count})"
        )

    def test_history_cleared_after_deadlock(self, af_env):
        """After deadlock reset, position history should be empty."""
        executor = ScriptedExecutor(af_env)
        executor.reset()

        state  = af_env.base_env.state
        n_agents = af_env.mdp.num_players

        for _ in range(DEADLOCK_STEPS):
            executor._check_deadlock(state, n_agents)

        # History for both agents must have been cleared (len == 0).
        for agent_id in range(n_agents):
            assert len(executor._pos_history[agent_id]) == 0, (
                f"Agent {agent_id}: position history not cleared after deadlock"
            )

    def test_no_retrigger_before_threshold(self, af_env):
        """AF must not re-trigger before DEADLOCK_STEPS - 1 stuck steps."""
        executor = ScriptedExecutor(af_env)
        executor.reset()

        initial_trigger_count = af_env._af_trigger_count
        state  = af_env.base_env.state
        n_agents = af_env.mdp.num_players

        for _ in range(DEADLOCK_STEPS - 1):
            executor._check_deadlock(state, n_agents)

        assert af_env._af_trigger_count == initial_trigger_count, (
            "AF was re-triggered prematurely (before hitting the deadlock threshold)"
        )
