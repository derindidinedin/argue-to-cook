"""
Unit tests for af/reward_shaping.py.

No overcooked_ai dependency - all states are mock OvercookedAFState objects.

Run with: pytest tests/test_reward_shaping.py -v
"""

import pytest
from af.framework import Intention, OvercookedAFState, PotState
from af.reward_shaping import PotentialFunction, MAX_STEPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    steps: int = 5,
    soup_is_ready: bool = False,
    agent_holding: list | None = None,
    onions_in_pot: int = 0,
) -> OvercookedAFState:
    """Return a two-agent mock state with uniform step counts."""
    is_cooking = (onions_in_pot == 3 and not soup_is_ready)
    return OvercookedAFState(
        agent_positions=[(0, 0), (0, 1)],
        agent_holding=[None, None] if agent_holding is None else agent_holding,
        pot_states=[PotState(onion_count=onions_in_pot, is_cooking=is_cooking, is_ready=soup_is_ready)],
        num_agents_assigned={i: 0 for i in Intention},
        agent_steps_to_target=[
            {i: steps for i in Intention},
            {i: steps for i in Intention},
        ],
    )


def _profile(i0: Intention, i1: Intention) -> dict:
    return {0: i0, 1: i1}


FO = Intention.FETCH_ONION
FD = Intention.FETCH_DISH
PS = Intention.PICKUP_SOUP
DS = Intention.DELIVER_SOUP
W  = Intention.WAIT


# ---------------------------------------------------------------------------
# Individual component in isolation
# ---------------------------------------------------------------------------

class TestIndividual:
    """Ablate coalition and load_ready; verify individual component."""

    def _pot(self, **kw):
        return PotentialFunction(ablate={"coalition", "load_ready"}, **kw)

    def test_both_assigned_fetch_onion(self):
        # steps=5, MAX_STEPS=10 → 1 - 5/10 = 0.5 per agent → mean = 0.5
        pot = self._pot()
        result = pot.phi(_state(steps=5), _profile(FO, FO))
        assert abs(result - 0.5) < 1e-9

    def test_wait_agent_contributes_zero(self):
        # One FO (0.5), one Wait (0) → mean = 0.25
        pot = self._pot()
        result = pot.phi(_state(steps=5), _profile(FO, W))
        assert abs(result - 0.25) < 1e-9

    def test_both_wait_gives_zero(self):
        pot = self._pot()
        result = pot.phi(_state(steps=5), _profile(W, W))
        assert result == 0.0

    def test_closer_agent_scores_higher(self):
        # steps=2 → 1 - 2/10 = 0.8  >  steps=8 → 1 - 8/10 = 0.2
        pot = self._pot()
        close = pot.phi(_state(steps=2), _profile(FO, FO))
        far   = pot.phi(_state(steps=8), _profile(FO, FO))
        assert close > far

    def test_individual_clamped_at_zero_when_very_far(self):
        # steps > MAX_STEPS → max(0, 1 - 20/10) = max(0, -1) = 0
        pot = self._pot()
        result = pot.phi(_state(steps=20), _profile(FO, FO))
        assert result == 0.0

    def test_alpha_scales_individual(self):
        pot_1 = self._pot(alpha=1.0)
        pot_2 = self._pot(alpha=2.0)
        s = _state(steps=5)
        p = _profile(FO, FO)
        assert abs(pot_2.phi(s, p) - 2.0 * pot_1.phi(s, p)) < 1e-9


# ---------------------------------------------------------------------------
# Coalition component in isolation
# ---------------------------------------------------------------------------

class TestCoalition:
    """Ablate individual and load_ready; verify coalition component."""

    def _pot(self, **kw):
        return PotentialFunction(ablate={"individual", "load_ready"}, **kw)

    def test_distinct_non_wait_gives_one(self):
        result = self._pot().phi(_state(), _profile(FO, FD))
        assert result == 1.0

    def test_distinct_non_wait_all_pairs(self):
        pot = self._pot()
        s = _state()
        for a in [FO, FD, PS, DS]:
            for b in [FO, FD, PS, DS]:
                if a is not b:
                    assert pot.phi(s, _profile(a, b)) == 1.0

    def test_one_wait_gives_half(self):
        pot = self._pot()
        assert pot.phi(_state(), _profile(FO, W)) == 0.5
        assert pot.phi(_state(), _profile(W, FD)) == 0.5

    def test_both_wait_gives_zero(self):
        assert self._pot().phi(_state(), _profile(W, W)) == 0.0

    def test_same_non_wait_gives_zero(self):
        pot = self._pot()
        for intention in [FO, FD, PS, DS]:
            assert pot.phi(_state(), _profile(intention, intention)) == 0.0

    def test_beta_scales_coalition(self):
        pot_1 = self._pot(beta=1.0)
        pot_2 = self._pot(beta=3.0)
        s, p = _state(), _profile(FO, FD)
        assert abs(pot_2.phi(s, p) - 3.0 * pot_1.phi(s, p)) < 1e-9


# ---------------------------------------------------------------------------
# Load-ready component in isolation
# ---------------------------------------------------------------------------

class TestLoadReady:
    """Ablate individual and coalition; verify load_ready component."""

    def _pot(self, **kw):
        return PotentialFunction(ablate={"individual", "coalition"}, **kw)

    def test_one_dish_soup_ready(self):
        s = _state(soup_is_ready=True, agent_holding=["dish", None])
        # 1 dish / 2 agents = 0.5
        assert abs(self._pot().phi(s, _profile(W, W)) - 0.5) < 1e-9

    def test_two_dishes_soup_ready(self):
        s = _state(soup_is_ready=True, agent_holding=["dish", "dish"])
        # 2 dishes / 2 agents = 1.0
        assert abs(self._pot().phi(s, _profile(W, W)) - 1.0) < 1e-9

    def test_dish_held_soup_not_ready(self):
        # soup not ready → load_ready = 0 regardless of holding
        s = _state(soup_is_ready=False, agent_holding=["dish", None])
        assert self._pot().phi(s, _profile(W, W)) == 0.0

    def test_no_dish_soup_ready(self):
        s = _state(soup_is_ready=True, agent_holding=[None, None])
        assert self._pot().phi(s, _profile(W, W)) == 0.0

    def test_delta_scales_load_ready(self):
        pot_1 = self._pot(delta=1.0)
        pot_2 = self._pot(delta=4.0)
        s = _state(soup_is_ready=True, agent_holding=["dish", None])
        assert abs(pot_2.phi(s, _profile(W, W)) - 4.0 * pot_1.phi(s, _profile(W, W))) < 1e-9


# ---------------------------------------------------------------------------
# Shaping bonus
# ---------------------------------------------------------------------------

class TestShapingBonus:
    def test_positive_when_moving_closer(self):
        """Moving an agent closer to its target raises phi → positive bonus."""
        pot = PotentialFunction(ablate={"coalition", "load_ready"})
        state_far  = _state(steps=8)
        state_near = _state(steps=2)
        p = _profile(FO, FO)
        bonus = pot.shaping_bonus(state_far, state_near, p, p)
        assert bonus > 0.0

    def test_negative_when_moving_farther(self):
        pot = PotentialFunction(ablate={"coalition", "load_ready"})
        state_near = _state(steps=2)
        state_far  = _state(steps=8)
        p = _profile(FO, FO)
        bonus = pot.shaping_bonus(state_near, state_far, p, p)
        assert bonus < 0.0

    def test_formula_is_gamma_phi_next_minus_phi_curr(self):
        pot = PotentialFunction()
        s1 = _state(steps=5)
        s2 = _state(steps=3)
        p1 = _profile(FO, FD)
        p2 = _profile(FO, W)
        gamma = 0.99
        expected = gamma * pot.phi(s2, p2) - pot.phi(s1, p1)
        result = pot.shaping_bonus(s1, s2, p1, p2, gamma=gamma)
        assert abs(result - expected) < 1e-9

    def test_zero_bonus_when_phi_unchanged(self):
        """Same state, same profile → bonus ≈ (gamma - 1) * phi, only zero when phi = 0."""
        pot = PotentialFunction(ablate={"individual", "coalition", "load_ready"})
        s = _state()
        p = _profile(W, W)
        # phi = 0 everywhere when all ablated
        assert pot.shaping_bonus(s, s, p, p) == 0.0


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

class TestAblation:
    def test_ablate_all_gives_zero(self):
        pot = PotentialFunction(ablate={"individual", "coalition", "load_ready"})
        for steps in [1, 5, 10, 20]:
            for i0 in [FO, FD, W]:
                for i1 in [FO, DS, W]:
                    s = _state(steps=steps, soup_is_ready=True, agent_holding=["dish", None])
                    assert pot.phi(s, _profile(i0, i1)) == 0.0

    def test_ablate_individual_leaves_coalition_and_load_ready(self):
        pot = PotentialFunction(ablate={"individual"})
        s = _state(soup_is_ready=True, agent_holding=["dish", None])
        p = _profile(FO, FD)
        # coalition = 1.0, load_ready = 0.5, individual = 0
        expected = 1.0 * 1.0 + 1.0 * 0.5
        assert abs(pot.phi(s, p) - expected) < 1e-9

    def test_ablate_coalition_leaves_individual_and_load_ready(self):
        pot = PotentialFunction(ablate={"coalition"})
        s = _state(steps=5, soup_is_ready=True, agent_holding=["dish", None])
        p = _profile(FO, FO)
        # individual = 0.5, load_ready = 0.5, coalition = 0
        expected = 1.0 * 0.5 + 1.0 * 0.5
        assert abs(pot.phi(s, p) - expected) < 1e-9

    def test_invalid_ablation_key_raises(self):
        with pytest.raises(ValueError):
            PotentialFunction(ablate={"typo_key"})


# ---------------------------------------------------------------------------
# phi non-negativity (Wait guarantee)
# ---------------------------------------------------------------------------

class TestPhiNonNegative:
    def test_phi_nonnegative_all_states(self):
        """phi must be >= 0 for all component combinations and profiles."""
        pot = PotentialFunction()
        states = [
            _state(steps=1),
            _state(steps=5),
            _state(steps=20),   # beyond MAX_STEPS → individual clamped to 0
            _state(soup_is_ready=True, agent_holding=["dish", "soup"]),
        ]
        profiles = [
            _profile(FO, FD),
            _profile(FO, W),
            _profile(W, W),
            _profile(DS, DS),
        ]
        for s in states:
            for p in profiles:
                result = pot.phi(s, p)
                assert result >= 0.0, f"phi went negative ({result}) for state={s}, profile={p}"

    def test_shaping_bonus_can_be_negative_but_phi_stays_nonneg(self):
        """The bonus itself can be negative; only phi must be >= 0."""
        pot = PotentialFunction(ablate={"coalition", "load_ready"})
        s_good = _state(steps=2)
        s_bad  = _state(steps=9)
        p = _profile(FO, FO)
        assert pot.phi(s_good, p) >= 0.0
        assert pot.phi(s_bad, p) >= 0.0
        # But bonus when going from good → bad is negative
        assert pot.shaping_bonus(s_good, s_bad, p, p) < 0.0
