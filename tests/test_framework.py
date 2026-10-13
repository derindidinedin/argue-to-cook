"""
Unit tests for the argumentation framework (af/framework.py) and scoring
(af/scoring.py). No overcooked_ai dependency - all states are mock objects.

Run with: pytest tests/test_framework.py -v
"""

import numpy as np
import pytest

from af.framework import (
    Argument,
    ArgumentationFramework,
    Intention,
    OvercookedAFState,
    PotState,
    Strategy,
    _is_exclusive_inter_agent,
    check_preconditions,
    compute_acceptable_sets,
    generate_arguments,
    generate_attacks_with_state,
    resolve_vaf,
    select_profile,
)
from af.scoring import score_argument, DEPENDENCY_COUNT


# ---------------------------------------------------------------------------
# Helpers - mock state factories
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> OvercookedAFState:
    """
    Default two-agent state: empty pot, no one holding anything,
    no previous AF assignments, large dummy step counts.

    Convenience bridge: accepts onions_in_pot (int) and soup_is_ready (bool)
    and converts them to a single-element pot_states list so existing tests
    don't need to be rewritten for the per-pot redesign.
    """
    onions_in_pot = overrides.pop("onions_in_pot", 0)
    soup_is_ready = overrides.pop("soup_is_ready", False)
    # 3 onions in pot and not yet ready → pot is cooking
    is_cooking = (onions_in_pot == 3 and not soup_is_ready)
    defaults = dict(
        agent_positions=[(0, 0), (0, 1)],
        agent_holding=[None, None],
        pot_states=[PotState(onion_count=onions_in_pot, is_cooking=is_cooking, is_ready=soup_is_ready)],
        num_agents_assigned={i: 0 for i in Intention},
        agent_steps_to_target=[
            {i: 10 for i in Intention},
            {i: 10 for i in Intention},
        ],
    )
    defaults.update(overrides)
    return OvercookedAFState(**defaults)


def _weights(w1=1.0, w2=0.0, w3=0.0) -> np.ndarray:
    return np.array([w1, w2, w3], dtype=float)


# ---------------------------------------------------------------------------
# Precondition tests
# ---------------------------------------------------------------------------

class TestPreconditions:
    def test_wait_always_true(self):
        for onions in range(4):
            for holding in [None, "onion", "dish", "soup"]:
                state = _base_state(
                    agent_holding=[holding, None], onions_in_pot=onions
                )
                assert check_preconditions(Intention.WAIT, 0, state)

    def test_fetch_onion_active_when_pot_empty(self):
        state = _base_state(onions_in_pot=0, agent_holding=[None, None])
        assert check_preconditions(Intention.FETCH_ONION, 0, state)

    def test_fetch_onion_inactive_when_pot_full(self):
        state = _base_state(onions_in_pot=3, agent_holding=[None, None])
        assert not check_preconditions(Intention.FETCH_ONION, 0, state)

    def test_fetch_onion_inactive_when_slot_already_assigned(self):
        """Only 1 onion slot remains AND 1 agent already assigned."""
        state = _base_state(
            onions_in_pot=2,
            agent_holding=[None, None],
            num_agents_assigned={Intention.FETCH_ONION: 1},
        )
        # onions_needed=1, assigned=1 → not (1 > 1)
        assert not check_preconditions(Intention.FETCH_ONION, 0, state)

    def test_fetch_onion_inactive_when_holding(self):
        state = _base_state(onions_in_pot=0, agent_holding=["onion", None])
        assert not check_preconditions(Intention.FETCH_ONION, 0, state)

    def test_fetch_dish_active(self):
        state = _base_state(
            onions_in_pot=3,
            agent_holding=[None, None],
            num_agents_assigned={Intention.FETCH_DISH: 0},
        )
        assert check_preconditions(Intention.FETCH_DISH, 0, state)

    def test_fetch_dish_inactive_pot_not_full(self):
        state = _base_state(
            onions_in_pot=2,
            agent_holding=[None, None],
            num_agents_assigned={Intention.FETCH_DISH: 0},
        )
        assert not check_preconditions(Intention.FETCH_DISH, 0, state)

    def test_fetch_dish_inactive_when_agent_already_assigned(self):
        state = _base_state(
            onions_in_pot=3,
            agent_holding=[None, None],
            num_agents_assigned={Intention.FETCH_DISH: 1},
        )
        assert not check_preconditions(Intention.FETCH_DISH, 0, state)

    def test_fetch_dish_inactive_when_agent_holding_dish(self):
        state = _base_state(
            onions_in_pot=3,
            agent_holding=["dish", None],
            num_agents_assigned={Intention.FETCH_DISH: 0},
        )
        assert not check_preconditions(Intention.FETCH_DISH, 0, state)

    def test_pickup_soup_active(self):
        state = _base_state(
            agent_holding=["dish", None], soup_is_ready=True
        )
        assert check_preconditions(Intention.PICKUP_SOUP, 0, state)

    def test_pickup_soup_inactive_no_dish(self):
        state = _base_state(agent_holding=[None, None], soup_is_ready=True)
        assert not check_preconditions(Intention.PICKUP_SOUP, 0, state)

    def test_pickup_soup_inactive_not_ready(self):
        state = _base_state(
            agent_holding=["dish", None], soup_is_ready=False
        )
        assert not check_preconditions(Intention.PICKUP_SOUP, 0, state)

    def test_deliver_soup_active(self):
        state = _base_state(agent_holding=["soup", None])
        assert check_preconditions(Intention.DELIVER_SOUP, 0, state)

    def test_deliver_soup_inactive_no_soup(self):
        state = _base_state(agent_holding=[None, None])
        assert not check_preconditions(Intention.DELIVER_SOUP, 0, state)


# ---------------------------------------------------------------------------
# Attack generation tests
# ---------------------------------------------------------------------------

class TestAttacks:
    def test_intra_agent_mutual_attack(self):
        """Two active arguments for the same agent must mutually attack."""
        state = _base_state(onions_in_pot=0, agent_holding=[None, None])
        args = generate_arguments(state, num_agents=2)
        agent0_args = [a for a in args if a.agent_id == 0]
        assert len(agent0_args) >= 2  # at least FetchOnion and Wait

        attacks = generate_attacks_with_state(args, state)
        attacker_target_pairs = {(a.agent_id, b.agent_id) for a, b in attacks}
        # Attacks between agent 0 arguments → both are agent 0
        self_attacks = [(a, b) for a, b in attacks if a.agent_id == b.agent_id == 0]
        assert len(self_attacks) >= 2  # at least (FO→W, W→FO)

    def test_inter_agent_attack_on_exclusive_slot(self):
        """When only 1 onion slot remains, both agents' FetchOnion args attack each other."""
        state = _base_state(
            onions_in_pot=2,   # 1 slot left
            agent_holding=[None, None],
            num_agents_assigned={Intention.FETCH_ONION: 0},
        )
        args = generate_arguments(state, num_agents=2)
        fo_args = [a for a in args if a.intention is Intention.FETCH_ONION]
        assert len(fo_args) == 2, "Both agents should have FetchOnion active"

        attacks = generate_attacks_with_state(args, state)
        inter = [
            (a, b) for a, b in attacks
            if a.agent_id != b.agent_id
            and a.intention is Intention.FETCH_ONION
        ]
        assert len(inter) == 2  # mutual attack

    def test_no_inter_agent_attack_when_multiple_slots(self):
        """When 2+ onion slots remain, both agents can fetch onions - no inter-agent attack."""
        state = _base_state(
            onions_in_pot=0,   # 3 slots left
            agent_holding=[None, None],
            num_agents_assigned={Intention.FETCH_ONION: 0},
        )
        args = generate_arguments(state, num_agents=2)
        attacks = generate_attacks_with_state(args, state)
        inter_fo = [
            (a, b) for a, b in attacks
            if a.agent_id != b.agent_id
            and a.intention is Intention.FETCH_ONION
        ]
        assert inter_fo == []

    def test_wait_never_inter_agent_exclusive(self):
        """WAIT must never be classified inter-agent-exclusive, across every
        state - this is the load-bearing precondition of the non-emptiness
        theorem (Wait(A) and Wait(B) must never inter-attack). Enforced by an
        explicit branch in _is_exclusive_inter_agent, guarded here so a future
        edit to that function cannot silently break the theorem."""
        for onions in range(4):
            for ready in (False, True):
                for holdings in ([None, None], ["dish", "soup"], ["soup", "dish"]):
                    state = _base_state(
                        onions_in_pot=onions,
                        soup_is_ready=ready,
                        agent_holding=holdings,
                    )
                    assert _is_exclusive_inter_agent(Intention.WAIT, state) is False, (
                        f"WAIT wrongly exclusive for onions={onions}, "
                        f"ready={ready}, holdings={holdings}"
                    )

        # And end-to-end: no inter-agent attack ever targets a WAIT argument.
        for onions in range(4):
            state = _base_state(onions_in_pot=onions)
            args = generate_arguments(state, num_agents=2)
            attacks = generate_attacks_with_state(args, state)
            wait_inter = [
                (a, b) for a, b in attacks
                if a.agent_id != b.agent_id
                and (a.intention is Intention.WAIT or b.intention is Intention.WAIT)
            ]
            assert wait_inter == [], f"WAIT drawn into inter-agent attack, onions={onions}"


# ---------------------------------------------------------------------------
# VAF resolution tests
# ---------------------------------------------------------------------------

class TestVAFResolution:
    def test_higher_score_defeats_lower(self):
        """In a mutual attack, the higher-scored argument should win."""
        arg_a = Argument(agent_id=0, intention=Intention.FETCH_ONION, score=0.9)
        arg_b = Argument(agent_id=0, intention=Intention.WAIT, score=0.1)
        attacks = [(arg_a, arg_b), (arg_b, arg_a)]

        successful = resolve_vaf([arg_a, arg_b], attacks)
        assert (arg_a, arg_b) in successful
        assert (arg_b, arg_a) not in successful

    def test_lower_scored_arg_does_not_defeat_higher(self):
        arg_a = Argument(agent_id=0, intention=Intention.FETCH_ONION, score=0.2)
        arg_b = Argument(agent_id=0, intention=Intention.WAIT, score=0.8)
        attacks = [(arg_a, arg_b), (arg_b, arg_a)]

        successful = resolve_vaf([arg_a, arg_b], attacks)
        assert (arg_b, arg_a) in successful
        assert (arg_a, arg_b) not in successful

    def test_equal_scores_both_succeed(self):
        """Equal-scored mutual attack - both attacks succeed (genuine conflict under the framework's attack rules)."""
        arg_a = Argument(agent_id=0, intention=Intention.FETCH_ONION, score=0.5)
        arg_b = Argument(agent_id=0, intention=Intention.WAIT, score=0.5)
        attacks = [(arg_a, arg_b), (arg_b, arg_a)]

        successful = resolve_vaf([arg_a, arg_b], attacks)
        assert (arg_a, arg_b) in successful
        assert (arg_b, arg_a) in successful

    def test_one_directional_attack_always_succeeds(self):
        arg_a = Argument(agent_id=0, intention=Intention.FETCH_ONION, score=0.1)
        arg_b = Argument(agent_id=1, intention=Intention.WAIT, score=0.9)
        attacks = [(arg_a, arg_b)]  # one-directional

        successful = resolve_vaf([arg_a, arg_b], attacks)
        assert (arg_a, arg_b) in successful


# ---------------------------------------------------------------------------
# Acceptable set computation tests
# ---------------------------------------------------------------------------

class TestAcceptableSets:
    def test_worked_example(self):
        """
        Worked example from the thesis:
          Pot has 2 onions in it (1 slot remaining).
          Agent A is near the dispenser: FetchOnion score 0.9.
          Agent B is far: FetchOnion score 0.3.
          Both have Wait (score 0.0).

        Expected: the only valid acceptable set is {FetchOnion(A), Wait(B)}.
        """
        state = _base_state(
            onions_in_pot=2,
            agent_holding=[None, None],
            num_agents_assigned={Intention.FETCH_ONION: 0},
        )
        args = generate_arguments(state, num_agents=2)

        # Set scores
        for a in args:
            if a.intention is Intention.FETCH_ONION and a.agent_id == 0:
                a.score = 0.9
            elif a.intention is Intention.FETCH_ONION and a.agent_id == 1:
                a.score = 0.3
            else:
                a.score = 0.0  # Wait

        attacks = generate_attacks_with_state(args, state)
        successful = resolve_vaf(args, attacks)
        acceptable = compute_acceptable_sets(args, successful, num_agents=2)

        # Extract intention pairs
        profiles = [
            tuple(sorted((a.intention for a in s), key=lambda x: x.value))
            for s in acceptable
        ]

        # Must contain FetchOnion(A) + Wait(B)
        fo_wait = tuple(sorted(
            [Intention.FETCH_ONION, Intention.WAIT], key=lambda x: x.value
        ))
        assert fo_wait in profiles, f"Expected FetchOnion+Wait profile, got: {profiles}"

        # FetchOnion(B) + anything should NOT be in an acceptable set
        # because FO(B) is defeated by FO(A)
        for s in acceptable:
            for a in s:
                if a.agent_id == 1 and a.intention is Intention.FETCH_ONION:
                    pytest.fail(
                        f"FetchOnion(agent=1) survived into an acceptable set: {s}"
                    )

    def test_wait_guarantee_empty_pot(self):
        """Acceptable sets must always be non-empty (Wait theorem)."""
        state = _base_state(onions_in_pot=0, agent_holding=[None, None])
        args = generate_arguments(state, num_agents=2)
        for a in args:
            a.score = score_argument(a, state, _weights(1.0, 0.0, 0.0))
        attacks = generate_attacks_with_state(args, state)
        successful = resolve_vaf(args, attacks)
        acceptable = compute_acceptable_sets(args, successful, num_agents=2)
        assert len(acceptable) > 0

    def test_wait_guarantee_full_pot_soup_ready(self):
        """Wait theorem holds in a complex state."""
        state = _base_state(
            onions_in_pot=3,
            soup_is_ready=True,
            agent_holding=["dish", "soup"],
        )
        args = generate_arguments(state, num_agents=2)
        for a in args:
            a.score = score_argument(a, state, _weights(1.0, 1.0, 0.5))
        attacks = generate_attacks_with_state(args, state)
        successful = resolve_vaf(args, attacks)
        acceptable = compute_acceptable_sets(args, successful, num_agents=2)
        assert len(acceptable) > 0

    def test_wait_wait_always_valid(self):
        """The pair {Wait(0), Wait(1)} must always be an acceptable set."""
        for onions in range(4):
            state = _base_state(onions_in_pot=onions)
            args = generate_arguments(state, num_agents=2)
            # Give all non-Wait args score 999 to be adversarial
            for a in args:
                a.score = 0.0 if a.intention is Intention.WAIT else 999.0
            attacks = generate_attacks_with_state(args, state)
            successful = resolve_vaf(args, attacks)
            acceptable = compute_acceptable_sets(args, successful, num_agents=2)

            wait_sets = [
                s for s in acceptable
                if all(a.intention is Intention.WAIT for a in s)
            ]
            assert wait_sets, (
                f"Wait+Wait set missing for onions_in_pot={onions}"
            )


class TestAllZeroScores:
    """
    The all-zero-score case: every active argument (Wait and non-Wait alike)
    scores exactly 0.0. This is reachable in practice (e.g. weights near zero
    at init, or a score driven to the >=0 clamp). It probes the true content
    of the non-emptiness theorem and whether the profile 'falls back to Wait'.
    """

    def _all_zero_exclusive_state(self) -> OvercookedAFState:
        """
        One pot with exactly one onion slot left: FetchOnion is both ACTIVE
        (slot open, agents empty-handed) and EXCLUSIVE (only one agent is
        useful), so FetchOnion(0) and FetchOnion(1) inter-attack.
        """
        return OvercookedAFState(
            agent_positions=[(0, 0), (0, 1)],
            agent_holding=[None, None],
            pot_states=[PotState(onion_count=2, is_cooking=False, is_ready=False)],
            num_agents_assigned={i: 0 for i in Intention},
            agent_steps_to_target=[
                {i: 10 for i in Intention},
                {i: 10 for i in Intention},
            ],
        )

    def test_all_zero_still_non_empty_and_contains_wait_wait(self):
        """Non-emptiness holds under all-zero scores, and {Wait, Wait} is among
        the acceptable sets (independent of any score comparison)."""
        state = self._all_zero_exclusive_state()
        assert state.total_onions_needed == 1  # FetchOnion exclusive

        args = generate_arguments(state, num_agents=2)
        for a in args:
            a.score = 0.0
        # Sanity: both agents have exactly {FetchOnion, Wait} active
        active = {(a.agent_id, a.intention) for a in args}
        assert active == {
            (0, Intention.FETCH_ONION), (0, Intention.WAIT),
            (1, Intention.FETCH_ONION), (1, Intention.WAIT),
        }

        attacks = generate_attacks_with_state(args, state)
        successful = resolve_vaf(args, attacks)
        acceptable = compute_acceptable_sets(args, successful, num_agents=2)

        assert acceptable, "acceptable set empty under all-zero scores"
        assert any(
            all(a.intention is Intention.WAIT for a in s) for s in acceptable
        ), "{Wait, Wait} must be present even when all scores tie at 0.0"

    def test_all_zero_aggressive_does_not_fall_back_to_wait(self):
        """HONEST NUANCE: with all scores tied at 0.0, the aggressive strategy
        does NOT return {Wait, Wait}. Every acceptable set sums to 0.0, so
        max() returns the first in enumeration order; because the Intention
        enum lists WAIT last, a FetchOnion-bearing set is returned first.
        Non-emptiness != 'falls back to Wait'."""
        state = self._all_zero_exclusive_state()
        af = ArgumentationFramework(num_agents=2, strategy=Strategy.AGGRESSIVE)
        profile = af.run(state, lambda arg, s: 0.0)

        # A non-Wait intention IS assigned, so this is not a Wait fallback.
        assert Intention.FETCH_ONION in profile.values()
        assert profile == {0: Intention.FETCH_ONION, 1: Intention.WAIT}

    def test_all_zero_conservative_falls_back_to_wait(self):
        """The conservative strategy DOES return {Wait, Wait} here: Wait appears
        in more acceptable sets (2 of 3) than either FetchOnion (1 of 3), so it
        wins on appearance count before the score tie-break is ever reached.
        Whether the profile 'falls back to Wait' is therefore strategy-dependent,
        not a property of the theorem."""
        state = self._all_zero_exclusive_state()
        af = ArgumentationFramework(num_agents=2, strategy=Strategy.CONSERVATIVE)
        profile = af.run(state, lambda arg, s: 0.0)

        assert profile == {0: Intention.WAIT, 1: Intention.WAIT}


# ---------------------------------------------------------------------------
# Profile selection tests
# ---------------------------------------------------------------------------

class TestProfileSelection:
    def _make_set(self, i0: Intention, s0: float, i1: Intention, s1: float):
        return [
            Argument(agent_id=0, intention=i0, score=s0),
            Argument(agent_id=1, intention=i1, score=s1),
        ]

    def test_aggressive_picks_max_total_score(self):
        set_a = self._make_set(Intention.FETCH_ONION, 0.9, Intention.WAIT, 0.0)  # total 0.9
        set_b = self._make_set(Intention.WAIT, 0.0, Intention.WAIT, 0.0)          # total 0.0
        profile = select_profile([set_a, set_b], Strategy.AGGRESSIVE)
        assert profile[0] is Intention.FETCH_ONION
        assert profile[1] is Intention.WAIT

    def test_conservative_picks_most_frequent(self):
        # FetchOnion(0) appears in 2 of 2 sets; Wait(0) appears in 0 of 2 sets
        set_a = self._make_set(Intention.FETCH_ONION, 0.5, Intention.WAIT, 0.0)
        set_b = self._make_set(Intention.FETCH_ONION, 0.5, Intention.FETCH_DISH, 0.8)
        profile = select_profile([set_a, set_b], Strategy.CONSERVATIVE)
        assert profile[0] is Intention.FETCH_ONION  # appears in 2/2

    def test_empty_sets_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            select_profile([], Strategy.AGGRESSIVE)


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    def test_wait_always_zero(self):
        state = _base_state()
        arg = Argument(agent_id=0, intention=Intention.WAIT)
        for w in [_weights(1, 1, 1), _weights(0, 0, 0), _weights(99, 99, 99)]:
            assert score_argument(arg, state, w) == 0.0

    def test_score_decreases_with_distance(self):
        """Closer agent should score higher (w1 drives 1/d term)."""
        state_close = _base_state(
            agent_steps_to_target=[
                {Intention.FETCH_ONION: 2}, {Intention.FETCH_ONION: 20}
            ]
        )
        state_far = _base_state(
            agent_steps_to_target=[
                {Intention.FETCH_ONION: 20}, {Intention.FETCH_ONION: 2}
            ]
        )
        arg = Argument(agent_id=0, intention=Intention.FETCH_ONION)
        w = _weights(1.0, 0.0, 0.0)
        assert score_argument(arg, state_close, w) > score_argument(arg, state_far, w)

    def test_score_clamped_to_zero(self):
        """Congestion (high w3, high rho) must not produce negative scores."""
        state = _base_state(
            agent_steps_to_target=[
                {Intention.FETCH_ONION: 100}, {Intention.FETCH_ONION: 100}
            ],
            num_agents_assigned={Intention.FETCH_ONION: 10},
        )
        arg = Argument(agent_id=0, intention=Intention.FETCH_ONION)
        w = _weights(0.0, 0.0, 99.0)  # massive congestion penalty
        score = score_argument(arg, state, w)
        assert score >= 0.0, f"Score went negative: {score}"

    def test_dependency_count_ordering(self):
        """FetchOnion (r=3) should outscore FetchDish (r=2) when r dominates."""
        state = _base_state(
            onions_in_pot=3,
            agent_holding=[None, None],
            agent_steps_to_target=[
                {
                    Intention.FETCH_ONION: 10,
                    Intention.FETCH_DISH: 10,
                    Intention.WAIT: 10,
                },
                {Intention.FETCH_ONION: 10},
            ],
            num_agents_assigned={Intention.FETCH_DISH: 0},
        )
        fo = Argument(agent_id=0, intention=Intention.FETCH_ONION)
        fd = Argument(agent_id=0, intention=Intention.FETCH_DISH)
        w = _weights(0.0, 1.0, 0.0)  # only r term matters
        assert score_argument(fo, state, w) > score_argument(fd, state, w)

    def test_dependency_counts_correct(self):
        assert DEPENDENCY_COUNT[Intention.FETCH_ONION] == 3
        assert DEPENDENCY_COUNT[Intention.FETCH_DISH] == 2
        assert DEPENDENCY_COUNT[Intention.PICKUP_SOUP] == 1
        assert DEPENDENCY_COUNT[Intention.DELIVER_SOUP] == 0
        assert DEPENDENCY_COUNT[Intention.WAIT] == 0


# ---------------------------------------------------------------------------
# End-to-end ArgumentationFramework tests
# ---------------------------------------------------------------------------

class TestAF:
    def test_af_returns_profile_for_every_agent(self):
        state = _base_state(onions_in_pot=0, agent_holding=[None, None])
        af = ArgumentationFramework(num_agents=2, strategy=Strategy.AGGRESSIVE)
        w = _weights(1.0, 1.0, 0.5)
        profile = af.run(state, lambda arg, s: score_argument(arg, s, w))
        assert set(profile.keys()) == {0, 1}
        for intention in profile.values():
            assert isinstance(intention, Intention)

    def test_af_profile_covers_every_agent(self):
        """run() must return one Intention per agent regardless of state."""
        state = _base_state(
            onions_in_pot=3,
            soup_is_ready=False,
            agent_holding=[None, None],
            num_agents_assigned={Intention.FETCH_DISH: 0},
        )
        af = ArgumentationFramework(num_agents=2, strategy=Strategy.AGGRESSIVE)
        w = _weights(1.0, 1.0, 0.5)
        profile = af.run(state, lambda arg, s: score_argument(arg, s, w))
        assert set(profile.keys()) == {0, 1}
        assert all(isinstance(v, Intention) for v in profile.values())
