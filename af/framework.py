"""
Core argumentation framework for multi-agent Overcooked coordination.

Implements a Value-based Argumentation Framework (VAF) following
Bench-Capon (2003). Arguments are created from the current state, and the
attack relations are computed from that same state. Nothing here imports
overcooked_ai. All state arrives through OvercookedAFState, so the framework
can be run on its own without an Overcooked environment.

Order of operations:
  generate_arguments -> generate_attacks_with_state -> (set scores externally)
  -> resolve_vaf -> compute_acceptable_sets -> select_profile
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from enum import Enum, auto
from itertools import product
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Intention(Enum):
    FETCH_ONION = auto()
    FETCH_DISH = auto()
    PICKUP_SOUP = auto()
    DELIVER_SOUP = auto()
    WAIT = auto()


class Strategy(Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PotState:
    """State of a single pot. A layout may have any number of pots."""
    onion_count: int      # 0..3 ingredients currently in the pot
    is_cooking: bool      # 3 onions in, timer running, not yet ready
    is_ready: bool        # cooking finished, soup ready to be plated


@dataclasses.dataclass
class OvercookedAFState:
    """
    State information used by the argumentation framework and scoring.

    A layout may hold any number of pots, so preconditions must never assume
    there is exactly one.
    """
    agent_positions: List[Tuple[int, int]]      # retained for diagnostics, currently unused
    agent_holding: List[Optional[str]]          # None | 'onion' | 'dish' | 'soup'
    pot_states: List[PotState]
    num_agents_assigned: Dict[Intention, int]   # from the previous framework run
    agent_steps_to_target: List[Dict[Intention, int]]   # planner steps, used by scoring

    @property
    def any_pot_ready(self) -> bool:
        """True if any pot has a fully cooked soup ready to plate."""
        return any(p.is_ready for p in self.pot_states)

    @property
    def any_pot_needs_onions(self) -> bool:
        """True if any pot has fewer than 3 onions and isn't cooking/ready."""
        return any(
            p.onion_count < 3 and not p.is_cooking and not p.is_ready
            for p in self.pot_states
        )

    @property
    def any_pot_full(self) -> bool:
        """True if any pot has all 3 onions in (cooking or ready)."""
        return any(p.onion_count >= 3 for p in self.pot_states)

    @property
    def total_onions_needed(self) -> int:
        """Total onion slots still open across pots not cooking/ready."""
        return sum(
            3 - p.onion_count
            for p in self.pot_states
            if not p.is_cooking and not p.is_ready
        )

    @property
    def num_ready_pots(self) -> int:
        return sum(1 for p in self.pot_states if p.is_ready)


# ---------------------------------------------------------------------------
# Argument
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Argument:
    agent_id: int
    intention: Intention
    score: float = 0.0          # set by scoring module before resolve_vaf

    def __hash__(self):
        return hash((self.agent_id, self.intention))

    def __eq__(self, other):
        if not isinstance(other, Argument):
            return NotImplemented
        return self.agent_id == other.agent_id and self.intention == other.intention

    def __repr__(self):
        return f"Arg({self.intention.name}, agent={self.agent_id}, score={self.score:.3f})"


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def check_preconditions(
    intention: Intention,
    agent_id: int,
    state: OvercookedAFState,
) -> bool:
    holding = state.agent_holding[agent_id]
    assigned = state.num_agents_assigned

    if intention is Intention.WAIT:
        return True

    if intention is Intention.FETCH_ONION:
        return (
            state.any_pot_needs_onions
            and state.total_onions_needed > assigned.get(Intention.FETCH_ONION, 0)
            and holding is None
        )

    if intention is Intention.FETCH_DISH:
        return (
            state.any_pot_full
            and assigned.get(Intention.FETCH_DISH, 0) == 0
            and holding is None
        )

    if intention is Intention.PICKUP_SOUP:
        return holding == "dish" and state.any_pot_ready

    if intention is Intention.DELIVER_SOUP:
        return holding == "soup"

    return False  # unreachable


# ---------------------------------------------------------------------------
# Argument generation
# ---------------------------------------------------------------------------

def generate_arguments(
    state: OvercookedAFState,
    num_agents: int,
) -> List[Argument]:
    """Return all active Argument objects for the current state."""
    args: List[Argument] = []
    for agent_id in range(num_agents):
        for intention in Intention:
            if check_preconditions(intention, agent_id, state):
                args.append(Argument(agent_id=agent_id, intention=intention))
    return args


# ---------------------------------------------------------------------------
# Attack generation
# ---------------------------------------------------------------------------

Attack = Tuple[Argument, Argument]  # (attacker, target)


def _is_exclusive_inter_agent(intention: Intention, state: OvercookedAFState) -> bool:
    """
    Return True when only one agent can usefully pursue this intention
    simultaneously - i.e., the precondition is mutually exclusive between agents.

    FetchOnion: only when exactly 1 onion slot remains across all pots (so
                two agents would both want the slot but only one is needed).
    FetchDish:  always exclusive - precondition blocks a second assignment.
    PickupSoup: exclusive only when fewer than 2 pots are ready - with 2+
                ready pots, two agents can each plate a different soup.
    DeliverSoup: each agent holding soup can deliver independently - not exclusive.
    Wait:       never exclusive.
    """
    if intention is Intention.FETCH_ONION:
        return state.total_onions_needed == 1
    if intention is Intention.FETCH_DISH:
        return True
    if intention is Intention.PICKUP_SOUP:
        return state.num_ready_pots < 2
    if intention is Intention.WAIT:
        # WAIT is never inter-agent-exclusive. This is a load-bearing invariant:
        # the "acceptable set is never empty" theorem relies on Wait(A) and
        # Wait(B) never inter-attacking. Enforced explicitly rather than by
        # fall-through so a future edit cannot silently break the theorem.
        return False
    return False


def generate_attacks_with_state(
    arguments: List[Argument],
    state: OvercookedAFState,
) -> List[Attack]:
    """
    Generate intra-agent and inter-agent attacks.

    Intra-agent: all active arguments for the same agent mutually attack each
                 other (an agent can only pursue one intention at a time).
    Inter-agent: arguments from different agents attack each other iff they
                 share the same intention AND _is_exclusive_inter_agent returns
                 True for that intention in the current state.
    """
    attacks: List[Attack] = []
    n = len(arguments)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = arguments[i], arguments[j]

            # Intra-agent
            if a.agent_id == b.agent_id and a.intention is not b.intention:
                attacks.append((a, b))
                continue

            # Inter-agent
            if (
                a.agent_id != b.agent_id
                and a.intention is b.intention
                and _is_exclusive_inter_agent(a.intention, state)
            ):
                attacks.append((a, b))

    return attacks


# ---------------------------------------------------------------------------
# VAF resolution
# ---------------------------------------------------------------------------

def resolve_vaf(
    arguments: List[Argument],
    attacks: List[Attack],
) -> List[Attack]:
    """
    Apply VAF semantics (Bench-Capon 2003).

    In a mutual attack between arguments A and B:
      - if score(A) > score(B): A defeats B, so attack (A->B) succeeds and (B->A) fails
      - if score(B) > score(A): B defeats A
      - if score(A) == score(B): both attacks succeed (genuine conflict, both eliminated)

    Returns only the successful (defeating) attacks.
    """
    attack_set: Set[Tuple[int, int]] = {
        (id(a), id(b)) for a, b in attacks
    }

    successful: List[Attack] = []

    for attacker, target in attacks:
        reverse = (id(target), id(attacker))
        if reverse in attack_set:
            # Mutual attack: attacker succeeds if its score >= target's score.
            # When scores are equal this makes both directions succeed, which is
            # the correct VAF behaviour, since a genuine conflict eliminates both.
            if attacker.score >= target.score:
                successful.append((attacker, target))
        else:
            # One-directional attack always succeeds
            successful.append((attacker, target))

    return successful


# ---------------------------------------------------------------------------
# Acceptable sets
# ---------------------------------------------------------------------------

def compute_acceptable_sets(
    arguments: List[Argument],
    successful_attacks: List[Attack],
    num_agents: int,
) -> List[List[Argument]]:
    """
    Enumerate admissible sets of one argument per agent.

    Admissibility is checked against INTER-AGENT successful attacks only.
    Intra-agent conflicts are resolved by the VAF step; since each candidate
    set contains exactly one argument per agent, intra-agent conflicts never
    arise within a candidate. Applying intra-agent defeats here would wrongly
    eliminate Wait (which always scores 0 and loses to any non-Wait argument
    of the same agent), violating the Wait-always-survives theorem.

    A set S is acceptable iff:
    - Conflict-free (inter-agent): no arg in S is inter-agent defeated by
      another arg in S.
    - Self-defending (inter-agent): for every inter-agent attacker B of A
      in S where B is outside S, some arg in S inter-agent defeats B.
    """
    # Only inter-agent successful attacks matter for acceptable-set admissibility
    inter_attacks = [
        (a, t) for a, t in successful_attacks if a.agent_id != t.agent_id
    ]

    # Group arguments by agent
    by_agent: Dict[int, List[Argument]] = {i: [] for i in range(num_agents)}
    for arg in arguments:
        by_agent[arg.agent_id].append(arg)

    # Index for fast lookup
    inter_attacked_by: Dict[Argument, Set[Argument]] = {arg: set() for arg in arguments}
    inter_defeats: Dict[Argument, Set[Argument]] = {arg: set() for arg in arguments}
    for attacker, target in inter_attacks:
        inter_attacked_by[target].add(attacker)
        inter_defeats[attacker].add(target)

    def is_valid_set(candidate: List[Argument]) -> bool:
        cset = set(candidate)

        # Conflict-free: no member inter-defeats another member
        for attacker, target in inter_attacks:
            if attacker in cset and target in cset:
                return False

        # Self-defending: every external inter-agent attacker must be
        # counter-defeated by something inside the set
        for arg in candidate:
            for attacker in inter_attacked_by[arg]:
                if attacker not in cset:
                    if not any(attacker in inter_defeats[c] for c in candidate):
                        return False

        return True

    # Generate all combinations (one arg per agent)
    choices = [by_agent[i] for i in range(num_agents)]
    return [list(c) for c in product(*choices) if is_valid_set(list(c))]


# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------

def select_profile(
    acceptable_sets: List[List[Argument]],
    strategy: Strategy = Strategy.AGGRESSIVE,
) -> Dict[int, Intention]:
    """
    Choose one acceptable set and return the intention profile.

    Aggressive:   pick the set with highest total argument score.
    Conservative: per agent, pick the argument that appears in the most
                  acceptable sets; tie-break by argument score.
    """
    if not acceptable_sets:
        raise ValueError("No acceptable sets - Wait theorem violated")

    if strategy is Strategy.AGGRESSIVE:
        best = max(acceptable_sets, key=lambda s: sum(a.score for a in s))
        return {a.agent_id: a.intention for a in best}

    # Conservative
    appearance: Counter[Argument] = Counter()
    for s in acceptable_sets:
        for arg in s:
            appearance[arg] += 1

    # Determine num_agents from the sets
    num_agents = len(acceptable_sets[0])
    profile: Dict[int, Intention] = {}
    for agent_id in range(num_agents):
        # dict.fromkeys deduplicates while keeping enumeration order, so ties
        # in the key below always break the same way from run to run
        candidates = dict.fromkeys(
            arg
            for s in acceptable_sets
            for arg in s
            if arg.agent_id == agent_id
        )
        best_arg = max(candidates, key=lambda a: (appearance[a], a.score))
        profile[agent_id] = best_arg.intention

    return profile


# ---------------------------------------------------------------------------
# Top-level class
# ---------------------------------------------------------------------------

class ArgumentationFramework:
    """
    Wraps the whole sequence above.

    Usage:
        af = ArgumentationFramework(num_agents=2, strategy=Strategy.AGGRESSIVE)
        # Before calling run(), the caller must populate
        # state.agent_steps_to_target with planner distances and
        # supply a scoring function so scores can be set on Arguments.
        profile = af.run(state, score_fn)

    score_fn: callable(arg: Argument, state: OvercookedAFState) -> float
    """

    def __init__(
        self,
        num_agents: int = 2,
        strategy: Strategy = Strategy.AGGRESSIVE,
    ):
        self.num_agents = num_agents
        self.strategy = strategy

    def run(
        self,
        state: OvercookedAFState,
        score_fn,
    ) -> Dict[int, Intention]:
        """
        Run the whole sequence and return the intention profile.

        Returns {agent_id: Intention} mapping.
        Guaranteed to be non-empty (Wait theorem).
        """
        arguments = generate_arguments(state, self.num_agents)

        # Set scores on Argument objects in-place
        for arg in arguments:
            arg.score = score_fn(arg, state)

        attacks = generate_attacks_with_state(arguments, state)
        successful = resolve_vaf(arguments, attacks)
        acceptable_sets = compute_acceptable_sets(arguments, successful, self.num_agents)

        return select_profile(acceptable_sets, self.strategy)
