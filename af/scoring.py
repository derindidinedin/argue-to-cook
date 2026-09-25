"""
Argument scoring for the value-based argumentation framework.

Each active argument receives the score

    sigma = max(0, w1 * (1 / d) + w2 * c - w3 * rho)

where

  d     Planner step count to the argument's target, at least 1, so the w1 term
        rewards proximity.
  c     Downstream subtasks that depend on the intention: 3 for FetchOnion, 2 for
        FetchDish, 1 for PickupSoup, and 0 for DeliverSoup.
  rho   Agents already assigned the same intention, so a crowded intention scores
        lower.

Wait has no target, so it scores zero without going through the formula. A
negative score is set to zero, so Wait ties with any argument whose penalty
outweighs its benefit.

Weights arrive from the caller as a plain numpy array [w1, w2, w3]. The optional
bias is used only by the intention bias mechanism.
"""

from __future__ import annotations

import numpy as np

from af.framework import Argument, Intention, OvercookedAFState

# Downstream dependency count c(intention)
# = number of uncompleted subtasks that depend on this intention completing
DEPENDENCY_COUNT: dict[Intention, int] = {
    Intention.FETCH_ONION:  3,
    Intention.FETCH_DISH:   2,
    Intention.PICKUP_SOUP:  1,
    Intention.DELIVER_SOUP: 0,
    Intention.WAIT:         0,
}

_DEFAULT_LARGE_D = 100  # fallback if steps_to_target entry is missing


def score_argument(
    arg: Argument,
    state: OvercookedAFState,
    weights: np.ndarray,
    bias: float = 0.0,
) -> float:
    """
    Calculate an argument's non-negative score.

    Parameters
    ----------
    arg
        The argument to score.
    state
        The framework's view of the current game state.
    weights
        Non-negative array of [w1, w2, w3].
    bias
        Optional per-intention bias, used only by the intention bias mechanism. Wait
        ignores it.

    Returns
    -------
    float
        The score, never below zero.
    """
    if arg.intention is Intention.WAIT:
        return 0.0

    w1, w2, w3 = map(float, weights)

    agent_distances = state.agent_steps_to_target[arg.agent_id]
    d = max(agent_distances.get(arg.intention, _DEFAULT_LARGE_D), 1)
    c = DEPENDENCY_COUNT[arg.intention]
    # counted from the profile of the previous trigger, this agent included when
    # it held the intention itself
    rho = state.num_agents_assigned.get(arg.intention, 0)

    sigma = w1 * (1.0 / d) + w2 * c - w3 * rho + float(bias)
    return max(0.0, sigma)
