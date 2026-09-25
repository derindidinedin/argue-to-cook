"""
Potential-based reward shaping for the Overcooked argumentation framework.
See af/scoring.py for the scoring function.

Phi scores the current intention profile as a single scalar, recomputed at every
time step. It is the sum of three components, each of which can be removed
independently for ablation.

    Phi(s) = phi_purpose(s) + phi_commit(s) + phi_ready(s)

  phi_purpose  Individual purpose (_individual, ablation key 'individual'). Mean
               over agents of proximity to the assigned target, 1 at the target
               and falling to 0 at ten steps away. Wait contributes 0.
  phi_commit   Coalition commitment (_coalition, key 'coalition'). 1.0 if both
               agents hold distinct intentions other than Wait, 0.5 if exactly
               one does, 0.0 otherwise.
  phi_ready    Load-ready bonus (_load_ready, key 'load_ready'). Fraction of
               agents holding a dish while a pot is ready.

The potential evaluates the profile that scoring selected. It does not sum the
argument scores used to select it.

The term added to the environment reward is gamma * Phi(s') - Phi(s), following
Ng et al. (1999), with gamma the agent's discount factor. Because the scoring
weights change during training, the mapping from state to profile can also
change, making this dynamic potential-based shaping in the sense of Devlin and
Kudenko (2012). Section 4.4.2 of the thesis gives the invariance argument.
"""

from __future__ import annotations

from typing import Dict, Set

from af.framework import Intention, OvercookedAFState

# Distance at which proximity reaches zero. A fallback distance passed in for an
# unreachable target should be at least this, so that it contributes nothing.
MAX_STEPS: int = 10

_VALID_ABLATIONS: Set[str] = {"individual", "coalition", "load_ready"}


class PotentialFunction:
    """
    Compute Phi(s) for a state and an intention profile.

    Parameters
    ----------
    num_agents
        Number of agents.
    alpha, beta, delta
        Coefficients on the three components. All experiments use the unit
        defaults.
    ablate
        Component names to remove. Ablating all three makes phi() return 0.0 in
        every state, which is how the reward-shaping channel is switched off.
    """

    def __init__(
        self,
        num_agents: int = 2,
        alpha: float = 1.0,
        beta: float = 1.0,
        delta: float = 1.0,
        ablate: Set[str] | None = None,
    ) -> None:
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.ablate: Set[str] = set() if ablate is None else set(ablate)

        unknown = self.ablate - _VALID_ABLATIONS
        if unknown:
            raise ValueError(
                f"Unknown ablation keys: {unknown!r}. Valid: {_VALID_ABLATIONS!r}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def phi(
        self,
        state: OvercookedAFState,
        profile: Dict[int, Intention],
    ) -> float:
        """Return Phi(s) for the given state and intention profile."""
        value = 0.0
        if "individual" not in self.ablate:
            value += self.alpha * self._individual(state, profile)
        if "coalition" not in self.ablate:
            value += self.beta * self._coalition(profile)
        if "load_ready" not in self.ablate:
            value += self.delta * self._load_ready(state)
        return value

    def shaping_bonus(
        self,
        state: OvercookedAFState,
        next_state: OvercookedAFState,
        profile: Dict[int, Intention],
        next_profile: Dict[int, Intention],
        gamma: float = 0.99,
    ) -> float:
        """
        Return the shaping term gamma * Phi(s') - Phi(s).

        gamma must be the agent's discount factor. The training environment
        computes the same expression inline (af/env_wrapper.py), since it
        already holds Phi(s) from the previous step.
        """
        return gamma * self.phi(next_state, next_profile) - self.phi(state, profile)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _individual(
        self,
        state: OvercookedAFState,
        profile: Dict[int, Intention],
    ) -> float:
        """
        Individual purpose, phi_purpose.

            phi_purpose = mean_i( max(0, 1 - steps_i / MAX_STEPS) )

        Wait contributes 0. The mean is over all agents, not only the assigned
        ones, so an idle agent lowers the component.
        """
        total = 0.0
        for agent_id in range(self.num_agents):
            intention = profile.get(agent_id, Intention.WAIT)
            if intention is Intention.WAIT:
                continue
            d = state.agent_steps_to_target[agent_id].get(intention, MAX_STEPS)
            total += max(0.0, 1.0 - d / MAX_STEPS)
        return total / self.num_agents

    def _coalition(self, profile: Dict[int, Intention]) -> float:
        """
        Coalition commitment, phi_commit, rewarding complementary assignment.

        1.0  every agent holds a non-Wait intention and all are distinct
        0.5  exactly one agent holds a non-Wait intention
        0.0  otherwise, meaning agents duplicate an intention or all wait
        """
        intentions = [profile.get(i, Intention.WAIT) for i in range(self.num_agents)]
        non_wait = [i for i in intentions if i is not Intention.WAIT]

        if len(non_wait) == self.num_agents and len(set(non_wait)) == self.num_agents:
            return 1.0
        if len(non_wait) == 1:
            return 0.5
        return 0.0

    def _load_ready(self, state: OvercookedAFState) -> float:
        """
        Load-ready bonus, phi_ready.

        The fraction of agents holding a dish while a pot is ready. This offsets
        the penalty in phi_purpose that an agent incurs by fetching a dish
        before the soup has finished cooking.
        """
        if not state.any_pot_ready:
            return 0.0
        count = sum(1 for h in state.agent_holding if h == "dish")
        return count / self.num_agents
