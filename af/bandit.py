"""
Outcome-driven intention biases for argument scoring.

BanditWeights keeps one bias per intention and updates it from completion
outcomes collected during an episode. The bias is added to the argument score:

    score(A) = base_score(A) + theta[A.intention]

Wait's bias is fixed at zero, providing a stable reference point for the other
intentions. The environment records whether each adopted intention was completed
before reassignment and passes these outcomes to update() once per episode.

Update, batched once per episode from the per-arm outcome lists:

    advantage   = mean(outcomes[k]) - baseline[k]
    theta[k]    = clip(theta[k] + lr*advantage, theta_min, theta_max)
    baseline[k] = (1 - beta_b)*baseline[k] + beta_b*mean(outcomes[k])
    theta[WAIT] = 0.0

Outcome convention, assembled by OvercookedAFEnv:
    +1  the assigned intention completed before the adoption ended
    -1  adopted but not completed before it was reassigned
Wait is never credited or debited.

The module depends only on numpy and the Intention enum, so it is testable
without Overcooked or Stable-Baselines3.
"""

from __future__ import annotations

import numpy as np

from af.framework import Intention

DEFAULT_LR = 0.05
DEFAULT_THETA_MIN = -1.0
DEFAULT_THETA_MAX = 1.0
DEFAULT_BETA_BASELINE = 0.1


class BanditWeights:
    """Maintain and update a bounded bias for each intention."""

    def __init__(
        self,
        arms: list[Intention] | None = None,
        lr: float = DEFAULT_LR,
        theta_min: float = DEFAULT_THETA_MIN,
        theta_max: float = DEFAULT_THETA_MAX,
        beta_baseline: float = DEFAULT_BETA_BASELINE,
    ) -> None:
        self.arms = list(arms) if arms is not None else list(Intention)
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("Bandit arms must be unique")
        self._index = {a: i for i, a in enumerate(self.arms)}
        if Intention.WAIT not in self._index:
            raise ValueError("WAIT must be one of the bandit arms (it is pinned to 0.0)")
        self._wait_idx = self._index[Intention.WAIT]

        if lr < 0:
            raise ValueError("lr must be non-negative")
        if theta_min > theta_max:
            raise ValueError("theta_min must not exceed theta_max")
        if not 0.0 <= beta_baseline <= 1.0:
            raise ValueError("beta_baseline must be between 0 and 1")

        self.lr = float(lr)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.beta_baseline = float(beta_baseline)

        n = len(self.arms)
        self.theta = np.zeros(n, dtype=np.float64)
        self.baseline = np.zeros(n, dtype=np.float64)

    def bias(self, intention: Intention) -> float:
        """Return theta[intention]; 0.0 for arms this bandit does not track."""
        idx = self._index.get(intention)
        return 0.0 if idx is None else float(self.theta[idx])

    def update(self, outcomes: dict[Intention, list[float]]) -> None:
        """Update tracked arms from one episode's completion outcomes."""
        for intention, outs in outcomes.items():
            if not outs:
                continue
            if intention not in self._index:
                raise ValueError(f"Outcome provided for untracked arm: {intention!r}")
            if intention is Intention.WAIT:
                continue  # never credited or debited
            k = self._index[intention]
            mean_out = float(np.mean(outs))
            advantage = mean_out - self.baseline[k]
            self.theta[k] = float(
                np.clip(self.theta[k] + self.lr * advantage, self.theta_min, self.theta_max)
            )
            self.baseline[k] = (
                (1.0 - self.beta_baseline) * self.baseline[k]
                + self.beta_baseline * mean_out
            )
        self.theta[self._wait_idx] = 0.0  # pinned, always

    def theta_dict(self) -> dict[Intention, float]:
        """{Intention: theta} snapshot (for logging)."""
        return {a: float(self.theta[i]) for a, i in self._index.items()}
