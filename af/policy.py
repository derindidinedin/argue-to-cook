"""
Stable-Baselines3 ActorCriticPolicy with an argument-weight network.
See af/scoring.py for the scoring function.

The weight network is a small MLP on the policy's features. It outputs the three
scoring weights w1, w2 and w3, passed through a softplus so they are never
negative. They reach the environment through a shared WeightStore.

ArgRL-Full, ArgRL-Obs and ArgRL-Shape use this policy.

The weight head receives no gradient from task performance. The only gradient it
receives is a penalty on the size of its own outputs, set by WEIGHT_REG_COEF.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


# ---------------------------------------------------------------------------
# Shared mutable weight container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WeightStore:
    """
    Holds the three scoring weights and is shared by the policy and the env.

    Both are handed the same object, so a write on one side is a read on the
    other. That only works inside one process, which is why training does not use
    SubprocVecEnv (analysis/train.py).

    The values start at one, which is what the ArgRL-Frozen condition uses, and
    never go negative, since the policy applies softplus.
    """
    weights: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float64)
    )


# ---------------------------------------------------------------------------
# Custom policy
# ---------------------------------------------------------------------------

class ArgumentWeightPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy augmented with the argument weight network.

    Architecture::

        obs --> features_extractor --> features
                                        |          |
                                 mlp_extractor   weight_head (small MLP)
                                  /       \           |
                           action_net  value_net   softplus
                                |          |           |
                             actions    values    w1, w2, w3

    The weight branch is a separate output and does not feed the action or value
    networks, so it cannot directly affect the action selected in the same
    forward pass. Its outputs are written to the shared WeightStore and can
    influence later intention profiles, observations and shaping rewards. During
    forward(), which Stable-Baselines3 uses for rollout collection, the branch
    runs without gradient tracking.

    Parameters:
    weight_store : WeightStore | None
        Shared container to write computed weights into after each forward pass.
        If None, weights are computed but not shared (for use without AF env).
    """

    # L2 penalty on the weights, folded into the entropy term.
    # Every run uses an entropy coefficient of 0.01, so the penalty enters the
    # PPO loss at 0.01 * 1e-4 = 1e-6.
    WEIGHT_REG_COEF: float = 1e-4

    def __init__(
        self,
        *args,
        weight_store: WeightStore | None = None,
        **kwargs,
    ) -> None:
        self._weight_store = weight_store
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        # Build the extra head before super(), which makes the optimiser out of
        # self.parameters(). Anything added later would be left out of it
        self.weight_head_mlp = th.nn.Sequential(
            th.nn.Linear(self.features_dim, 32),
            th.nn.Tanh(),
            th.nn.Linear(32, 3),
        ).to(self.device)
        super()._build(lr_schedule)

    # ------------------------------------------------------------------
    # Weight utilities
    # ------------------------------------------------------------------

    def _weight_head_output(self, features: th.Tensor) -> th.Tensor:
        """Return the three weights for each item in the batch. Never negative."""
        return F.softplus(self.weight_head_mlp(features))

    def _write_to_store(self, weights_tensor: th.Tensor) -> None:
        """Average the weights over the batch and put them in the shared store."""
        if self._weight_store is not None:
            w = weights_tensor.detach().mean(dim=0).cpu().numpy().astype(np.float64)
            self._weight_store.weights = w

    # ------------------------------------------------------------------
    # Overridden forward passes
    # ------------------------------------------------------------------

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Actor-critic forward pass.

        Also runs the weight network and writes the result to WeightStore,
        so OvercookedAFEnv reads current values at the next framework trigger.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
            weight_features = features
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            weight_features = pi_features

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        with th.no_grad():
            weights = self._weight_head_output(weight_features)
        self._write_to_store(weights)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        """
        Evaluate actions during PPO gradient updates.

        The weight network runs with gradient tracking and writes its outputs to
        the shared store. The environment does not step during these updates, so
        framework triggers see the current store values once rollout collection
        resumes.

        An L2 penalty on the weight outputs is subtracted from the entropy. This
        is the only gradient received by the weight network.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
            weight_features = features
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            weight_features = pi_features

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        # Unlike in forward(), the weight network is part of the graph here
        weights = self._weight_head_output(weight_features)  # (batch, 3)

        # L2 penalty on the weights, subtracted from the entropy
        if entropy is not None:
            weight_reg = self.WEIGHT_REG_COEF * weights.pow(2).sum(dim=-1)
            entropy = entropy - weight_reg

        self._write_to_store(weights)

        return values, log_prob, entropy

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        # WeightStore is not serialisable.
        # The caller re-supplies it on load.
        data["weight_store"] = None
        return data
