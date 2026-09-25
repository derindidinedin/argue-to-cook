"""
Gymnasium wrappers for two-agent Overcooked training.

OvercookedGymEnv provides the base joint environment. OvercookedAFEnv adds an
intention profile and potential-based shaping. Both track soups delivered
separately from the training reward, and expose it in terminal step info so
callbacks can log it.

The base training reward is the sparse delivery reward plus Overcooked's own
intermediate rewards, which every condition receives. The framework's shaping
term is the only reward component that differs between conditions.

Observation size follows the layout. In Cramped Room it is 1040 without the
profile and 1050 with it. The action space is MultiDiscrete([6, 6]), one
discrete action per agent.
"""

import math

import numpy as np
import gymnasium
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MotionPlanner

from af.framework import (
    ArgumentationFramework,
    Intention,
    OvercookedAFState,
    PotState,
    Strategy,
)
from af.scoring import score_argument
from af.reward_shaping import PotentialFunction

DELIVERY_REWARD = 20  # value of each onion soup delivery in cramped_room

# Used when the planner finds no path (e.g. PickupSoup when no soup is ready).
# Must be >= reward_shaping.MAX_STEPS so the individual potential component
# clamps to 0, and >= 1 so the score formula 1/d never divides by zero.
_FALLBACK_STEPS = 20

# Ordered list of intentions used for one-hot profile encoding.
_INTENTION_LIST = [
    Intention.FETCH_ONION,
    Intention.FETCH_DISH,
    Intention.PICKUP_SOUP,
    Intention.DELIVER_SOUP,
    Intention.WAIT,
]
_INTENTION_INDEX = {intention: i for i, intention in enumerate(_INTENTION_LIST)}
_N_INTENTIONS = len(_INTENTION_LIST)


def _held_name(held_object) -> str | None:
    """Overcooked held object -> 'onion'|'dish'|'soup'|None."""
    return None if held_object is None else held_object.name


def _intention_completed(intention: Intention, prev_name, curr_name) -> bool:
    """
    True when the assigned agent's held-object transition this step satisfies
    the completion signature of ``intention``.

    FETCH_ONION  : nothing -> holding onion  (fetched an onion)
    FETCH_DISH   : nothing -> holding dish
    PICKUP_SOUP  : holding dish -> holding soup  (plated a soup)
    DELIVER_SOUP : holding soup -> nothing       (delivered)
    WAIT / other : never completes

    Attribution uses pickup for FETCH_ONION (rather than pot-placement) because
    the AF reassigns an onion-holding agent off FETCH_ONION at the next trigger,
    so pickup is the last step at which the FETCH_ONION assignment is still live.
    Evaluated against the PRE-trigger assignment, before the AF re-runs.
    """
    if intention is Intention.FETCH_ONION:
        return prev_name is None and curr_name == "onion"
    if intention is Intention.FETCH_DISH:
        return prev_name is None and curr_name == "dish"
    if intention is Intention.PICKUP_SOUP:
        return prev_name == "dish" and curr_name == "soup"
    if intention is Intention.DELIVER_SOUP:
        return prev_name == "soup" and curr_name is None
    return False


class OvercookedGymEnv(gymnasium.Env):
    """
    Two-agent Overcooked wrapper compatible with SB3 PPO.

    Both agents share a single joint policy: the policy receives the full
    flattened state and outputs a MultiDiscrete([6, 6]) joint action.
    """

    metadata = {"render_modes": []}

    def __init__(self, layout_name: str = "cramped_room", horizon: int = 400):
        super().__init__()
        self.layout_name = layout_name
        self.horizon = horizon

        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        # Observation: lossless state encoding stacked for all agents, then flattened.
        # Shape per agent: lossless_state_encoding_shape = [grid_w, grid_h, n_features]
        # With 2 agents: (2, grid_w, grid_h, n_features) flattened to a 1-D vector.
        enc_shape = self.mdp.get_lossless_state_encoding_shape()  # e.g. [5, 4, 26]
        n_agents = self.mdp.num_players                     # 2
        flat_obs_dim = n_agents * int(np.prod(enc_shape))

        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(flat_obs_dim,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [Action.NUM_ACTIONS] * n_agents, dtype=np.int64
        )

        # Episode accumulators - reset in reset()
        self._soups_delivered: int = 0
        self._ep_sparse_reward: float = 0.0
        self._ep_shaped_reward: float = 0.0
        self._ep_length: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env.reset()
        self._soups_delivered = 0
        self._ep_sparse_reward = 0.0
        self._ep_shaped_reward = 0.0
        self._ep_length = 0
        return self._get_obs(), {}

    def step(self, action):
        joint_action = tuple(Action.ALL_ACTIONS[int(a)] for a in action)
        _, sparse_reward, done, info = self.base_env.step(joint_action)

        sparse_reward = float(sparse_reward)
        shaped_reward = float(sum(info.get("shaped_r_by_agent", [0.0, 0.0])))
        soups_this_step = round(sparse_reward / DELIVERY_REWARD)

        self._soups_delivered += soups_this_step
        self._ep_sparse_reward += sparse_reward
        self._ep_shaped_reward += shaped_reward
        self._ep_length += 1

        training_reward = self._training_reward(sparse_reward, shaped_reward, info)

        obs = self._get_obs()
        terminated = False
        truncated = bool(done)

        step_info = {
            "sparse_reward": sparse_reward,
            "shaped_reward": shaped_reward,
            "soups_this_step": soups_this_step,
        }
        if done:
            step_info["soups_delivered"] = self._soups_delivered
            step_info["ep_sparse_reward"] = self._ep_sparse_reward
            step_info["ep_shaped_reward"] = self._ep_shaped_reward
            step_info["ep_length"] = self._ep_length

        return obs, training_reward, terminated, truncated, step_info

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _training_reward(
        self, sparse_reward: float, shaped_reward: float, info: dict
    ) -> float:
        """Return the reward signal used for training. Override in AF variants."""
        # shaped_reward = sum(info["shaped_r_by_agent"]) - Overcooked's built-in
        # heuristic shaping (+3 onion-in-pot, +5 dish-pickup / soup-plated).
        # Added to all conditions so PPO has a positive intermediate signal.
        return sparse_reward + shaped_reward

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        enc = self.mdp.lossless_state_encoding(self.base_env.state)
        return np.array(enc, dtype=np.float32).flatten()


# ---------------------------------------------------------------------------
# AF + shaping subclass
# ---------------------------------------------------------------------------

class OvercookedAFEnv(OvercookedGymEnv):
    """
    Overcooked environment with argumentation-based intention profiles.

    The framework runs on reset, whenever an agent's held object changes, and
    on every step for an agent assigned Wait. The resulting profile can be
    appended to the observation and turned into a potential-based shaping term,
    which is added to the base training reward.

    Parameters
    ----------
    strategy : str
        'aggressive' or 'conservative'.
    weights : np.ndarray, shape (3,)
        [w1, w2, w3] for the argument scoring function when weight_store is
        not provided. Must be non-negative. Defaults to ones.
    ablate : set of str
        Potential-function components to remove.
        Valid values: 'individual', 'coalition', 'load_ready'.
    weight_store : WeightStore | None
        Shared container produced by ArgumentWeightPolicy. When provided, the
        env reads weights from it at each framework trigger instead of using
        the fixed ``weights`` array. Pass the same object to the env and the
        policy so scoring sees current values during rollout collection.
    potential_scale : float
        Multiplier on the potential, and so on the shaping term.
    gamma : float
        Discount factor in the shaping term gamma * Phi(s') - Phi(s). Policy
        invariance (Ng et al. 1999) holds only when this is the discount factor
        the learner uses, so analysis/train.py passes PPO's gamma here.
    """

    def __init__(
        self,
        layout_name: str = "cramped_room",
        horizon: int = 400,
        strategy: str = "aggressive",
        weights: np.ndarray | None = None,
        ablate: set | None = None,
        weight_store=None,  # WeightStore | None - avoids circular import
        include_profile_in_obs: bool = True,
        bandit=None,  # af.bandit.BanditWeights | None
        potential_scale: float = 1.0,
        gamma: float = 0.99,
    ) -> None:
        # Initialise profile stub before super().__init__() so that _get_obs()
        # (which may be called during super's __init__ indirectly) is safe.
        self._profile: dict[int, Intention] = {}
        self._phi: float = 0.0
        self._af_trigger_count: int = 0
        self._intention_completion_count: int = 0
        # When False, the framework still runs and still shapes the reward,
        # but the intention profile is not appended to the observation. This
        # isolates the reward-shaping channel from the observation channel,
        # which is the argrl_shape condition.
        self._include_profile_in_obs = include_profile_in_obs

        # Bandit weight learning (optional). When set, the AF score gets a
        # per-intention theta bias and the env attributes per-argument outcomes
        # (adoption/completion) to update theta once per episode.
        self._bandit = bandit
        self._episode_outcomes: dict = {}
        self._adopted: dict[int, Intention] = {}
        self._adopted_completed: dict[int, bool] = {}

        super().__init__(layout_name=layout_name, horizon=horizon)

        n_agents = self.mdp.num_players
        self._profile = {i: Intention.WAIT for i in range(n_agents)}

        # Extend observation space only when the profile is exposed: base 1040
        # + profile one-hot 10 = 1050.  With the profile hidden, the space
        # stays at the ppo_baseline base dimension (1040).
        if self._include_profile_in_obs:
            base_dim = self.observation_space.shape[0]
            profile_dim = _N_INTENTIONS * n_agents
            self.observation_space = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(base_dim + profile_dim,),
                dtype=np.float32,
            )

        _strategy = (
            Strategy.AGGRESSIVE if strategy == "aggressive"
            else Strategy.CONSERVATIVE if strategy == "conservative"
            else None
        )
        if _strategy is None:
            raise ValueError(
                f"Unknown strategy {strategy!r}, expected 'aggressive' or 'conservative'"
            )
        self._af = ArgumentationFramework(num_agents=n_agents, strategy=_strategy)

        self._weights = (
            np.ones(3, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        self._weight_store = weight_store  # WeightStore | None
        self._gamma = float(gamma)
        # potential_scale multiplies every component, so Phi and the shaping term
        # gamma * Phi(s') - Phi(s) scale with it. 1.0 is the confirmatory setting.
        self._pot_fn = PotentialFunction(
            num_agents=n_agents,
            alpha=potential_scale,
            beta=potential_scale,
            delta=potential_scale,
            ablate=set() if ablate is None else set(ablate),
        )

        # Motion planner, built once per env and reused for every plan.
        self._motion_planner = MotionPlanner(self.mdp)
        # Static target lists (layout-fixed, state-independent).
        self._onion_dispensers = self.mdp.get_onion_dispenser_locations()
        self._dish_dispensers = self.mdp.get_dish_dispenser_locations()
        self._serving_locations = self.mdp.get_serving_locations()

    # ------------------------------------------------------------------
    # Gymnasium API overrides
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        n_agents = self.mdp.num_players

        # Initialise AF state before calling super() so _get_obs() is safe
        self._profile = {i: Intention.WAIT for i in range(n_agents)}
        self._af_trigger_count = 0
        self._intention_completion_count = 0
        self._phi = 0.0

        # Parent handles base_env.reset() and episode accumulator resets
        super().reset(seed=seed, options=options)

        # Run AF once immediately on the fresh state
        self._run_af()
        self._af_trigger_count = 1

        # Initialise bandit per-episode tracking against the opening profile.
        if self._bandit is not None:
            self._episode_outcomes = {}
            self._adopted = dict(self._profile)
            self._adopted_completed = {i: False for i in range(n_agents)}

        af_state = self._extract_af_state()
        self._phi = self._pot_fn.phi(af_state, self._profile)

        return self._get_obs(), {}

    def step(self, action):
        # Pre-step: snapshot held objects for completion detection
        prev_held = [p.held_object for p in self.base_env.state.players]

        # Execute joint action in overcooked
        joint_action = tuple(Action.ALL_ACTIONS[int(a)] for a in action)
        _, sparse_reward, done, info = self.base_env.step(joint_action)

        sparse_reward = float(sparse_reward)
        soups_this_step = round(sparse_reward / DELIVERY_REWARD)
        self._soups_delivered += soups_this_step
        self._ep_sparse_reward += sparse_reward
        self._ep_length += 1

        # Detect completions; re-trigger AF as needed
        n_agents = self.mdp.num_players
        curr_held = [p.held_object for p in self.base_env.state.players]
        should_trigger = False

        for agent_id in range(n_agents):
            held_changed = prev_held[agent_id] != curr_held[agent_id]
            if held_changed:
                should_trigger = True
                # Counts held-object changes, which stand in for completions.
                # A change can also happen without the assigned intention
                # completing, so this is an upper bound on completions.
                self._intention_completion_count += 1
            if self._profile.get(agent_id) is Intention.WAIT:
                should_trigger = True

        # Bandit: mark whether each agent's CURRENTLY-ADOPTED intention completed
        # this step. Evaluated against the pre-trigger assignment (self._adopted),
        # before the AF re-runs below.
        if self._bandit is not None:
            for agent_id in range(n_agents):
                adopted = self._adopted.get(agent_id)
                if adopted is not None and _intention_completed(
                    adopted,
                    _held_name(prev_held[agent_id]),
                    _held_name(curr_held[agent_id]),
                ):
                    self._adopted_completed[agent_id] = True

        if should_trigger:
            self._run_af()
            self._af_trigger_count += 1

        # Bandit: close any adoption whose intention changed after the re-trigger,
        # emitting its outcome (+1 completed / -1 not). Same intention => the
        # adoption continues and no outcome is emitted (WAIT never scores).
        if self._bandit is not None:
            for agent_id in range(n_agents):
                new_intent = self._profile.get(agent_id)
                old_intent = self._adopted.get(agent_id)
                if new_intent is not old_intent:
                    if old_intent is not None and old_intent is not Intention.WAIT:
                        outcome = 1.0 if self._adopted_completed.get(agent_id) else -1.0
                        self._episode_outcomes.setdefault(old_intent, []).append(outcome)
                    self._adopted[agent_id] = new_intent
                    self._adopted_completed[agent_id] = False

        # Potential-based shaping bonus
        af_state = self._extract_af_state()
        phi_next = self._pot_fn.phi(af_state, self._profile)
        shaping = self._gamma * phi_next - self._phi
        self._phi = phi_next

        # Stack intermediate task reward (Overcooked built-in heuristic shaping)
        # on top of potential shaping.  Gives PPO a positive learning signal
        # before it ever reaches a full delivery.
        intermediate = float(sum(info.get("shaped_r_by_agent", [0.0, 0.0])))
        training_reward = sparse_reward + intermediate + shaping
        self._ep_shaped_reward += shaping

        # Bandit: on episode end, close all still-open adoptions and apply one
        # batched per-episode update from the accumulated outcomes.
        if done and self._bandit is not None:
            for agent_id in range(n_agents):
                old_intent = self._adopted.get(agent_id)
                if old_intent is not None and old_intent is not Intention.WAIT:
                    outcome = 1.0 if self._adopted_completed.get(agent_id) else -1.0
                    self._episode_outcomes.setdefault(old_intent, []).append(outcome)
            self._bandit.update(self._episode_outcomes)
            self._episode_outcomes = {}

        obs = self._get_obs()
        terminated = False
        truncated = bool(done)

        step_info = {
            "sparse_reward": sparse_reward,
            "shaped_reward": shaping,
            "soups_this_step": soups_this_step,
        }
        if done:
            step_info["soups_delivered"] = self._soups_delivered
            step_info["ep_sparse_reward"] = self._ep_sparse_reward
            step_info["ep_shaped_reward"] = self._ep_shaped_reward
            step_info["ep_length"] = self._ep_length
            step_info["af_trigger_count"] = self._af_trigger_count
            step_info["intention_completion_count"] = self._intention_completion_count
            if self._bandit is not None:
                step_info["bandit_theta"] = self._bandit.theta_dict()

        return obs, training_reward, terminated, truncated, step_info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()                  # 1040-dim
        if not self._include_profile_in_obs:
            return base
        profile_enc = self._encode_profile()        # 10-dim
        return np.concatenate([base, profile_enc])

    def _encode_profile(self) -> np.ndarray:
        n_agents = self.mdp.num_players
        enc = np.zeros(_N_INTENTIONS * n_agents, dtype=np.float32)
        for agent_id in range(n_agents):
            intention = self._profile.get(agent_id, Intention.WAIT)
            idx = agent_id * _N_INTENTIONS + _INTENTION_INDEX[intention]
            enc[idx] = 1.0
        return enc

    # ------------------------------------------------------------------
    # AF internals
    # ------------------------------------------------------------------

    def _run_af(self) -> None:
        """Run the framework and update self._profile."""
        af_state = self._extract_af_state()
        # Use network-produced weights (no task-reward gradient; L2 shrinkage
        # only - not learned from the task) from the shared store when available;
        # fall back to the fixed weights array otherwise.
        if self._weight_store is not None:
            weights = self._weight_store.weights
        else:
            weights = self._weights
        if self._bandit is not None:
            score_fn = lambda arg, s: score_argument(
                arg, s, weights, bias=self._bandit.bias(arg.intention)
            )
        else:
            score_fn = lambda arg, s: score_argument(arg, s, weights)
        self._profile = self._af.run(af_state, score_fn=score_fn)

    def _extract_af_state(self) -> OvercookedAFState:
        """Convert the current overcooked_ai state into OvercookedAFState."""
        state = self.base_env.state
        n_agents = self.mdp.num_players

        # Agent positions (x, y) tuples
        agent_positions = [p.position for p in state.players]

        # What each agent is holding: None | 'onion' | 'dish' | 'soup'
        agent_holding = [
            None if p.held_object is None else p.held_object.name
            for p in state.players
        ]

        # Pot contents: one PotState entry per pot regardless of fill level.
        # Empty pots (no SoupState object in state.objects yet) are represented
        # as PotState(0, False, False). Using a per-pot list rather than an
        # aggregated scalar is required for correctness on multi-pot layouts
        # (e.g. coordination_ring has 2 pots) where aggregate onion counts
        # can exceed 3 and break the FetchDish == 3 precondition check.
        pot_locations = self.mdp.get_pot_locations()
        pot_states = []
        for pot_loc in pot_locations:
            if pot_loc in state.objects:
                pot = state.objects[pot_loc]
                pot_states.append(PotState(
                    onion_count=pot.ingredients.count("onion"),
                    is_cooking=pot.is_cooking,
                    is_ready=pot.is_ready,
                ))
            else:
                pot_states.append(PotState(onion_count=0, is_cooking=False, is_ready=False))

        # Planner step counts from each agent to each intention's target.
        agent_steps_to_target = self._compute_steps_to_target(state)

        # Assignment counts from the current profile
        num_agents_assigned: dict[Intention, int] = {i: 0 for i in Intention}
        for intention in self._profile.values():
            num_agents_assigned[intention] += 1

        return OvercookedAFState(
            agent_positions=list(agent_positions),
            agent_holding=agent_holding,
            pot_states=pot_states,
            num_agents_assigned=num_agents_assigned,
            agent_steps_to_target=agent_steps_to_target,
        )

    def _compute_steps_to_target(self, state) -> list:
        """
        Return motion-planner step counts (including final interact) from each agent's
        current pos_and_or to each intention's nearest target location.

        FETCH_ONION  -> nearest onion dispenser (layout-fixed)
        FETCH_DISH   -> nearest dish dispenser (layout-fixed)
        PICKUP_SOUP  -> nearest ready pot (dynamic, fallback when none ready)
        DELIVER_SOUP -> nearest serving counter (layout-fixed)
        WAIT         -> 0 always

        Uses _FALLBACK_STEPS (20) when no valid target exists or path is
        unreachable, ensuring the score/potential contribution stays at zero.
        """
        mp = self._motion_planner

        # Ready pots are the only state-dependent target list.
        pot_states = self.mdp.get_pot_states(state)
        ready_pots = self.mdp.get_ready_pots(pot_states)

        static_targets = {
            Intention.FETCH_ONION: self._onion_dispensers,
            Intention.FETCH_DISH: self._dish_dispensers,
            Intention.PICKUP_SOUP: ready_pots,
            Intention.DELIVER_SOUP: self._serving_locations,
        }

        results = []
        for player in state.players:
            pos_and_or = player.pos_and_or
            steps: dict[Intention, int] = {}
            for intention in Intention:
                if intention is Intention.WAIT:
                    steps[intention] = 0
                    continue
                targets = static_targets[intention]
                cost = mp.min_cost_to_feature(pos_and_or, targets)
                steps[intention] = (
                    int(cost) if math.isfinite(cost) else _FALLBACK_STEPS
                )
            results.append(steps)

        return results
