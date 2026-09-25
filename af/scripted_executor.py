"""
Deterministic scripted policy for OvercookedAFEnv.

The executor converts each agent's current intention into primitive actions
using MotionPlanner. Agents carrying objects are routed to the corresponding
deposit target. It also avoids immediate collisions and re-runs the framework
after prolonged lack of movement.

The target rules live in _resolve_targets and _targets_for_held. There are no
stochastic elements, so the behaviour is fixed by the state and the profile.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import Action

if TYPE_CHECKING:
    from af.env_wrapper import OvercookedAFEnv

from af.framework import Intention

# Index shorthands
STAY_IDX = Action.ACTION_TO_INDEX[Action.STAY]
INTERACT_IDX = Action.ACTION_TO_INDEX[Action.INTERACT]

DEADLOCK_STEPS = 10  # re-trigger AF after this many steps with no position change


class ScriptedExecutor:
    """
    Wraps OvercookedAFEnv and provides `act() -> np.ndarray([a0, a1])`.

    The executor reads the current AF profile from env._profile, resolves
    each agent's assigned intention to a target tile, uses MotionPlanner to
    obtain the next primitive action, and applies a light collision check.
    """

    def __init__(self, env: "OvercookedAFEnv") -> None:
        self._env = env
        n = env.mdp.num_players
        # Recent positions per agent (deque of (x,y)); capped at DEADLOCK_STEPS
        self._pos_history: list[deque] = [
            deque(maxlen=DEADLOCK_STEPS) for _ in range(n)
        ]

    def reset(self) -> None:
        """Clear position history at the start of a new episode."""
        for dq in self._pos_history:
            dq.clear()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self) -> np.ndarray:
        """
        Compute joint action for the current env state.

        Returns a numpy array of shape (n_agents,) with integer action indices.
        """
        env = self._env
        state = env.base_env.state
        n_agents = env.mdp.num_players
        profile = env._profile

        # 1. Update position history; re-trigger AF if any agent is stuck.
        self._check_deadlock(state, n_agents)

        # Current positions used to compute per-agent blocked cells.
        current_pos = [state.players[i].position for i in range(n_agents)]

        # 2. Compute action for each agent, each avoiding the other agent's
        #    current cell to prevent swap deadlocks.
        actions = np.full(n_agents, STAY_IDX, dtype=np.int64)
        next_positions = []
        for agent_id in range(n_agents):
            intention = profile.get(agent_id, Intention.WAIT)
            other_pos = current_pos[1 - agent_id] if n_agents == 2 else None
            act_idx, next_pos = self._plan_action(
                agent_id, intention, state, blocked_pos=other_pos
            )
            actions[agent_id] = act_idx
            next_positions.append(next_pos)

        # 3. Same-cell collision: if both agents would move to the same cell,
        #    the higher-id agent yields.
        if (
            n_agents >= 2
            and next_positions[0] is not None
            and next_positions[1] is not None
            and next_positions[0] == next_positions[1]
        ):
            actions[1] = STAY_IDX

        return actions

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _plan_action(
        self,
        agent_id: int,
        intention: Intention,
        state,
        blocked_pos: tuple | None = None,
    ) -> tuple[int, tuple | None]:
        """
        Return the next action and the resulting position for one agent.

        next_position is the agent's (x, y) after a movement step, or None for
        STAY and INTERACT, and is used for collision detection.

        An agent carrying an object has its destination set by that object
        rather than by its intention, since the preconditions block every
        non-Wait intention while the hands are full. Plans whose first step
        enters blocked_pos are skipped.
        """
        env = self._env
        player = state.players[agent_id]
        held = player.held_object

        # Carrying override: holding something means the AF intention is stale
        # (FetchOnion is blocked while holding anything).  Route to deposit target.
        if held is not None:
            held_name = held.name  # 'onion', 'dish', 'soup'
            targets = self._targets_for_held(held_name, state)
            if targets:
                return self._navigate_to(
                    player, targets, env._motion_planner, blocked_pos=blocked_pos
                )
            return STAY_IDX, None

        # Start-cooking override: in the new-dynamics Overcooked mode, a full pot
        # does NOT begin cooking automatically. An empty-handed agent must
        # interact with it first.  This fires before any AF intention so that the
        # empty-handed agent starts cooking before being routed elsewhere.
        pot_states_now = env.mdp.get_pot_states(state)
        full_unstarted = env.mdp.get_full_but_not_cooking_pots(pot_states_now)
        if full_unstarted:
            return self._navigate_to(
                player, full_unstarted, env._motion_planner, blocked_pos=blocked_pos
            )

        if intention is Intention.WAIT:
            return STAY_IDX, None

        # FETCH_ONION guard: do not pick up another onion when the onions already
        # held by other agents, plus those in the pot, cover every remaining pot
        # slot. This prevents both agents targeting the last free slot at the
        # same time, which would leave one of them with an unusable extra onion.
        #
        # Instead of STAY (which would park the idle agent on the pot's only
        # motion goal and block the onion carrier), navigate toward the nearest
        # onion dispenser as a holding position.  If the agent is already at the
        # dispenser's motion goal, suppress the interact to avoid picking up.
        if intention is Intention.FETCH_ONION:
            pot_onions = sum(
                len([i for i in state.objects[loc].ingredients if i == "onion"])
                for loc in env.mdp.get_pot_locations()
                if loc in state.objects
            )
            in_flight = sum(
                1 for p in state.players
                if p.held_object is not None and p.held_object.name == "onion"
            )
            if 3 - pot_onions - in_flight <= 0:
                return self._approach_without_interacting(
                    player,
                    env.mdp.get_onion_dispenser_locations(),
                    env._motion_planner,
                    blocked_pos=blocked_pos,
                )

        # FETCH_DISH guard: analogous to the FETCH_ONION guard above.
        # Only one dish is needed per batch.  If another agent is already
        # holding a dish or a plated soup, route to the dish dispenser as a
        # holding position but suppress the interact so we don't pick up a
        # second dish.
        if intention is Intention.FETCH_DISH:
            dishes_held = sum(
                1 for p in state.players
                if p.held_object is not None
                and p.held_object.name in ("dish", "soup")
            )
            if dishes_held >= 1:
                return self._approach_without_interacting(
                    player,
                    env.mdp.get_dish_dispenser_locations(),
                    env._motion_planner,
                    blocked_pos=blocked_pos,
                )

        mp = env._motion_planner

        targets = self._resolve_targets(intention, state)
        if not targets:
            return STAY_IDX, None

        return self._navigate_to(player, targets, mp, blocked_pos=blocked_pos)

    def _approach_without_interacting(
        self,
        player,
        targets: list,
        mp,
        blocked_pos: tuple | None = None,
    ) -> tuple[int, tuple | None]:
        """
        Navigate toward `targets` but never pick anything up.

        Used by the guards below, where the agent should hold position near a
        dispenser without taking a second onion or dish.
        """
        act_idx, next_pos = self._navigate_to(
            player, targets, mp, blocked_pos=blocked_pos
        )
        if act_idx == INTERACT_IDX:
            return STAY_IDX, None
        return act_idx, next_pos

    def _navigate_to(
        self,
        player,
        targets: list,
        mp,
        blocked_pos: tuple | None = None,
    ) -> tuple[int, tuple | None]:
        """
        Find the cheapest reachable motion goal across all `targets` and return
        (action_index, next_position).

        Plans whose first action steps into `blocked_pos` are skipped; the next
        cheapest plan is tried instead.  If ALL direct plans are blocked, a
        sidestep is attempted: one step in any free direction that is not
        blocked_pos, picking the direction that minimises remaining cost to goal.
        Falls back to (STAY, None) only if no sidestep is possible.
        """
        start = player.pos_and_or  # ((x, y), (dx, dy))

        # Collect all candidate plans and sort cheapest-first.
        all_plans: list = []
        for tile in targets:
            for goal in mp.motion_goals_for_pos.get(tile, []):
                try:
                    plan = mp.get_plan(start, goal)
                    all_plans.append(plan)
                except KeyError:
                    continue

        all_plans.sort(key=lambda p: p[2])

        for plan in all_plans:
            if not plan[0]:
                continue

            raw_action = plan[0][0]   # tuple (dx,dy) or 'interact'
            act_idx = Action.ACTION_TO_INDEX[raw_action]

            if isinstance(raw_action, tuple):
                px, py = player.position
                dx, dy = raw_action
                next_pos: tuple | None = (px + dx, py + dy)
            else:
                next_pos = None

            # Skip this plan if its first step leads into the blocked cell.
            if blocked_pos is not None and next_pos == blocked_pos:
                continue

            return act_idx, next_pos

        # All direct plans are blocked.  Try a single sidestep toward the goal.
        if blocked_pos is not None and targets:
            return self._sidestep(player, targets, mp, blocked_pos)

        return STAY_IDX, None

    def _sidestep(
        self,
        player,
        targets: list,
        mp,
        blocked_pos: tuple,
    ) -> tuple[int, tuple | None]:
        """
        Move one step in the free direction that minimises remaining plan cost
        to the nearest motion goal, without entering blocked_pos.

        Used when all direct plans require stepping into the other agent's cell.
        """
        terrain = self._env.mdp.terrain_mtx
        px, py = player.position

        best_cost = float("inf")
        best_act = STAY_IDX
        best_next: tuple | None = None

        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            if (nx, ny) == blocked_pos:
                continue
            if ny < 0 or ny >= len(terrain) or nx < 0 or nx >= len(terrain[ny]):
                continue
            if terrain[ny][nx] != " ":
                continue
            # Cost from candidate cell (orientation = move direction) to nearest goal.
            for tile in targets:
                for goal in mp.motion_goals_for_pos.get(tile, []):
                    try:
                        side_plan = mp.get_plan(((nx, ny), (dx, dy)), goal)
                        cost = 1 + side_plan[2]
                        if cost < best_cost:
                            best_cost = cost
                            best_act = Action.ACTION_TO_INDEX[(dx, dy)]
                            best_next = (nx, ny)
                    except KeyError:
                        continue

        return best_act, best_next

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def _targets_for_held(self, held_name: str, state) -> list[tuple]:
        """
        Return deposit target tiles for an agent that is already carrying an item.

        onion  -> pots that can still accept an onion (not cooking, not ready).
                 Returns [] when all pots are full; the agent then waits with
                 the onion until a pot becomes available after soup delivery.
        dish   -> nearest ready pot, fallback to any pot (plate or pre-position)
        soup   -> serving counter (deliver)
        """
        mdp = self._env.mdp
        if held_name == "onion":
            # Only route to pots that can still accept another onion.
            # In the new-dynamics mode a full pot (3 onions) is not yet cooking
            # until an empty-handed agent starts it; it cannot accept more onions.
            # Cooking or ready pots also cannot accept onions.
            pot_states = mdp.get_pot_states(state)
            unavailable = (
                set(mdp.get_full_but_not_cooking_pots(pot_states))
                | set(mdp.get_cooking_pots(pot_states))
                | set(mdp.get_ready_pots(pot_states))
            )
            return [p for p in mdp.get_pot_locations() if p not in unavailable]
        if held_name == "dish":
            pot_states = mdp.get_pot_states(state)
            ready = mdp.get_ready_pots(pot_states)
            if ready:
                return ready
            cooking = mdp.get_cooking_pots(pot_states)
            if cooking:
                # Pot is cooking: wait near it so we can plate immediately.
                return mdp.get_pot_locations()
            # Pot is empty or being filled: clear the pot area so onion-carriers
            # can deposit.  Wait near the serving counter instead.
            return mdp.get_serving_locations()
        if held_name == "soup":
            return mdp.get_serving_locations()
        return []

    def _resolve_targets(self, intention: Intention, state) -> list[tuple]:
        """
        Return the list of tile positions that the agent should navigate to
        in order to complete this intention.
        """
        env = self._env
        mdp = env.mdp

        if intention is Intention.FETCH_ONION:
            return mdp.get_onion_dispenser_locations()

        if intention is Intention.FETCH_DISH:
            return mdp.get_dish_dispenser_locations()

        if intention is Intention.PICKUP_SOUP:
            # Navigate to ready pot; fall back to any pot if none ready yet.
            pot_states = mdp.get_pot_states(state)
            ready = mdp.get_ready_pots(pot_states)
            return ready if ready else mdp.get_pot_locations()

        if intention is Intention.DELIVER_SOUP:
            return mdp.get_serving_locations()

        # WAIT (and any unexpected value)
        return []

    # ------------------------------------------------------------------
    # Deadlock detection
    # ------------------------------------------------------------------

    def _check_deadlock(self, state, n_agents: int) -> None:
        """
        Update position history; re-trigger AF for any stuck agent.

        An agent is considered stuck if the same (x, y) has been the only
        value in its position history for the last DEADLOCK_STEPS steps.
        """
        env = self._env
        for agent_id in range(n_agents):
            pos = state.players[agent_id].position
            self._pos_history[agent_id].append(pos)

            if len(self._pos_history[agent_id]) == DEADLOCK_STEPS:
                unique_positions = set(self._pos_history[agent_id])
                if len(unique_positions) == 1:
                    # Stuck: force an AF refresh and clear history
                    env._run_af()
                    env._af_trigger_count += 1
                    self._pos_history[agent_id].clear()


# ---------------------------------------------------------------------------
# Exploratory evaluation of intention sources
# ---------------------------------------------------------------------------
#
# Run from the repository root:
#
#     python -m af.scripted_executor
#
# The executor is supplied with three intention sources in turn and evaluated
# over six runs of fifty episodes each. It never learns, so its performance
# depends only on the intentions it is given. Nothing is read from results/,
# since the episodes are generated here rather than during training.
#
# The seed reaches only the random source. The executor and the environment are
# deterministic, so the six runs under the framework and the fixed role split
# repeat the same episodes and their standard deviation is zero by
# construction. For those two the runs are a reproducibility check rather than
# independent samples.

if __name__ == "__main__":
    import dataclasses
    import types

    from scipy.stats import bootstrap

    from af.env_wrapper import OvercookedAFEnv
    from af.framework import check_preconditions

    LAYOUT = "cramped_room"
    HORIZON = 400
    STRATEGY = "aggressive"
    WEIGHTS = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    SEEDS = list(range(6))
    N_EVAL_EPISODES = 50
    N_BOOT = 10_000
    BOOT_SEED = 42

    SOURCES = [
        ("af",     "Framework intentions"),
        ("static", "Fixed role split"),
        ("random", "Random valid intentions"),
    ]

    def apply_intention_source(env, source: str, seed: int) -> None:
        """
        Replace the profile the framework would produce, leaving trigger timing,
        the counters and the executor untouched.

        af      the framework itself, so nothing is replaced
        static  FetchOnion for one agent and FetchDish for the other, always
        random  a uniformly random valid intention per agent at every trigger
        """
        if source == "af":
            return

        if source == "static":
            def _run_af_static(self):
                self._profile = {0: Intention.FETCH_ONION, 1: Intention.FETCH_DISH}
            env._run_af = types.MethodType(_run_af_static, env)
            return

        if source == "random":
            rng = np.random.RandomState(seed)

            def _run_af_random(self):
                af_state = self._extract_af_state()
                profile, assigned = {}, {}
                for agent_id in range(self.mdp.num_players):
                    # Each agent sees the assignments already made, so its
                    # preconditions are checked against a consistent state.
                    updated = dataclasses.replace(
                        af_state, num_agents_assigned=dict(assigned)
                    )
                    valid = [i for i in Intention
                             if check_preconditions(i, agent_id, updated)]
                    chosen = rng.choice(valid)   # Wait is always valid
                    profile[agent_id] = chosen
                    assigned[chosen] = assigned.get(chosen, 0) + 1
                self._profile = profile

            env._run_af = types.MethodType(_run_af_random, env)
            return

        raise ValueError(f"Unknown intention source: {source!r}")

    def run_once(source: str, seed: int) -> float:
        """
        Mean soups delivered per episode for one run.

        The seed is used only by the random source. The other two sources
        ignore it and produce the same episodes every time.
        """
        env = OvercookedAFEnv(
            layout_name=LAYOUT,
            horizon=HORIZON,
            strategy=STRATEGY,
            weights=WEIGHTS,
            weight_store=None,
        )
        apply_intention_source(env, source, seed)
        executor = ScriptedExecutor(env)

        soups = []
        for _ in range(N_EVAL_EPISODES):
            executor.reset()
            env.reset()
            done = False
            while not done:
                _, _, terminated, truncated, info = env.step(executor.act())
                done = terminated or truncated
            soups.append(info["soups_delivered"])
        return float(np.mean(soups))

    def bca(values: np.ndarray) -> str:
        """BCa 95% interval on the mean, or a note when the sample is constant."""
        if float(values.std()) == 0.0:
            return "constant, no interval"
        res = bootstrap(
            (values,),
            statistic=lambda x, axis: np.mean(x, axis=axis),
            n_resamples=N_BOOT,
            confidence_level=0.95,
            method="BCa",
            random_state=BOOT_SEED,
        )
        return f"[{res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f}]"

    print(f"Scripted executor, {LAYOUT}, {len(SEEDS)} runs x {N_EVAL_EPISODES} episodes")
    print("The seed reaches only the random source, so the other two repeat "
          "identical episodes.\n")
    print(f"{'Intention source':<26}{'Mean':>8}{'SD':>8}   BCa 95%")

    for source, label in SOURCES:
        per_run = np.array([run_once(source, s) for s in SEEDS])
        print(f"{label:<26}{per_run.mean():>8.3f}{per_run.std(ddof=1):>8.3f}   {bca(per_run)}")
