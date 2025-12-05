# module3_planning/graphplan.py

from typing import List, Set, Tuple
from dataclasses import dataclass

from .planning_domain import Action, build_domain_for_acc3, Proposition


@dataclass
class PlanGraph:
    proposition_layers: List[Set[Proposition]]
    action_layers: List[List[Action]]


def build_plan_graph(
    initial_state: Set[Proposition],
    actions: List[Action],
    goals: Set[Proposition],
    max_levels: int = 8,
) -> Tuple[PlanGraph, int]:
    """
    Build a simple planning graph until:
    - goals are all present in a proposition layer, or
    - max_levels is reached.

    Mutexes are ignored (OK for this small domain).
    """

    proposition_layers: List[Set[Proposition]] = [set(initial_state)]
    action_layers: List[List[Action]] = []

    for level in range(max_levels):
        current_props = proposition_layers[-1]

        # 1. Applicable actions at this level
        applicable_actions: List[Action] = []
        for action in actions:
            if action.preconditions.issubset(current_props):
                applicable_actions.append(action)

        action_layers.append(applicable_actions)

        # 2. Next proposition layer = current + all add_effects
        next_props = set(current_props)
        for act in applicable_actions:
            next_props |= act.add_effects

        proposition_layers.append(next_props)

        # 3. Check if goals are reached
        if goals.issubset(next_props):
            return PlanGraph(proposition_layers, action_layers), level + 1

    raise ValueError("Goals not reachable within max_levels in planning graph.")


def extract_linear_plan(
    plan_graph: PlanGraph,
    goal_layer_index: int,
    goals: Set[Proposition],
) -> List[Action]:
    """
    Backward extraction of a linear plan from the planning graph.

    Greedy and simplified:
    - Move backwards from the goal layer
    - For each goal proposition, look for an action in the previous action layer
      that adds it; add that action and its preconditions as subgoals.
    """

    proposition_layers = plan_graph.proposition_layers
    action_layers = plan_graph.action_layers

    chosen_actions: List[Action] = []
    current_goals = set(goals)

    for level in range(goal_layer_index, 0, -1):
        actions_at_prev = action_layers[level - 1]
        new_subgoals: Set[Proposition] = set()

        for g in current_goals:
            # If goal already true at previous proposition layer, no new action required
            if g in proposition_layers[level - 1]:
                continue

            supporter = None
            for a in actions_at_prev:
                if g in a.add_effects:
                    supporter = a
                    break

            if supporter is None:
                # Assume carried forward from earlier layer or initial state
                continue

            chosen_actions.append(supporter)
            new_subgoals |= supporter.preconditions

        current_goals |= new_subgoals

    # Reverse into forward order and remove duplicates while preserving order
    chosen_actions.reverse()
    seen = set()
    final_plan: List[Action] = []
    for a in chosen_actions:
        if a.name not in seen:
            final_plan.append(a)
            seen.add(a.name)

    return final_plan


def compute_strict_plan_for_acc3() -> List[Action]:
    """
    Build domain, construct plan graph, and return a strict action sequence
    for serving ACC3 and taking the patient to H1.
    """
    initial_state, actions, goals = build_domain_for_acc3()
    pg, goal_layer_index = build_plan_graph(initial_state, actions, goals, max_levels=8)
    plan = extract_linear_plan(pg, goal_layer_index, goals)
    return plan


if __name__ == "__main__":
    plan = compute_strict_plan_for_acc3()
    print("=== Strict GraphPlan-style plan for ACC3 ===")
    for step, a in enumerate(plan, start=1):
        print(f"{step}. {a.name}")
