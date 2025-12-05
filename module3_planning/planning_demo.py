# module3_planning/planning_demo.py

from .graphplan import compute_strict_plan_for_acc3
from .pop_planner import build_pop_plan_for_acc3, pretty_print_pop_plan


def run_module3_demo():
    print("########################################################")
    print("Module 3: Adaptive Emergency Response Planning (GraphPlan + POP)")
    print("########################################################\n")

    # 1) Strict GraphPlan-style plan
    print("=== Strict plan using simplified GraphPlan ===\n")
    strict_plan = compute_strict_plan_for_acc3()
    for i, a in enumerate(strict_plan, start=1):
        print(f"{i}. {a.name}")
    print(
        "\nThis strict plan assumes a fixed view of the world "
        "(rainy conditions, H2 CT offline, no unexpected updates) "
        "and produces a single linear sequence of actions.\n"
    )

    # 2) POP partial-order plan with contingencies
    pop_plan = build_pop_plan_for_acc3()
    pretty_print_pop_plan(pop_plan)
    print(
        "In the POP representation, early actions such as drone reconnaissance, "
        "pre-deployment, and traffic diversion can overlap in time. After "
        "StartTransportToAccident, the plan deliberately keeps a branching point:\n"
        "  - If CTAvailable(H2) is reported while A1 is en route, reroute mid-way to H2.\n"
        "  - If CTOffline(H2) persists, continue and deliver the patient to H1.\n"
    )


if __name__ == "__main__":
    run_module3_demo()
