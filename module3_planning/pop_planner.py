# module3_planning/pop_planner.py

from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict

from .planning_domain import build_domain_for_acc3, Action as DomainAction, Proposition


@dataclass
class POPAction:
    id: str
    name: str
    preconditions: Set[Proposition]
    effects: Set[Proposition]


@dataclass
class POPPlan:
    actions: Dict[str, POPAction] = field(default_factory=dict)
    orderings: List[Tuple[str, str]] = field(default_factory=list)  # (before, after)
    causal_links: List[Tuple[str, Proposition, str]] = field(default_factory=list)  # (producer, prop, consumer)
    branches: List[Dict[str, object]] = field(default_factory=list)  # conditional branches


def build_pop_plan_for_acc3() -> POPPlan:
    """
    Construct a partial-order plan for servicing ACC3 with contingencies:

    Base flow:
      Start -> Drone Recon, Request Diversion, Predeploy A1, Notify H1
      Then StartTransportToAccident
      Then either DeliverPatientToH1 or RerouteMidJourneyToH2

    Branching:
      IF CTAvailable(H2) THEN RerouteMidJourneyToH2
      ELSE (CTOffline(H2)) THEN DeliverPatientToH1
    """

    initial_state, domain_actions, goals = build_domain_for_acc3()
    plan = POPPlan()

    # Artificial Start and Finish
    start = POPAction(
        id="Start",
        name="Start",
        preconditions=set(),
        effects=set(initial_state),
    )
    finish = POPAction(
        id="Finish",
        name="Finish",
        preconditions={p for p in goals},  # requires AccidentServed and PatientAt(ACC3,H1) in strict case
        effects=set(),
    )
    plan.actions[start.id] = start
    plan.actions[finish.id] = finish

    # Helper: find domain actions by prefix
    def find_da(prefix: str) -> DomainAction:
        for da in domain_actions:
            if da.name.startswith(prefix):
                return da
        raise ValueError(f"Domain action with prefix '{prefix}' not found")

    da_drone = find_da("TriggerDroneRecon")
    da_predeploy = find_da("PreDeployAmbulanceNearHotspot")
    da_notify_h1 = find_da("NotifyHospital")
    da_request_div = find_da("RequestTrafficDiversion")
    da_start_transport = find_da("StartTransportToAccident")
    da_deliver_h1 = find_da("DeliverPatientToH1")
    da_reroute_h2 = find_da("RerouteMidJourneyToH2")

    # Wrap into POPActions
    def wrap(da: DomainAction, id_suffix: str) -> POPAction:
        return POPAction(
            id=id_suffix,
            name=da.name,
            preconditions=set(da.preconditions),
            effects=set(da.add_effects),
        )

    a_drone = wrap(da_drone, "DroneRecon")
    a_predeploy = wrap(da_predeploy, "PredeployA1")
    a_notify_h1 = wrap(da_notify_h1, "NotifyH1")
    a_request_div = wrap(da_request_div, "RequestDiversion")
    a_start_transport = wrap(da_start_transport, "StartTransport")
    a_deliver_h1 = wrap(da_deliver_h1, "DeliverH1")
    a_reroute_h2 = wrap(da_reroute_h2, "RerouteH2")

    for a in [a_drone, a_predeploy, a_notify_h1, a_request_div, a_start_transport, a_deliver_h1, a_reroute_h2]:
        plan.actions[a.id] = a

    # ------------------ Orderings (partial, not fully linear) ------------------

    # Start before everything
    for aid in plan.actions:
        if aid != "Start":
            plan.orderings.append(("Start", aid))

    # Pre-transport phase can be parallel, but must precede StartTransport:
    plan.orderings.append((a_drone.id, a_start_transport.id))
    plan.orderings.append((a_predeploy.id, a_start_transport.id))
    plan.orderings.append((a_request_div.id, a_start_transport.id))

    # NotifyH1 must happen before DeliverH1
    plan.orderings.append((a_notify_h1.id, a_deliver_h1.id))

    # StartTransport must happen before both possible destination actions
    plan.orderings.append((a_start_transport.id, a_deliver_h1.id))
    plan.orderings.append((a_start_transport.id, a_reroute_h2.id))

    # Destination actions come before Finish
    plan.orderings.append((a_deliver_h1.id, "Finish"))
    plan.orderings.append((a_reroute_h2.id, "Finish"))

    # ------------------ Causal links (simplified) ------------------

    # Start provides initial facts for pre-phase actions
    for a in [a_drone, a_predeploy, a_notify_h1, a_request_div]:
        for p in a.preconditions:
            if p in start.effects:
                plan.causal_links.append(("Start", p, a.id))

    # DroneRecon provides AccidentLocationConfirmed(ACC3) to StartTransport
    for p in a_start_transport.preconditions:
        if "AccidentLocationConfirmed" in p:
            plan.causal_links.append((a_drone.id, p, a_start_transport.id))

    # Predeploy provides AmbulanceAt(A1, nayapalli_chowk) to StartTransport
    for p in a_start_transport.preconditions:
        if "AmbulanceAt" in p and "nayapalli_chowk" in p:
            plan.causal_links.append((a_predeploy.id, p, a_start_transport.id))

    # RequestDiversion provides TrafficDiverted(stadium_corridor) to StartTransport
    for p in a_start_transport.preconditions:
        if "TrafficDiverted(stadium_corridor)" == p:
            plan.causal_links.append((a_request_div.id, p, a_start_transport.id))

    # NotifyH1 provides HospitalNotified(H1) to DeliverH1
    for p in a_deliver_h1.preconditions:
        if "HospitalNotified" in p:
            plan.causal_links.append((a_notify_h1.id, p, a_deliver_h1.id))

    # StartTransport provides AmbulanceEnRoute(A1,ACC3) to both DeliverH1 and RerouteH2
    for p in a_deliver_h1.preconditions:
        if "AmbulanceEnRoute" in p:
            plan.causal_links.append((a_start_transport.id, p, a_deliver_h1.id))
    for p in a_reroute_h2.preconditions:
        if "AmbulanceEnRoute" in p:
            plan.causal_links.append((a_start_transport.id, p, a_reroute_h2.id))

    # ------------------ Contingency branches ------------------

    plan.branches.append({
        "condition": "CTAvailable(H2)",
        "actions": [a_reroute_h2.id],
        "description": "If CT scanner at H2 becomes available while A1 is en route, reroute mid-way to H2 for faster imaging."
    })

    plan.branches.append({
        "condition": "CTOffline(H2)",
        "actions": [a_deliver_h1.id],
        "description": "If H2 CT remains offline, continue with baseline plan and deliver the patient to H1 (trauma centre)."
    })

    return plan


def pretty_print_pop_plan(plan: POPPlan) -> None:
    print("=== POP Partial-Order Plan for ACC3 with Contingent Routing ===\n")

    print("Actions:")
    for aid, act in plan.actions.items():
        print(f"- {aid}: {act.name}")
        if act.preconditions:
            print(f"    Pre: {sorted(act.preconditions)}")
        if act.effects:
            print(f"    Eff: {sorted(act.effects)}")
    print()

    print("Ordering constraints (before -> after):")
    for before, after in plan.orderings:
        print(f"  {before} -> {after}")
    print()

    print("Causal links (producer --[condition]--> consumer):")
    for prod, cond, cons in plan.causal_links:
        print(f"  {prod} --[{cond}]--> {cons}")
    print()

    print("Contingency branches (flexible branching points):")
    for branch in plan.branches:
        cond = branch["condition"]
        actions = branch["actions"]
        desc = branch.get("description", "")
        print(f"  IF {cond}:")
        print(f"    THEN execute actions: {actions}")
        if desc:
            print(f"    ({desc})")
    print()
