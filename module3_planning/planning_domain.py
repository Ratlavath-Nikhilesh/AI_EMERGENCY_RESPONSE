# module3_planning/planning_domain.py

from dataclasses import dataclass
from typing import Set, List, Tuple

from common.scenario import create_base_scenario
from common.entities import Ambulance, Hospital, Accident
from common.constants import HospitalCTStatus

Proposition = str


@dataclass
class Action:
    name: str
    preconditions: Set[Proposition]
    add_effects: Set[Proposition]
    del_effects: Set[Proposition]


def _find_by_id(objects, obj_id: str):
    for o in objects:
        if getattr(o, "id", None) == obj_id:
            return o
    raise ValueError(f"Object with id {obj_id} not found")


def build_domain_for_acc3() -> Tuple[Set[Proposition], List[Action], Set[Proposition]]:
    """
    Construct a STRIPS-like planning domain around:
    - Ambulance A1 at hospital H1 (trauma)
    - Accident ACC3 (SUV rollover near airport_approach)
    - H2 initially with CT offline (from scenario)

    Returns:
        initial_state: set of propositions
        actions: list of available ground actions
        goal_state: set of goal propositions (strict plan: patient at H1)
    """

    scenario = create_base_scenario()
    hospitals: List[Hospital] = scenario["hospitals"]
    ambulances: List[Ambulance] = scenario["ambulances"]
    accidents: List[Accident] = scenario["accidents"]

    h1 = _find_by_id(hospitals, "H1")
    h2 = _find_by_id(hospitals, "H2")
    a1 = _find_by_id(ambulances, "A1")
    acc3 = _find_by_id(accidents, "ACC3")

    # ----------------------------------------------------
    # 1. Initial State Propositions
    # ----------------------------------------------------
    initial_state: Set[Proposition] = set()

    # Ambulance status
    initial_state.add(f"AmbulanceFree({a1.id})")
    initial_state.add(f"AmbulanceAt({a1.id}, {a1.current_node})")  # hospital_h1

    # Accident ACC3 status
    initial_state.add(f"AccidentReported({acc3.id})")
    initial_state.add(f"AccidentHighRisk({acc3.id})")    # from Module 1 intuition
    initial_state.add(f"AccidentNotServed({acc3.id})")

    # Hospital info
    initial_state.add(f"Hospital({h1.id})")
    initial_state.add(f"Hospital({h2.id})")
    initial_state.add(f"TraumaCenter({h1.id})")

    # CT status: from scenario H2 is offline now; assume H1 CT is available
    initial_state.add(f"CTAvailable({h1.id})")
    if h2.ct_status == HospitalCTStatus.OFFLINE:
        initial_state.add(f"CTOffline({h2.id})")
    else:
        initial_state.add(f"CTAvailable({h2.id})")

    # Environment info
    initial_state.add("RainyConditions")
    initial_state.add("StadiumTrafficLikely")

    # Control room
    initial_state.add("ControlRoomOperational")

    # ----------------------------------------------------
    # 2. Actions
    # ----------------------------------------------------

    actions: List[Action] = []

    # 2.1 TriggerDroneRecon(ACC3)
    actions.append(
        Action(
            name=f"TriggerDroneRecon({acc3.id})",
            preconditions={
                "ControlRoomOperational",
                f"AccidentReported({acc3.id})",
            },
            add_effects={
                f"DroneReconDone({acc3.id})",
                f"AccidentLocationConfirmed({acc3.id})",
            },
            del_effects=set(),
        )
    )

    # 2.2 RequestTrafficDiversion(stadium_corridor)
    actions.append(
        Action(
            name="RequestTrafficDiversion(stadium_corridor)",
            preconditions={
                "ControlRoomOperational",
                "StadiumTrafficLikely",
            },
            add_effects={
                "TrafficDiverted(stadium_corridor)",
            },
            del_effects=set(),
        )
    )

    # 2.3 PreDeployAmbulanceNearHotspot(A1, nayapalli_chowk)
    hotspot_node = "nayapalli_chowk"
    actions.append(
        Action(
            name=f"PreDeployAmbulanceNearHotspot({a1.id}, {hotspot_node})",
            preconditions={
                f"AmbulanceFree({a1.id})",
                f"AmbulanceAt({a1.id}, {a1.current_node})",
            },
            add_effects={
                f"AmbulanceAt({a1.id}, {hotspot_node})",
                f"AmbulancePredeployed({a1.id})",
            },
            del_effects={
                f"AmbulanceAt({a1.id}, {a1.current_node})",
            },
        )
    )

    # 2.4 NotifyHospital(H1)
    actions.append(
        Action(
            name=f"NotifyHospital({h1.id})",
            preconditions={
                "ControlRoomOperational",
                f"AccidentHighRisk({acc3.id})",
                f"Hospital({h1.id})",
            },
            add_effects={
                f"HospitalNotified({h1.id})",
            },
            del_effects=set(),
        )
    )

    # 2.5 StartTransportToAccident(A1, ACC3)
    actions.append(
        Action(
            name=f"StartTransportToAccident({a1.id}, {acc3.id})",
            preconditions={
                f"AmbulanceFree({a1.id})",
                f"AmbulanceAt({a1.id}, {hotspot_node})",
                f"AccidentLocationConfirmed({acc3.id})",
                "TrafficDiverted(stadium_corridor)",
            },
            add_effects={
                f"AmbulanceEnRoute({a1.id}, {acc3.id})",
                f"AmbulanceBusy({a1.id})",
            },
            del_effects={
                f"AmbulanceFree({a1.id})",
            },
        )
    )

    # 2.6 DeliverPatientToH1(A1, ACC3, H1)
    actions.append(
        Action(
            name=f"DeliverPatientToH1({a1.id}, {acc3.id}, {h1.id})",
            preconditions={
                f"AmbulanceEnRoute({a1.id}, {acc3.id})",
                f"HospitalNotified({h1.id})",
            },
            add_effects={
                f"PatientAt({acc3.id}, {h1.id})",
                f"AccidentServed({acc3.id})",
                f"AmbulanceFree({a1.id})",
                f"AmbulanceAt({a1.id}, {h1.node})",
            },
            del_effects={
                f"AccidentNotServed({acc3.id})",
                f"AmbulanceEnRoute({a1.id}, {acc3.id})",
                f"AmbulanceBusy({a1.id})",
            },
        )
    )

    # 2.7 RerouteMidJourneyToH2(A1, ACC3, H2) – POP contingency action
    actions.append(
        Action(
            name=f"RerouteMidJourneyToH2({a1.id}, {acc3.id}, {h2.id})",
            preconditions={
                f"AmbulanceEnRoute({a1.id}, {acc3.id})",
                f"CTAvailable({h2.id})",
            },
            add_effects={
                f"PatientAt({acc3.id}, {h2.id})",
                f"AccidentServed({acc3.id})",
                f"AmbulanceFree({a1.id})",
            },
            del_effects={
                f"AccidentNotServed({acc3.id})",
                f"AmbulanceEnRoute({a1.id}, {acc3.id})",
                f"AmbulanceBusy({a1.id})",
            },
        )
    )

    # ----------------------------------------------------
    # 3. Goal State (for strict GraphPlan)
    # ----------------------------------------------------
    # Strict plan commits to taking the patient to H1 under current assumptions.
    goal_state: Set[Proposition] = {
        f"AccidentServed({acc3.id})",
        f"PatientAt({acc3.id}, {h1.id})",
    }

    return initial_state, actions, goal_state
