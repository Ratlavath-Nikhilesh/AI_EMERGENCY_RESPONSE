# module5_llm/demo_llm_summary.py

from .prompt_templates import DecisionContext
from .summarizer import generate_briefing


def build_non_trivial_context() -> DecisionContext:
    """
    Constructs a realistic non-trivial scenario using the Bhubaneswar story
    from the problem statement.
    """

    active_incidents = [
        {
            "id": "ACC3",
            "description": "SUV rollover with airbags deployed",
            "location": "Airport approach road, southbound lane",
            "severity_estimate": "Critical polytrauma with possible head injury",
            "severity_confidence": 0.82,
            "eta_minutes": 8.0,
        },
        {
            "id": "ACC2",
            "description": "Car–auto collision with suspected limb fractures",
            "location": "Rupali Square junction",
            "severity_estimate": "Moderate trauma",
            "severity_confidence": 0.65,
            "eta_minutes": 12.0,
        },
    ]

    dispatched_units = [
        {
            "ambulance_id": "A1",
            "incident_id": "ACC3",
            "start_location": "pre-deployed near Nayapalli chowk (between H1 and stadium cluster)",
            "route_summary": "Nayapalli → Airport Road via stadium diversion corridor",
            "target_hospital_id": "H1",
            "eta_to_hospital_min": 18.0,
        },
        {
            "ambulance_id": "A2",
            "incident_id": "ACC2",
            "start_location": "H2 base near Rupali",
            "route_summary": "Direct via Rupali Square arterial road",
            "target_hospital_id": "H2",
            "eta_to_hospital_min": 10.0,
        },
    ]

    hospital_status = [
        {
            "id": "H1",
            "distance_km": 11.0,
            "trauma_capability": "Level-1 trauma centre with neurosurgery on call",
            "ct_status": "CT available",
            "load_level": "High but within safe capacity",
        },
        {
            "id": "H2",
            "distance_km": 6.0,
            "trauma_capability": "General emergency unit",
            "ct_status": "CT offline for the next 30–40 minutes",
            "load_level": "Moderate crowding at triage",
        },
    ]

    chosen_hospital_reason = (
        "Although H2 is geographically closer to the airport corridor, its CT scanner is "
        "currently offline and neurosurgical support is limited. Given the high-speed rollover "
        "mechanism and strong suspicion of intracranial injury, H1 was chosen as the primary "
        "destination despite the additional distance, to avoid secondary transfer delays and "
        "ensure direct access to trauma imaging and neurosurgical intervention."
    )

    planning_summary = (
        "The GraphPlan sequence recommends early pre-deployment of A1 near the stadium-airport "
        "corridor, activation of stadium traffic diversions, and direct routing of the critical "
        "case to a trauma-capable centre. POP contingencies keep open the option of diverting "
        "to H2 only if CT comes online and H1’s load reaches unsafe levels."
    )

    rl_policy_comment = (
        "The reinforcement learning agent supports pre-positioning A1 near predicted hotspots, "
        "prioritising the ACC3 critical case when it occurs, and favouring routes that trade a "
        "small increase in travel distance for a significant reduction in expected delay due to "
        "rain and match-related congestion."
    )

    risk_forecast = (
        "there is a high risk of occult internal bleeding and airway compromise in the ACC3 "
        "patient over the next 20–30 minutes, with a non-trivial chance of deterioration en route. "
        "Operationally, there is a moderate probability of a second moderate-severity incident "
        "around Rupali or the stadium exit within the next 30 minutes based on historical match-day data."
    )

    safety_notes = (
        "This briefing is based on current sensor feeds, historical congestion patterns, and predictive "
        "models. It does not replace on-scene clinical judgement or the authority of the duty medical "
        "officer. Any significant deviation in patient condition, road closures, or hospital capacity "
        "should trigger a rapid re-evaluation of the plan."
    )

    return DecisionContext(
        current_time="20:23 hrs",
        weather="intense rain with reduced visibility",
        road_condition="severely congested around stadium and main corridors",
        active_incidents=active_incidents,
        dispatched_units=dispatched_units,
        hospital_status=hospital_status,
        chosen_hospital_id="H1",
        chosen_hospital_reason=chosen_hospital_reason,
        planning_summary=planning_summary,
        rl_policy_comment=rl_policy_comment,
        risk_forecast=risk_forecast,
        safety_notes=safety_notes,
    )


def run_demo(use_llm: bool = False):
    """
    Run a non-trivial test case for Module 5.

    Set use_llm=True if you have an OpenAI API key configured and want
    to see a real LLM-generated briefing.
    """
    context = build_non_trivial_context()
    briefing = generate_briefing(context, use_llm=use_llm)

    print("####################################################")
    print("Module 5: LLM-Based Control Room Summary – Demo Case")
    print("####################################################\n")
    print(briefing)


if __name__ == "__main__":
    # By default, use fallback (no external API). Set to True if LLM is available.
    run_demo(use_llm=False)
