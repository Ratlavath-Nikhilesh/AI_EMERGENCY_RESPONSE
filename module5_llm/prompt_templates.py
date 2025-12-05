# module5_llm/prompt_templates.py

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DecisionContext:
    """
    Structured information coming from Modules 1–4.
    This is what the LLM will use to generate the briefing.
    """
    current_time: str
    weather: str
    road_condition: str

    # Active incidents (at least one)
    active_incidents: List[Dict[str, Any]]

    # Dispatch & routing decisions actually taken
    dispatched_units: List[Dict[str, Any]]

    # Hospital status and selection details
    hospital_status: List[Dict[str, Any]]
    chosen_hospital_id: str
    chosen_hospital_reason: str

    # Planning and RL information for justification
    planning_summary: str            # from GraphPlan / POP
    rl_policy_comment: str           # from Module 4 (why this action sequence)
    risk_forecast: str               # future risk / escalation notes from Module 1
    safety_notes: str                # disclaimers, operator reminders, etc.


def build_system_prompt() -> str:
    """
    System prompt: defines role, tone, and safety constraints.
    You will submit this text in the report.
    """
    return (
        "You are an emergency medical control room decision-support assistant. "
        "You generate short, professional briefings for duty medical officers and "
        "dispatch supervisors during active incidents.\n\n"
        "Your briefings must:\n"
        "- Be concise but complete (8–14 sentences or around 150–250 words).\n"
        "- Sound professional, time-critical, and responsibility-aware.\n"
        "- Avoid casual language, jokes, emojis, or speculation beyond the data.\n"
        "- State uncertainty explicitly using phrases like 'based on current data' or "
        "'with medium confidence' instead of sounding overconfident.\n"
        "- Explain, in plain language, WHY a particular ambulance was dispatched, "
        "WHY a hospital was chosen (even if farther), and what the main predicted "
        "clinical and operational risks are.\n"
        "- Include contingencies where relevant (e.g., what to do if CT scanner "
        "remains offline or traffic worsens).\n"
        "- Never override human authority: always treat your output as a recommendation, "
        "not an order.\n"
    )


def build_user_prompt(context: DecisionContext) -> str:
    """
    User prompt template: injects structured context into natural-language instructions.
    You will also submit this (structure + example filling) in the report.
    """
    # Format active incidents
    incidents_text_lines = []
    for inc in context.active_incidents:
        incidents_text_lines.append(
            f"- ID {inc['id']}: {inc['description']} at {inc['location']} "
            f"(severity estimate: {inc['severity_estimate']} with "
            f"{inc['severity_confidence']*100:.0f}% confidence; "
            f"ETA if dispatched now: {inc.get('eta_minutes', 'N/A')} min)"
        )
    incidents_text = "\n".join(incidents_text_lines)

    # Dispatched units
    dispatched_text_lines = []
    for d in context.dispatched_units:
        dispatched_text_lines.append(
            f"- Ambulance {d['ambulance_id']} dispatched to {d['incident_id']} from "
            f"{d['start_location']} using route via {d['route_summary']}; "
            f"target hospital: {d['target_hospital_id']} "
            f"(expected arrival in {d['eta_to_hospital_min']} min)."
        )
    dispatched_text = "\n".join(dispatched_text_lines)

    # Hospital status
    hospital_status_lines = []
    for h in context.hospital_status:
        hospital_status_lines.append(
            f"- {h['id']}: distance {h['distance_km']} km, "
            f"trauma capability: {h['trauma_capability']}, "
            f"CT status: {h['ct_status']}, "
            f"current load: {h['load_level']}."
        )
    hospital_status_text = "\n".join(hospital_status_lines)

    user_prompt = f"""
Current time: {context.current_time}
Weather and roads: {context.weather}; road condition: {context.road_condition}

Active incidents:
{incidents_text}

Dispatch and routing decisions taken:
{dispatched_text}

Hospital status:
{hospital_status_text}

Chosen hospital for the primary high-risk case: {context.chosen_hospital_id}
Reason for choosing this hospital (internal system explanation):
{context.chosen_hospital_reason}

Planning layer (GraphPlan / POP) summary:
{context.planning_summary}

Reinforcement learning policy behaviour summary:
{context.rl_policy_comment}

Bayesian future risk / escalation forecast:
{context.risk_forecast}

Safety and responsibility notes:
{context.safety_notes}

TASK:
Using the information above, write a single operator-ready briefing suitable
for being read aloud or quickly scanned on screen by the duty doctor and
dispatch supervisor.

Structure your response as 3–5 short paragraphs with clear topic flow:
1) Situation overview.
2) Actions already taken (ambulance dispatch, traffic measures).
3) Rationale for ambulance and hospital choices, tying together traffic, severity,
   and hospital capability, including why a farther hospital may still be preferred.
4) Key clinical and operational risks over the next 30–60 minutes, and contingency
   plans (e.g., if CT remains offline, if traffic worsens, if a second critical
   incident occurs).

Use neutral, professional language. Do NOT invent additional facts that are not
consistent with the input. Where data is uncertain, clearly state this.
"""
    return user_prompt.strip()
