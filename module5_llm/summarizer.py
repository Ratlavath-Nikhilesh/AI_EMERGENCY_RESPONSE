# module5_llm/summarizer.py

import os
from typing import Optional

from .prompt_templates import DecisionContext, build_system_prompt, build_user_prompt


def call_openai_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Helper to call an OpenAI chat model.
    If OPENAI_API_KEY or openai package is not available, raises an ImportError.

    You can change the model name as per your account (e.g., 'gpt-4.1-mini').
    """
    try:
        import openai  # type: ignore
    except ImportError as e:
        raise ImportError(
            "openai package not installed. Install with 'pip install openai' "
            "or disable LLM calling."
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Please export it as an environment variable."
        )

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or 'gpt-4.1-mini' / your chosen model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=400,
    )
    return response["choices"][0]["message"]["content"].strip()


def generate_briefing(
    context: DecisionContext,
    use_llm: bool = True,
) -> str:
    """
    Main entry-point for Module 5.

    - If use_llm == True and OpenAI is configured, calls the LLM with the
      engineered prompt.
    - Otherwise, falls back to a deterministic, template-based summary,
      so the code remains runnable in offline / lab environments.
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(context)

    if use_llm:
        try:
            return call_openai_chat(system_prompt, user_prompt)
        except Exception as e:
            # Fallback to deterministic output if LLM not available
            print(f"[WARN] LLM call failed ({e}); using fallback templated briefing.")
            return fallback_briefing(context)

    # If LLM use is disabled
    return fallback_briefing(context)


def fallback_briefing(context: DecisionContext) -> str:
    """
    Non-LLM fallback: still professional, but purely template-based.
    Used only if there is no API access during demonstration.
    """
    primary_inc = context.active_incidents[0] if context.active_incidents else {}
    primary_dispatch = context.dispatched_units[0] if context.dispatched_units else {}

    lines = []

    # Situation overview
    lines.append(
        f"As of {context.current_time}, under {context.weather} with "
        f"{context.road_condition} road conditions, the control room is managing "
        f"{len(context.active_incidents)} active incident(s)."
    )
    if primary_inc:
        lines.append(
            f"The primary case is {primary_inc.get('description', 'an incident')} "
            f"at {primary_inc.get('location', 'unknown location')}, with a "
            f"current severity estimate of {primary_inc.get('severity_estimate', 'unknown')} "
            f"(confidence approximately {primary_inc.get('severity_confidence', 0.0)*100:.0f}%)."
        )

    # Actions taken
    if primary_dispatch:
        lines.append(
            f"Ambulance {primary_dispatch.get('ambulance_id', 'A1')} has been dispatched "
            f"from {primary_dispatch.get('start_location', 'its base')} towards incident "
            f"{primary_dispatch.get('incident_id', 'N/A')} via "
            f"{primary_dispatch.get('route_summary', 'the current least-delay route')}, "
            f"with an expected hospital arrival time of "
            f"{primary_dispatch.get('eta_to_hospital_min', 'N/A')} minutes."
        )

    # Hospital rationale
    lines.append(
        f"The current destination hospital is {context.chosen_hospital_id}. "
        f"{context.chosen_hospital_reason}"
    )

    # Planning & RL rationale
    lines.append(
        f"Planning modules (GraphPlan/POP) recommend this sequence as a safe baseline, "
        f"while the reinforcement learning policy supports proactive positioning and "
        f"prioritisation under congested traffic. {context.planning_summary} "
        f"{context.rl_policy_comment}"
    )

    # Risk forecast
    lines.append(
        f"Based on the Bayesian risk forecast, {context.risk_forecast}"
    )

    # Safety notes
    lines.append(
        f"Operators are reminded that these recommendations are advisory and final "
        f"clinical and operational decisions remain with the senior duty officer. "
        f"{context.safety_notes}"
    )

    return " ".join(lines)
