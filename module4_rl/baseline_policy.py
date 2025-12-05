# module4_rl/baseline_policy.py

from typing import Tuple

from .env import (
    EmergencyResponseEnv,
    ACC_NONE,
    ACT_IMMEDIATE_DISPATCH,
    ACT_DELAY_DISPATCH,
)


def baseline_policy(state: Tuple[int, ...]) -> int:
    """
    Simple hand-crafted policy:
    - If there is an active accident, always dispatch immediately.
    - Otherwise, do nothing (delay).
    """
    # State encoding:
    # (time_step, a1_status, a2_status, active_acc_loc, active_severity,
    #  predicted_hotspot, road_condition, hospital_load)
    active_acc_loc = state[3]
    if active_acc_loc != ACC_NONE:
        return ACT_IMMEDIATE_DISPATCH
    else:
        return ACT_DELAY_DISPATCH


def evaluate_baseline(
    num_episodes: int = 200,
    max_steps: int = 10,
    seed: int = 123,
):
    env = EmergencyResponseEnv(max_steps=max_steps, seed=seed)

    total_rewards = 0.0
    total_response_time = 0.0
    total_served = 0
    total_critical = 0
    critical_within = 0

    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0

        done = False
        while not done:
            action = baseline_policy(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state

        total_rewards += ep_reward
        total_response_time += info["total_response_time"]
        total_served += info["served_accidents"]
        total_critical += info["critical_total"]
        critical_within += info["critical_within_threshold"]

    avg_reward = total_rewards / num_episodes
    avg_response_time = total_response_time / max(total_served, 1)
    crit_within_rate = (
        critical_within / total_critical if total_critical > 0 else 0.0
    )

    return {
        "avg_reward": avg_reward,
        "avg_response_time": avg_response_time,
        "served_accidents": total_served / num_episodes,
        "critical_within_rate": crit_within_rate,
    }
