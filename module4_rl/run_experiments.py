# module4_rl/run_experiments.py

from typing import Dict, Any

from .env import EmergencyResponseEnv, ACC_NONE
from .q_learning import train_q_learning, QLearningAgent
from .baseline_policy import evaluate_baseline
from .env import N_ACTIONS


def evaluate_agent(
    agent: QLearningAgent,
    num_episodes: int = 200,
    max_steps: int = 10,
    seed: int = 999,
) -> Dict[str, Any]:
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
            # Greedy action (no exploration)
            q_values = [agent._get_q(state, a) for a in range(N_ACTIONS)]
            max_q = max(q_values)
            best_actions = [a for a, qv in enumerate(q_values) if qv == max_q]
            action = agent.rng.choice(best_actions)

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


def run_module4_experiments():
    print("###############################################")
    print("Module 4: RL for Proactive Resource Allocation")
    print("###############################################\n")

    # 1. Train Q-learning agent
    print("Training Q-learning agent...")
    agent, rewards = train_q_learning(num_episodes=3000, max_steps=10, seed=42)
    print(f"Training completed. Last 10 episode rewards: {rewards[-10:]}\n")

    # 2. Evaluate baseline
    print("Evaluating baseline (always dispatch immediately)...")
    baseline_metrics = evaluate_baseline(num_episodes=300, max_steps=10, seed=123)
    print("Baseline metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.3f}")
    print()

    # 3. Evaluate learned Q-policy
    print("Evaluating Q-learning policy...")
    rl_metrics = evaluate_agent(agent, num_episodes=300, max_steps=10, seed=999)
    print("Q-learning metrics:")
    for k, v in rl_metrics.items():
        print(f"  {k}: {v:.3f}")
    print()

    print("Comparison:")
    print("  - avg_reward:     RL vs Baseline =", rl_metrics["avg_reward"], "vs", baseline_metrics["avg_reward"])
    print("  - avg_resp_time:  RL vs Baseline =", rl_metrics["avg_response_time"], "vs", baseline_metrics["avg_response_time"])
    print("  - crit_within_rate RL vs Baseline =", rl_metrics["critical_within_rate"], "vs", baseline_metrics["critical_within_rate"])
    print("\nInterpretation:")
    print("  A better RL policy should achieve higher average reward,")
    print("  lower average response time, and a higher fraction of critical")
    print("  cases served within the time threshold, compared to the naive")
    print("  'dispatch immediately to nearest' baseline.")


if __name__ == "__main__":
    run_module4_experiments()
