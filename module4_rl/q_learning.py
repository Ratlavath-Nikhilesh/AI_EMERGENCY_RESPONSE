# module4_rl/q_learning.py

from typing import Dict, Tuple, Any, List
import random

import numpy as np

from .env import EmergencyResponseEnv, N_ACTIONS


State = Tuple[int, ...]
Action = int
QTable = Dict[Tuple[State, Action], float]


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q: QTable = {}

    def _get_q(self, state: State, action: Action) -> float:
        return self.q.get((state, action), 0.0)

    def _set_q(self, state: State, action: Action, value: float):
        self.q[(state, action)] = value

    def select_action(self, state: State, training: bool = True) -> Action:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(N_ACTIONS)

        # Greedy action
        q_values = [self._get_q(state, a) for a in range(N_ACTIONS)]
        max_q = max(q_values)
        best_actions = [a for a, qv in enumerate(q_values) if qv == max_q]
        return self.rng.choice(best_actions)

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ):
        old_q = self._get_q(state, action)
        if done:
            target = reward
        else:
            next_q_values = [self._get_q(next_state, a) for a in range(N_ACTIONS)]
            target = reward + self.gamma * max(next_q_values)

        new_q = old_q + self.alpha * (target - old_q)
        self._set_q(state, action, new_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(
    num_episodes: int = 3000,
    max_steps: int = 10,
    seed: int = 0,
) -> Tuple[QLearningAgent, List[float]]:
    env = EmergencyResponseEnv(max_steps=max_steps, seed=seed)
    agent = QLearningAgent(seed=seed)

    episode_rewards: List[float] = []

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    return agent, episode_rewards
