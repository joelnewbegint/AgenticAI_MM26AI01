import os
import matplotlib.pyplot as plt
import numpy as np
from my_agent import BaselineAgent, MyAgent


class MockEnvironment:
    """Non-stationary Bernoulli Multi-Armed Bandit Environment."""

    def __init__(self, n_actions: int = 5, switch_interval: int = 300, seed: int = None):
        self.n_actions = n_actions
        self.switch_interval = switch_interval
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.p_rewards = np.zeros(n_actions)
        self._switch_environment()

    def _switch_environment(self) -> None:
        self.p_rewards = self.rng.uniform(0.1, 0.9, size=self.n_actions)

    def step(self, action: int):
        self.step_count += 1
        if self.step_count % self.switch_interval == 0:
            self._switch_environment()

        p_chosen = self.p_rewards[action]
        reward = 1.0 if self.rng.rand() < p_chosen else 0.0

        best_p = float(np.max(self.p_rewards))
        expected_regret = best_p - p_chosen

        return reward, expected_regret, best_p, p_chosen


def run_trial(agent_class, n_actions, total_steps, switch_interval, seed):
    env = MockEnvironment(n_actions=n_actions, switch_interval=switch_interval, seed=seed)
    agent = agent_class(n_actions=n_actions)

    rewards = np.zeros(total_steps)
    regrets = np.zeros(total_steps)

    for step in range(total_steps):
        action = agent.act()
        reward, expected_regret, _, _ = env.step(action)
        agent.update(action, reward)

        rewards[step] = reward
        regrets[step] = expected_regret

    return np.cumsum(rewards), np.cumsum(regrets)


def run_benchmark(n_trials=20, total_steps=1200, n_actions=5, switch_interval=300):
    print(f"Running {n_trials} trials across BaselineAgent and MyAgent...")

    baseline_cum_rewards = np.zeros((n_trials, total_steps))
    baseline_cum_regrets = np.zeros((n_trials, total_steps))

    myagent_cum_rewards = np.zeros((n_trials, total_steps))
    myagent_cum_regrets = np.zeros((n_trials, total_steps))

    for trial in range(n_trials):
        seed = 42 + trial
        b_rew, b_reg = run_trial(
            BaselineAgent, n_actions, total_steps, switch_interval, seed
        )
        m_rew, m_reg = run_trial(MyAgent, n_actions, total_steps, switch_interval, seed)

        baseline_cum_rewards[trial] = b_rew
        baseline_cum_regrets[trial] = b_reg
        myagent_cum_rewards[trial] = m_rew
        myagent_cum_regrets[trial] = m_reg

    # Plot results
    steps = np.arange(1, total_steps + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Regret plot
    ax1.plot(
        steps,
        np.mean(baseline_cum_regrets, axis=0),
        label="BaselineAgent (Stationary TS)",
        color="red",
    )
    ax1.fill_between(
        steps,
        np.mean(baseline_cum_regrets, axis=0) - np.std(baseline_cum_regrets, axis=0),
        np.mean(baseline_cum_regrets, axis=0) + np.std(baseline_cum_regrets, axis=0),
        color="red",
        alpha=0.15,
    )

    ax1.plot(
        steps,
        np.mean(myagent_cum_regrets, axis=0),
        label="MyAgent (Discounted TS + Page-Hinkley)",
        color="blue",
    )
    ax1.fill_between(
        steps,
        np.mean(myagent_cum_regrets, axis=0) - np.std(myagent_cum_regrets, axis=0),
        np.mean(myagent_cum_regrets, axis=0) + np.std(myagent_cum_regrets, axis=0),
        color="blue",
        alpha=0.15,
    )

    ax1.set_title("Cumulative Expected Regret (Lower is Better)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Expected Regret")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Reward plot
    ax2.plot(
        steps,
        np.mean(baseline_cum_rewards, axis=0),
        label="BaselineAgent",
        color="red",
    )
    ax2.plot(steps, np.mean(myagent_cum_rewards, axis=0), label="MyAgent", color="blue")

    ax2.set_title("Cumulative Rewards (Higher is Better)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Total Reward")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path = "benchmark_result.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved simulation benchmark chart to {output_path}")

    # Summary Statistics
    print("\n" + "=" * 50)
    print("FINAL BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    print(
        f"BaselineAgent Mean Regret: {np.mean(baseline_cum_regrets[:, -1]):.2f} ± {np.std(baseline_cum_regrets[:, -1]):.2f}"
    )
    print(
        f"MyAgent Mean Regret:       {np.mean(myagent_cum_regrets[:, -1]):.2f} ± {np.std(myagent_cum_regrets[:, -1]):.2f}"
    )
    print(
        f"BaselineAgent Mean Reward: {np.mean(baseline_cum_rewards[:, -1]):.2f} ± {np.std(baseline_cum_rewards[:, -1]):.2f}"
    )
    print(
        f"MyAgent Mean Reward:       {np.mean(myagent_cum_rewards[:, -1]):.2f} ± {np.std(myagent_cum_rewards[:, -1]):.2f}"
    )
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()