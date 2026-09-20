import numpy as np
from base_agent import BaseAgent


class BaselineAgent(BaseAgent):
    """Standard Thompson Sampling for stationary Bernoulli Multi-Armed Bandits.

    Maintains standard Beta(alpha, beta) posteriors without discounting or
    change detection.
    """

    def __init__(self, n_actions: int):
        super().__init__(n_actions)
        self.reset()

    def reset(self) -> None:
        self.alpha = np.ones(self.n_actions, dtype=np.float64)
        self.beta = np.ones(self.n_actions, dtype=np.float64)

    def act(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        if reward > 0:
            self.alpha[action] += 1.0
        else:
            self.beta[action] += 1.0


class MyAgent(BaseAgent):
    """Adaptive Multi-Armed Bandit Agent for Non-Stationary Environments.

    Combines:
    1. Discounted Thompson Sampling (gradual adaptation to slow drift).
    2. Cumulative Page-Hinkley Change-Point Detection (rapid adaptation to abrupt shifts).
    """

    def __init__(
        self,
        n_actions: int,
        gamma: float = 0.985,
        ph_delta: float = 0.01,
        ph_threshold: float = 4.0,
        reset_strength: float = 0.2,
    ):
        super().__init__(n_actions)

        if not (0.0 < gamma < 1.0):
            raise ValueError("gamma must be in the open interval (0, 1).")
        if ph_delta <= 0:
            raise ValueError("ph_delta must be positive.")
        if ph_threshold <= 0:
            raise ValueError("ph_threshold must be positive.")
        if not (0.0 <= reset_strength <= 1.0):
            raise ValueError("reset_strength must be in the range [0, 1].")

        self.gamma = gamma
        self.ph_delta = ph_delta
        self.ph_threshold = ph_threshold
        self.reset_strength = reset_strength

        self.reset()

    def reset(self) -> None:
        # Prior Beta(1,1) for each action
        self.alpha = np.ones(self.n_actions, dtype=np.float64)
        self.beta = np.ones(self.n_actions, dtype=np.float64)

        # Page-Hinkley cumulative state variables
        self.ph_count = 0
        self.ph_mean = 0.0
        self.ph_sum = 0.0
        self.ph_min = 0.0

    def act(self) -> int:
        # Prevent non-positive values due to float precision limits
        safe_alpha = np.maximum(self.alpha, 1e-5)
        safe_beta = np.maximum(self.beta, 1e-5)

        samples = np.random.beta(safe_alpha, safe_beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        # 1. Discount historical observations towards Beta(1,1) prior foundation
        self.alpha = 1.0 + (self.alpha - 1.0) * self.gamma
        self.beta = 1.0 + (self.beta - 1.0) * self.gamma

        # 2. Add current observation
        if reward > 0:
            self.alpha[action] += 1.0
        else:
            self.beta[action] += 1.0

        # 3. Update Page-Hinkley Change-Point Detector
        self.ph_count += 1
        self.ph_mean += (reward - self.ph_mean) / self.ph_count
        deviation = reward - self.ph_mean - self.ph_delta
        self.ph_sum += deviation
        self.ph_min = min(self.ph_min, self.ph_sum)

        # 4. Detect change and perform partial reset if threshold exceeded
        if (self.ph_sum - self.ph_min) > self.ph_threshold:
            self._handle_change_detected()

    def _handle_change_detected(self) -> None:
        # Partially decay historical knowledge back towards Beta(1,1)
        self.alpha = 1.0 + (self.alpha - 1.0) * self.reset_strength
        self.beta = 1.0 + (self.beta - 1.0) * self.reset_strength

        # Reset detector cumulative statistics
        self.ph_count = 0
        self.ph_mean = 0.0
        self.ph_sum = 0.0
        self.ph_min = 0.0