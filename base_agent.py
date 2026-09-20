from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract Base Class for Multi-Armed Bandit Agents."""

    def __init__(self, n_actions: int):
        if n_actions <= 0:
            raise ValueError("n_actions must be a positive integer.")
        self.n_actions = n_actions

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal state of the agent."""
        pass

    @abstractmethod
    def act(self) -> int:
        """Select an action based on current knowledge/policy.

        Returns:
            int: Index of chosen action in [0, n_actions - 1].
        """
        pass

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """Update internal estimates after taking an action and receiving a reward.

        Args:
            action (int): The action taken.
            reward (float): Observed reward (typically 0 or 1 for Bernoulli bandit).
        """
        pass