import numpy as np
import pytest
from my_agent import BaselineAgent, MyAgent


def test_agent_initialization():
    agent = MyAgent(n_actions=5)
    assert agent.n_actions == 5
    assert len(agent.alpha) == 5
    assert len(agent.beta) == 5
    assert np.all(agent.alpha == 1.0)
    assert np.all(agent.beta == 1.0)


def test_act_returns_valid_index():
    for n_actions in [2, 5, 10]:
        agent = MyAgent(n_actions=n_actions)
        for _ in range(100):
            action = agent.act()
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < n_actions


def test_update_accepts_valid_rewards():
    agent = MyAgent(n_actions=3)
    agent.update(0, 1.0)
    agent.update(1, 0.0)
    assert np.all(agent.alpha > 0)
    assert np.all(agent.beta > 0)


def test_reset_clears_state():
    agent = MyAgent(n_actions=4)
    agent.update(0, 1.0)
    agent.update(0, 1.0)
    agent.reset()
    assert np.all(agent.alpha == 1.0)
    assert np.all(agent.beta == 1.0)
    assert agent.ph_count == 0


def test_numeric_stability():
    agent = MyAgent(n_actions=3)
    for _ in range(2000):
        action = agent.act()
        agent.update(action, np.random.choice([0.0, 1.0]))

    assert not np.isnan(agent.alpha).any()
    assert not np.isnan(agent.beta).any()
    assert not np.isinf(agent.alpha).any()
    assert np.all(agent.alpha > 0)
    assert np.all(agent.beta > 0)


def test_baseline_agent():
    agent = BaselineAgent(n_actions=4)
    act = agent.act()
    assert 0 <= act < 4
    agent.update(act, 1.0)
    assert agent.alpha[act] == 2.0