# Change-Point Adaptive Multi-Armed Bandit Agent

**Project:** AgenticAI_MM26AI01  
**Framework:** Non-Stationary Multi-Armed Bandit with Discounted Thompson Sampling & Page-Hinkley Change-Point Detection.

---

## 1. Problem Statement

In classic Multi-Armed Bandit (MAB) problems, reward distributions for actions remain constant over time. In real-world applications (e.g., ad placement, user recommendation, financial allocation), environments are **non-stationary**: action probabilities change abruptly or drift over time without warning.

The agent only observes:
- The action it took.
- The binary reward received ($r \in \{0, 1\}$).

The agent has no prior information regarding when environment distribution shifts occur, which action distribution changed, or the true underlying probabilities.

---

## 2. Solution Architecture

The agent combines two complementary adaptation mechanisms: