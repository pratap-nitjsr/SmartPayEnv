import math
from dataclasses import dataclass, field
from typing import List


# -----------------------------
# Routing Efficacy Grader
# -----------------------------
@dataclass
class RoutingEfficacyGrader:
    """
    Grades routing decisions on DECISION QUALITY, not luck.

    v3 fix: uses deterministic `expected_outcome` (gateway_rate × user_history)
    instead of a binary random `success` flag.  The agent now gets a reliable,
    learnable gradient: pick the best gateway for this user → score goes up,
    regardless of the random draw that determines whether the tx actually cleared.

    Weights:
      alpha  – outcome scale (maps expected_outcome [0,1] → [-alpha, +alpha])
      beta   – cost penalty per dollar spent
      gamma  – retry penalty per retry attempt
      delta  – decision-quality bonus (how close to optimal gateway?)
    """
    alpha: float = 1.2
    beta: float  = 0.15
    gamma: float = 0.4
    delta: float = 0.8

    def evaluate(
        self,
        expected_outcome: float,
        cost: float,
        retries: int,
        chosen_gateway: int,
        gateway_rates: List[float],
    ) -> float:
        """
        Compute a fully DETERMINISTIC routing score in [0, 1].

        Args:
            expected_outcome: gateway_rates[chosen] * user_history_score — the
                              deterministic success probability given state+action.
                              Maps [0, 1] → outcome_term in [-alpha, +alpha].
            cost:             Total gateway cost incurred.
            retries:          Number of retries used.
            chosen_gateway:   Index of the gateway the agent chose.
            gateway_rates:    Current success-rate estimates for all gateways.
        """
        best_rate        = max(gateway_rates) if gateway_rates else 1.0
        chosen_rate      = gateway_rates[chosen_gateway] if gateway_rates else 1.0
        decision_quality = (chosen_rate / best_rate) if best_rate > 0 else 0.0

        # Deterministic: map expected_outcome [0,1] → [-alpha, +alpha]
        outcome_term = self.alpha * (2.0 * expected_outcome - 1.0)
        penalty      = (self.beta * cost) + (self.gamma * retries)

        raw_score = outcome_term - penalty + (self.delta * decision_quality)
        # Strictly between (0, 1)
        return max(0.001, min(0.999, self._sigmoid(raw_score)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))


# -----------------------------
# Fraud Detection Grader
# -----------------------------
class FraudDetectionGrader:
    """
    Grades fraud blocking accuracy using normalized Matthews Correlation
    Coefficient (MCC), mapped to [0, 1].
    """
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def add_step(self, predicted_block: bool, actual_fraud: bool) -> None:
        """Update confusion matrix."""
        if predicted_block and actual_fraud:
            self.tp += 1
        elif predicted_block and not actual_fraud:
            self.fp += 1
        elif not predicted_block and actual_fraud:
            self.fn += 1
        else:
            self.tn += 1

    def evaluate(self) -> float:
        """
        Compute normalized MCC → [0, 1].
        Returns 0.5 (neutral) when denominator is zero (all same class).
        """
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = math.sqrt(
            (self.tp + self.fp) *
            (self.tp + self.fn) *
            (self.tn + self.fp) *
            (self.tn + self.fn)
        )
        if denominator == 0:
            return 0.5  # Neutral — insufficient data to compute MCC
        mcc = numerator / denominator
        score = (mcc + 1.0) / 2.0  # Normalize [-1, 1] → [0, 1]
        return max(0.001, min(0.999, score))


# -----------------------------
# User Retention Grader
# -----------------------------
class UserRetentionGrader:
    """
    Models user churn using exponential decay driven by consecutive failures.
    """
    def __init__(self, churn_rate: float = 0.1, initial_users: int = 100):
        self.churn_rate = churn_rate
        self.total_users = initial_users
        self.survived_users = float(initial_users)

    def add_step(self, consecutive_failures: int) -> None:
        """Model user drop-off from consecutive transaction failures."""
        if consecutive_failures <= 0:
            return
        hazard = 1.0 - math.exp(-self.churn_rate * (consecutive_failures ** 2))
        lost = self.survived_users * hazard
        self.survived_users = max(0.0, self.survived_users - lost)

    def evaluate(self) -> float:
        """Return retention ratio strictly in (0, 1)."""
        score = self.survived_users / self.total_users
        return max(0.001, min(0.999, score))


# -----------------------------
# Combined Reward Function
# -----------------------------
def process_combined_reward(
    route_score: float,
    fraud_detected: bool,
    false_positive: bool,
    retries: int
) -> float:
    """
    Combines signals into a single reward score [0, 1].
    Used for the payment_optimization task.
    """
    fraud_bonus   =  1.5 if fraud_detected  else 0.0
    false_penalty = -2.0 if false_positive  else 0.0
    retry_penalty = -0.2 * retries

    raw = route_score + fraud_bonus + false_penalty + retry_penalty
    score = 1.0 / (1.0 + math.exp(-raw))
    return max(0.001, min(0.999, score))