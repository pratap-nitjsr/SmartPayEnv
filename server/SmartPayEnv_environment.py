# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SmartPayEnv v3 — Advanced Fintech Reality Layer.

High-fidelity benchmark for RL agents in the payment domain.
Features: 3D Secure (3DS), Chargeback Delays, BIN Affinity, Dynamic Costs, & Cohorts.
"""

import numpy as np
from collections import deque
from uuid import uuid4
from dataclasses import dataclass, field

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import SmartpayenvAction, SmartpayenvObservation
except (ImportError, ValueError):
    from models import SmartpayenvAction, SmartpayenvObservation

try:
    from .graders import RoutingEfficacyGrader, FraudDetectionGrader, UserRetentionGrader
except (ImportError, ValueError):
    from server.graders import RoutingEfficacyGrader, FraudDetectionGrader, UserRetentionGrader


# ── Configuration Constants ────────────────────────────────────────────
GATEWAY_COST_FIXED = [0.10, 0.30, 0.50]   # Flat fee per tx
GATEWAY_FEE_PCT    = [0.02, 0.025, 0.035] # % of amount

# BIN Affinity: Multiplier for success_prob based on [GatewayIndex][BIN_Category]
# Values aligned with the agent's Knowledge-Rich Prompt in inference.py
BIN_AFFINITY = [
    [0.95, 0.80, 0.70, 0.60, 0.50, 0.90, 0.75, 0.65, 0.55, 0.85], # Gateway 0
    [0.60, 0.95, 0.80, 0.70, 0.60, 0.55, 0.90, 0.75, 0.65, 0.50], # Gateway 1
    [0.50, 0.60, 0.95, 0.85, 0.75, 0.50, 0.60, 0.95, 0.85, 0.75]  # Gateway 2
]

GATEWAY_RETRY_PENALTY = 0.2

DIFFICULTY_CONFIG = {
    0: {   # easy
        "fraud_base_rate":    0.02,
        "instability":        0.05,
        "churn_rate":         0.05,
    },
    1: {   # medium
        "fraud_base_rate":    0.15,
        "instability":        0.15,
        "churn_rate":         0.15,
    },
    2: {   # hard
        "fraud_base_rate":    0.25,
        "instability":        0.30,
        "churn_rate":         0.25,
    },
}

@dataclass
class State:
    episode_id: str
    step_count: int
    consecutive_failures: int = 0
    fraud_wave_drift: float = 0.0
    market_volatility: float = 0.0
    chargeback_queue: list = field(default_factory=list)


class _GatewayState:
    """State machine for one payment gateway with realistic drift."""
    def __init__(self, base_rate: float, instability: float, rng: np.random.Generator):
        self.base_rate  = base_rate
        self.instability = instability
        self._rng       = rng
        self.state      = "normal"
        self._countdown = 0
        self.current_rate = base_rate

    def step(self) -> None:
        if self.state == "normal":
            if self._rng.random() < self.instability:
                self.state      = "degraded"
                self._countdown = int(self._rng.integers(3, 10))
                self.current_rate = self.base_rate * self._rng.uniform(0.2, 0.5)
        elif self.state == "degraded":
            self._countdown -= 1
            if self._countdown <= 0:
                self.state        = "recovering"
                self._countdown   = int(self._rng.integers(2, 5))
        elif self.state == "recovering":
            self._countdown -= 1
            self.current_rate = min(self.base_rate, self.current_rate + (self.base_rate - self.current_rate) * 0.4)
            if self._countdown <= 0:
                self.state        = "normal"
                self.current_rate = self.base_rate
        
        if self.state == "normal":
            noise = self._rng.normal(0, 0.01)
            self.current_rate = float(np.clip(self.current_rate + noise, 0.1, 1.0))


class SmartpayenvEnvironment(Environment):
    """
    Production-grade Payment Environment.
    Models the 'Messy Reality': 3DS friction, delayed chargeback risk, 
    bank affinity, and user segments.
    """
    def __init__(self):
        self._state        = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count  = 0
        self._difficulty   = 0
        self._cfg          = DIFFICULTY_CONFIG[0]
        self._rng          = np.random.default_rng()
        self._gateways     = []
        self.route_grader     = RoutingEfficacyGrader()
        self.fraud_grader     = FraudDetectionGrader()
        self.retention_grader = UserRetentionGrader()
        self._velocity_buffer = deque(maxlen=5)
        self.current_obs   = None

    def _init_gateways(self) -> None:
        instability = self._cfg["instability"]
        self._gateways = [
            _GatewayState(0.96, instability, self._rng),
            _GatewayState(0.98, instability, self._rng),
            _GatewayState(0.99, instability, self._rng),
        ]

    def _generate_transaction(self) -> SmartpayenvObservation:
        # 1. Advanced Diurnal Cycle (UTC)
        # Peak Fraud: 01:00 - 05:00. Peak Volume: 12:00 - 20:00
        hour = int(self._state.step_count % 24)
        is_night = (1 <= hour <= 5)
        
        # 2. User Segments (Cohorts)
        segment = int(self._rng.choice([0, 1, 2], p=[0.25, 0.60, 0.15])) # 0=New, 1=Existing, 2=Premium
        
        # Segment behavioral traits
        fraud_mult = {0: 1.8, 1: 1.0, 2: 0.3}[segment]
        history_mu  = {0: 0.3, 1: 0.7, 2: 0.9}[segment]
        
        # 3. Correlated Merchant Categories (MCC)
        mcc = int(self._rng.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.2]))
        
        # MCC-Amount Correlation
        amount_mu = {0: 4.0, 1: 4.5, 2: 6.5, 3: 7.0, 4: 5.0, 5: 3.0}[mcc]
        amount = float(self._rng.lognormal(mean=amount_mu, sigma=0.8))
        
        # 4. Statistical Fraud Model
        wave_drift = self._state.fraud_wave_drift
        category_risk = {0: 0.02, 1: 0.05, 2: 0.15, 3: 0.08, 4: 0.25, 5: 0.12}[mcc]
        
        base_risk = self._cfg["fraud_base_rate"] + wave_drift + category_risk
        if is_night: base_risk += 0.25 # Night surge
        
        is_international = bool(self._rng.random() < (0.4 if mcc == 3 else 0.15))
        device_type = int(self._rng.choice([0, 1, 2], p=[0.5, 0.4, 0.1])) # 0=Mobile, 1=Web, 2=Unknown
        
        final_risk = base_risk + (0.15 if is_international else 0.0)
        final_risk += (0.2 if device_type == 2 else 0.0)
        
        fraud_risk_score = float(np.clip(final_risk * fraud_mult, 0.01, 0.99))
        user_history_score = float(np.clip(self._rng.normal(history_mu, 0.15), 0.1, 1.0))

        # 5. Other Transactional Features
        bin_category = int(self._rng.integers(0, 10))
        card_present = bool(self._rng.random() > 0.6 if is_night else 0.3)
        
        # Velocity and Fraud Risk (History Buffer)
        velocity = float(np.clip(self._rng.random() * 0.2 + (0.5 if is_night else 0.0), 0.1, 0.9))

        return SmartpayenvObservation(
            amount=amount,
            merchant_category=mcc,
            is_international=is_international,
            card_present=card_present,
            user_type=0, 
            user_segment=segment,
            user_history_score=user_history_score,
            device_type=device_type,
            bin_category=bin_category,
            transaction_velocity=velocity,
            time_of_day=hour,
            gateway_success_rates=[g.current_rate for g in self._gateways],
            gateway_states=[g.state for g in self._gateways],
            fraud_risk_score=fraud_risk_score,
            previous_failures=self._state.consecutive_failures,
            difficulty=self._difficulty,
            reward=0.5,
            done=False,
            task_routing_score=0.5,
            task_fraud_mcc_score=0.5,
            task_retention_score=0.5,
        )

    def reset(self, difficulty: int = 0) -> SmartpayenvObservation:
        self._difficulty = int(np.clip(difficulty, 0, 2))
        self._cfg        = DIFFICULTY_CONFIG[self._difficulty]
        self._state      = State(episode_id=str(uuid4()), step_count=0)
        self._init_gateways()
        self.route_grader     = RoutingEfficacyGrader()
        self.fraud_grader     = FraudDetectionGrader()
        self.retention_grader = UserRetentionGrader(churn_rate=self._cfg["churn_rate"])
        self._velocity_buffer.clear()
        self.current_obs = self._generate_transaction()
        return self.current_obs

    def step(self, action: SmartpayenvAction) -> SmartpayenvObservation:
        self._state.step_count += 1
        if self.current_obs is None: self.reset()
        
        obs = self.current_obs
        assert obs is not None # Satisfy type checker
        # 0. Stochastic Reality Drift
        # Fraud Wave: base rate drifts every step
        if self._state.step_count % 5 == 0:
            drift = self._rng.normal(0, 0.05)
            self._state.fraud_wave_drift = np.clip(self._state.fraud_wave_drift + drift, -0.1, 0.2)
        
        # Systemic Volatility: 5% chance of market-wide degradation
        if self._rng.random() < 0.05:
            for g in self._gateways:
                if g.state == "normal":
                    g.state = "degraded"
                    g._countdown = int(self._rng.integers(4, 9))
                    g.current_rate = g.current_rate * 0.7

        for gw in self._gateways: gw.step()

        # 1. 3DS / Action Logic
        is_fraud     = (obs.fraud_risk_score >= 0.65)
        action_block = (action.fraud_decision == 1)
        action_3ds   = (action.fraud_decision == 2)
        
        self.fraud_grader.add_step(action_block or action_3ds, is_fraud)

        done = False
        success = False
        retries = 0
        gateway = action.gateway
        total_cost = 0.0
        cb_penalty_this_step = 0.0

        if action_block:
            route_score = obs.fraud_risk_score if is_fraud else (obs.fraud_risk_score * 0.3)
            done = True
        else:
            gw_rates = [g.current_rate for g in self._gateways]
            
            # BIN Affinity & 3DS Support
            affinity = BIN_AFFINITY[gateway][obs.bin_category]
            
            # Extreme Reality Scaling: mismatched BINs now fail aggressively
            if affinity < 0.9:
                affinity = affinity * 0.15 # Harsh penalty for subpar routing
                
            # 3DS reduces remaining fraud risk by 90%
            eff_fraud_risk = obs.fraud_risk_score * (0.1 if action_3ds else 1.0)
            expected_outcome = gw_rates[gateway] * (1.0 - eff_fraud_risk) * affinity
            expected_outcome = float(np.clip(expected_outcome, 0.05, 1.0))

            # Simulate outcome (Friction varies by segment: New = high distrust/abandonment)
            abandon_prob = {0: 0.25, 1: 0.10, 2: 0.05}[obs.user_segment]
            if action_3ds and self._rng.random() < abandon_prob:
                success = False # User abandonment
            else:
                success = bool(self._rng.random() < expected_outcome)

            if not success and action.retry_strategy == 1 and not action_3ds:
                retries += 1
                gateway  = (gateway + 1) % 3
                affinity = BIN_AFFINITY[gateway][obs.bin_category]
                expected_outcome = gw_rates[gateway] * (1.0 - obs.fraud_risk_score) * affinity
                success = bool(self._rng.random() < expected_outcome)

            # Dynamic Cost: % + flat
            total_cost = (obs.amount * GATEWAY_FEE_PCT[gateway]) + GATEWAY_COST_FIXED[gateway]
            if retries > 0:
                total_cost += (obs.amount * GATEWAY_FEE_PCT[action.gateway]) + GATEWAY_COST_FIXED[action.gateway]

            route_score = self.route_grader.evaluate(
                expected_outcome=expected_outcome,
                cost=total_cost,
                retries=retries,
                chosen_gateway=action.gateway,
                gateway_rates=gw_rates,
            )

            # Success Logic
            if success:
                self._state.consecutive_failures = 0
            else:
                self._state.consecutive_failures += 1
                self.retention_grader.add_step(self._state.consecutive_failures)

            # Churn Impact (Friction/Failure)
            if action_3ds: 
                self.retention_grader.add_step(1) # Friction bump
                
            # Delayed Chargeback: undetected fraud hit later (unless protected by 3DS)
            if success and is_fraud and not action_3ds:
                delay = self._rng.integers(20, 45)
                self._state.chargeback_queue.append((self._state.step_count + delay, obs.amount + 20.0))

        # Process maturation
        cb_amt: float = 0.0
        pending = []
        for mat, pen in self._state.chargeback_queue:
            if self._state.step_count >= mat: 
                cb_amt = cb_amt + float(pen)
            else: 
                pending.append((mat, pen))
        self._state.chargeback_queue = pending

        # Finalize
        self.current_obs = self._generate_transaction()
        self.current_obs.gateway_success_rates = [g.current_rate for g in self._gateways]
        self.current_obs.gateway_states        = [g.state for g in self._gateways]
        self.current_obs.chargeback_penalty_applied = cb_amt
        
        if done or self._state.step_count >= 100: self.current_obs.done = True
        
        fs = self.fraud_grader.evaluate()
        rs = self.retention_grader.evaluate()
        base_reward = (0.4 * route_score) + (0.4 * fs) + (0.2 * rs)
        
        # Norm punishment for chargebacks
        final_reward = base_reward - (cb_amt / 150.0)
        self.current_obs.reward = float(np.clip(final_reward, 0.001, 0.999))
        
        self.current_obs.task_routing_score = route_score
        self.current_obs.task_fraud_mcc_score = fs
        self.current_obs.task_retention_score = rs

        return self.current_obs

    @property
    def state(self) -> State:
        return self._state
