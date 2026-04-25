# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SmartPayEnv — Advanced Fintech Reality Layer.

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
    from .utils import LogLoader
except (ImportError, ValueError):
    from server.graders import RoutingEfficacyGrader, FraudDetectionGrader, UserRetentionGrader
    from server.utils import LogLoader


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
    health_lag_buffer: deque = field(default_factory=lambda: deque(maxlen=3)) # 2-step lag
    true_fraud_risk: float = 0.0
    simulation_hour: int = 0
    active_events: dict = field(default_factory=dict) # e.g. {"fraud_spike": 10, "outage": 5}
    log_cursor: int = 0
    review_queue: list = field(default_factory=list) # [{ 'step': int, 'is_fraud': bool, 'amount': float }]
    curriculum_level: float = 0.0
    policy_skill_estimate: float = 0.5
    challenger_skill: float = 0.55
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=25))
    recent_route_scores: deque = field(default_factory=lambda: deque(maxlen=25))
    recent_fraud_scores: deque = field(default_factory=lambda: deque(maxlen=25))
    recent_retention_scores: deque = field(default_factory=lambda: deque(maxlen=25))
    anti_gaming_alerts: int = 0


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
        self._log_loader   = LogLoader()
        self._pattern_queue = deque()
        self._meta_curriculum_enabled = True

    def _init_gateways(self) -> None:
        instability = self._cfg["instability"]
        self._gateways = [
            _GatewayState(0.96, instability, self._rng),
            _GatewayState(0.98, instability, self._rng),
            _GatewayState(0.99, instability, self._rng),
        ]

    def _generate_transaction(self) -> SmartpayenvObservation:
        # Check if we have a queued pattern to replay
        if self._pattern_queue:
            log_entry = self._pattern_queue.popleft()
        else:
            # Sample sequentially from logs to maintain temporal correlation
            noise = {0: 0.05, 1: 0.15, 2: 0.3}[self._difficulty]
            log_entry = self._log_loader.sample(index=self._state.log_cursor, noise_level=noise)
            self._state.log_cursor += 1

        if log_entry is None:
            # Fallback to random if logs fail (shouldn't happen)
            return self._generate_fallback_transaction()

        true_risk = float(log_entry["fraud_risk_score"])
        self._state.true_fraud_risk = true_risk

        return SmartpayenvObservation(
            amount=float(log_entry["amount"]),
            merchant_category=int(log_entry["merchant_category"]),
            is_international=bool(log_entry["is_international"]),
            card_present=bool(log_entry["card_present"]),
            user_type=0, 
            user_segment=int(log_entry["user_segment"]),
            user_history_score=float(log_entry["user_history_score"]),
            device_type=int(log_entry["device_type"]),
            bin_category=int(log_entry["bin_category"]),
            transaction_velocity=float(log_entry["transaction_velocity"]),
            time_of_day=int(log_entry["time_of_day"]),
            gateway_success_rates=[g.current_rate for g in self._gateways],
            gateway_states=[g.state for g in self._gateways],
            observed_fraud_risk=self._get_noisy_risk(float(log_entry["fraud_risk_score"])),
            previous_failures=self._state.consecutive_failures,
            difficulty=self._difficulty,
            reward=0.5,
            done=False,
            task_routing_score=0.5,
            task_fraud_mcc_score=0.5,
            task_retention_score=0.5,
        )

    def _get_noisy_risk(self, true_risk: float) -> float:
        """Adds Gaussian noise to the true risk score."""
        noise = self._rng.normal(0, 0.1)
        return float(np.clip(true_risk + noise, 0.01, 0.99))

    def _generate_fallback_transaction(self) -> SmartpayenvObservation:
        # Original logic as fallback
        hour = int(self._state.step_count % 24)
        segment = int(self._rng.choice([0, 1, 2], p=[0.25, 0.60, 0.15]))
        mcc = int(self._rng.choice([0, 1, 2, 3, 4, 5]))
        amount = float(self._rng.lognormal(mean=4.0, sigma=0.8))
        
        self._state.true_fraud_risk = 0.1
        return SmartpayenvObservation(
            amount=amount,
            merchant_category=mcc,
            is_international=False,
            card_present=True,
            user_type=0,
            user_segment=segment,
            user_history_score=0.8,
            device_type=0,
            bin_category=0,
            transaction_velocity=0.5,
            time_of_day=hour,
            gateway_success_rates=[0.9, 0.9, 0.9],
            gateway_states=["normal", "normal", "normal"],
            observed_fraud_risk=0.1,
            previous_failures=0,
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
        # Random initial cursor for variety, but then sequential within episode
        self._state.log_cursor = self._rng.integers(0, 100000) 
        self._init_gateways()
        self.route_grader     = RoutingEfficacyGrader()
        self.fraud_grader     = FraudDetectionGrader()
        self.retention_grader = UserRetentionGrader(churn_rate=self._cfg["churn_rate"])
        self._velocity_buffer.clear()
        self.current_obs = self._generate_transaction()
        # Synchronize simulation clock with the log's starting hour
        self._state.simulation_hour = self.current_obs.time_of_day
        self._state.curriculum_level = float(self._difficulty)
        self._state.policy_skill_estimate = 0.5
        self._state.challenger_skill = 0.55 + (0.08 * self._difficulty)
        self._state.anti_gaming_alerts = 0
        return self.current_obs

    def _curriculum_multiplier(self) -> float:
        return 1.0 + (0.15 * self._state.curriculum_level)

    def _update_self_play_curriculum(self, route_score: float, fraud_score: float, retention_score: float) -> None:
        """
        Theme-4 core: self-improvement loop inspired by league training.
        The policy competes against a moving challenger and environment complexity
        scales with sustained performance.
        """
        self._state.recent_route_scores.append(route_score)
        self._state.recent_fraud_scores.append(fraud_score)
        self._state.recent_retention_scores.append(retention_score)
        perf = (0.45 * route_score) + (0.35 * fraud_score) + (0.20 * retention_score)
        self._state.recent_rewards.append(perf)

        if not self._state.recent_rewards:
            return

        rolling_perf = float(np.mean(self._state.recent_rewards))
        skill_delta = 0.08 * (rolling_perf - 0.5)
        self._state.policy_skill_estimate = float(np.clip(self._state.policy_skill_estimate + skill_delta, 0.05, 0.99))

        # PFSP-inspired challenger adaptation: keep matches near policy frontier.
        gap = self._state.policy_skill_estimate - self._state.challenger_skill
        self._state.challenger_skill = float(np.clip(self._state.challenger_skill + (0.06 * gap), 0.05, 0.99))

        if self._meta_curriculum_enabled and len(self._state.recent_rewards) >= 8:
            if rolling_perf > 0.72:
                self._state.curriculum_level = float(np.clip(self._state.curriculum_level + 0.12, 0.0, 2.0))
            elif rolling_perf < 0.45:
                self._state.curriculum_level = float(np.clip(self._state.curriculum_level - 0.08, 0.0, 2.0))

    def step(self, action: SmartpayenvAction) -> SmartpayenvObservation:
        self._state.step_count += 1
        
        # Advance hour every 20 steps
        if self._state.step_count % 20 == 0:
            self._state.simulation_hour = (self._state.simulation_hour + 1) % 24
        
        if self.current_obs is None: self.reset()
        
        obs = self.current_obs
        assert obs is not None 

        # 0. Temporal Event Management
        # Decay active events (Safer way to delete items)
        self._state.active_events = {e: d - 1 for e, d in self._state.active_events.items() if d > 1}

        # Randomly trigger a systemic gateway outage (Event Correlation)
        if self._rng.random() < 0.01:
            self._state.active_events["systemic_outage"] = self._rng.integers(5, 15)
            # Force multiple gateways into "degraded" state
            for gw in self._gateways:
                if self._rng.random() < 0.7:
                    gw.state = "degraded"
                    gw._countdown = self._state.active_events["systemic_outage"]
                    gw.current_rate = gw.base_rate * 0.1

        # 0. Gateway Health Lag Update
        current_health = {
            "rates": [g.current_rate for g in self._gateways],
            "states": [g.state for g in self._gateways]
        }
        self._state.health_lag_buffer.append(current_health)

        if self._state.step_count % 10 == 0 and self._rng.random() < 0.2:
            # Inject a "Fraud Surge" pattern from logs
            surge_logs = self._log_loader.get_pattern("fraud_surge", count=5)
            self._pattern_queue.extend(surge_logs)

        # Curriculum-driven stress events (self-improvement pressure).
        if self._rng.random() < (0.01 * self._curriculum_multiplier()):
            self._state.active_events["adversarial_shift"] = int(self._rng.integers(4, 12))

        for gw in self._gateways: gw.step()

        # 1. 3DS / Action Logic
        is_fraud      = (self._state.true_fraud_risk >= 0.65)
        action_block  = (action.fraud_decision == 1)
        action_3ds    = (action.fraud_decision == 2)
        action_review = (action.fraud_decision == 3)
        
        self.fraud_grader.add_step(action_block or action_3ds or action_review, is_fraud)

        done = False
        success = False
        retries = 0
        gateway = action.gateway
        total_cost = 0.0
        cb_penalty_this_step = 0.0

        if action_block:
            route_score = self._state.true_fraud_risk if is_fraud else (self._state.true_fraud_risk * 0.3)
            done = True
        elif action_review:
            # Manual Review: Costly but accurate delay
            total_cost += 5.0 # High internal cost for human time
            delay = self._rng.integers(10, 25)
            self._state.review_queue.append({
                'maturation': self._state.step_count + delay,
                'is_fraud': is_fraud,
                'amount': obs.amount
            })
            route_score = 0.5 # Neutral immediate feedback
            success = False # Held in review
        else:
            gw_rates = [g.current_rate for g in self._gateways]
            
            # BIN Affinity & 3DS Support
            affinity = BIN_AFFINITY[gateway][obs.bin_category]
            
            # Extreme Reality Scaling: mismatched BINs now fail aggressively
            if affinity < 0.9:
                affinity = affinity * 0.15 # Harsh penalty for subpar routing
                
            # 3DS reduces remaining fraud risk by 90%
            eff_fraud_risk = self._state.true_fraud_risk * (0.1 if action_3ds else 1.0)
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
                expected_outcome = gw_rates[gateway] * (1.0 - self._state.true_fraud_risk) * affinity
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
        for maturation_step, penalty_amount in self._state.chargeback_queue:
            if self._state.step_count >= maturation_step: 
                cb_amt += float(penalty_amount)
            else: 
                pending.append((maturation_step, penalty_amount))
        self._state.chargeback_queue = pending

        # 3. Apply Lagged Health to Next Observation
        # Use first item in buffer for 2-step lag if buffer is full
        lagged_health = self._state.health_lag_buffer[0] if len(self._state.health_lag_buffer) >= 3 else current_health
        
        self.current_obs = self._generate_transaction()
        self.current_obs.time_of_day = self._state.simulation_hour
        self.current_obs.gateway_success_rates = lagged_health["rates"]
        self.current_obs.gateway_states        = lagged_health["states"]
        self.current_obs.chargeback_penalty_applied = cb_amt
        
        # Process and report matured Manual Reviews
        matured_reviews = []
        remaining_reviews = []
        for r in self._state.review_queue:
            if self._state.step_count >= r['maturation']:
                matured_reviews.append({
                    'amount': r['amount'],
                    'is_fraud': r['is_fraud'],
                    'outcome': 'rejected' if r['is_fraud'] else 'accepted'
                })
            else:
                remaining_reviews.append(r)
        self._state.review_queue = remaining_reviews
        self.current_obs.review_resolutions = matured_reviews
        
        if done or self._state.step_count >= 100: self.current_obs.done = True
        
        fs = self.fraud_grader.evaluate()
        rs = self.retention_grader.evaluate()
        base_reward = (0.4 * route_score) + (0.4 * fs) + (0.2 * rs)

        # League-style regret: penalize underperforming against moving challenger.
        challenger_regret = max(0.0, self._state.challenger_skill - base_reward)
        regret_penalty = 0.35 * challenger_regret

        # Anti-gaming check: repeatedly overusing manual review without quality gains.
        gaming_penalty = 0.0
        if action.fraud_decision == 3 and fs < 0.55 and rs < 0.6:
            self._state.anti_gaming_alerts += 1
            gaming_penalty = min(0.12, 0.02 * self._state.anti_gaming_alerts)

        # Curriculum bonus: reward robust performance under higher difficulty pressure.
        robustness_bonus = 0.06 * self._state.curriculum_level * max(0.0, base_reward - 0.55)

        # Norm punishment for delayed liabilities + self-improvement terms.
        final_reward = base_reward - (cb_amt / 150.0) - regret_penalty - gaming_penalty + robustness_bonus
        self.current_obs.reward = float(np.clip(final_reward, 0.001, 0.999))
        
        self.current_obs.task_routing_score = route_score
        self.current_obs.task_fraud_mcc_score = fs
        self.current_obs.task_retention_score = rs
        self._update_self_play_curriculum(route_score, fs, rs)

        self.current_obs.metadata = {
            "theme": "self_improvement",
            "curriculum_level": round(self._state.curriculum_level, 4),
            "policy_skill_estimate": round(self._state.policy_skill_estimate, 4),
            "challenger_skill": round(self._state.challenger_skill, 4),
            "challenger_regret": round(challenger_regret, 4),
            "gaming_penalty": round(gaming_penalty, 4),
            "robustness_bonus": round(robustness_bonus, 4),
            "anti_gaming_alerts": int(self._state.anti_gaming_alerts),
            "active_events": dict(self._state.active_events),
        }

        return self.current_obs

    def simulate(self, action: SmartpayenvAction) -> SmartpayenvObservation:
        """
        Simulates an action without advancing the true environment state.
        Allows agents to explore 'what-if' scenarios from the same state.
        """
        import copy
        
        # 1. Full State Backup
        # Note: We backup the entire current_obs and _state object.
        # We also need to backup the graders because they track cumulative stats.
        backup_state = copy.deepcopy(self._state)
        backup_obs   = copy.deepcopy(self.current_obs)
        backup_g_route     = copy.deepcopy(self.route_grader)
        backup_g_fraud     = copy.deepcopy(self.fraud_grader)
        backup_g_retention = copy.deepcopy(self.retention_grader)
        
        # Backup Gateway internal dynamics
        backup_gateways_data = []
        for g in self._gateways:
            backup_gateways_data.append({
                'state':        g.state,
                'countdown':    g._countdown,
                'current_rate': g.current_rate
            })

        # Backup RNG State to ensure determinism during simulation if needed
        # Or alternatively, allow simulation to have its own random paths
        rng_state = self._rng.bit_generator.state

        # 2. Execute ephemeral step
        sim_obs = copy.deepcopy(self.step(action))

        # 3. Restore Reality
        self._state      = backup_state
        self.current_obs = backup_obs
        self.route_grader     = backup_g_route
        self.fraud_grader     = backup_g_fraud
        self.retention_grader = backup_g_retention

        for i, g in enumerate(self._gateways):
            d = backup_gateways_data[i]
            g.state        = d['state']
            g._countdown   = d['countdown']
            g.current_rate = d['current_rate']
            
        self._rng.bit_generator.state = rng_state

        return sim_obs

    @property
    def state(self) -> State:
        return self._state
