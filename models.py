# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smartpayenv Environment.

Rich, production-inspired payment transaction observation and action types.
"""

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


class SmartpayenvAction(Action):
    """
    Agent action for one payment transaction step.

    gateway:         Which payment gateway to attempt (0=GatewayA cheap, 1=GatewayB balanced, 2=GatewayC premium)
    retry_strategy:  0=no retry on failure, 1=failover to next gateway
    fraud_decision:  0=allow transaction, 1=block transaction (ends episode)
    """
    gateway: int = Field(default=0, description="0=GatewayA (cheap), 1=GatewayB (balanced), 2=GatewayC (premium)")
    retry_strategy: int = Field(default=0, description="0=No Retry, 1=Failover to next gateway on failure")
    fraud_decision: int = Field(default=0, description="0=Allow, 1=Block (end episode), 2=Challenge (3DS / MFA)")


class SmartpayenvObservation(Observation):
    """
    Rich observation for one incoming payment transaction.

    Includes multi-factor signals that a real payment intelligence
    system would use: merchant context, device fingerprinting,
    transaction velocity, international flag, and gateway health.
    """
    # ── Transaction context ────────────────────────────────────────────
    amount: float = Field(default=0.0, description="Transaction amount in USD")
    merchant_category: int = Field(
        default=0,
        description="Merchant category: 0=grocery, 1=travel, 2=electronics, 3=dining, 4=gaming, 5=other"
    )
    is_international: bool = Field(default=False, description="Cross-border transaction flag")
    card_present: bool = Field(default=True, description="Card physically present (lowers fraud risk)")

    # ── User / device signals ──────────────────────────────────────────
    user_type: int = Field(default=0, description="Derived risk tier: 0=Normal, 1=Risky, 2=Fraud")
    user_segment: int = Field(default=1, description="Cohort: 0=New/Guest, 1=Existing, 2=Premium/VIP")
    user_history_score: float = Field(default=1.0, description="Normalized user reliability score [0,1]")
    device_type: int = Field(default=0, description="0=mobile, 1=desktop, 2=tablet")
    bin_category: int = Field(default=0, description="Bank Identification Number category (0-9)")
    transaction_velocity: float = Field(
        default=0.0,
        description="Normalized count of transactions in the last 5 steps [0,1]"
    )

    # ── Temporal ──────────────────────────────────────────────────────
    time_of_day: int = Field(default=0, description="Hour of day 0–23")

    # ── Gateway health ────────────────────────────────────────────────
    gateway_success_rates: list[float] = Field(
        default_factory=list,
        description="Current success-rate estimates for [GatewayA, GatewayB, GatewayC]"
    )
    gateway_states: list[str] = Field(
        default_factory=list,
        description="Health state for each gateway: 'normal' | 'degraded' | 'recovering'"
    )

    # ── Risk scores ───────────────────────────────────────────────────
    fraud_risk_score: float = Field(
        default=0.0,
        description="Continuous multi-factor fraud risk [0,1] (higher = more suspicious)"
    )

    # ── Episode tracking ──────────────────────────────────────────────
    previous_failures: int = Field(default=0, description="Consecutive failed transactions in this episode")
    difficulty: int = Field(default=0, description="Episode difficulty tier: 0=easy, 1=medium, 2=hard")

    # ── Step outputs ──────────────────────────────────────────────────
    reward: float = Field(default=0.0, description="Combined step reward [0,1]")
    done: bool = Field(default=False, description="Episode done flag")
    chargeback_penalty_applied: float = Field(default=0.0, description="Penalty deducted this step from a past transaction chargeback")

    # Per-task scores — declared as first-class fields so openenv framework serializes them
    task_routing_score: float = Field(default=0.0, description="Routing efficacy score [0,1]")
    task_fraud_mcc_score: float = Field(default=0.0, description="Fraud detection MCC score [0,1]")
    task_retention_score: float = Field(default=1.0, description="User retention score [0,1]")

    # Metadata dict for backward compatibility / agent introspection
    metadata: dict = Field(default_factory=dict, description="Per-task score breakdown")
