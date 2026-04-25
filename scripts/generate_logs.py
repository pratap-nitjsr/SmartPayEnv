import argparse
import json
import os
from collections import defaultdict, deque

import numpy as np


LOCATIONS = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata", "Europe", "Singapore"]
SEGMENT_LABELS = {0: "new", 1: "existing", 2: "premium"}
BASE_MCC_DIST = [0.30, 0.20, 0.10, 0.10, 0.10, 0.20]
HIGH_RISK_MCCS = {2, 4, 5}
RISKY_HOURS = {1, 2, 3, 4, 5}


def _time_bucket(hour: int) -> str:
    if 0 <= hour <= 5:
        return "night"
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    return "evening"


def _sample_user_profiles(rng: np.random.Generator, n_users: int) -> list[dict]:
    profiles: list[dict] = []
    for uid in range(n_users):
        segment = int(rng.choice([0, 1, 2], p=[0.30, 0.55, 0.15]))
        traveler = bool(rng.random() < {0: 0.08, 1: 0.15, 2: 0.35}[segment])
        home = str(rng.choice(LOCATIONS[:7]))
        preferred_mcc = int(rng.choice([0, 1, 3, 5], p=[0.35, 0.25, 0.20, 0.20]))
        profiles.append(
            {
                "user_id": f"user_{uid}",
                "user_segment": segment,
                "frequent_traveler": traveler,
                "home_location": home,
                "preferred_mcc": preferred_mcc,
                "base_device_type": int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10])),
                "base_spend_mu": {0: 3.8, 1: 4.5, 2: 5.0}[segment],
                "base_spend_sigma": {0: 0.70, 1: 0.75, 2: 0.85}[segment],
                "history_base": {0: 0.35, 1: 0.72, 2: 0.88}[segment],
            }
        )
    return profiles


def _normal_transaction(
    rng: np.random.Generator,
    profile: dict,
    hour: int,
    user_recent_times: deque,
    user_recent_amounts: deque,
) -> dict:
    mcc_probs = np.array(BASE_MCC_DIST, dtype=float)
    mcc_probs[profile["preferred_mcc"]] += 0.18
    mcc_probs = mcc_probs / mcc_probs.sum()
    mcc = int(rng.choice([0, 1, 2, 3, 4, 5], p=mcc_probs))

    amount = float(rng.lognormal(mean=profile["base_spend_mu"], sigma=profile["base_spend_sigma"]))
    if mcc in HIGH_RISK_MCCS:
        amount *= 1.35

    location = profile["home_location"]
    is_international = False
    if profile["frequent_traveler"] and rng.random() < 0.10:
        location = str(rng.choice(["Europe", "Singapore"]))
        is_international = True

    device_type = profile["base_device_type"]
    if rng.random() < 0.07:
        device_type = int(rng.choice([0, 1, 2]))

    velocity = float(min(12, len([t for t in user_recent_times if hour - t <= 1])))
    velocity_norm = float(np.clip(velocity / 10.0, 0.05, 0.98))

    risk = 0.02
    risk += 0.06 if hour in RISKY_HOURS else 0.0
    risk += 0.05 if mcc in HIGH_RISK_MCCS else 0.0
    risk += 0.04 if device_type != profile["base_device_type"] else 0.0
    risk += 0.03 if is_international else 0.0
    risk += 0.08 * velocity_norm
    risk += rng.normal(0.0, 0.02)

    return {
        "amount": float(np.clip(amount, 5.0, 150000.0)),
        "currency": "INR",
        "time": _time_bucket(hour),
        "merchant_category": mcc,
        "location": location,
        "is_international": is_international,
        "card_present": bool(rng.random() > 0.45),
        "user_segment": profile["user_segment"],
        "user_history_score": float(np.clip(rng.normal(profile["history_base"], 0.12), 0.05, 1.0)),
        "device_type": device_type,
        "ip_risk": float(np.clip(rng.normal(0.10 if location == profile["home_location"] else 0.45, 0.08), 0.01, 0.99)),
        "bin_category": int(rng.integers(0, 10)),
        "time_of_day": int(hour),
        "transaction_velocity": velocity_norm,
        "fraud_risk_score": float(np.clip(risk, 0.01, 0.99)),
        "fraud_strategy": "none",
        "event_marker": None,
        "is_fraud": False,
    }


def _fraud_agent_strategy_mix(
    rng: np.random.Generator,
    attack_level: float,
) -> list[str]:
    templates = [
        ("high_value_spike", 0.20),
        ("velocity_burst", 0.22),
        ("geo_anomaly", 0.16),
        ("device_spoof", 0.18),
        ("split_transactions", 0.14),
        ("low_risk_disguise", 0.10),
    ]
    weights = np.array([w for _, w in templates], dtype=float)
    # Self-improving fraud agent: shifts toward stealth blends as defender hardens.
    stealth_boost = min(0.18, 0.06 * attack_level)
    weights[5] += stealth_boost
    weights[4] += stealth_boost * 0.8
    weights = weights / weights.sum()

    k = 1 if attack_level < 1.0 else (2 if rng.random() < 0.75 else 3)
    selected = rng.choice([name for name, _ in templates], size=k, replace=False, p=weights)
    return list(selected)


def _apply_fraud_strategy(
    rng: np.random.Generator,
    tx: dict,
    profile: dict,
    strategies: list[str],
) -> list[dict]:
    tx = dict(tx)
    event_markers = []

    for s in strategies:
        if s == "high_value_spike":
            tx["amount"] = float(min(200000.0, tx["amount"] * rng.uniform(6.0, 18.0)))
            event_markers.append("high_value_spike")
        elif s == "velocity_burst":
            tx["transaction_velocity"] = float(np.clip(tx["transaction_velocity"] + rng.uniform(0.45, 0.85), 0.1, 0.99))
            event_markers.append("velocity_burst")
        elif s == "geo_anomaly":
            tx["location"] = str(rng.choice(["Europe", "Singapore"]))
            tx["is_international"] = True
            tx["ip_risk"] = float(np.clip(tx["ip_risk"] + rng.uniform(0.25, 0.50), 0.01, 0.99))
            event_markers.append("geo_anomaly")
        elif s == "device_spoof":
            tx["device_type"] = int((profile["base_device_type"] + int(rng.integers(1, 3))) % 3)
            tx["card_present"] = False
            tx["ip_risk"] = float(np.clip(tx["ip_risk"] + rng.uniform(0.18, 0.35), 0.01, 0.99))
            event_markers.append("device_spoof")
        elif s == "split_transactions":
            # Converted to multiple low-value events that preserve a high total.
            pieces = int(rng.integers(4, 10))
            each_amount = float(max(1500.0, tx["amount"] * rng.uniform(0.10, 0.22)))
            generated = []
            for _ in range(pieces):
                p = dict(tx)
                p["amount"] = each_amount
                p["transaction_velocity"] = float(np.clip(tx["transaction_velocity"] + rng.uniform(0.35, 0.55), 0.1, 0.99))
                p["event_marker"] = "split_transactions"
                p["fraud_strategy"] = "split_transactions"
                p["is_fraud"] = True
                risk = p["fraud_risk_score"] + rng.uniform(0.18, 0.32)
                p["fraud_risk_score"] = float(np.clip(risk, 0.01, 0.99))
                generated.append(p)
            return generated
        elif s == "low_risk_disguise":
            # Fraud tries to look normal: lower explicit risk while preserving anomalies elsewhere.
            tx["amount"] = float(np.clip(tx["amount"] * rng.uniform(0.18, 0.35), 250.0, 12000.0))
            tx["merchant_category"] = int(rng.choice([0, 1, 3], p=[0.5, 0.3, 0.2]))
            tx["fraud_risk_score"] = float(np.clip(tx["fraud_risk_score"] - rng.uniform(0.08, 0.20), 0.02, 0.80))
            event_markers.append("low_risk_disguise")

    tx["fraud_strategy"] = "+".join(strategies)
    tx["event_marker"] = "|".join(event_markers) if event_markers else "fraud_pattern"
    tx["is_fraud"] = True
    tx["fraud_risk_score"] = float(np.clip(tx["fraud_risk_score"] + rng.uniform(0.18, 0.42), 0.01, 0.99))
    return [tx]


def generate_logs(
    output_path: str = "data/transactions_log.jsonl",
    num_transactions: int = 15000,
    n_users: int = 4000,
    seed: int = 7,
    base_fraud_rate: float = 0.08,
) -> None:
    """
    Generates realistic synthetic payment logs with an evolving fraud adversary.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profiles = _sample_user_profiles(rng, n_users=n_users)
    user_recent_times: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    user_recent_amounts: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))

    current_hour = 0
    steps_per_hour = 90
    global_attack_level = 0.0
    defender_pressure = 0.0

    records_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        while records_written < num_transactions:
            if records_written % steps_per_hour == 0:
                current_hour = (current_hour + 1) % 24

            profile = profiles[int(rng.integers(0, len(profiles)))]
            uid = profile["user_id"]

            base_tx = _normal_transaction(
                rng=rng,
                profile=profile,
                hour=current_hour,
                user_recent_times=user_recent_times[uid],
                user_recent_amounts=user_recent_amounts[uid],
            )

            fraud_p = base_fraud_rate + (0.05 if current_hour in RISKY_HOURS else 0.0) + (0.07 * global_attack_level)
            fraud_p = float(np.clip(fraud_p, 0.01, 0.55))
            is_attack = bool(rng.random() < fraud_p)

            if is_attack:
                strategies = _fraud_agent_strategy_mix(rng, attack_level=global_attack_level)
                txs = _apply_fraud_strategy(rng, base_tx, profile, strategies)
            else:
                txs = [base_tx]

            for tx in txs:
                tx["user_id"] = uid
                tx["user_profile"] = {
                    "segment": SEGMENT_LABELS[profile["user_segment"]],
                    "frequent_traveler": profile["frequent_traveler"],
                    "home_location": profile["home_location"],
                }
                tx["attack_level"] = round(float(global_attack_level), 4)
                tx["defender_pressure"] = round(float(defender_pressure), 4)
                f.write(json.dumps(tx) + "\n")
                records_written += 1

                user_recent_times[uid].append(current_hour)
                user_recent_amounts[uid].append(tx["amount"])
                if records_written >= num_transactions:
                    break

            # Self-improvement dynamics:
            # when fraud is frequently obvious, increase defender pressure;
            # when stealth fraud appears, raise attack sophistication.
            if is_attack and any("low_risk_disguise" in t.get("fraud_strategy", "") for t in txs):
                global_attack_level = float(np.clip(global_attack_level + 0.015, 0.0, 3.0))
            elif is_attack:
                defender_pressure = float(np.clip(defender_pressure + 0.010, 0.0, 2.5))
            else:
                global_attack_level = float(np.clip(global_attack_level + 0.002 - (0.001 * defender_pressure), 0.0, 3.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SmartPayEnv transaction logs.")
    parser.add_argument("--output", default="data/transactions_log.jsonl", help="Output JSONL file path")
    parser.add_argument("--num-transactions", type=int, default=15000, help="Number of transactions")
    parser.add_argument("--n-users", type=int, default=4000, help="Number of synthetic users")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--base-fraud-rate", type=float, default=0.08, help="Baseline fraud probability")
    args = parser.parse_args()

    generate_logs(
        output_path=args.output,
        num_transactions=args.num_transactions,
        n_users=args.n_users,
        seed=args.seed,
        base_fraud_rate=args.base_fraud_rate,
    )
    print(f"Generated {args.num_transactions} synthetic transactions at {args.output}")
