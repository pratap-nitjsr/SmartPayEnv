import json
import numpy as np
import os
from uuid import uuid4

def generate_logs(output_path="data/transactions_log.jsonl", num_transactions=5000):
    rng = np.random.default_rng()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    current_hour = 0
    steps_per_hour = 100 # average density
    active_spike_countdown = 0
    
    with open(output_path, "w") as f:
        for i in range(num_transactions):
            # Advance time every ~100 transactions
            if i % steps_per_hour == 0:
                current_hour = (current_hour + 1) % 24
                
            # Randomly start a fraud spike (correlated event)
            if active_spike_countdown <= 0 and rng.random() < 0.005:
                active_spike_countdown = rng.integers(20, 50)
            
            # 1. Hour of day (Diurnal pattern)
            hour = current_hour
            
            # 2. Segment & MCC
            segment = int(rng.choice([0, 1, 2], p=[0.25, 0.60, 0.15]))
            mcc = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.2]))
            
            # 3. Fraud Risk with Correlation (Spikes)
            is_night = (1 <= hour <= 5)
            base_risk = {0: 0.02, 1: 0.05, 2: 0.15, 3: 0.08, 4: 0.25, 5: 0.12}[mcc]
            
            risk_boost = 0.0
            if active_spike_countdown > 0:
                risk_boost = 0.4 # Persistent spike
                active_spike_countdown -= 1
            elif is_night:
                risk_boost = 0.2
                
            final_risk = base_risk + risk_boost + rng.uniform(-0.05, 0.05)
            fraud_risk_score = float(np.clip(final_risk * {0: 1.8, 1: 1.0, 2: 0.3}[segment], 0.01, 0.99))
            
            # 4. Transaction Details
            amount = float(rng.lognormal(mean={0: 4.0, 1: 4.5, 2: 6.5, 3: 7.0, 4: 5.0, 5: 3.0}[mcc], sigma=0.8))
            bin_category = int(rng.integers(0, 10))
            is_international = bool(rng.random() < (0.4 if mcc == 3 else 0.15))
            
            log_entry = {
                "amount": amount,
                "merchant_category": mcc,
                "is_international": is_international,
                "card_present": bool(rng.random() > 0.5),
                "user_segment": segment,
                "user_history_score": float(np.clip(rng.normal({0: 0.3, 1: 0.7, 2: 0.9}[segment], 0.15), 0.1, 1.0)),
                "device_type": int(rng.choice([0, 1, 2], p=[0.5, 0.4, 0.1])),
                "bin_category": bin_category,
                "time_of_day": hour,
                "transaction_velocity": float(np.clip(rng.random() * 0.2 + (0.5 if active_spike_countdown > 0 else 0.0), 0.1, 0.9)),
                "fraud_risk_score": fraud_risk_score,
                "event_marker": "fraud_spike" if active_spike_countdown > 0 else None
            }
            f.write(json.dumps(log_entry) + "\n")

if __name__ == "__main__":
    generate_logs(num_transactions=5000)
    print("Sequential logs with correlated events generated.")
