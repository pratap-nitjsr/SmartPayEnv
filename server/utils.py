import json
import random
import os

class LogLoader:
    def __init__(self, log_path="data/transactions_log.jsonl"):
        self.log_path = log_path
        self.logs = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    self.logs.append(json.loads(line))
        else:
            print(f"Warning: Log file {log_path} not found.")

    def sample(self, index=None, noise_level=0.05):
        if not self.logs:
            return None
        
        if index is not None:
            entry = self.logs[index % len(self.logs)].copy()
        else:
            entry = random.choice(self.logs).copy()
        
        # Inject noise into float fields
        if noise_level > 0:
            for key in ["amount", "fraud_risk_score", "user_history_score", "transaction_velocity"]:
                if key in entry:
                    noise = random.uniform(-noise_level, noise_level)
                    entry[key] = max(0.01, entry[key] * (1 + noise))
                    
        return entry

    def get_pattern(self, pattern_type="fraud_surge", count=10):
        """Returns a subset of logs matching a certain pattern."""
        if not self.logs:
            return []
            
        if pattern_type == "fraud_surge":
            # Filter for high fraud risk
            candidates = [l for l in self.logs if l.get("fraud_risk_score", 0) > 0.5]
        elif pattern_type == "stealth_fraud":
            candidates = [
                l for l in self.logs
                if l.get("is_fraud", False)
                and "low_risk_disguise" in str(l.get("fraud_strategy", ""))
            ]
        elif pattern_type == "velocity_attack":
            candidates = [l for l in self.logs if float(l.get("transaction_velocity", 0.0)) > 0.7]
        elif pattern_type == "premium_only":
            candidates = [l for l in self.logs if l.get("user_segment") == 2]
        else:
            candidates = self.logs
            
        if not candidates:
            return [random.choice(self.logs) for _ in range(count)]
            
        return [random.choice(candidates) for _ in range(count)]
