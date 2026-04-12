import requests
import json
import time

URL = "http://localhost:7860"

def test_temporal():
    # 1. Reset
    res = requests.post(f"{URL}/reset", json={"difficulty": 1})
    obs = res.json().get("observation")
    last_hour = obs.get("time_of_day")
    
    print(f"Initial Hour: {last_hour}")
    
    correlated_failures = 0
    high_velocity_count = 0
    
    for i in range(100):
        # Action doesn't matter much for this test
        res = requests.post(f"{URL}/step", json={"action": {"gateway": 0, "fraud_decision": 0, "retry_strategy": 0}})
        data = res.json()
        obs = data.get("observation")
        
        hour = obs.get("time_of_day")
        states = obs.get("gateway_states")
        
        # Check hour progression
        if hour != last_hour:
            print(f"Hour advanced to {hour}")
            last_hour = hour
            
        # Check correlation (Systemic Outage)
        down_count = sum(1 for s in states if s != "normal")
        if down_count >= 2:
            correlated_failures += 1
            print(f"Step {i}: Cluster failure detected! States: {states}")
        
        # Velocity might be high during fraud spikes
        # Actually transaction_velocity is in observation? Let's check model.py
        # No, it's not in observation yet. Let's check models.py
        
    print(f"Correlated failures detected: {correlated_failures}")

if __name__ == "__main__":
    try:
        test_temporal()
    except Exception as e:
        print(f"Failed to connect to server: {e}. Make sure it is running.")
