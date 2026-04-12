import sys
import os
import time

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.SmartPayEnv_environment import SmartpayenvEnvironment
from models import SmartpayenvAction

def test_partial_obs():
    env = SmartpayenvEnvironment()
    obs = env.reset()
    
    print("--- STEP 0 (Initial) ---")
    print(f"Observed Risk: {obs.observed_fraud_risk:.4f}")
    print(f"True Risk (Hidden): {env._state.true_fraud_risk:.4f}")
    print(f"Gateway Rates: {obs.gateway_success_rates}")
    
    # Store initial rates
    initial_rates = env.current_obs.gateway_success_rates.copy()
    
    for i in range(1, 10):
        # Force a change in gateway rates to see the lag
        for g in env._gateways:
            g.current_rate = min(1.0, g.current_rate + 0.01) # Slowly drift up
            
        action = SmartpayenvAction(gateway=0, fraud_decision=0, retry_strategy=0)
        obs = env.step(action)
        
        print(f"\n--- STEP {i} ---")
        print(f"Observed Risk: {obs.observed_fraud_risk:.4f} (True: {env._state.true_fraud_risk:.4f})")
        print(f"Observed Health: {obs.gateway_success_rates}")
        print(f"Hidden Real Health: {[g.current_rate for g in env._gateways]}")
        
if __name__ == "__main__":
    test_partial_obs()
