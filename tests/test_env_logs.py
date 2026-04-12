import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.SmartPayEnv_environment import SmartpayenvEnvironment
from models import SmartpayenvAction

def test_env():
    env = SmartpayenvEnvironment()
    obs = env.reset()
    print(f"Initial Obs: Amount={obs.amount}, Segment={obs.user_segment}, FraudRisk={obs.fraud_risk_score}")
    
    for i in range(20):
        action = SmartpayenvAction(gateway=0, fraud_decision=0, retry_strategy=0)
        obs = env.step(action)
        print(f"Step {i+1}: Amount={obs.amount:.2f}, FraudRisk={obs.fraud_risk_score:.2f}, Hour={obs.time_of_day}")
        if env._pattern_queue:
            print(f"  [Pattern Queued: {len(env._pattern_queue)} items remaining]")

if __name__ == "__main__":
    test_env()
