import json
from typing import Dict, Any
import requests

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SmartpayenvAction, SmartpayenvObservation

class SmartpayenvEnv(EnvClient[SmartpayenvAction, SmartpayenvObservation, State]):
    def _step_payload(self, action: SmartpayenvAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[SmartpayenvObservation]:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=SmartpayenvObservation(**obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

def main():
    import random
    base_url = "http://localhost:7860"
    print("Environment resetting...")
    
    # 1. Reset
    response = requests.post(f"{base_url}/reset")
    if response.status_code != 200:
        print(f"Error connecting to server. Error code: {response.status_code}")
        return
        
    obs_data = response.json()
    obs = SmartpayenvObservation(**obs_data)
    total_reward = 0
    
    for step in range(50):
        # Basic strategy 
        gateway = 2 if obs.amount > 10000 else random.randint(0, 1)
        retry_strategy = 1 if gateway != 2 else 0
        fraud_decision = 1 if obs.fraud_risk_score > 0.8 else 0
        
        action = SmartpayenvAction(
            gateway=gateway,
            retry_strategy=retry_strategy,
            fraud_decision=fraud_decision
        )
        
        # 2. Step
        res = requests.post(
            f"{base_url}/step",
            json=action.model_dump()
        )
        
        step_res = res.json()
        obs = SmartpayenvObservation(**step_res["observation"])
        reward = step_res.get("reward", 0.0)
        done = step_res.get("done", False)
        
        total_reward += reward
        
        print(f"Step {step+1}:")
        print(f"  Action taken: gateway={action.gateway},  fraud_decision={action.fraud_decision}")
        print(f"  Reward received: {reward:.2f}")
        print(f"  Next State details: Amount={obs.amount:.2f}, FraudRisk={obs.fraud_risk_score:.2f}")

        if done:
            print("Episode done!")
            break
            
    print(f"Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
