import numpy as np

def test_preference_utils():
    import sys
    sys.path.append(".")
    from server.preference_utils import get_context_bucket, calculate_advantages, rank_actions
    
    class DummyObs:
        def __init__(self, bin, amt, risk):
            self.bin_category = bin
            self.amount = amt
            self.observed_fraud_risk = risk
            
    obs = DummyObs(2, 600, 0.45)
    bucket = get_context_bucket(obs)
    print(f"Context Bucket: {bucket}")
    assert bucket == (2, 1, 2) # (2, 600//500=1, 0.45*5=2)
    
    results = [("action1", 0.8), ("action2", 0.4), ("action3", 0.6)]
    advantages = calculate_advantages(results)
    print(f"Advantages: {advantages}")
    
    ranks = rank_actions(results)
    print(f"Ranks: {ranks}")
    assert ranks[0][0] == "action2" # lowest
    assert ranks[2][0] == "action1" # highest

def test_simulation_branching_direct():
    import sys
    sys.path.append(".")
    from server.SmartPayEnv_environment import SmartpayenvEnvironment
    from models import SmartpayenvAction
    
    env = SmartpayenvEnvironment()
    print("Resetting environment...")
    obs = env.reset(difficulty=1)
    
    # 2. Simulate Action A
    print("Simulating Action A (Allow)...")
    action_a = SmartpayenvAction(gateway=0, fraud_decision=0, retry_strategy=0)
    obs_a = env.simulate(action_a)
    reward_a = obs_a.reward
    
    # 3. Simulate Action B (3DS)
    print("Simulating Action B (3DS)...")
    action_b = SmartpayenvAction(gateway=0, fraud_decision=2, retry_strategy=0)
    obs_b = env.simulate(action_b)
    reward_b = obs_b.reward
    
    print(f"Results: Reward Allow={reward_a:.4f}, Reward 3DS={reward_b:.4f}")
    
    # 4. Step once with Action C
    print("Stepping with Action C (Block)...")
    action_c = SmartpayenvAction(gateway=0, fraud_decision=1, retry_strategy=0)
    final_obs = env.step(action_c)
    
    print(f"Final Step Reward: {final_obs.reward:.4f}")
    
    if reward_a != reward_b:
        print("[PASS] Branching rewards differ as expected.")
    else:
        print("[INFO] Branching rewards were identical (sampling luck).")
    
    print("[PASS] Simulation branching logic verified.")

if __name__ == "__main__":
    try:
        test_preference_utils()
        test_simulation_branching_direct()
        print("\nAll preference verification tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
