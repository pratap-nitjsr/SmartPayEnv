import os
import json
import textwrap
from typing import List, Optional
import requests
from openai import OpenAI
import dotenv
import numpy as np

dotenv.load_dotenv()

# Environment variables mapping
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-token")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5
ENV_URL = "http://localhost:7860"
BENCHMARK = os.getenv("BENCHMARK", "SmartPayEnv")
DIFFICULTY_LABELS = {0: "EASY", 1: "MEDIUM", 2: "HARD"}

# Environmental Knowledge Injection
AFFINITY_INFO = {
    "Gateway_0_Affinity": [0.95, 0.80, 0.70, 0.60, 0.50, 0.90, 0.75, 0.65, 0.55, 0.85],
    "Gateway_1_Affinity": [0.60, 0.95, 0.80, 0.70, 0.60, 0.55, 0.90, 0.75, 0.65, 0.50],
    "Gateway_2_Affinity": [0.50, 0.60, 0.95, 0.85, 0.75, 0.50, 0.60, 0.95, 0.85, 0.75]
}

SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are a Self-Optimizing Payment Intelligence agent.
    
    ### KNOWLEDGE BASE:
    1. BIN Affinity Matrix (Success Probability multipliers):
       {json.dumps(AFFINITY_INFO, indent=2)}
       Note: Using a gateway with affinity < 0.9 incurs an 'Extreme Reality' penalty (x0.15 effectiveness).
    
    2. Merchant Risk Profiles (MCC):
       - 2 (Electronics) & 4 (Gambling): High Risk
       - 5 (Digital Goods): Med-High Risk
       - 0 (Retail) & 1 (Services): Low Risk
    
    3. Diurnal Cycle (UTC):
       - Hours 01:00-05:00: Severe Fraud Surge (Attack period).
       - Segment 0 (New): High distrust/abandonment during 3DS challenges.
    
    ### ACTION SCHEMA:
    Respond with EXACTLY ONE JSON object:
    {{
        "thought": "Reasoning based on current BIN category vs Affinity Matrix and Risk Score",
        "gateway": 0|1|2,
        "retry_strategy": 0|1,
        "fraud_decision": 0(Allow)|1(Block)|2(3DS Challenge)
    }}
    """
).strip()

def log_start(task: str, env: str, model: str, difficulty: str) -> None:
    print(f"[START] difficulty={difficulty} task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str], thought: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    thought_val = f" thought=\"{thought}\"" if thought else ""
    print(
        f"[STEP] step={step} action={action}{thought_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, step: int, obs: dict, last_reward: float) -> dict:
    user_prompt = textwrap.dedent(
        f"""
        Step: {step}
        Observation (State): {json.dumps(obs)}
        Last Reward: {last_reward:.2f}
        Send your JSON action now.
        """
    ).strip()
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        text = (completion.choices[0].message.content or "").strip()
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx+1]
            
        action_data = json.loads(text)
        return {
            "thought": str(action_data.get("thought", "N/A")),
            "gateway": int(action_data.get("gateway", 0)),
            "retry_strategy": int(action_data.get("retry_strategy", 0)),
            "fraud_decision": int(action_data.get("fraud_decision", 0))
        }
    except Exception as exc:
        return {
            "thought": f"Fallback: {exc}",
            "gateway": 0,
            "retry_strategy": 1,
            "fraud_decision": 0
        }

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    TASK_CONFIG = [
        ("routing_efficacy", 0),
        ("user_retention", 1),
        ("fraud_detection", 1),
        ("payment_optimization", 2)
    ]
    
    for task_name, diff_level in TASK_CONFIG:
        diff_label = DIFFICULTY_LABELS[diff_level]
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME, difficulty=diff_label)
        try:
            res = requests.post(f"{ENV_URL}/reset", json={"difficulty": diff_level})
            obs = res.json().get("observation", res.json())
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                action_data = get_model_action(client, step, obs, last_reward)
                thought = action_data.pop("thought")
                action_dict = action_data
                action_str = json.dumps(action_dict).replace(" ", "")

                step_res = requests.post(f"{ENV_URL}/step", json={"action": action_dict})
                if step_res.status_code == 200:
                    step_data = step_res.json()
                    obs = step_data.get("observation", step_data)
                    
                    if task_name == "routing_efficacy": reward = obs.get("task_routing_score", 0.0)
                    elif task_name == "fraud_detection": reward = obs.get("task_fraud_mcc_score", 0.0)
                    elif task_name == "user_retention": reward = obs.get("task_retention_score", 0.0)
                    else: reward = step_data.get("reward", 0.0)
                    
                    done = step_data.get("done", False)
                    log_step(step, action_str, reward, done, None, thought)
                    rewards.append(reward)
                    last_reward = reward
                    steps_taken = step
                    if done: break
                else:
                    log_step(step, action_str, 0.0, True, f"HTTP {step_res.status_code}")
                    break
                        
            score = sum(rewards) / len(rewards) if rewards else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    main()
