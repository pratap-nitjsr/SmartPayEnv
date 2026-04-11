import os
import json
import textwrap
from typing import List, Optional
import requests
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# Environment variables mapping as per instructions
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-token")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

# Task definitions ordered by incremental difficulty
# 1. Routing: choosing the best gateway (deterministic decision)
# 2. Retention: keeping success rate high to prevent churn (temporal impact)
# 3. Fraud: context-aware blocking (highest stakes, incorrect block ends episode)
# 4. Optimization: balancing all objectives (Expert task)
TASKS = ["routing_efficacy", "user_retention", "fraud_detection", "payment_optimization"]
DIFFICULTIES = [0, 1, 2] # 0=Easy, 1=Medium, 2=Hard
DIFFICULTY_LABELS = {0: "EASY", 1: "MEDIUM", 2: "HARD"}
BENCHMARK = os.getenv("BENCHMARK", "SmartPayEnv")
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5  # target normalized score in [0, 1]

ENV_URL = "http://localhost:7860"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Self-Optimizing Payment Intelligence agent interacting with the SPIS environment.
    Each turn you must send an action to route a transaction or block fraud.
    Respond with EXACTLY ONE valid JSON object — no quotes, no markdown blocks, no prefixes.
    Keys required:
    "gateway" (integer: 0, 1, or 2)
    "retry_strategy" (integer: 0 or 1)
    "fraud_decision" (integer: 0=Allow, 1=Block (ends episode), 2=Challenge/3DS)
    Note: 3DS reduces fraud risk significantly but adds 15% abandonment failure and a retention penalty.
    BIN affinity and User Segments (New/Existing/Premium) now affect success rates.
    """
).strip()

def log_start(task: str, env: str, model: str, difficulty: str) -> None:
    print(f"[START] difficulty={difficulty} task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
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
        Send your next JSON action.
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
        
        # Simple extraction helper in case of markdown bloat
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx+1]
            
        action_data = json.loads(text)
        return {
            "gateway": int(action_data.get("gateway", 1)),
            "retry_strategy": int(action_data.get("retry_strategy", 0)),
            "fraud_decision": int(action_data.get("fraud_decision", 0))
        }
    except Exception as exc:
        # Fallback heuristic logic if LLM fails
        return {
            "gateway": 2 if obs.get("amount", 0) > 10000 else 0,
            "retry_strategy": 1,
            "fraud_decision": 1 if obs.get("fraud_risk_score", 0) > 0.8 else 0
        }

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for diff_level in DIFFICULTIES:
        diff_label = DIFFICULTY_LABELS[diff_level]
        
        for task_name in TASKS:
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME, difficulty=diff_label)

            try:
                # Reset Env with the specific difficulty level
                res = requests.post(f"{ENV_URL}/reset", json={"difficulty": diff_level})
                if res.status_code != 200:
                    # Fallback for environments that don't support JSON in reset yet
                    res = requests.post(f"{ENV_URL}/reset")
                    if res.status_code != 200:
                        raise ConnectionError("Server did not return 200 on /reset")
                    
                obs = res.json()
                # If wrapped in 'observation' key (depends on framework version)
                if isinstance(obs, dict) and "observation" in obs:
                    obs = obs["observation"]
                
                last_reward = 0.0

                for step in range(1, MAX_STEPS + 1):
                    action_dict = get_model_action(client, step, obs, last_reward)
                    action_str = json.dumps(action_dict).replace(" ", "")

                    # Step Env
                    error = None
                    done = False
                    reward = 0.0
                    try:
                        step_res = requests.post(f"{ENV_URL}/step", json={"action": action_dict})
                        if step_res.status_code == 200:
                            step_data = step_res.json()
                            # openenv wraps response: {"observation": {...}, "reward": ..., "done": ...}
                            obs = step_data.get("observation", step_data)

                            # Per-task scores are declared fields on the observation
                            if task_name == "routing_efficacy":
                                reward = obs.get("task_routing_score", 0.0)
                            elif task_name == "fraud_detection":
                                reward = obs.get("task_fraud_mcc_score", 0.0)
                            elif task_name == "user_retention":
                                reward = obs.get("task_retention_score", 0.0)
                            else:
                                # payment_optimization: use combined reward at top level
                                reward = step_data.get("reward", obs.get("reward", 0.0))

                            done = step_data.get("done", obs.get("done", False))
                        else:
                            error = f"HTTP {step_res.status_code}"
                    except Exception as e:
                        error = str(e)
                        done = True

                    rewards.append(reward)
                    steps_taken = step
                    last_reward = reward

                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    if done:
                        break
                        
                score = sum(rewards) / len(rewards) if rewards else 0.0
                score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
                success = score >= SUCCESS_SCORE_THRESHOLD

            except Exception as e:
                print(f"[DEBUG] Execution error: {e}", flush=True)
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
