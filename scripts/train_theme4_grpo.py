"""
Theme-4 training starter for SmartPayEnv.

This script demonstrates a novel self-improvement loop:
1) sample K candidate actions per observation
2) score each candidate with /simulate rewards (group-relative signal)
3) collect best/worst pairs for preference-style post-training

It is intentionally lightweight so teams can run it in Colab with TRL/Unsloth.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import requests


ENV_URL = "http://localhost:7860"
MAX_STEPS = 200
GROUP_SIZE = 8


@dataclass
class RolloutExample:
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float


def _action_candidates() -> list[dict[str, int]]:
    all_actions: list[dict[str, int]] = []
    for gateway in (0, 1, 2):
        for fraud_decision in (0, 1, 2, 3):
            for retry_strategy in (0, 1):
                all_actions.append(
                    {
                        "gateway": gateway,
                        "fraud_decision": fraud_decision,
                        "retry_strategy": retry_strategy,
                    }
                )
    random.shuffle(all_actions)
    return all_actions


def _simulate_reward(action: dict[str, int]) -> float:
    response = requests.post(f"{ENV_URL}/simulate", json={"action": action}, timeout=30)
    response.raise_for_status()
    obs = response.json()
    return float(obs.get("reward", 0.0))


def _step(action: dict[str, int]) -> dict[str, Any]:
    response = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    response.raise_for_status()
    return response.json()


def _reset(difficulty: int = 2) -> dict[str, Any]:
    response = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("observation", payload)


def collect_group_relative_pairs(max_steps: int = MAX_STEPS, group_size: int = GROUP_SIZE) -> list[RolloutExample]:
    obs = _reset(difficulty=2)
    dataset: list[RolloutExample] = []
    actions_pool = _action_candidates()

    for _ in range(max_steps):
        sampled = random.sample(actions_pool, k=min(group_size, len(actions_pool)))
        scored: list[tuple[dict[str, int], float]] = []

        for action in sampled:
            try:
                reward = _simulate_reward(action)
                scored.append((action, reward))
            except requests.RequestException:
                continue

        if len(scored) < 2:
            break

        scored.sort(key=lambda x: x[1], reverse=True)
        best_action, best_reward = scored[0]
        worst_action, worst_reward = scored[-1]

        prompt = (
            "SmartPayEnv observation:\n"
            f"{json.dumps(obs, sort_keys=True)}\n"
            "Return one action JSON with fields: gateway, fraud_decision, retry_strategy."
        )

        dataset.append(
            RolloutExample(
                prompt=prompt,
                chosen=json.dumps(best_action, sort_keys=True),
                rejected=json.dumps(worst_action, sort_keys=True),
                chosen_reward=best_reward,
                rejected_reward=worst_reward,
            )
        )

        step_payload = _step(best_action)
        obs = step_payload.get("observation", step_payload)
        if bool(obs.get("done", False)):
            obs = _reset(difficulty=2)

    return dataset


def export_jsonl(dataset: list[RolloutExample], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(
                json.dumps(
                    {
                        "prompt": row.prompt,
                        "chosen": row.chosen,
                        "rejected": row.rejected,
                        "chosen_reward": row.chosen_reward,
                        "rejected_reward": row.rejected_reward,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    data = collect_group_relative_pairs()
    export_jsonl(data, "theme4_grpo_pairs.jsonl")
    print(f"Collected {len(data)} preference pairs into theme4_grpo_pairs.jsonl")
