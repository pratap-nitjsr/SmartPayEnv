"""
Comprehensive tests for SmartPayEnv v2 graders, data generation, and environment.
Run from the repo root:  python test_graders.py
"""
import sys, math
sys.path.insert(0, ".")
sys.path.insert(0, "./server")

import numpy as np
from server.graders import (
    RoutingEfficacyGrader,
    FraudDetectionGrader,
    UserRetentionGrader,
    process_combined_reward,
)
from server.SmartPayEnv_environment import SmartpayenvEnvironment, DIFFICULTY_CONFIG
from models import SmartpayenvAction

SEP = "=" * 60

# ── 1. RoutingEfficacyGrader (deterministic expected_outcome) ────────
print(f"\n{SEP}\n[1] RoutingEfficacyGrader — deterministic expected_outcome\n{SEP}")
rg = RoutingEfficacyGrader()

gw_rates = [0.70, 0.85, 0.95]   # GatewayC is best (index 2)

# Optimal choice: choose best gateway, high expected outcome
s_opt  = rg.evaluate(expected_outcome=0.90, cost=0.5, retries=0, chosen_gateway=2, gateway_rates=gw_rates)
# Suboptimal choice: choose worst gateway, same exp outcome for fairness (though in practice it would be lower)
s_sub  = rg.evaluate(expected_outcome=0.90, cost=0.5, retries=0, chosen_gateway=0, gateway_rates=gw_rates)
# Optimal choice, low expected outcome
s_low  = rg.evaluate(expected_outcome=0.20, cost=0.5, retries=0, chosen_gateway=2, gateway_rates=gw_rates)
# Worst: suboptimal + low outcome + retry + expensive
s_bad  = rg.evaluate(expected_outcome=0.10, cost=4.0, retries=2, chosen_gateway=0, gateway_rates=gw_rates)

print(f"  optimal gw + high outcome  → {s_opt:.4f}")
print(f"  suboptimal gw + same cost  → {s_sub:.4f}  (lower: worse gateway choice)")
print(f"  optimal gw + low outcome   → {s_low:.4f}  (mid)")
print(f"  worst case                 → {s_bad:.4f}  (expect lowest)")

for s in [s_opt, s_sub, s_low, s_bad]:
    assert 0.0 <= s <= 1.0, f"Out of [0,1]: {s}"
assert s_opt > s_sub, "Optimal gateway should outscore suboptimal"
assert s_opt > s_low, "High expected outcome should outscore low"
assert s_low > s_bad, "Any reasonable choice beats the worst case"

# DETERMINISM check: same inputs must always give same score
assert rg.evaluate(0.7, 1.5, 0, 1, gw_rates) == rg.evaluate(0.7, 1.5, 0, 1, gw_rates), "Not deterministic!"
print("  ✅ RoutingEfficacyGrader deterministic OK")

# ── 2. FraudDetectionGrader ──────────────────────────────────
print(f"\n{SEP}\n[2] FraudDetectionGrader\n{SEP}")
fg = FraudDetectionGrader()
for _ in range(70): fg.add_step(False, False)
for _ in range(30): fg.add_step(True,  True)
assert abs(fg.evaluate() - 1.0) < 1e-9, f"Perfect: {fg.evaluate()}"

fg2 = FraudDetectionGrader()
for _ in range(70): fg2.add_step(True,  False)
for _ in range(30): fg2.add_step(False, True)
assert abs(fg2.evaluate() - 0.0) < 1e-9, f"Worst: {fg2.evaluate()}"

fg3 = FraudDetectionGrader()
for _ in range(100): fg3.add_step(True, True)
assert abs(fg3.evaluate() - 0.5) < 1e-9, f"Neutral: {fg3.evaluate()}"

print(f"  perfect=1.0 worst=0.0 neutral=0.5  ✅")

# ── 3. UserRetentionGrader ───────────────────────────────────
print(f"\n{SEP}\n[3] UserRetentionGrader\n{SEP}")
urg = UserRetentionGrader(churn_rate=0.1, initial_users=100)
assert abs(urg.evaluate() - 1.0) < 1e-9
urg.add_step(0); assert abs(urg.evaluate() - 1.0) < 1e-9
urg.add_step(3); assert urg.evaluate() < 1.0
print(f"  initial=1.0, no-failure=1.0, 3-failures={urg.evaluate():.4f}  ✅")

# ── 4. process_combined_reward ────────────────────────────────
print(f"\n{SEP}\n[4] process_combined_reward\n{SEP}")
r_best  = process_combined_reward(1.0, True,  False, 0)
r_worst = process_combined_reward(0.0, False, True,  5)
assert 0.0 <= r_best  <= 1.0
assert 0.0 <= r_worst <= 1.0
assert r_best > r_worst
print(f"  best={r_best:.4f}  worst={r_worst:.4f}  ✅")

# ── 5. Multi-factor fraud risk ────────────────────────────────
print(f"\n{SEP}\n[5] Multi-factor fraud risk via environment\n{SEP}")
rng_seed = np.random.default_rng(42)
env = SmartpayenvEnvironment()

# Collect 200 transactions in easy mode and check fraud_risk ranges
env.reset(difficulty=0)
risks_easy = []
for _ in range(50):
    obs = env._generate_transaction()
    risks_easy.append(obs.fraud_risk_score)
    assert 0.0 <= obs.fraud_risk_score <= 1.0
    assert obs.merchant_category in range(6)
    assert obs.device_type in (0, 1, 2)
    assert isinstance(obs.is_international, bool)
    assert isinstance(obs.card_present, bool)

env.reset(difficulty=2)
risks_hard = []
for _ in range(50):
    obs = env._generate_transaction()
    risks_hard.append(obs.fraud_risk_score)

mean_easy = sum(risks_easy) / len(risks_easy)
mean_hard  = sum(risks_hard) / len(risks_hard)
print(f"  avg fraud_risk easy={mean_easy:.3f}  hard={mean_hard:.3f}")
assert mean_hard > mean_easy, "Hard mode should have higher avg fraud risk"
print("  ✅ Multi-factor fraud + difficulty scaling OK")

# ── 6. Gateway state machine ──────────────────────────────────
print(f"\n{SEP}\n[6] Gateway state machine\n{SEP}")
env.reset(difficulty=2)   # high degrade_p for quick test
states_seen = set()
for _ in range(80):
    for gw in env._gateways:
        gw.step()
        states_seen.add(gw.state)
        assert 0.0 <= gw.current_rate <= 1.0

print(f"  States observed: {states_seen}")
assert "degraded" in states_seen or "recovering" in states_seen, \
    "Hard mode should see degraded/recovering states"
print("  ✅ Gateway state machine OK")

# ── 7. Transaction velocity tracking ─────────────────────────
print(f"\n{SEP}\n[7] Transaction velocity tracking\n{SEP}")
env.reset(difficulty=0)
velocities = []
for _ in range(20):
    obs = env._generate_transaction()
    velocities.append(obs.transaction_velocity)
    assert 0.0 <= obs.transaction_velocity <= 1.0

print(f"  velocity range: [{min(velocities):.2f}, {max(velocities):.2f}]  ✅")

# ── 8. Episode smoke test — all 3 difficulty tiers ───────────
print(f"\n{SEP}\n[8] Full episode smoke test (15 steps × 3 difficulties)\n{SEP}")
for diff in [0, 1, 2]:
    obs = env.reset(difficulty=diff)
    assert obs.difficulty == diff
    rewards = []
    for step in range(15):
        action = SmartpayenvAction(
            gateway=int(np.argmax(obs.gateway_success_rates)),  # always choose best gw
            retry_strategy=1,
            fraud_decision=1 if obs.fraud_risk_score > 0.65 else 0,
        )
        obs = env.step(action)
        assert 0.0 <= obs.reward <= 1.0, f"reward out of [0,1]: {obs.reward}"
        assert 0.0 <= obs.task_routing_score <= 1.0
        assert 0.0 <= obs.task_fraud_mcc_score <= 1.0
        assert 0.0 <= obs.task_retention_score <= 1.0
        rewards.append(obs.reward)
        if obs.done:
            break
    avg = sum(rewards) / len(rewards)
    print(f"  difficulty={diff}: {len(rewards)} steps, avg_reward={avg:.4f}")
    assert any(r > 0 for r in rewards), "All rewards are still 0!"

print(f"\n  ✅ All difficulty tiers produce non-zero rewards")

# ── 9. Block → done=True immediately ─────────────────────────
print(f"\n{SEP}\n[9] fraud_decision=1 ends episode immediately\n{SEP}")
env.reset(difficulty=0)
obs = env.step(SmartpayenvAction(gateway=0, retry_strategy=0, fraud_decision=1))
assert obs.done is True, f"Expected done=True after block, got {obs.done}"
print(f"  Block step done={obs.done}  ✅")

print(f"\n{SEP}")
print("  ALL TESTS PASSED ✅")
print(f"{SEP}\n")
