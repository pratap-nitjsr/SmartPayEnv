import numpy as np
from typing import List, Tuple, Any

def get_context_bucket(obs: Any) -> Tuple[int, int, int]:
    """
    Discretizes the observation into a context bucket for preference learning.
    
    Args:
        obs: SmartpayenvObservation object or dict
    
    Returns:
        tuple: (bin_category, amount_bucket, risk_bucket)
    """
    # Extract values whether obs is a class or dict
    if hasattr(obs, 'bin_category'):
        bin_cat = int(obs.bin_category)
        amount = float(obs.amount)
        risk = float(obs.observed_fraud_risk)
    else:
        bin_cat = int(obs.get('bin_category', 0))
        amount = float(obs.get('amount', 0))
        risk = float(obs.get('observed_fraud_risk', 0))

    return (
        bin_cat,
        int(amount // 500),         # Bucket amounts by $500
        int(np.clip(risk * 5, 0, 4)) # Risk buckets 0–4
    )

def calculate_advantages(results: List[Tuple[Any, float]], baseline: float = 0.5) -> List[Tuple[Any, float]]:
    """
    Calculates standardized advantage scores from simulation results.
    
    Args:
        results: List of (action, reward) tuples
        baseline: Neutral reward baseline
        
    Returns:
        List of (action, advantage) tuples
    """
    if not results:
        return []
        
    scores = [r for _, r in results]
    
    if len(scores) < 2:
        # If only one action, advantage is relative to baseline
        return [(results[0][0], results[0][1] - baseline)]
        
    mean = np.mean(scores)
    std = np.std(scores) + 1e-6 # Avoid div by zero
    
    return [(a, (r - mean) / std) for (a, r) in results]

def rank_actions(results: List[Tuple[Any, float]]) -> List[Tuple[Any, int]]:
    """
    Ranks actions by reward (higher index = better).
    """
    sorted_results = sorted(results, key=lambda x: x[1])
    return [(a, i) for i, (a, _) in enumerate(sorted_results)]
