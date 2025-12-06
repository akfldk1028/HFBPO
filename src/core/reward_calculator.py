import numpy as np
import math
from typing import Dict, Any

class RewardCalculator:
    """
    Calculates the Reward Score based on YouTube Analytics metrics.
    Formula: R = alpha*c + beta*w + gamma*e + delta*s + eta*u
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights from user example
        self.weights = weights or {
            "alpha": 0.2, # CTR
            "beta": 0.4,  # Watch Time (%)
            "gamma": 0.2, # Engagement
            "delta": 0.1, # Sentiment
            "eta": 0.1    # Subscribers
        }

    def calculate_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Compute the scalar reward from raw metrics.
        """
        # 1. Extract Raw Metrics
        views = metrics.get("views", 0)
        impressions = metrics.get("impressions", 0)
        likes = metrics.get("likes", 0)
        comments = metrics.get("comments", 0)
        shares = metrics.get("shares", 0)
        avg_view_percentage = metrics.get("avg_view_percentage", 0.0) # 0.0 to 1.0 (or 0-100, need to check)
        subscribers_gained = metrics.get("subscribers_gained", 0)
        subscribers_lost = metrics.get("subscribers_lost", 0)
        sentiment_mean = metrics.get("sentiment_mean", 0.0) # -1.0 to 1.0

        # Handle potential zero division
        if views == 0:
            return 0.0

        # 2. Calculate Rates (Normalized 0~1)
        
        # c: CTR
        # If impressions is 0, CTR is 0.
        ctr = (views / impressions) if impressions > 0 else 0.0
        ctr = min(1.0, max(0.0, ctr)) # Clip to [0, 1]

        # w: Watch Percentage
        # Assuming input is 0.0-1.0. If 0-100, divide by 100.
        # Usually YouTube API returns 0.45 for 45%.
        w = avg_view_percentage
        if w > 1.0: w /= 100.0 # Auto-correct if percent
        w = min(1.0, max(0.0, w))

        # e: Engagement Rate
        # (likes + comments + shares) / views
        # This can exceed 1.0 theoretically, so we clip or use log-sigmoid if needed.
        # User said "0~1로 스케일", so we clip for simplicity or use a soft cap.
        engagement_count = likes + comments + shares
        e = engagement_count / views
        e = min(1.0, max(0.0, e))

        # s: Sentiment Normalized
        # Map [-1, 1] -> [0, 1]
        # s' = (sentiment + 1) / 2
        s = (sentiment_mean + 1.0) / 2.0
        s = min(1.0, max(0.0, s))

        # u: Subscriber Rate
        # (gained - lost) / views
        # This is usually very small. User said "대략 작은 값, 나중에 스케일 조정".
        # Let's multiply by a factor (e.g., 100) to make it comparable, or just use raw.
        # For now, raw ratio clipped.
        net_subs = subscribers_gained - subscribers_lost
        u = net_subs / views
        # u can be negative.
        # Map to [0, 1]? Or allow negative reward impact?
        # User formula implies summation. If u is negative, it reduces reward.
        # But for "0~1 normalization", let's assume we want a positive score.
        # Let's keep it simple: if negative, 0. Or just add it as is (small penalty).
        # Given the weights, let's clip to [-1, 1] then map to [0, 1] or just clip 0.
        # Let's assume u is small positive usually.
        u = max(0.0, u * 10) # Boost factor 10, clip 0.
        u = min(1.0, u)

        # 3. Weighted Sum
        R = (self.weights["alpha"] * ctr +
             self.weights["beta"] * w +
             self.weights["gamma"] * e +
             self.weights["delta"] * s +
             self.weights["eta"] * u)

        return float(R)

    def normalize_log_min_max(self, value: float, min_val: float, max_val: float) -> float:
        """
        Helper for Log + Min-Max Scaling (for Views etc if needed later)
        v' = (log(1+v) - min_log) / (max_log - min_log)
        """
        if value <= 0: return 0.0
        log_v = math.log(1 + value)
        log_min = math.log(1 + min_val)
        log_max = math.log(1 + max_val)
        
        if log_max == log_min:
            return 0.0
            
        return (log_v - log_min) / (log_max - log_min)
