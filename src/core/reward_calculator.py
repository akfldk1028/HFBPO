import numpy as np
import math
from typing import Dict, Any


class RewardCalculator:
    """YouTube 지표 → 스칼라 보상 변환"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "ctr": 0.2,
            "retention": 0.4,  # 가장 중요
            "engagement": 0.2,
            "sentiment": 0.1,
            "subscriber": 0.1
        }

    def calculate_reward(self, metrics: Dict[str, Any]) -> float:
        """
        R = w1*CTR + w2*Retention + w3*Engagement + w4*Sentiment + w5*Subscriber
        """
        views = metrics.get("views", 0)
        impressions = metrics.get("impressions", 0)
        likes = metrics.get("likes", 0)
        comments = metrics.get("comments", 0)
        shares = metrics.get("shares", 0)
        avg_view_pct = metrics.get("avg_view_percentage", 0.0)
        subs_gained = metrics.get("subscribers_gained", 0)
        subs_lost = metrics.get("subscribers_lost", 0)
        sentiment = metrics.get("sentiment_mean", 0.0)

        if views == 0:
            return 0.0

        # CTR: views / impressions
        ctr = (views / impressions) if impressions > 0 else 0.0
        ctr = np.clip(ctr, 0, 1)

        # Retention: 시청 유지율 (0~1 또는 0~100)
        retention = avg_view_pct / 100.0 if avg_view_pct > 1 else avg_view_pct
        retention = np.clip(retention, 0, 1)

        # Engagement: (likes + comments + shares) / views
        engagement = (likes + comments + shares) / views
        engagement = np.clip(engagement, 0, 1)

        # Sentiment: [-1, 1] → [0, 1]
        sent_norm = (sentiment + 1.0) / 2.0
        sent_norm = np.clip(sent_norm, 0, 1)

        # Subscriber: net_subs / views (부스트 x10)
        net_subs = subs_gained - subs_lost
        sub_rate = np.clip(max(0, net_subs / views * 10), 0, 1)

        # 가중합
        R = (self.weights["ctr"] * ctr +
             self.weights["retention"] * retention +
             self.weights["engagement"] * engagement +
             self.weights["sentiment"] * sent_norm +
             self.weights["subscriber"] * sub_rate)

        return float(R)
