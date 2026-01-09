"""
CatProject Reward Calculator
정규화된 보상 계산 + 다중 시점 보상 + 채널 단계별 가중치

Critical Features:
- Min-Max 정규화로 공정한 가중치 적용
- 다중 시점 보상 (6h/24h/72h) - 바이럴 잠재력 반영
- 채널 성장 단계별 가중치 자동 조정
"""
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class YouTubeMetrics:
    """YouTube Analytics 메트릭"""
    views: int
    likes: int
    comments: int
    shares: int
    average_view_duration: float  # 초
    video_duration: float  # 초
    subscribers_gained: int
    impressions: int = 0  # CTR 계산용
    impression_click_through_rate: float = 0.0  # 직접 제공되면 사용


class CatRewardCalculator:
    """
    CatProject 보상 계산기

    - 정규화된 메트릭 계산
    - 채널 단계별 가중치 적용
    - 다중 시점 보상 지원
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: config.json 파일 경로
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'config.json'
            )

        self.config = self._load_config(config_path)
        self.normalization = self.config.get('normalization', {})
        self.reward_weights = self.config.get('reward_weights', {})
        self.multi_horizon = self.config.get('multi_horizon_reward', {})

    def _load_config(self, path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[RewardCalculator] Error loading config: {e}")
            return {}

    def normalize(self, value: float, metric_name: str) -> float:
        """
        메트릭 정규화 (0~1 범위)

        Args:
            value: 원본 값
            metric_name: 메트릭 이름 (retention_rate, ctr, etc.)

        Returns:
            정규화된 값 (0~1)
        """
        if metric_name not in self.normalization:
            return min(1.0, max(0.0, value))

        norm_config = self.normalization[metric_name]
        min_val = norm_config.get('min', 0)
        max_val = norm_config.get('max', 1)

        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        return min(1.0, max(0.0, normalized))

    def get_weights(self, subscriber_count: int) -> Dict[str, float]:
        """
        채널 구독자 수에 따른 가중치 반환

        Args:
            subscriber_count: 현재 채널 구독자 수

        Returns:
            가중치 딕셔너리
        """
        phase_a = self.reward_weights.get('phase_a_early', {})
        phase_b = self.reward_weights.get('phase_b_growth', {})
        phase_c = self.reward_weights.get('phase_c_stable', {})

        if subscriber_count < phase_a.get('subscriber_threshold', 1000):
            return phase_a.get('weights', {})
        elif subscriber_count < phase_b.get('subscriber_threshold', 100000):
            return phase_b.get('weights', {})
        else:
            return phase_c.get('weights', {})

    def calculate_metrics(self, metrics: YouTubeMetrics) -> Dict[str, float]:
        """
        원시 메트릭에서 계산된 비율 반환

        Args:
            metrics: YouTube 메트릭 데이터

        Returns:
            계산된 비율 딕셔너리
        """
        # Retention rate
        if metrics.video_duration > 0:
            retention_rate = metrics.average_view_duration / metrics.video_duration
        else:
            retention_rate = 0.5

        # CTR (Click-Through Rate)
        if metrics.impression_click_through_rate > 0:
            ctr = metrics.impression_click_through_rate
        elif metrics.impressions > 0:
            ctr = metrics.views / metrics.impressions
        else:
            ctr = 0.05  # 기본값

        # Engagement rate
        if metrics.views > 0:
            engagement_rate = (metrics.likes + metrics.comments) / metrics.views
        else:
            engagement_rate = 0.05

        # Share rate
        if metrics.views > 0:
            share_rate = metrics.shares / metrics.views
        else:
            share_rate = 0.005

        return {
            'retention_rate': retention_rate,
            'ctr': ctr,
            'engagement_rate': engagement_rate,
            'share_rate': share_rate,
            'subscribers_gained': metrics.subscribers_gained
        }

    def calculate_reward(
        self,
        metrics: YouTubeMetrics,
        subscriber_count: int = 0
    ) -> Dict[str, Any]:
        """
        단일 시점 보상 계산

        Args:
            metrics: YouTube 메트릭
            subscriber_count: 채널 구독자 수 (가중치 결정용)

        Returns:
            보상 정보 딕셔너리
        """
        # 메트릭 계산
        calculated = self.calculate_metrics(metrics)

        # 정규화
        normalized = {
            'retention': self.normalize(calculated['retention_rate'], 'retention_rate'),
            'ctr': self.normalize(calculated['ctr'], 'ctr'),
            'engagement': self.normalize(calculated['engagement_rate'], 'engagement_rate'),
            'share': self.normalize(calculated['share_rate'], 'share_rate'),
            'growth': self.normalize(calculated['subscribers_gained'], 'subscribers_gained')
        }

        # 가중치
        weights = self.get_weights(subscriber_count)

        # 가중 합계
        reward = (
            weights.get('retention', 0.30) * normalized['retention'] +
            weights.get('ctr', 0.10) * normalized['ctr'] +
            weights.get('engagement', 0.20) * normalized['engagement'] +
            weights.get('share', 0.20) * normalized['share'] +
            weights.get('growth', 0.20) * normalized['growth']
        )

        return {
            'reward': round(reward, 4),
            'raw_metrics': calculated,
            'normalized_metrics': normalized,
            'weights_used': weights,
            'subscriber_count': subscriber_count
        }

    def calculate_multi_horizon_reward(
        self,
        metrics_6h: Optional[YouTubeMetrics],
        metrics_24h: Optional[YouTubeMetrics],
        metrics_72h: Optional[YouTubeMetrics],
        subscriber_count: int = 0
    ) -> Dict[str, Any]:
        """
        다중 시점 보상 계산 (6h/24h/72h)

        바이럴 영상은 72시간 후에 폭발하는 경향이 있으므로
        단일 시점 보상보다 더 정확한 학습 신호 제공

        Args:
            metrics_6h: 6시간 후 메트릭
            metrics_24h: 24시간 후 메트릭
            metrics_72h: 72시간 후 메트릭
            subscriber_count: 채널 구독자 수

        Returns:
            다중 시점 가중 보상
        """
        if not self.multi_horizon.get('enabled', True):
            # 다중 시점 비활성화 시 가장 최신 메트릭 사용
            metrics = metrics_72h or metrics_24h or metrics_6h
            if metrics:
                return self.calculate_reward(metrics, subscriber_count)
            return {'reward': 0.5, 'error': 'No metrics provided'}

        horizon_weights = self.multi_horizon.get('weights', {
            '6h': 0.20,
            '24h': 0.40,
            '72h': 0.40
        })

        rewards = {}
        details = {}

        # 각 시점별 보상 계산
        if metrics_6h:
            result_6h = self.calculate_reward(metrics_6h, subscriber_count)
            rewards['6h'] = result_6h['reward']
            details['6h'] = result_6h

        if metrics_24h:
            result_24h = self.calculate_reward(metrics_24h, subscriber_count)
            rewards['24h'] = result_24h['reward']
            details['24h'] = result_24h

        if metrics_72h:
            result_72h = self.calculate_reward(metrics_72h, subscriber_count)
            rewards['72h'] = result_72h['reward']
            details['72h'] = result_72h

        # 가중 평균 계산
        total_weight = 0
        weighted_sum = 0

        for horizon, reward in rewards.items():
            weight = horizon_weights.get(horizon, 0.33)
            weighted_sum += weight * reward
            total_weight += weight

        if total_weight > 0:
            final_reward = weighted_sum / total_weight
        else:
            final_reward = 0.5

        return {
            'final_reward': round(final_reward, 4),
            'horizon_rewards': rewards,
            'horizon_weights': horizon_weights,
            'details': details,
            'subscriber_count': subscriber_count
        }


# 유틸리티 함수
def calculate_reward_from_dict(
    metrics_dict: Dict[str, Any],
    subscriber_count: int = 0,
    config_path: str = None
) -> float:
    """
    딕셔너리에서 직접 보상 계산 (API 호출용)

    Args:
        metrics_dict: {
            'views': int,
            'likes': int,
            'comments': int,
            'shares': int,
            'average_view_duration': float,
            'video_duration': float,
            'subscribers_gained': int,
            'impressions': int (optional),
            'ctr': float (optional)
        }
        subscriber_count: 채널 구독자 수
        config_path: config.json 경로

    Returns:
        정규화된 보상 (0~1)
    """
    calculator = CatRewardCalculator(config_path)

    metrics = YouTubeMetrics(
        views=metrics_dict.get('views', 0),
        likes=metrics_dict.get('likes', 0),
        comments=metrics_dict.get('comments', 0),
        shares=metrics_dict.get('shares', 0),
        average_view_duration=metrics_dict.get('average_view_duration', 0),
        video_duration=metrics_dict.get('video_duration', 24),  # 3씬 × 8초
        subscribers_gained=metrics_dict.get('subscribers_gained', 0),
        impressions=metrics_dict.get('impressions', 0),
        impression_click_through_rate=metrics_dict.get('ctr', 0)
    )

    result = calculator.calculate_reward(metrics, subscriber_count)
    return result['reward']


# 테스트
if __name__ == "__main__":
    calculator = CatRewardCalculator()

    print("=== CatRewardCalculator 테스트 ===\n")

    # 테스트 메트릭
    test_metrics = YouTubeMetrics(
        views=15000,
        likes=1200,
        comments=85,
        shares=45,
        average_view_duration=18.5,  # 초
        video_duration=24,  # 3씬 × 8초
        subscribers_gained=12,
        impressions=200000,
        impression_click_through_rate=0.075
    )

    print("=== 단일 시점 보상 계산 ===")

    # 초기 채널 (구독자 500)
    result_early = calculator.calculate_reward(test_metrics, subscriber_count=500)
    print(f"\n[초기 채널 - 구독자 500명]")
    print(f"  Final Reward: {result_early['reward']}")
    print(f"  Weights: {result_early['weights_used']}")

    # 성장 채널 (구독자 50000)
    result_growth = calculator.calculate_reward(test_metrics, subscriber_count=50000)
    print(f"\n[성장 채널 - 구독자 50,000명]")
    print(f"  Final Reward: {result_growth['reward']}")
    print(f"  Weights: {result_growth['weights_used']}")

    # 안정 채널 (구독자 200000)
    result_stable = calculator.calculate_reward(test_metrics, subscriber_count=200000)
    print(f"\n[안정 채널 - 구독자 200,000명]")
    print(f"  Final Reward: {result_stable['reward']}")
    print(f"  Weights: {result_stable['weights_used']}")

    print("\n=== 다중 시점 보상 계산 ===")

    # 6시간 후 메트릭 (초기)
    metrics_6h = YouTubeMetrics(
        views=2000, likes=150, comments=10, shares=5,
        average_view_duration=15, video_duration=24,
        subscribers_gained=2, impressions=30000
    )

    # 24시간 후 메트릭
    metrics_24h = YouTubeMetrics(
        views=8000, likes=600, comments=40, shares=20,
        average_view_duration=17, video_duration=24,
        subscribers_gained=6, impressions=100000
    )

    # 72시간 후 메트릭 (바이럴!)
    metrics_72h = YouTubeMetrics(
        views=25000, likes=2000, comments=150, shares=100,
        average_view_duration=19, video_duration=24,
        subscribers_gained=25, impressions=300000
    )

    multi_result = calculator.calculate_multi_horizon_reward(
        metrics_6h, metrics_24h, metrics_72h,
        subscriber_count=5000
    )

    print(f"\n[다중 시점 보상]")
    print(f"  6h reward: {multi_result['horizon_rewards'].get('6h', 'N/A')}")
    print(f"  24h reward: {multi_result['horizon_rewards'].get('24h', 'N/A')}")
    print(f"  72h reward: {multi_result['horizon_rewards'].get('72h', 'N/A')}")
    print(f"  Final (weighted): {multi_result['final_reward']}")
