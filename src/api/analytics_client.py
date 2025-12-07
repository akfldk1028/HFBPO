import requests
from typing import Dict, Any


class AnalyticsClient:
    """YouTube Analytics API 클라이언트 (Cloud Run 백엔드 연동)"""

    def __init__(self, base_url: str = "https://short-video-maker-7qtnitbuvq-uc.a.run.app"):
        self.base_url = base_url.rstrip('/')

    def get_video_analytics(self, video_id: str, channel_name: str = "ATT") -> Dict[str, Any]:
        """영상별 Analytics 조회"""
        url = f"{self.base_url}/api/analytics/video/{video_id}/full"
        params = {"channelName": channel_name}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "views": data.get("views", 0),
                "impressions": data.get("impressions", 0),
                "likes": data.get("likes", 0),
                "comments": data.get("comments", 0),
                "shares": data.get("shares", 0),
                "avg_view_percentage": data.get("avgViewPercentage", data.get("avg_view_percentage", 0.0)),
                "subscribers_gained": data.get("subscribersGained", 0),
                "subscribers_lost": data.get("subscribersLost", 0),
                "sentiment_mean": data.get("sentimentMean", 0.0)
            }

        except Exception as e:
            print(f"[Analytics] Error: {e}")
            return {}
