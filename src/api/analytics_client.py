import requests
from typing import Dict, List, Any, Optional

class AnalyticsClient:
    """
    Client for fetching YouTube Analytics data from the Video Backend (Cloud Run).
    """
    
    def __init__(self, base_url: str = "https://short-video-maker-7qtnitbuvq-uc.a.run.app"):
        self.base_url = base_url.rstrip('/')

    def get_video_analytics(self, video_id: str, channel_name: str = "ATT") -> Dict[str, Any]:
        """
        Fetch full analytics for a specific video.
        GET /api/analytics/video/{videoId}/full?channelName={channelName}
        """
        url = f"{self.base_url}/api/analytics/video/{video_id}/full"
        params = {"channelName": channel_name}
        
        print(f"[Analytics] Fetching data for {video_id}...")
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant metrics from the response structure
            # Assuming the API returns a flat dict or nested 'analytics' key
            # Based on user description, it returns raw metrics.
            # We map them to standard keys for RewardCalculator.
            
            # Example response mapping (adjust based on actual API response)
            metrics = {
                "views": data.get("views", 0),
                "impressions": data.get("impressions", 0),
                "likes": data.get("likes", 0),
                "comments": data.get("comments", 0),
                "shares": data.get("shares", 0),
                "avg_view_percentage": data.get("avgViewPercentage", 0.0), # CamelCase likely
                "subscribers_gained": data.get("subscribersGained", 0),
                "subscribers_lost": data.get("subscribersLost", 0),
                "sentiment_mean": data.get("sentimentMean", 0.0)
            }
            
            # Fallback for snake_case if API uses that
            if "avg_view_percentage" in data:
                metrics["avg_view_percentage"] = data["avg_view_percentage"]
            
            return metrics
            
        except Exception as e:
            print(f"[Analytics] Error fetching analytics for {video_id}: {e}")
            return {}
