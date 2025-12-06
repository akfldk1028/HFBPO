import requests
import json
import time
from typing import Dict, Optional, Any

class YTBClient:
    """
    Client for communicating with the YTB Node.js Server
    """
    
    def __init__(self, base_url: str = "http://localhost:3123"):
        self.base_url = base_url.rstrip('/')

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

    def generate_nano_banana_video(self, prompt: str, orientation: str = "portrait") -> Dict[str, Any]:
        """
        Call YTB API to generate a video using NanoBanana mode
        """
        url = f"{self.base_url}/api/video/nano-banana"
        payload = {
            "scenes": [
                {
                    "text": prompt, # This might be used for TTS
                    "imageData": {
                        "prompt": prompt,
                        "style": "cinematic",
                        "mood": "dynamic"
                    }
                }
            ],
            "config": {
                "orientation": orientation,
                "videoSource": "ffmpeg" # or veo, depending on what we want
            }
        }
        
        print(f"[YTB Client] Requesting NanoBanana Video: {prompt}")
        return self._post_request(url, payload)

    def generate_veo_video(self, prompt: str, orientation: str = "portrait") -> Dict[str, Any]:
        """
        Call YTB API to generate a video using VEO3 mode
        """
        url = f"{self.base_url}/api/video/veo3"
        payload = {
            "prompt": prompt,
            "orientation": orientation
            # Note: Need to verify exact payload structure for veo3 endpoint
        }
        
        print(f"[YTB Client] Requesting VEO3 Video: {prompt}")
        return self._post_request(url, payload)

    def upload_to_youtube(self, video_path: str, title: str, description: str) -> Dict[str, Any]:
        """
        Call YTB API to upload a video
        """
        url = f"{self.base_url}/api/youtube/upload"
        payload = {
            "videoPath": video_path,
            "title": title,
            "description": description
        }
        return self._post_request(url, payload)

    def _post_request(self, url: str, payload: Dict) -> Dict[str, Any]:
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[YTB Client] Error calling {url}: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    client = YTBClient()
    if client.check_health():
        print("YTB Server is online!")
    else:
        print("YTB Server is OFFLINE. Please run 'npm start' in YTB directory.")
