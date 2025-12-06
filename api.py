"""
HFBPO FastAPI Server (고정 토픽 버전)
- /generate: 고정된 ARM에서 프롬프트 생성 → N8N 전송
- /reward: YouTube 성과 피드백 수신 (수동)
- /update-policy: Google Sheets + YouTube Analytics 기반 자동 피드백 루프
- /arms: 고정된 ARM 목록 조회
"""
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from src.generators.rapo_generator import RapoGenerator
from src.api.google_sheets_client import GoogleSheetsClient
from src.api.analytics_client import AnalyticsClient
from src.core.reward_calculator import RewardCalculator

load_dotenv("YTB/.env")

app = FastAPI(title="HFBPO API", description="Human Feedback Bandit Prompt Optimization (Fixed Topic)")

# 컴포넌트 초기화 (서버 시작 시 고정 토픽으로 ARM 생성)
generator = RapoGenerator(
    fixed_topic=os.getenv("FIXED_TOPIC", "디즈니 공주의 일상")
)
sheets_client = GoogleSheetsClient(
    credentials_path=os.getenv("GOOGLE_SHEETS_CREDENTIALS", "GOOGLE_SHEETS_CREDENTIALS.json"),
    sheet_id=os.getenv("GOOGLE_SHEET_ID")
)
analytics_client = AnalyticsClient()
reward_calc = RewardCalculator()

# N8N Webhook URL
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "https://cgxr.app.n8n.cloud/webhook/video-trigger")


class GenerateRequest(BaseModel):
    topic: Optional[str] = None  # 생략하면 고정 토픽 사용
    send_to_n8n: bool = True


class GenerateResponse(BaseModel):
    prompt: str
    combination_key: str
    place: str
    verb: str
    scenario: str
    estimated_reward: float
    candidates_count: int
    n8n_sent: bool = False


class RewardRequest(BaseModel):
    combination_key: str
    reward: float  # 0~1


class RewardResponse(BaseModel):
    success: bool
    message: str


@app.get("/")
def root():
    return {
        "service": "HFBPO",
        "description": "Human Feedback Bandit Prompt Optimization (Fixed Topic)",
        "fixed_topic": generator.fixed_topic,
        "fixed_arm_count": len(generator.fixed_candidates),
        "endpoints": {
            "/generate": "POST - 고정 ARM에서 프롬프트 생성",
            "/reward": "POST - 성과 보상 피드백 (수동)",
            "/update-policy": "POST - YouTube Analytics 기반 자동 피드백 루프",
            "/arms": "GET - 고정된 ARM 목록",
            "/stats": "GET - 밴딧 통계"
        }
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_prompt(req: GenerateRequest):
    """고정된 ARM에서 프롬프트 생성"""
    # 토픽 생략 시 고정 토픽 사용
    topic = req.topic.strip() if req.topic else generator.fixed_topic

    result = generator.generate(topic=topic)

    n8n_sent = False
    if req.send_to_n8n:
        try:
            payload = {
                "topic": topic,
                "place": result["place"],
                "verb": result["verb"],
                "scenario": result["scenario"],
                "combination_key": result["combination_key"],
                "estimated_reward": result["estimated_reward"],
                "candidates_count": result["candidates_count"]
            }
            response = requests.post(
                N8N_WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            n8n_sent = response.status_code == 200
        except Exception as e:
            print(f"[API] N8N 전송 실패: {e}")

    return GenerateResponse(
        prompt=result["prompt"],
        combination_key=result["combination_key"],
        place=result["place"],
        verb=result["verb"],
        scenario=result["scenario"],
        estimated_reward=result["estimated_reward"],
        candidates_count=result["candidates_count"],
        n8n_sent=n8n_sent
    )


@app.post("/reward", response_model=RewardResponse)
def update_reward(req: RewardRequest):
    """YouTube 성과 피드백으로 보상 업데이트"""
    if not req.combination_key:
        raise HTTPException(status_code=400, detail="combination_key가 비어있습니다")

    if not 0.0 <= req.reward <= 1.0:
        raise HTTPException(status_code=400, detail="reward는 0~1 사이여야 합니다")

    try:
        generator.update_reward(req.combination_key, req.reward)
        return RewardResponse(
            success=True,
            message=f"보상 {req.reward} 적용됨: {req.combination_key}"
        )
    except Exception as e:
        return RewardResponse(
            success=False,
            message=f"오류: {str(e)}"
        )


@app.get("/arms")
def get_arms():
    """고정된 ARM 목록 조회"""
    return {
        "fixed_topic": generator.fixed_topic,
        "total_arms": len(generator.fixed_candidates),
        "places": generator.fixed_places,
        "verbs": generator.fixed_verbs,
        "scenarios": generator.fixed_scenarios,
        "combinations": [
            f"{p}|{v}|{s}" for p, v, s in generator.fixed_candidates
        ]
    }


@app.get("/stats")
def get_stats():
    """밴딧 통계"""
    top_combinations = generator.get_top_combinations(10)
    return {
        "fixed_topic": generator.fixed_topic,
        "total_arms": len(generator.fixed_candidates),
        "learned_combinations": len(generator.bandit.combinations),
        "top_10": top_combinations
    }


class UpdatePolicyResponse(BaseModel):
    processed_count: int
    details: List[dict]


@app.post("/update-policy", response_model=UpdatePolicyResponse)
def update_policy():
    """
    자동 피드백 루프:
    1. Google Sheets에서 PENDING 상태 & 6시간 경과한 영상 조회
    2. YouTube Analytics에서 성과 지표 수집
    3. RewardCalculator로 0~1 점수 계산
    4. Bandit 가중치 업데이트
    5. Sheet 상태 업데이트 (DONE)
    """
    if not sheets_client.client:
        raise HTTPException(
            status_code=503,
            detail="Google Sheets Client not initialized (Check GOOGLE_SHEETS_CREDENTIALS)"
        )

    pending_videos = sheets_client.get_pending_videos()
    processed_details = []

    for video in pending_videos:
        video_id = str(video.get("VideoID"))
        combination_key = video.get("CombinationKey", "")
        row_index = video.get("row_index")

        if not combination_key:
            print(f"[UpdatePolicy] Skipping {video_id}: No combination_key")
            continue

        # YouTube Analytics에서 성과 수집
        metrics = analytics_client.get_video_analytics(video_id)
        if not metrics:
            print(f"[UpdatePolicy] Skipping {video_id}: No metrics found")
            continue

        # RewardCalculator로 점수 계산
        reward = reward_calc.calculate_reward(metrics)

        # Bandit 업데이트
        generator.update_reward(combination_key, reward)

        # Sheet 상태 업데이트
        sheets_client.update_video_status(row_index, "DONE", reward)

        processed_details.append({
            "video_id": video_id,
            "combination_key": combination_key,
            "reward": round(reward, 4),
            "metrics": metrics
        })

        print(f"[UpdatePolicy] Processed {video_id}: reward={reward:.4f}")

    return UpdatePolicyResponse(
        processed_count=len(processed_details),
        details=processed_details
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)