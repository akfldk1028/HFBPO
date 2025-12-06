"""
HFBPO FastAPI Server
- POST /generate: 토픽 → 프롬프트 생성
- POST /reward: 단일 보상 업데이트
- POST /batch-reward: 다중 보상 업데이트
- POST /calculate-reward: Analytics → reward 변환
- POST /update-policy: Google Sheets + Analytics 자동 피드백 루프
- GET /stats: 통계 조회
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

from src.generators.rapo_generator import RapoGenerator
from src.api.google_sheets_client import GoogleSheetsClient
from src.api.analytics_client import AnalyticsClient
from src.core.reward_calculator import RewardCalculator

# .env 로드
load_dotenv(project_root / ".env")
load_dotenv("/mnt/d/Data/00_Personal/YTB/short-video-maker/.env")  # fallback

app = FastAPI(
    title="HFBPO API",
    description="Human Feedback Bandit Prompt Optimization",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Generator 초기화 (서버 시작 시 한 번만)
generator: Optional[RapoGenerator] = None

# Google Sheets & Analytics 클라이언트
sheets_client: Optional[GoogleSheetsClient] = None
analytics_client: Optional[AnalyticsClient] = None
reward_calc: Optional[RewardCalculator] = None


def get_generator() -> RapoGenerator:
    global generator
    if generator is None:
        generator = RapoGenerator()
    return generator


def get_sheets_client() -> GoogleSheetsClient:
    global sheets_client
    if sheets_client is None:
        sheets_client = GoogleSheetsClient(
            credentials_path=os.getenv("GOOGLE_SHEETS_CREDENTIALS", "GOOGLE_SHEETS_CREDENTIALS.json"),
            sheet_id=os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")
        )
    return sheets_client


def get_analytics_client() -> AnalyticsClient:
    global analytics_client
    if analytics_client is None:
        analytics_client = AnalyticsClient()
    return analytics_client


def get_reward_calc() -> RewardCalculator:
    global reward_calc
    if reward_calc is None:
        reward_calc = RewardCalculator()
    return reward_calc


# ============== Request/Response Models ==============

class GenerateRequest(BaseModel):
    topic: str = Field(..., description="프롬프트 생성할 토픽", example="한강 야경")


class GenerateResponse(BaseModel):
    prompt: str
    combination_key: str
    place: str
    verb: str
    scenario: str
    estimated_reward: float
    candidates_count: int


class RewardRequest(BaseModel):
    combination_key: str = Field(..., description="HFBPO 조합 키", example="han_river|zoom|romantic")
    reward: float = Field(..., ge=0.0, le=1.0, description="보상값 (0~1)", example=0.75)


class RewardResponse(BaseModel):
    success: bool
    message: str
    combination_key: str
    new_alpha: Optional[float] = None
    new_beta: Optional[float] = None


class BatchRewardItem(BaseModel):
    combination_key: str
    reward: float = Field(..., ge=0.0, le=1.0)


class BatchRewardRequest(BaseModel):
    rewards: List[BatchRewardItem]


class BatchRewardResponse(BaseModel):
    success: bool
    updated_count: int
    failed: List[str] = []


class CalculateRewardRequest(BaseModel):
    views: int = Field(0, description="조회수")
    likes: int = Field(0, description="좋아요")
    comments: int = Field(0, description="댓글")
    shares: int = Field(0, description="공유")
    average_watch_percentage: float = Field(0, description="평균 시청률 (%)")
    subscribers_gained: int = Field(0, description="구독자 증가")


class RewardBreakdown(BaseModel):
    ctr_score: float
    watch_score: float
    engagement_score: float
    growth_score: float


class CalculateRewardResponse(BaseModel):
    reward: float
    breakdown: RewardBreakdown


# ============== Endpoints ==============

@app.get("/")
def root():
    return {
        "service": "HFBPO",
        "version": "1.1.0",
        "description": "Human Feedback Bandit Prompt Optimization",
        "endpoints": {
            "POST /generate": "토픽 → 프롬프트 생성",
            "POST /reward": "단일 보상 업데이트",
            "POST /batch-reward": "다중 보상 업데이트",
            "POST /calculate-reward": "Analytics → reward 변환",
            "POST /update-policy": "Google Sheets + Analytics 자동 피드백 루프",
            "GET /stats": "상위 조합 통계"
        }
    }


@app.get("/generate", response_model=GenerateResponse)
def generate_prompt_get(topic: str):
    """
    [GET] 토픽을 받아 최적화된 프롬프트 생성

    사용법: /generate?topic=한강야경

    - RAPO Pipeline: Retrieval → Selection → Generation
    - Thompson Sampling으로 최적 조합 선택
    """
    if not topic.strip():
        raise HTTPException(status_code=400, detail="토픽이 비어있습니다")

    gen = get_generator()
    result = gen.generate(topic=topic)

    return GenerateResponse(
        prompt=result["prompt"],
        combination_key=result["combination_key"],
        place=result["place"],
        verb=result["verb"],
        scenario=result["scenario"],
        estimated_reward=result["estimated_reward"],
        candidates_count=result["candidates_count"]
    )


@app.post("/generate", response_model=GenerateResponse)
def generate_prompt_post(req: GenerateRequest):
    """
    [POST] 토픽을 받아 최적화된 프롬프트 생성

    - RAPO Pipeline: Retrieval → Selection → Generation
    - Thompson Sampling으로 최적 조합 선택
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="토픽이 비어있습니다")

    gen = get_generator()
    result = gen.generate(topic=req.topic)

    return GenerateResponse(
        prompt=result["prompt"],
        combination_key=result["combination_key"],
        place=result["place"],
        verb=result["verb"],
        scenario=result["scenario"],
        estimated_reward=result["estimated_reward"],
        candidates_count=result["candidates_count"]
    )


@app.post("/reward", response_model=RewardResponse)
def update_reward(req: RewardRequest):
    """
    YouTube 성과 피드백으로 보상 업데이트

    - Thompson Sampling의 alpha/beta 업데이트
    - alpha += reward, beta += (1 - reward)
    """
    if not req.combination_key:
        raise HTTPException(status_code=400, detail="combination_key가 비어있습니다")

    gen = get_generator()

    try:
        gen.update_reward(req.combination_key, req.reward)

        # 업데이트된 alpha/beta 값 조회
        state = gen.bandit.combinations.get(req.combination_key, {})

        return RewardResponse(
            success=True,
            message=f"보상 {req.reward} 적용됨",
            combination_key=req.combination_key,
            new_alpha=state.get("alpha"),
            new_beta=state.get("beta")
        )
    except Exception as e:
        return RewardResponse(
            success=False,
            message=f"오류: {str(e)}",
            combination_key=req.combination_key
        )


@app.post("/batch-reward", response_model=BatchRewardResponse)
def batch_update_reward(req: BatchRewardRequest):
    """
    여러 비디오의 보상을 한번에 업데이트
    """
    gen = get_generator()
    updated = 0
    failed = []

    for item in req.rewards:
        try:
            gen.update_reward(item.combination_key, item.reward)
            updated += 1
        except Exception as e:
            failed.append(f"{item.combination_key}: {str(e)}")

    return BatchRewardResponse(
        success=len(failed) == 0,
        updated_count=updated,
        failed=failed
    )


@app.post("/calculate-reward", response_model=CalculateRewardResponse)
def calculate_reward(req: CalculateRewardRequest):
    """
    YouTube Analytics 데이터를 0~1 reward 값으로 변환

    공식:
    - CTR score (20%): views 기반 (임시로 engagement rate 사용)
    - Watch score (40%): 평균 시청률
    - Engagement score (20%): 좋아요, 댓글, 공유
    - Growth score (20%): 구독자 증가
    """
    # 가중치
    W_CTR = 0.20
    W_WATCH = 0.40
    W_ENGAGEMENT = 0.20
    W_GROWTH = 0.20

    # CTR score (engagement rate로 대체)
    if req.views > 0:
        engagement_rate = (req.likes + req.comments + req.shares) / req.views
        ctr_score = min(engagement_rate / 0.05, 1.0)  # 5% = 만점
    else:
        ctr_score = 0.0

    # Watch score
    watch_score = min(req.average_watch_percentage / 100, 1.0)

    # Engagement score (가중 합계)
    if req.views > 0:
        weighted_engagement = (req.likes + req.comments * 5 + req.shares * 10) / req.views
        engagement_score = min(weighted_engagement / 0.1, 1.0)  # 10% = 만점
    else:
        engagement_score = 0.0

    # Growth score
    growth_score = min(req.subscribers_gained / 10, 1.0)  # 10명 = 만점

    # 최종 reward
    reward = (
        W_CTR * ctr_score +
        W_WATCH * watch_score +
        W_ENGAGEMENT * engagement_score +
        W_GROWTH * growth_score
    )

    return CalculateRewardResponse(
        reward=round(reward, 4),
        breakdown=RewardBreakdown(
            ctr_score=round(ctr_score * W_CTR, 4),
            watch_score=round(watch_score * W_WATCH, 4),
            engagement_score=round(engagement_score * W_ENGAGEMENT, 4),
            growth_score=round(growth_score * W_GROWTH, 4)
        )
    )


@app.get("/stats")
def get_stats():
    """
    밴딧 통계 - 상위 조합 및 전체 학습 상태
    """
    gen = get_generator()
    top_combinations = gen.get_top_combinations(10)

    return {
        "total_learned_combinations": len(gen.bandit.combinations),
        "retriever_stats": {
            "places": 153,
            "verbs": 62,
            "scenarios": 81
        },
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
    sheets = get_sheets_client()
    analytics = get_analytics_client()
    calc = get_reward_calc()
    gen = get_generator()

    if not sheets.client:
        raise HTTPException(
            status_code=503,
            detail="Google Sheets Client not initialized (Check GOOGLE_SHEETS_CREDENTIALS)"
        )

    pending_videos = sheets.get_pending_videos()
    processed_details = []

    for video in pending_videos:
        video_id = str(video.get("VideoID"))
        combination_key = video.get("CombinationKey", "")
        row_index = video.get("row_index")

        if not combination_key:
            print(f"[UpdatePolicy] Skipping {video_id}: No combination_key")
            continue

        # YouTube Analytics에서 성과 수집
        metrics = analytics.get_video_analytics(video_id)
        if not metrics:
            print(f"[UpdatePolicy] Skipping {video_id}: No metrics found")
            continue

        # RewardCalculator로 점수 계산
        reward = calc.calculate_reward(metrics)

        # Bandit 업데이트
        gen.update_reward(combination_key, reward)

        # Sheet 상태 업데이트
        sheets.update_video_status(row_index, "DONE", reward)

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


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
