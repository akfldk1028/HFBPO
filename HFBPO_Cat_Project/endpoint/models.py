"""
CatProject API Models
Pydantic 요청/응답 스키마 정의
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class SeriesType(str, Enum):
    """시리즈 타입"""
    DAILY_LIFE = "daily_life"
    KAMI_SOLO = "kami_solo"
    DALGI_SOLO = "dalgi_solo"
    SPECIAL = "special"


# ============ 조합 관련 ============

class CombinationInfo(BaseModel):
    """조합 정보"""
    combination_key: str = Field(..., description="hook|conflict|ending 형식의 조합 키")
    hook_type: str
    conflict_type: str
    ending_type: str
    hook_info: Dict[str, Any] = {}
    conflict_info: Dict[str, Any] = {}
    ending_info: Dict[str, Any] = {}
    estimated_reward: float = Field(..., ge=0, le=1)
    compatibility_score: float = Field(..., ge=0, le=1)
    selection_mode: str = Field(..., description="random_exploration, factored_exploration, exploitation_exploit, etc.")
    trial_number: int


class SelectRequest(BaseModel):
    """조합 선택 요청"""
    series: SeriesType = SeriesType.DAILY_LIFE
    theme: str = "커플 일상"


class SelectResponse(BaseModel):
    """조합 선택 응답"""
    combination: CombinationInfo
    message: str = "Combination selected successfully"


# ============ 에피소드 관련 ============

class SceneInfo(BaseModel):
    """씬 정보"""
    characterIds: List[str]
    text: str
    textEnglish: str
    scenePrompt: str
    duration: int = 8


class EpisodeInfo(BaseModel):
    """에피소드 정보"""
    titleText: Dict[str, str]
    scenes: List[SceneInfo]
    soundEffects: List[Dict[str, Any]] = []
    youtubeTitle: str
    youtubeDescription: str
    _meta: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    """에피소드 생성 요청"""
    combination_key: str = Field(..., description="hook|conflict|ending 형식")
    theme: str = "커플 일상"
    series: SeriesType = SeriesType.DAILY_LIFE


class GenerateResponse(BaseModel):
    """에피소드 생성 응답"""
    combination_key: str
    episode: Dict[str, Any]
    api_body: Dict[str, Any] = Field(..., description="ShortVideoMaker API 형식")


class FullGenerateResponse(BaseModel):
    """조합 선택 + 에피소드 생성 통합 응답"""
    combination: CombinationInfo
    episode: Dict[str, Any]
    api_body: Dict[str, Any]


# ============ 보상 업데이트 관련 ============

class YouTubeMetricsInput(BaseModel):
    """YouTube 메트릭 입력"""
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    average_view_duration: float = 0  # 초
    video_duration: float = 24  # 기본 24초 (3씬 × 8초)
    subscribers_gained: int = 0
    impressions: int = 0
    ctr: float = 0  # 직접 제공되면 사용


class RewardUpdateRequest(BaseModel):
    """보상 업데이트 요청 (단일 시점)"""
    combination_key: str
    reward: float = Field(..., ge=0, le=1, description="정규화된 보상 (0~1)")


class MultiHorizonRewardRequest(BaseModel):
    """다중 시점 보상 업데이트 요청"""
    combination_key: str
    video_id: Optional[str] = None
    subscriber_count: int = 0
    metrics_6h: Optional[YouTubeMetricsInput] = None
    metrics_24h: Optional[YouTubeMetricsInput] = None
    metrics_72h: Optional[YouTubeMetricsInput] = None


class RewardUpdateResponse(BaseModel):
    """보상 업데이트 응답"""
    combination_key: str
    reward: float
    new_alpha: float
    new_beta: float
    new_mean: float
    total_trials: int


# ============ 통계/조회 관련 ============

class TopCombination(BaseModel):
    """상위 조합 정보"""
    key: str
    hook_type: str
    conflict_type: str
    ending_type: str
    mean_reward: float
    alpha: float
    beta: float
    trials: int


class StatsResponse(BaseModel):
    """학습 통계 응답"""
    total_trials: int
    unique_combinations: int
    total_possible: int
    coverage: float  # 퍼센트
    mean_reward: float
    max_reward: float
    min_reward: float
    top_3: List[TopCombination]


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = "ok"
    project: str = "CatProject"
    total_trials: int
    unique_combinations: int
