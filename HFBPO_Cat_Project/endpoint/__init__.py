"""
CatProject API Endpoint Package

Usage:
    # 독립 실행
    uvicorn endpoint.cat_app:app --port 8001

    # 기존 server.py에 mount
    from HFBPO_Cat_Project.endpoint import cat_router
    app.include_router(cat_router)
"""

from .cat_router import router as cat_router
from .cat_app import app as cat_app
from .models import (
    SelectRequest, SelectResponse, CombinationInfo,
    GenerateRequest, GenerateResponse, FullGenerateResponse,
    RewardUpdateRequest, MultiHorizonRewardRequest, RewardUpdateResponse,
    StatsResponse, TopCombination, HealthResponse,
    SeriesType, YouTubeMetricsInput
)
from .dependencies import (
    get_components, get_agent, get_calculator, get_generator,
    CatProjectComponents
)

__all__ = [
    # Router & App
    'cat_router',
    'cat_app',

    # Models
    'SelectRequest', 'SelectResponse', 'CombinationInfo',
    'GenerateRequest', 'GenerateResponse', 'FullGenerateResponse',
    'RewardUpdateRequest', 'MultiHorizonRewardRequest', 'RewardUpdateResponse',
    'StatsResponse', 'TopCombination', 'HealthResponse',
    'SeriesType', 'YouTubeMetricsInput',

    # Dependencies
    'get_components', 'get_agent', 'get_calculator', 'get_generator',
    'CatProjectComponents'
]

__version__ = '1.0.0'
