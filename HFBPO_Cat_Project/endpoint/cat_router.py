"""
CatProject API Router
FastAPI APIRouter - 기존 server.py에 mount 가능
"""
import os
import sys
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional

# 경로 설정 (직접 실행 지원)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT_DIR = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ENDPOINT_DIR not in sys.path:
    sys.path.insert(0, ENDPOINT_DIR)

# Import 호환성 처리 (패키지 vs 직접 실행)
try:
    from .models import (
        SelectRequest, SelectResponse, CombinationInfo,
        GenerateRequest, GenerateResponse, FullGenerateResponse,
        RewardUpdateRequest, MultiHorizonRewardRequest, RewardUpdateResponse,
        StatsResponse, TopCombination, HealthResponse,
        SeriesType
    )
    from .dependencies import (
        get_components, get_agent, get_calculator, get_generator,
        create_youtube_metrics, CatProjectComponents
    )
except ImportError:
    from models import (
        SelectRequest, SelectResponse, CombinationInfo,
        GenerateRequest, GenerateResponse, FullGenerateResponse,
        RewardUpdateRequest, MultiHorizonRewardRequest, RewardUpdateResponse,
        StatsResponse, TopCombination, HealthResponse,
        SeriesType
    )
    from dependencies import (
        get_components, get_agent, get_calculator, get_generator,
        create_youtube_metrics, CatProjectComponents
    )


def pydantic_to_dict(model):
    """Pydantic v1/v2 호환 dict 변환"""
    if model is None:
        return None
    if hasattr(model, 'model_dump'):
        return model.model_dump()  # Pydantic v2
    return model.dict()  # Pydantic v1


router = APIRouter(prefix="/cat", tags=["CatProject"])


# ============ Health Check ============

@router.get("/health", response_model=HealthResponse)
def health_check(components: CatProjectComponents = Depends(get_components)):
    """
    CatProject 상태 확인

    Returns:
        status, project name, 학습 상태
    """
    return HealthResponse(
        status="ok",
        project="CatProject",
        total_trials=components.agent.total_trials,
        unique_combinations=len(components.agent.combinations)
    )


# ============ 조합 선택 ============

@router.get("/select", response_model=SelectResponse)
def select_combination(
    series: SeriesType = Query(SeriesType.DAILY_LIFE, description="시리즈 타입"),
    components: CatProjectComponents = Depends(get_components)
):
    """
    Thompson Sampling으로 조합 선택

    - trials < 30: 완전 랜덤 탐색
    - 30 <= trials < 60: Factored 탐색
    - trials >= 60: Thompson Sampling (10% 탐색)

    Args:
        series: 시리즈 타입 (daily_life, kami_solo, etc.)

    Returns:
        선택된 조합 정보
    """
    try:
        result = components.agent.select_combination(series=series.value)

        combination = CombinationInfo(
            combination_key=result['combination_key'],
            hook_type=result['hook_type'],
            conflict_type=result['conflict_type'],
            ending_type=result['ending_type'],
            hook_info=result.get('hook_info', {}),
            conflict_info=result.get('conflict_info', {}),
            ending_info=result.get('ending_info', {}),
            estimated_reward=result['estimated_reward'],
            compatibility_score=result['compatibility_score'],
            selection_mode=result['selection_mode'],
            trial_number=result['trial_number']
        )

        return SelectResponse(
            combination=combination,
            message=f"Selected via {result['selection_mode']}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Selection failed: {str(e)}")


# ============ 에피소드 생성 ============

@router.post("/generate", response_model=GenerateResponse)
def generate_episode(
    request: GenerateRequest,
    components: CatProjectComponents = Depends(get_components)
):
    """
    지정된 조합으로 GPT 에피소드 생성

    Args:
        combination_key: hook|conflict|ending 형식
        theme: 에피소드 테마
        series: 시리즈 타입

    Returns:
        생성된 에피소드 + ShortVideoMaker API 형식
    """
    try:
        # 조합 정보 파싱
        parts = request.combination_key.split('|')
        if len(parts) != 3:
            raise HTTPException(
                status_code=400,
                detail="Invalid combination_key format. Expected: hook|conflict|ending"
            )

        hook, conflict, ending = parts

        # 조합 딕셔너리 구성
        combination = {
            'combination_key': request.combination_key,
            'hook_type': hook,
            'conflict_type': conflict,
            'ending_type': ending,
            'hook_info': components.agent.hook_types.get('modifiers', {}).get(hook, {}),
            'conflict_info': components.agent.conflict_types.get('modifiers', {}).get(conflict, {}),
            'ending_info': components.agent.ending_types.get('modifiers', {}).get(ending, {})
        }

        # 에피소드 생성
        episode = components.generator.generate_episode(
            combination=combination,
            theme=request.theme,
            series=request.series.value
        )

        # API 형식 변환
        api_body = components.generator.format_for_api(episode)

        return GenerateResponse(
            combination_key=request.combination_key,
            episode=episode,
            api_body=api_body
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/select-and-generate", response_model=FullGenerateResponse)
def select_and_generate(
    series: SeriesType = Query(SeriesType.DAILY_LIFE),
    theme: str = Query("커플 일상"),
    components: CatProjectComponents = Depends(get_components)
):
    """
    조합 선택 + 에피소드 생성 통합 엔드포인트

    n8n에서 단일 호출로 전체 프로세스 실행

    Args:
        series: 시리즈 타입
        theme: 에피소드 테마

    Returns:
        선택된 조합 + 생성된 에피소드 + API 형식
    """
    try:
        # 1. 조합 선택
        result = components.agent.select_combination(series=series.value)

        combination = CombinationInfo(
            combination_key=result['combination_key'],
            hook_type=result['hook_type'],
            conflict_type=result['conflict_type'],
            ending_type=result['ending_type'],
            hook_info=result.get('hook_info', {}),
            conflict_info=result.get('conflict_info', {}),
            ending_info=result.get('ending_info', {}),
            estimated_reward=result['estimated_reward'],
            compatibility_score=result['compatibility_score'],
            selection_mode=result['selection_mode'],
            trial_number=result['trial_number']
        )

        # 2. 에피소드 생성
        episode = components.generator.generate_episode(
            combination=result,
            theme=theme,
            series=series.value
        )

        # 3. API 형식 변환
        api_body = components.generator.format_for_api(episode)

        return FullGenerateResponse(
            combination=combination,
            episode=episode,
            api_body=api_body
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Select and generate failed: {str(e)}")


# ============ 보상 업데이트 ============

@router.post("/update", response_model=RewardUpdateResponse)
def update_reward(
    request: RewardUpdateRequest,
    components: CatProjectComponents = Depends(get_components)
):
    """
    조합에 보상 업데이트 (단일 시점)

    n8n에서 YouTube 메트릭 수집 후 계산된 보상 전달

    Args:
        combination_key: hook|conflict|ending 형식
        reward: 정규화된 보상 (0~1)

    Returns:
        업데이트된 조합 정보
    """
    try:
        # 보상 업데이트
        components.agent.update_reward(
            combination_key=request.combination_key,
            reward=request.reward
        )

        # 업데이트된 상태 조회
        stats = components.agent.combinations.get(request.combination_key, {})
        alpha = stats.get('alpha', 1)
        beta = stats.get('beta', 1)

        return RewardUpdateResponse(
            combination_key=request.combination_key,
            reward=request.reward,
            new_alpha=alpha,
            new_beta=beta,
            new_mean=alpha / (alpha + beta),
            total_trials=components.agent.total_trials
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.post("/update-multi-horizon", response_model=RewardUpdateResponse)
def update_multi_horizon_reward(
    request: MultiHorizonRewardRequest,
    components: CatProjectComponents = Depends(get_components)
):
    """
    다중 시점 보상 업데이트 (6h/24h/72h)

    바이럴 영상의 지연 효과를 반영한 가중 보상 계산

    Args:
        combination_key: 조합 키
        metrics_6h/24h/72h: 각 시점의 YouTube 메트릭
        subscriber_count: 채널 구독자 수 (가중치 결정용)

    Returns:
        업데이트된 조합 정보
    """
    try:
        # 메트릭 객체 생성 (Pydantic v1/v2 호환)
        m6h = create_youtube_metrics(pydantic_to_dict(request.metrics_6h)) if request.metrics_6h else None
        m24h = create_youtube_metrics(pydantic_to_dict(request.metrics_24h)) if request.metrics_24h else None
        m72h = create_youtube_metrics(pydantic_to_dict(request.metrics_72h)) if request.metrics_72h else None

        # 다중 시점 보상 계산
        result = components.calculator.calculate_multi_horizon_reward(
            metrics_6h=m6h,
            metrics_24h=m24h,
            metrics_72h=m72h,
            subscriber_count=request.subscriber_count
        )

        final_reward = result.get('final_reward', 0.5)

        # Bandit 업데이트
        components.agent.update_reward(
            combination_key=request.combination_key,
            reward=final_reward
        )

        # 업데이트된 상태
        stats = components.agent.combinations.get(request.combination_key, {})
        alpha = stats.get('alpha', 1)
        beta = stats.get('beta', 1)

        return RewardUpdateResponse(
            combination_key=request.combination_key,
            reward=final_reward,
            new_alpha=alpha,
            new_beta=beta,
            new_mean=alpha / (alpha + beta),
            total_trials=components.agent.total_trials
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-horizon update failed: {str(e)}")


# ============ 통계/조회 ============

@router.get("/stats", response_model=StatsResponse)
def get_stats(components: CatProjectComponents = Depends(get_components)):
    """
    현재 학습 상태 조회

    Returns:
        전체 통계 + Top 3 조합
    """
    try:
        stats = components.agent.get_stats()

        # Top 3 변환
        top_3 = []
        for item in stats.get('top_3', []):
            top_3.append(TopCombination(
                key=item['key'],
                hook_type=item['hook_type'],
                conflict_type=item['conflict_type'],
                ending_type=item['ending_type'],
                mean_reward=item['mean_reward'],
                alpha=item['alpha'],
                beta=item['beta'],
                trials=item['trials']
            ))

        return StatsResponse(
            total_trials=stats.get('total_trials', 0),
            unique_combinations=stats.get('unique_combinations', 0),
            total_possible=stats.get('total_possible', 180),
            coverage=stats.get('coverage', 0),
            mean_reward=stats.get('mean_reward', 0.5),
            max_reward=stats.get('max_reward', 0.5),
            min_reward=stats.get('min_reward', 0.5),
            top_3=top_3
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@router.get("/top/{n}", response_model=List[TopCombination])
def get_top_combinations(
    n: int = 10,
    components: CatProjectComponents = Depends(get_components)
):
    """
    성과 좋은 조합 Top N 조회

    Args:
        n: 반환할 조합 수 (기본 10)

    Returns:
        Top N 조합 리스트
    """
    try:
        top_list = components.agent.get_top_combinations(n)

        return [
            TopCombination(
                key=item['key'],
                hook_type=item['hook_type'],
                conflict_type=item['conflict_type'],
                ending_type=item['ending_type'],
                mean_reward=item['mean_reward'],
                alpha=item['alpha'],
                beta=item['beta'],
                trials=item['trials']
            )
            for item in top_list
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top combinations failed: {str(e)}")
