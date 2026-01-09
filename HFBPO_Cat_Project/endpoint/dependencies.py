"""
CatProject Dependencies
싱글톤 인스턴스 관리 및 의존성 주입
"""
import os
import sys
from functools import lru_cache
from typing import Optional

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import 호환성 처리 (패키지 vs 직접 실행)
try:
    # 직접 실행 또는 독립 패키지로 실행
    from src.cat_bandit_agent import CatBanditAgent
    from src.cat_reward_calculator import CatRewardCalculator, YouTubeMetrics
    from src.cat_episode_generator import CatEpisodeGenerator
except ModuleNotFoundError:
    # HFBPO 컨테이너에서 마운트될 때
    from HFBPO_Cat_Project.src.cat_bandit_agent import CatBanditAgent
    from HFBPO_Cat_Project.src.cat_reward_calculator import CatRewardCalculator, YouTubeMetrics
    from HFBPO_Cat_Project.src.cat_episode_generator import CatEpisodeGenerator


# 데이터 디렉토리 경로
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROMPTS_DIR = os.path.join(PROJECT_ROOT, 'prompts')


class CatProjectComponents:
    """
    CatProject 컴포넌트 싱글톤 컨테이너

    FastAPI 앱 lifecycle 동안 동일 인스턴스 유지
    """
    _instance: Optional['CatProjectComponents'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print("[CatProject] Initializing components...")

        # 컴포넌트 초기화
        self.agent = CatBanditAgent(data_dir=DATA_DIR)
        self.calculator = CatRewardCalculator(
            config_path=os.path.join(DATA_DIR, 'config.json')
        )
        self.generator = CatEpisodeGenerator(
            config_path=os.path.join(DATA_DIR, 'config.json'),
            prompts_dir=PROMPTS_DIR
        )

        self._initialized = True
        print(f"[CatProject] Components initialized")
        print(f"  - Agent: {len(self.agent.combinations)} saved combinations")
        print(f"  - Total trials: {self.agent.total_trials}")


@lru_cache()
def get_components() -> CatProjectComponents:
    """
    CatProject 컴포넌트 싱글톤 가져오기

    FastAPI Depends()에서 사용:
    ```python
    @app.get("/cat/select")
    def select(components: CatProjectComponents = Depends(get_components)):
        ...
    ```
    """
    return CatProjectComponents()


def get_agent() -> CatBanditAgent:
    """Bandit Agent 가져오기"""
    return get_components().agent


def get_calculator() -> CatRewardCalculator:
    """Reward Calculator 가져오기"""
    return get_components().calculator


def get_generator() -> CatEpisodeGenerator:
    """Episode Generator 가져오기"""
    return get_components().generator


def create_youtube_metrics(data: dict) -> YouTubeMetrics:
    """딕셔너리에서 YouTubeMetrics 객체 생성"""
    return YouTubeMetrics(
        views=data.get('views', 0),
        likes=data.get('likes', 0),
        comments=data.get('comments', 0),
        shares=data.get('shares', 0),
        average_view_duration=data.get('average_view_duration', 0),
        video_duration=data.get('video_duration', 24),
        subscribers_gained=data.get('subscribers_gained', 0),
        impressions=data.get('impressions', 0),
        impression_click_through_rate=data.get('ctr', 0)
    )


# 테스트
if __name__ == "__main__":
    print("=== Dependencies 테스트 ===\n")

    components = get_components()
    print(f"Agent: {type(components.agent).__name__}")
    print(f"Calculator: {type(components.calculator).__name__}")
    print(f"Generator: {type(components.generator).__name__}")

    # 싱글톤 확인
    components2 = get_components()
    print(f"\nSingleton check: {components is components2}")
