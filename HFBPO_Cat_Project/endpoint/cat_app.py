"""
CatProject Standalone API
독립 실행 가능한 FastAPI 앱

Usage:
    # 개발 서버 (패키지로 실행)
    cd HFBPO_Cat_Project
    uvicorn endpoint.cat_app:app --reload --port 8001

    # 직접 실행
    cd HFBPO_Cat_Project/endpoint
    python cat_app.py

    # Cloud Run (PORT 환경변수 자동 사용)
    uvicorn endpoint.cat_app:app --host 0.0.0.0
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 프로젝트 루트를 path에 추가 (직접 실행 지원)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT_DIR = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ENDPOINT_DIR not in sys.path:
    sys.path.insert(0, ENDPOINT_DIR)

# Import 호환성 처리 (패키지 vs 직접 실행)
try:
    # 패키지로 import될 때 (uvicorn endpoint.cat_app:app)
    from .cat_router import router as cat_router
    from .dependencies import get_components
except ImportError:
    # 직접 실행될 때 (python cat_app.py)
    from cat_router import router as cat_router
    from dependencies import get_components


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # Startup
    print("=" * 50)
    print("CatProject API Starting...")
    print("=" * 50)

    # 컴포넌트 초기화 (싱글톤)
    components = get_components()
    print(f"\nReady! Trials: {components.agent.total_trials}")

    yield

    # Shutdown
    print("\nCatProject API Shutting down...")


# FastAPI 앱 생성
app = FastAPI(
    title="CatProject HFBPO API",
    description="""
## CatProject - 고양이 커플 숏츠 MAB 최적화

Multi-Armed Bandit (Thompson Sampling) 기반으로
hook|conflict|ending 조합을 자동 최적화합니다.

### 주요 엔드포인트

- **GET /cat/select** - 조합 선택 (Thompson Sampling)
- **GET /cat/select-and-generate** - 조합 선택 + 에피소드 생성
- **POST /cat/update** - 보상 업데이트
- **POST /cat/update-multi-horizon** - 다중 시점 보상 업데이트
- **GET /cat/stats** - 학습 상태 조회

### n8n 연동

1. `/cat/select-and-generate` 호출 → 에피소드 생성
2. ShortVideoMaker로 영상 제작
3. YouTube 업로드
4. 6h/24h/72h 후 `/cat/update-multi-horizon` 호출
""",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router 등록
app.include_router(cat_router)


# 루트 엔드포인트
@app.get("/")
def root():
    """API 루트"""
    return {
        "project": "CatProject HFBPO",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "select": "GET /cat/select",
            "select_and_generate": "GET /cat/select-and-generate",
            "generate": "POST /cat/generate",
            "update": "POST /cat/update",
            "update_multi_horizon": "POST /cat/update-multi-horizon",
            "stats": "GET /cat/stats",
            "top_n": "GET /cat/top/{n}",
            "health": "GET /cat/health"
        }
    }


# 직접 실행
if __name__ == "__main__":
    import uvicorn

    # Cloud Run 호환: PORT 환경변수 사용
    port = int(os.environ.get("PORT", 8001))

    print("\n" + "=" * 50)
    print("Starting CatProject API Server")
    print("=" * 50)
    print(f"\nDocs: http://localhost:{port}/docs")
    print("=" * 50 + "\n")

    uvicorn.run(
        app,  # 직접 실행 시 app 객체 전달
        host="0.0.0.0",
        port=port,
        reload=False  # 직접 실행 시 reload 비활성화
    )
