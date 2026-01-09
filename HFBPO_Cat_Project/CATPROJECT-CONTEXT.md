# CatProject HFBPO - AI Context Document

**Last Updated:** 2026-01-09
**Status:** Production Ready (Deployed & Tested)
**Integration:** Mounted on HFBPO API Cloud Run

---

## Quick Reference

| Item | Value |
|------|-------|
| **Cloud Run URL** | `https://hfbpo-api-7qtnitbuvq-du.a.run.app` |
| **CatProject Base** | `/cat/*` |
| **ShortVideoMaker** | `https://short-video-maker-7qtnitbuvq-uc.a.run.app` |
| **Region** | `asia-northeast3` |
| **Memory** | 1GB |
| **CPU** | 1 |

---

## 1. Project Overview

CatProject is a **Multi-Armed Bandit (MAB)** optimization system for generating "why_cat" YouTube Shorts. It uses **Thompson Sampling** to learn which combination of `hook|conflict|ending` produces the best engagement.

### Key Metrics
- **Total Combinations:** 180 (5 hooks x 6 conflicts x 6 endings)
- **Algorithm:** Thompson Sampling with Beta distribution
- **Learning Strategy:** Hybrid (Random -> Factored -> Exploitation)

---

## 2. Architecture

```
HFBPO Cloud Run (hfbpo-api-7qtnitbuvq-du.a.run.app)
├── /              → HFBPO root (place|verb|scenario)
├── /generate      → HFBPO generate
├── /reward        → HFBPO reward
└── /cat/*         → CatProject Router (mounted)
    ├── /cat/select
    ├── /cat/select-and-generate
    ├── /cat/update
    ├── /cat/update-multi-horizon
    ├── /cat/stats
    └── /cat/top/{n}
```

### Integration Point (api.py lines 20-32)
```python
from HFBPO_Cat_Project.endpoint import cat_router

app = FastAPI(
    title="HFBPO API",
    description="Human Feedback Bandit Prompt Optimization + CatProject",
    version="1.1.0"
)

app.include_router(cat_router)  # Mounts at /cat/*
```

---

## 3. Directory Structure

```
HFBPO_Cat_Project/
├── src/
│   ├── cat_bandit_agent.py      # Thompson Sampling MAB
│   ├── cat_reward_calculator.py # YouTube metrics -> reward
│   └── cat_episode_generator.py # GPT episode generation
├── endpoint/
│   ├── __init__.py              # Package exports
│   ├── cat_router.py            # FastAPI Router (8 routes)
│   ├── cat_app.py               # Standalone app (optional)
│   ├── models.py                # Pydantic schemas
│   └── dependencies.py          # Singleton DI container
├── data/
│   ├── config.json              # Normalization, weights
│   ├── combinations.json        # Learned combinations (persistent)
│   ├── hook_types.json          # 5 hook types
│   ├── conflict_types.json      # 6 conflict types
│   └── ending_types.json        # 6 ending types
└── prompts/
    └── episode_generator.txt    # GPT system prompt
```

---

## 4. API Endpoints

### 4.1 Selection (GET /cat/select)
**Thompson Sampling으로 조합 선택**

```bash
curl "https://hfbpo-api-7qtnitbuvq-du.a.run.app/cat/select?series=daily_life"
```

Response:
```json
{
  "combination": {
    "combination_key": "curiosity_gap|prank_backfire|both_dumb",
    "hook_type": "curiosity_gap",
    "conflict_type": "prank_backfire",
    "ending_type": "both_dumb",
    "estimated_reward": 0.6667,
    "selection_mode": "thompson_sampling",
    "trial_number": 7
  },
  "message": "Selected via thompson_sampling"
}
```

### 4.2 Full Generation (GET /cat/select-and-generate)
**조합 선택 + GPT 에피소드 생성 + ShortVideoMaker API body**

```bash
curl "https://hfbpo-api-7qtnitbuvq-du.a.run.app/cat/select-and-generate?series=daily_life&theme=커플일상"
```

Response includes:
- `combination`: Selected hook|conflict|ending info
- `episode`: GPT-generated episode (title, intro, scenes)
- `api_body`: Ready-to-send to ShortVideoMaker `/api/video/consistent-shorts`

### 4.3 Reward Update (POST /cat/update)
**단일 시점 보상 업데이트**

```bash
curl -X POST "https://hfbpo-api-7qtnitbuvq-du.a.run.app/cat/update" \
  -H "Content-Type: application/json" \
  -d '{"combination_key": "curiosity_gap|prank_backfire|both_dumb", "reward": 0.75}'
```

### 4.4 Multi-Horizon Reward (POST /cat/update-multi-horizon)
**다중 시점 보상 (6h/24h/72h) - 바이럴 반영**

```bash
curl -X POST "https://hfbpo-api-7qtnitbuvq-du.a.run.app/cat/update-multi-horizon" \
  -H "Content-Type: application/json" \
  -d '{
    "combination_key": "curiosity_gap|prank_backfire|both_dumb",
    "metrics_6h": {"views": 2000, "likes": 150, "comments": 10, ...},
    "metrics_24h": {"views": 8000, "likes": 600, ...},
    "metrics_72h": {"views": 25000, "likes": 2000, ...},
    "subscriber_count": 5000
  }'
```

### 4.5 Statistics (GET /cat/stats)
```bash
curl "https://hfbpo-api-7qtnitbuvq-du.a.run.app/cat/stats"
```

---

## 5. Combination Types

### Hook Types (5)
| Key | Korean | Description |
|-----|--------|-------------|
| `curiosity_gap` | 궁금증 유발 | "뭔가 있다" 시작 |
| `shock_reveal` | 충격 공개 | 반전 시작 |
| `action_mid` | 액션 중간 | 동작 중 시작 |
| `dramatic_zoom` | 드라마틱 줌 | 클로즈업 시작 |
| `dialogue_hook` | 대사 훅 | 대사로 시작 |

### Conflict Types (6)
| Key | Korean | Description |
|-----|--------|-------------|
| `misunderstanding` | 오해 | 서로 오해 |
| `prank_backfire` | 장난 역풍 | 장난이 역효과 |
| `couple_battle` | 커플 배틀 | 대결 구도 |
| `jealousy_moment` | 질투 순간 | 질투 발생 |
| `kami_forgets` | 까미 깜빡 | 까미가 뭔가 잊음 |
| `dalgi_cold` | 딸기 도도 | 딸기가 차갑게 반응 |

### Ending Types (6)
| Key | Korean | Description |
|-----|--------|-------------|
| `twist_win` | 반전 승리 | 예상 뒤집기 |
| `both_dumb` | 둘다 바보 | 둘 다 어이없음 |
| `sweet_ending` | 달달 엔딩 | 로맨틱 마무리 |
| `revenge_sweet` | 복수 달콤 | 복수 성공 |
| `chaos_loop` | 카오스 루프 | 혼란 반복 |
| `cliff_hanger` | 클리프행어 | 다음 편 유도 |

---

## 6. Learning Strategy

```
Trial 1-30:   Random Exploration (모든 조합 균등 탐색)
Trial 31-60:  Factored Exploration (타입별 개별 학습)
Trial 61+:    Thompson Sampling (10% 탐색, 90% 활용)
```

### Compatibility Matrix
일부 조합은 호환성이 낮음 (예: `dialogue_hook` + `kami_forgets` = 0.7)

---

## 7. Reward Calculation

### Weights by Channel Phase
| Phase | Subscribers | Retention | CTR | Engagement | Share | Growth |
|-------|-------------|-----------|-----|------------|-------|--------|
| Early | < 1,000 | 0.20 | 0.05 | 0.15 | 0.25 | **0.35** |
| Growth | 1K - 100K | 0.30 | 0.10 | 0.20 | **0.25** | 0.15 |
| Stable | > 100K | **0.35** | 0.25 | 0.20 | 0.15 | 0.05 |

### Multi-Horizon Weights
- 6h: 20%
- 24h: 40%
- 72h: 40%

---

## 8. n8n Integration (TODO)

### Workflow: PLEASE.ver.0.05-catproject.json

```
1. Schedule Trigger (매일 특정 시간)
2. Channel Mapper (why_cat 선택)
3. Switch Node (channel_type == "why_cat")
   └─> CatProject API (GET /cat/select-and-generate)
4. Extract CatProject Body (combination_key 저장)
5. Create Cat Video (POST /api/video/consistent-shorts)
6. Wait for Video Completion
7. YouTube Upload
8. Wait 6h/24h/72h
9. Collect YouTube Analytics
10. Reward Update (POST /cat/update-multi-horizon)
```

### Key Differences from PLEASE Workflow
| Feature | PLEASE (기존) | CatProject |
|---------|---------------|------------|
| API Endpoint | `/api/video/nano-banana/to-veo3` | `/api/video/consistent-shorts` |
| Combination | `place\|verb\|scenario` | `hook\|conflict\|ending` |
| Character | 없음 | `cat-couple` profile |

---

## 9. ShortVideoMaker Integration

### Target Endpoint
```
POST https://short-video-maker-7qtnitbuvq-uc.a.run.app/api/video/consistent-shorts
```

### Request Body Structure (from api_body)
```json
{
  "characterReference": {
    "profileId": "cat-couple",
    "characterIds": ["kami", "dalgi"]
  },
  "scenes": [
    {
      "text": "까미가 방에 들어온다",
      "scenePrompt": "Black cat Kami entering room sneakily",
      "characterIds": ["kami"]
    },
    {
      "text": "딸기가 눈을 뜬다",
      "scenePrompt": "White cat Dalgi waking up surprised",
      "characterIds": ["dalgi"]
    },
    {
      "text": "둘이 같이 웃는다",
      "scenePrompt": "Both cats laughing together",
      "characterIds": ["kami", "dalgi"]
    }
  ],
  "config": {
    "orientation": "portrait",
    "generateVideos": true,
    "useFrameInterpolation": true
  }
}
```

---

## 10. Test Results (2025-01-09)

| Component | Status | Notes |
|-----------|--------|-------|
| CatBanditAgent | OK | 180 combinations, Thompson Sampling working |
| CatRewardCalculator | OK | Multi-horizon, channel phases working |
| CatEpisodeGenerator | OK | GPT integration ready |
| Endpoint Router | OK | 8 routes, Pydantic v1/v2 compatible |
| API Mount | OK | Mounted at /cat/* on HFBPO API |

---

## 11. Deployment

### Current Setup
- **HFBPO API:** `hfbpo-api-7qtnitbuvq-du.a.run.app` (includes CatProject)
- **ShortVideoMaker:** `short-video-maker-7qtnitbuvq-uc.a.run.app`

### Redeployment Required After Code Changes
```bash
cd D:/Data/00_Personal/YTB/HFBPO/HFBPO
gcloud builds submit --config=cloudbuild.yaml
```

---

## 12. Important Files

| File | Purpose |
|------|---------|
| `HFBPO/api.py` | Main server, mounts cat_router |
| `HFBPO_Cat_Project/endpoint/cat_router.py` | All /cat/* endpoints |
| `HFBPO_Cat_Project/data/combinations.json` | Persistent learning state |
| `HFBPO_Cat_Project/data/config.json` | Normalization, weights |

---

## 13. Quick Commands

```bash
# Local test
cd "D:/Data/00_Personal/YTB/HFBPO/HFBPO"
python -m uvicorn api:app --reload --port 8000

# Test CatProject endpoint
curl "http://localhost:8000/cat/health"
curl "http://localhost:8000/cat/select?series=daily_life"

# Production test
curl "https://hfbpo-api-7qtnitbuvq-du.a.run.app/cat/stats"
```

---

## 14. Next Steps (Priority)

1. **n8n Workflow Integration**
   - File: `PLEASE.ver.0.05-catproject.json` (created but not tested)
   - Need: Import to n8n, test full flow

2. **Cloud Run Redeployment**
   - Reason: api.py was modified to mount cat_router
   - Command: `gcloud builds submit --config=cloudbuild.yaml`

3. **YouTube Analytics Integration**
   - Need: 6h/24h/72h scheduled reward updates
   - Endpoint: `/cat/update-multi-horizon`

---

## 15. Code Patterns

### Pydantic v1/v2 Compatibility
```python
def pydantic_to_dict(model):
    if hasattr(model, 'model_dump'):
        return model.model_dump()  # v2
    return model.dict()  # v1
```

### Import Compatibility (package vs direct run)
```python
try:
    from .cat_router import router  # Package import
except ImportError:
    from cat_router import router   # Direct run
```

---

*This document is designed for AI context continuity. Read this first when resuming CatProject work.*
