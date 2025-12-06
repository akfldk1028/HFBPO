# HFBPO API 설계 문서

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        N8N (오케스트레이터)                       │
│  - 스케줄링                                                      │
│  - 워크플로우 관리                                                │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│   HFBPO (프롬프트 두뇌)   │     │  ShortVideoMaker (비디오 실행)   │
│                         │     │                                 │
│  - Thompson Sampling    │     │  - VEO3 영상 생성                │
│  - 프롬프트 최적화        │     │  - YouTube 업로드               │
│  - 보상 학습             │     │  - Google Sheets 기록           │
└─────────────────────────┘     └─────────────────────────────────┘
              ▲                               │
              │                               ▼
              │                 ┌─────────────────────────────────┐
              │                 │      Google Sheets (Videos)     │
              │                 │  - combination_key 저장          │
              │                 │  - YouTube Analytics 저장        │
              └─────────────────┴─────────────────────────────────┘
```

## 데이터 플로우

### Phase 1: 프롬프트 생성 + 비디오 생성

```
[1] N8N Schedule (매일 10:00)
         │
         ▼ POST /generate
[2] HFBPO API
         │ topic: "한강 야경"
         │
         ▼ RAPO Pipeline
[3] 프롬프트 생성
         │ - Retrieval: 유사 장소 검색
         │ - Selection: Thompson Sampling
         │ - Generation: GPT 다듬기
         │
         ▼ Response
[4] {
      "prompt": "A breathtaking night...",
      "combination_key": "han_river|zoom|romantic",
      "place": "han_river",
      "verb": "zoom",
      "scenario": "romantic",
      "estimated_reward": 0.72
    }
         │
         ▼ N8N extracts data
[5] POST /api/video/consistent-shorts
         │
         ▼ ShortVideoMaker
[6] 비디오 생성 + YouTube 업로드
         │ metadata: { combination_key: "han_river|zoom|romantic" }
         │
         ▼
[7] Google Sheets에 기록
    - videoId, youtubeUrl, combination_key, ...
```

### Phase 2: 보상 업데이트 (자동 학습)

```
[1] N8N Schedule (매일 18:00)
         │
         ▼ GET /api/sheet/videos?minAge=6h
[2] ShortVideoMaker API
         │ 6시간 이상 지난 비디오 목록
         │
         ▼ Loop each video
[3] GET /api/analytics/video/:videoId
         │ views, likes, watchTime, ...
         │
         ▼ Calculate reward
[4] reward = 0.2*CTR + 0.4*watchTime + 0.2*engagement + 0.2*growth
         │
         ▼ POST /reward
[5] HFBPO API
         │ { combination_key, reward }
         │
         ▼ Thompson Sampling Update
[6] alpha += reward
    beta += (1 - reward)
         │
         ▼
[7] 다음 프롬프트 선택에 반영됨!
```

## API 엔드포인트

### HFBPO API (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | 토픽 → 프롬프트 생성 |
| POST | `/reward` | 단일 보상 업데이트 |
| POST | `/batch-reward` | 다중 보상 업데이트 |
| GET | `/stats` | 상위 조합 통계 |
| POST | `/calculate-reward` | Analytics → reward 변환 |

### 요청/응답 스키마

#### POST /generate

```json
// Request
{
  "topic": "한강 야경"
}

// Response
{
  "prompt": "A breathtaking night at Han River...",
  "combination_key": "han_river|zoom|romantic",
  "place": "han_river",
  "verb": "zoom",
  "scenario": "romantic",
  "estimated_reward": 0.72,
  "candidates_count": 75
}
```

#### POST /reward

```json
// Request
{
  "combination_key": "han_river|zoom|romantic",
  "reward": 0.85
}

// Response
{
  "success": true,
  "message": "보상 0.85 적용됨",
  "new_alpha": 5.85,
  "new_beta": 2.15
}
```

#### POST /batch-reward

```json
// Request
{
  "rewards": [
    { "combination_key": "han_river|zoom|romantic", "reward": 0.85 },
    { "combination_key": "namsan|dolly|dreamy", "reward": 0.62 }
  ]
}

// Response
{
  "success": true,
  "updated_count": 2
}
```

#### POST /calculate-reward

```json
// Request
{
  "views": 1500,
  "likes": 120,
  "comments": 15,
  "shares": 8,
  "average_watch_percentage": 65,
  "subscribers_gained": 5
}

// Response
{
  "reward": 0.73,
  "breakdown": {
    "ctr_score": 0.15,
    "watch_score": 0.26,
    "engagement_score": 0.18,
    "growth_score": 0.14
  }
}
```

## ShortVideoMaker 수정 사항

### 1. consistent-shorts API 수정

```typescript
// 기존 metadata에 combination_key 추가
interface ConsistentShortsRequest {
  scenes: Scene[];
  config: RenderConfig;
  youtubeUpload?: YouTubeUploadConfig;
  metadata?: {
    combination_key?: string;  // HFBPO에서 받은 키
    // ...
  };
}
```

### 2. Google Sheets 컬럼 추가

| 컬럼 | 필드명 | 설명 |
|------|--------|------|
| AF | combinationKey | HFBPO 조합 키 |
| AG | rewardSent | 보상 전송 여부 |
| AH | rewardSentAt | 보상 전송 시간 |

## N8N 워크플로우

### 워크플로우 1: 비디오 생성

```
Schedule Trigger (10:00)
    ↓
Set Variables (토픽 설정)
    ↓
HTTP Request → HFBPO /generate
    ↓
Set Variables (prompt, combination_key 추출)
    ↓
HTTP Request → ShortVideoMaker /api/video/consistent-shorts
    ↓
IF success → Slack 알림
```

### 워크플로우 2: 보상 업데이트

```
Schedule Trigger (18:00)
    ↓
HTTP Request → ShortVideoMaker /api/sheet/videos?rewardSent=false&minAge=6h
    ↓
Loop Items
    ↓
HTTP Request → Analytics API
    ↓
Code Node (reward 계산)
    ↓
HTTP Request → HFBPO /reward
    ↓
HTTP Request → ShortVideoMaker (rewardSent=true 업데이트)
```

## 보상 계산 공식

```
reward = w1 * CTR_score
       + w2 * watch_score
       + w3 * engagement_score
       + w4 * growth_score

where:
  CTR_score = min(ctr / 0.05, 1.0)  # 5% CTR = 만점
  watch_score = average_watch_percentage / 100
  engagement_score = min((likes + comments*5 + shares*10) / views / 0.1, 1.0)
  growth_score = min(subscribers_gained / 10, 1.0)  # 10명 = 만점

weights (기본값):
  w1 = 0.20  # CTR
  w2 = 0.40  # 시청시간 (가장 중요)
  w3 = 0.20  # 참여도
  w4 = 0.20  # 구독자 증가
```

## 구현 우선순위

### Phase 1 (즉시)
- [ ] HFBPO api.py 리팩토링
- [ ] /batch-reward, /calculate-reward 추가
- [ ] ShortVideoMaker에 combination_key 전달 구조

### Phase 2 (다음)
- [ ] Google Sheets 컬럼 추가
- [ ] N8N 워크플로우 생성

### Phase 3 (개선)
- [ ] 보상 가중치 A/B 테스트
- [ ] 대시보드 구축

---

Last Updated: 2025-11-28
● 서버가 응답하지 않네요. 직접 토큰 파일을 업데이트하는 방법으로 진행하겠습니다.

  다른 방법: 브라우저에서 직접 OAuth 인증 후 토큰을 수동으로 업데이트

  1. 이 URL을 브라우저에서 열어주세요:
  https://accounts.google.com/o/oauth2/v2/auth?access_type=offline&scope=https%3A%2F%2Fwww.googleapis.c    
  om%2Fauth%2Fyoutube.upload%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutube%20https%3A%2F%2Fwww.go    
  ogleapis.com%2Fauth%2Fyt-analytics.readonly&prompt=consent&state=ATT&response_type=code&client_id=550    
  996044521-8luac0vqa8sj0jrpa68oi4lgq30k1nqc.apps.googleusercontent.com&redirect_uri=http%3A%2F%2Flocal    
  host%3A3124%2Fapi%2Fyoutube%2Fauth%2Fcallback

  2. CGXR 채널 계정으로 로그인
  3. 권한 승인 후 redirect 에러 페이지가 뜰 겁니다. 그 URL에서 code= 파라미터 값을 복사해주세요.

  그 코드를 주시면 제가 토큰을 교환하고 업데이트하겠습니다.