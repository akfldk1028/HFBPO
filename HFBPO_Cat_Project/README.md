# HFBPO CatProject

**Multi-Armed Bandit 기반 고양이 커플 숏츠 콘텐츠 최적화 시스템**

## 개요

CatProject는 Thompson Sampling 기반 MAB(Multi-Armed Bandit) 알고리즘을 사용하여 YouTube Shorts 콘텐츠의 Hook|Conflict|Ending 조합을 자동으로 최적화합니다.

### 핵심 기능
- **180개 조합 자동 탐색**: 5 hooks × 6 conflicts × 6 endings
- **정규화된 보상 계산**: Min-Max 정규화로 공정한 메트릭 비교
- **다중 시점 보상**: 6h/24h/72h 가중 평균 (바이럴 잠재력 반영)
- **채널 단계별 최적화**: 초기/성장/안정 단계별 가중치 자동 조정
- **GPT 기반 에피소드 생성**: Seed 제어로 일관된 출력

## 프로젝트 구조

```
HFBPO_Cat_Project/
├── data/
│   ├── config.json              # 전체 설정
│   ├── hook_type.json           # 5가지 Hook 타입
│   ├── conflict_type.json       # 6가지 Conflict 타입
│   ├── ending_type.json         # 6가지 Ending 타입
│   ├── compatibility_matrix.json # 조합 호환성 (어색한 조합 필터링)
│   └── bandit_state.json        # MAB 학습 상태 저장
├── prompts/
│   ├── system_message.md        # GPT 시스템 프롬프트
│   └── user_message_template.md # 에피소드 생성 템플릿
├── src/
│   ├── __init__.py
│   ├── cat_bandit_agent.py      # Thompson Sampling MAB
│   ├── cat_reward_calculator.py # 보상 계산기
│   └── cat_episode_generator.py # GPT 에피소드 생성기
└── README.md
```

## 설치 및 사용

### 환경 설정

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key"

# 의존성 설치
pip install openai numpy
```

### 기본 사용법

```python
from src import CatBanditAgent, CatRewardCalculator, CatEpisodeGenerator

# 1. MAB 에이전트로 조합 선택
agent = CatBanditAgent()
combination = agent.select_combination(series="daily_life")
print(f"Selected: {combination['combination_key']}")

# 2. GPT로 에피소드 생성
generator = CatEpisodeGenerator()
episode = generator.generate_episode(combination, theme="커플 일상")

# 3. ShortVideoMaker API 형식으로 변환
api_body = generator.format_for_api(episode)

# 4. 영상 제작 및 YouTube 업로드 후 메트릭 수집
# ... (영상 제작 프로세스)

# 5. 보상 계산 및 업데이트
calculator = CatRewardCalculator()
reward_result = calculator.calculate_multi_horizon_reward(
    metrics_6h, metrics_24h, metrics_72h,
    subscriber_count=5000
)

agent.update_reward(
    combination['combination_key'],
    reward_result['final_reward']
)
agent.save_state()
```

## 알고리즘 설계

### Hybrid Exploration Strategy

```
Trial 0-30:   완전 랜덤 탐색 (Cold Start 해결)
Trial 31-60:  Factored Exploration (Hook → Conflict → Ending 순차 탐색)
Trial 61+:    Thompson Sampling + 10% ε-greedy (Exploitation + 지속 탐색)
```

### 보상 정규화

각 메트릭을 0~1 범위로 정규화하여 공정한 가중 합계 계산:

| 메트릭 | Min | Max | 설명 |
|--------|-----|-----|------|
| retention_rate | 0.30 | 0.90 | 시청 유지율 |
| ctr | 0.02 | 0.15 | 클릭률 |
| engagement_rate | 0.03 | 0.15 | 참여율 (좋아요+댓글/조회수) |
| share_rate | 0.001 | 0.02 | 공유율 |
| subscribers_gained | 0 | 50 | 구독자 증가 |

### 채널 단계별 가중치

| 단계 | 구독자 | Retention | CTR | Engagement | Share | Growth |
|------|--------|-----------|-----|------------|-------|--------|
| 초기 | <1K | 0.20 | 0.05 | 0.15 | 0.25 | **0.35** |
| 성장 | 1K-100K | 0.30 | 0.10 | 0.20 | **0.25** | 0.15 |
| 안정 | >100K | **0.35** | **0.25** | 0.20 | 0.15 | 0.05 |

### 다중 시점 보상

바이럴 영상은 72시간 후 폭발하는 경향이 있어 단일 시점 측정의 한계 극복:

```
Final Reward = 0.20 × R(6h) + 0.40 × R(24h) + 0.40 × R(72h)
```

## 캐릭터 설정

### Kami (까미) 🐱⬛
- 검은 고양이
- 성격: 덜렁이, 깜빡쟁이, 순수함
- 역할: 문제 발생의 주체

### Dalgi (딸기) 🐱🍑
- 주황/복숭아색 고양이
- 성격: 현실적, 잔소리꾼, 다정함
- 역할: 현실적 리액션

## 조합 타입

### Hook Types (5개)
| Type | 설명 |
|------|------|
| shock_reveal | 충격적 표정/상황으로 시작 |
| question_hook | 궁금증 유발 질문 |
| time_pressure | 긴박한 시간 제한 상황 |
| contrast | 기대와 다른 반전 상황 |
| emotional_bait | 감정적 공감 유발 |

### Conflict Types (6개)
| Type | Kami 역할 | Dalgi 역할 |
|------|-----------|------------|
| kami_forgets | 실수 주인공 | 한숨 |
| dalgi_misunderstands | 억울함 | 오해 |
| both_confused | 혼란 | 혼란 |
| external_crisis | 문제 발견자 | 해결사 |
| jealousy | 질투 유발 | 질투 |
| communication_fail | 설명 실패 | 이해 실패 |

### Ending Types (6개)
| Type | 설명 |
|------|------|
| both_dumb | 둘 다 바보같은 상황 |
| sweet_resolution | 달콤한 화해 |
| plot_twist | 예상 못한 반전 |
| cliffhanger | 다음 편 기대감 |
| role_reversal | 역할 역전 |
| wholesome_moment | 훈훈한 순간 |

## 호환성 매트릭스

어색한 조합을 필터링하여 콘텐츠 품질 보장:

```json
{
  "shock_reveal": {
    "kami_forgets": 0.9,      // 좋은 조합
    "dalgi_misunderstands": 0.7,
    ...
  }
}
```

- 점수 < 0.5: 제외 권장
- 점수 0.5-0.7: 주의해서 사용
- 점수 > 0.7: 좋은 조합

## API 통합

### ShortVideoMaker API 형식

```python
api_body = generator.format_for_api(episode)
# Returns:
{
  "characterReference": {
    "profileId": "cat-couple",
    "characterIds": ["kami", "dalgi"]
  },
  "titleText": {...},
  "scenes": [...],
  "config": {
    "orientation": "portrait",
    "generateVideos": True,
    ...
  },
  "hfbpo": {
    "combinationKey": "shock_reveal|kami_forgets|both_dumb",
    "estimatedReward": 0.72
  }
}
```

## 수렴 예측

- **Naive 접근**: 180 조합 × 50회/조합 = 9,000 영상 (2.5년)
- **Hybrid + 호환성 필터링**: ~150 영상으로 상위 10개 조합 수렴 (3-4개월)

## 라이선스

InkMilk 내부 프로젝트
