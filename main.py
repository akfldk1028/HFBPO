"""
HFBPO Main - RAPO 기반 프롬프트 생성 → N8N 전송
"""
import os
import requests
from dotenv import load_dotenv

from src.generators.rapo_generator import RapoGenerator

# .env 로드
load_dotenv("YTB/.env")

# N8N Webhook URL (나중에 수정)
N8N_WEBHOOK_URL = "https://cgxr.app.n8n.cloud/webhook/video-trigger"


def main():
    print("=" * 50)
    print("HFBPO - RAPO 기반 프롬프트 생성기")
    print("=" * 50)

    # RAPO Generator 로드
    generator = RapoGenerator()
    print(f"\n[RAPO] 저장된 학습 조합: {len(generator.bandit.combinations)}개")

    # 토픽 입력
    topic = input("\n토픽 입력: ").strip()
    if not topic:
        topic = "디즈니 공주의 2025년 서울살이"
        print(f"기본 토픽 사용: {topic}")

    # 프롬프트 생성
    print(f"\n프롬프트 생성 중...")
    result = generator.generate(topic=topic)

    print(f"\n{'='*50}")
    print(f"[후보 조합] {result['candidates_count']}개 중 선택")
    print(f"[선택된 조합] {result['place']} | {result['verb']} | {result['scenario']}")
    print(f"[예상 보상] {result['estimated_reward']:.3f}")
    print(f"{'='*50}")
    print(f"\n[프롬프트]\n{result['prompt']}")
    print(f"\n[combination_key]\n{result['combination_key']}")

    # N8N으로 전송
    send = input("\nN8N으로 전송할까요? [y/N]: ").strip().lower()
    if send == "y":
        # 프롬프트 문장만 전송
        payload = {"prompt": result["prompt"]}
        try:
            response = requests.post(
                N8N_WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            print(f"\n전송 완료! Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
        except Exception as e:
            print(f"\n전송 실패: {e}")
    else:
        print("\n전송 취소됨")

    # 수동 보상 테스트 (개발용)
    test_reward = input("\n[테스트] 보상 입력 (0~1, 스킵은 Enter): ").strip()
    if test_reward:
        try:
            reward = float(test_reward)
            generator.update_reward(result["combination_key"], reward)
            print(f"보상 {reward} 적용됨")
        except:
            print("잘못된 입력")


if __name__ == "__main__":
    main()
