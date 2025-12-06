"""
RAPO 그래프 데이터 생성 스크립트
GPT로 주제별 place/verb/scenario 조합 생성 → CSV 추가
"""
import os
import csv
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("YTB/.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_PATH = "data/graph_test1.csv"

# 생성할 주제 리스트
TOPICS = [
    "카페에서의 오후",
    "비 오는 날의 창가",
    "놀이공원의 하루",
    "한강 야경",
    "벚꽃 거리 산책",
    "크리스마스 마켓",
    "일본 료칸 여행",
    "뉴욕 타임스퀘어",
    "파리 에펠탑 야경",
    "제주도 해변",
    "북극 오로라",
    "사막의 일몰",
    "열대 우림 탐험",
    "고성 탐방",
    "지하철 출퇴근",
    "옥상 바베큐 파티",
    "캠핑장의 밤",
    "수족관 데이트",
    "미술관 투어",
    "콘서트 현장",
]


def generate_combinations(topic: str, count: int = 3) -> list:
    """GPT로 주제에 맞는 place/verb/scenario 조합 생성"""

    prompt = f"""주제: "{topic}"

이 주제로 영상 프롬프트를 만들기 위한 요소들을 {count}개 생성해줘.

각 조합마다:
- place: 장소/배경 (영어, 2-3개)
- verb: 카메라 동작/행동 (영어, 2-3개)
- scenario: 분위기/느낌 (영어, 2-3개)

JSON 배열로만 답해:
[
  {{"place": ["장소1", "장소2"], "verb": ["동작1", "동작2"], "scenario": ["분위기1", "분위기2"]}},
  ...
]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()
        # JSON 파싱
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        return json.loads(content)

    except Exception as e:
        print(f"[Error] {topic}: {e}")
        return []


def append_to_csv(topic: str, combinations: list):
    """CSV에 데이터 추가"""

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for combo in combinations:
            place = str(combo.get("place", []))
            verb = str(combo.get("verb", []))
            scenario = str(combo.get("scenario", []))

            writer.writerow([topic, place, verb, scenario])


def main():
    print(f"=== RAPO 그래프 데이터 생성 ===")
    print(f"주제 수: {len(TOPICS)}")
    print(f"주제당 조합: 3개")
    print(f"예상 총 행: {len(TOPICS) * 3}개")
    print()

    total = 0
    for i, topic in enumerate(TOPICS):
        print(f"[{i+1}/{len(TOPICS)}] {topic}...", end=" ")

        combinations = generate_combinations(topic, count=3)

        if combinations:
            append_to_csv(topic, combinations)
            print(f"✓ {len(combinations)}개 추가")
            total += len(combinations)
        else:
            print("✗ 실패")

    print()
    print(f"=== 완료 ===")
    print(f"총 {total}개 행 추가됨")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
