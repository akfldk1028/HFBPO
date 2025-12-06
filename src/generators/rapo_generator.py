"""
RAPO Generator - Retrieval + 밴딧 조합 선택 + GPT 문장 다듬기

플로우:
1. Retriever가 토픽에 맞는 후보 조합 검색
2. Bandit이 후보 중 최적 조합 선택 (Thompson Sampling)
3. GPT가 문장으로 다듬기
"""
import os
from openai import OpenAI
from src.core.rl_agent import RapoBanditAgent
from src.rapo.retrieve_modifiers import ModifierRetriever


class RapoGenerator:
    """
    RAPO 기반 프롬프트 생성기

    1. Retriever가 토픽에서 관련 후보 검색
    2. 밴딧이 후보 중 최적 조합 선택
    3. GPT가 문장으로 다듬기

    fixed_topic 모드:
    - 서버 시작 시 고정 토픽으로 ARM(후보 조합)을 미리 생성
    - 매 요청마다 검색하지 않고 고정된 ARM에서 선택
    """

    def __init__(self, graph_dir: str = "data/graph_output", fixed_topic: str = None):
        self.retriever = ModifierRetriever(graph_dir=graph_dir)
        self.bandit = RapoBanditAgent()
        self.api_key = os.getenv("OPENAI_API_KEY")

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            print("[RapoGen] Warning: OPENAI_API_KEY not found")
            self.client = None

        # 고정 토픽 모드
        self.fixed_topic = fixed_topic
        self.fixed_places = []
        self.fixed_verbs = []
        self.fixed_scenarios = []
        self.fixed_candidates = []

        if fixed_topic:
            self._init_fixed_arms(fixed_topic)

    def _init_fixed_arms(self, topic: str):
        """고정 토픽으로 ARM(후보 조합) 미리 생성"""
        print(f"[RapoGen] Initializing fixed ARMs for topic: {topic}")
        retrieved = self.retriever.retrieve(
            topic=topic,
            top_k_places=3,
            top_k_verbs=5,
            top_k_scenarios=5
        )
        self.fixed_places = retrieved["places"]
        self.fixed_verbs = retrieved["verbs"]
        self.fixed_scenarios = retrieved["scenarios"]
        self.fixed_candidates = retrieved["combinations"]
        print(f"[RapoGen] Fixed ARMs initialized: {len(self.fixed_candidates)} combinations")

    def generate(self, topic: str = None, top_k_places: int = 3, top_k_verbs: int = 5, top_k_scenarios: int = 5) -> dict:
        """
        프롬프트 생성

        Args:
            topic: 입력 토픽 (예: "디즈니 공주 신혼방 투어")
                   fixed_topic 모드에서는 생략 가능
            top_k_places: 검색할 place 수
            top_k_verbs: 검색할 verb 수
            top_k_scenarios: 검색할 scenario 수

        Returns:
            {
                "prompt": "완성된 프롬프트 문장",
                "combination_key": "place|verb|scenario",
                "place": "...",
                "verb": "...",
                "scenario": "...",
                "estimated_reward": 0.xxx,
                "candidates_count": 후보 조합 수
            }
        """
        # 토픽 결정 (고정 토픽 모드 지원)
        use_topic = topic or self.fixed_topic
        if not use_topic:
            raise ValueError("topic이 필요합니다 (또는 fixed_topic 모드 사용)")

        # 고정 토픽 모드: 미리 생성된 ARM 사용
        if self.fixed_topic and self.fixed_candidates:
            candidates = self.fixed_candidates
        else:
            # 동적 검색 모드
            retrieved = self.retriever.retrieve(
                topic=use_topic,
                top_k_places=top_k_places,
                top_k_verbs=top_k_verbs,
                top_k_scenarios=top_k_scenarios
            )
            candidates = retrieved["combinations"]

        # 2. 밴딧이 후보 중에서 최적 조합 선택
        selection = self.bandit.select_combination(candidates=candidates)

        place = selection["place"]
        verb = selection["verb"]
        scenario = selection["scenario"]

        # 3. GPT로 문장 다듬기
        prompt = self._refine_with_gpt(use_topic, place, verb, scenario)

        return {
            "prompt": prompt,
            "combination_key": selection["combination_key"],
            "place": place,
            "verb": verb,
            "scenario": scenario,
            "estimated_reward": selection["estimated_reward"],
            "candidates_count": len(candidates)
        }

    def _refine_with_gpt(self, topic: str, place: str, verb: str, scenario: str) -> str:
        """GPT로 조합을 자연스러운 프롬프트로 변환"""

        if not self.client:
            # API 없으면 기본 형식
            return f"A {scenario} scene in {place}, camera {verb}. Topic: {topic}"

        system_prompt = """You are a video prompt expert.
Convert the given elements into a cinematic video prompt.
IMPORTANT: You MUST include the Topic keyword in the prompt.
Keep it under 50 words. Focus on visual details.
Output only the prompt, no explanation."""

        user_prompt = f"""Topic: {topic}
Place: {place}
Camera Action: {verb}
Mood/Scenario: {scenario}

Create a detailed video prompt combining these elements."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[RapoGen] GPT Error: {e}")
            return f"A {scenario} scene in {place}, camera {verb}. Topic: {topic}"

    def update_reward(self, combination_key: str, reward: float):
        """성과 기반 보상 업데이트"""
        self.bandit.update_reward(combination_key, reward)

    def get_top_combinations(self, n: int = 10):
        """성과 좋은 조합 조회"""
        return self.bandit.get_top_combinations(n)


# 테스트
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("YTB/.env")

    gen = RapoGenerator()

    print("=== RAPO Generator 테스트 ===\n")

    topic = "디즈니 공주 신혼방 투어"
    print(f"토픽: {topic}\n")

    for i in range(3):
        result = gen.generate(topic)
        print(f"[{i+1}] 조합: {result['place']} | {result['verb']} | {result['scenario']}")
        print(f"    프롬프트: {result['prompt']}")
        print(f"    예상 보상: {result['estimated_reward']:.3f}")
        print()
