"""
RAPO retrieve_modifiers.py
토픽 입력 → 관련 modifier(place, verb, scenario) 검색

역할:
- 토픽 문장을 임베딩
- 유사한 place top-K 검색
- 그래프에서 해당 place의 이웃(verb, scenario) 가져오기
- 각 이웃들 중 토픽과 유사한 것들만 필터링
"""
import os
import json
import numpy as np
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv("YTB/.env")


class GPTEmbedder:
    """OpenAI text-embedding-3-small 사용"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"

    def encode(self, text: str) -> list:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """코사인 유사도 계산"""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


class ModifierRetriever:
    """
    토픽 기반 modifier 검색기

    사용법:
        retriever = ModifierRetriever()
        candidates = retriever.retrieve("디즈니 공주 신혼방 투어")
        # candidates = {
        #     "places": ["princess bedroom", "royal castle", ...],
        #     "verbs": ["pan", "dolly in", ...],
        #     "scenarios": ["romantic", "dreamy", ...]
        # }
    """

    def __init__(self, graph_dir: str = "data/graph_output"):
        self.graph_dir = graph_dir
        self.model = GPTEmbedder()

        # 데이터 로드
        self._load_data()

    def _load_json(self, name: str):
        path = os.path.join(self.graph_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_data(self):
        """그래프 및 임베딩 데이터 로드"""
        print(f"[Retriever] Loading from {self.graph_dir}")

        # 인덱스 매핑
        self.place_to_idx = self._load_json("place_to_idx")
        self.verb_to_idx = self._load_json("verb_to_idx")
        self.scenario_to_idx = self._load_json("scenario_to_idx")

        # 역방향 매핑
        self.idx_to_place = {v: k for k, v in self.place_to_idx.items()}
        self.idx_to_verb = {v: k for k, v in self.verb_to_idx.items()}
        self.idx_to_scenario = {v: k for k, v in self.scenario_to_idx.items()}

        # 임베딩
        self.place_embed = self._load_json("place_embed")
        self.verb_embed = self._load_json("verb_words_embed")
        self.scenario_embed = self._load_json("scenario_words_embed")

        # 그래프
        place_verb_path = os.path.join(self.graph_dir, "graph_place_verb.graphml")
        place_scene_path = os.path.join(self.graph_dir, "graph_place_scene.graphml")

        if os.path.exists(place_verb_path):
            self.G_place_verb = nx.read_graphml(place_verb_path)
        else:
            self.G_place_verb = nx.Graph()

        if os.path.exists(place_scene_path):
            self.G_place_scene = nx.read_graphml(place_scene_path)
        else:
            self.G_place_scene = nx.Graph()

        print(f"[Retriever] Loaded {len(self.place_to_idx)} places, "
              f"{len(self.verb_to_idx)} verbs, {len(self.scenario_to_idx)} scenarios")

    def retrieve(
        self,
        topic: str,
        top_k_places: int = 3,
        top_k_verbs: int = 5,
        top_k_scenarios: int = 5
    ) -> Dict[str, List[str]]:
        """
        토픽에 맞는 modifier 후보 검색

        Args:
            topic: 입력 토픽 (예: "디즈니 공주 신혼방 투어")
            top_k_places: 검색할 place 수
            top_k_verbs: 검색할 verb 수
            top_k_scenarios: 검색할 scenario 수

        Returns:
            {
                "places": [...],
                "verbs": [...],
                "scenarios": [...],
                "combinations": [(place, verb, scenario), ...]
            }
        """
        if not self.place_embed:
            print("[Retriever] No embeddings loaded!")
            return {"places": [], "verbs": [], "scenarios": [], "combinations": []}

        # 1. 토픽 임베딩
        topic_emb = np.array(self.model.encode(topic))

        # 2. 유사한 place top-K 검색
        place_array = np.array(self.place_embed)
        place_sim = cosine_similarity(topic_emb.reshape(1, -1), place_array).flatten()
        top_place_indices = np.argsort(place_sim)[-top_k_places:][::-1].tolist()

        top_places = [self.idx_to_place[idx] for idx in top_place_indices]

        # 3. 각 place의 이웃에서 verb, scenario 수집
        candidate_verbs = set()
        candidate_scenarios = set()

        for place in top_places:
            # Place → Verb 이웃
            if self.G_place_verb.has_node(place):
                verbs = list(self.G_place_verb.neighbors(place))
                candidate_verbs.update(verbs)

            # Place → Scenario 이웃
            if self.G_place_scene.has_node(place):
                scenarios = list(self.G_place_scene.neighbors(place))
                candidate_scenarios.update(scenarios)

        # 4. 후보 verb들 중 토픽과 유사한 것만 필터링
        top_verbs = self._filter_by_similarity(
            list(candidate_verbs),
            self.verb_to_idx,
            self.verb_embed,
            topic_emb,
            top_k_verbs
        )

        # 5. 후보 scenario들 중 토픽과 유사한 것만 필터링
        top_scenarios = self._filter_by_similarity(
            list(candidate_scenarios),
            self.scenario_to_idx,
            self.scenario_embed,
            topic_emb,
            top_k_scenarios
        )

        # 6. 조합 생성
        combinations = []
        for place in top_places:
            for verb in top_verbs:
                for scenario in top_scenarios:
                    combinations.append((place, verb, scenario))

        return {
            "places": top_places,
            "verbs": top_verbs,
            "scenarios": top_scenarios,
            "combinations": combinations
        }

    def _filter_by_similarity(
        self,
        candidates: List[str],
        to_idx: Dict[str, int],
        embeddings: List,
        topic_emb: np.ndarray,
        top_k: int
    ) -> List[str]:
        """후보들 중 토픽과 유사한 것 top-K 선택"""
        if not candidates:
            return []

        # 후보들의 임베딩만 추출
        valid_candidates = []
        valid_embeds = []

        for c in candidates:
            if c in to_idx:
                idx = to_idx[c]
                if idx < len(embeddings):
                    valid_candidates.append(c)
                    valid_embeds.append(embeddings[idx])

        if not valid_embeds:
            return candidates[:top_k]

        # 유사도 계산
        embed_array = np.array(valid_embeds)
        sim = cosine_similarity(topic_emb.reshape(1, -1), embed_array).flatten()

        # Top-K 선택
        k = min(top_k, len(valid_candidates))
        top_indices = np.argsort(sim)[-k:][::-1].tolist()

        return [valid_candidates[i] for i in top_indices]


# 테스트
if __name__ == "__main__":
    retriever = ModifierRetriever()

    topics = [
        "디즈니 공주 신혼방 투어",
        "한강 야경",
        "카페에서의 오후"
    ]

    for topic in topics:
        print(f"\n{'='*50}")
        print(f"토픽: {topic}")
        print('='*50)

        result = retriever.retrieve(topic)

        print(f"\nPlaces ({len(result['places'])}개):")
        for p in result['places']:
            print(f"  - {p}")

        print(f"\nVerbs ({len(result['verbs'])}개):")
        for v in result['verbs']:
            print(f"  - {v}")

        print(f"\nScenarios ({len(result['scenarios'])}개):")
        for s in result['scenarios']:
            print(f"  - {s}")

        print(f"\n총 조합 수: {len(result['combinations'])}개")
