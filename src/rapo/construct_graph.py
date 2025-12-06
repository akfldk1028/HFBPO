"""
RAPO construct_graph.py
CSV에서 그래프 + 임베딩 생성 (한 번만 실행)

생성되는 파일:
- place_to_idx.json: place → index 매핑
- verb_to_idx.json: verb → index 매핑
- scenario_to_idx.json: scenario → index 매핑
- place_embed.json: place 임베딩 벡터
- verb_words_embed.json: verb 임베딩 벡터
- scenario_words_embed.json: scenario 임베딩 벡터
- place_in_sentence.json: place가 어떤 문장에 등장했는지
- verb_in_sentence.json: verb가 어떤 문장에 등장했는지
- scenario_in_sentence.json: scenario가 어떤 문장에 등장했는지
- graph_place_verb.graphml: place-verb 관계 그래프
- graph_place_scene.graphml: place-scenario 관계 그래프
- data_info.json: 통계 정보
"""
import os
import json
import ast
import pandas as pd
import networkx as nx
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

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


def construct_graph(
    csv_path: str = "data/graph_test1.csv",
    output_dir: str = "data/graph_output"
):
    """
    CSV에서 그래프와 임베딩 생성

    Args:
        csv_path: 입력 CSV 파일 경로
        output_dir: 출력 디렉토리
    """
    print(f"=== RAPO Graph Construction ===")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")

    # GPT 임베딩 모델 로드
    model = GPTEmbedder()
    print(f"Using OpenAI text-embedding-3-small")

    # 데이터 구조 초기화
    place_to_idx, verb_to_idx, scenario_to_idx = {}, {}, {}
    place_in_sentence = defaultdict(list)
    verb_in_sentence = defaultdict(list)
    scenario_in_sentence = defaultdict(list)
    place_embed, verb_words_embed, scenario_words_embed = [], [], []
    place_cache, verb_cache, scenario_cache = {}, {}, {}

    # 그래프 초기화
    G_place_verb = nx.Graph()
    G_place_scene = nx.Graph()

    # CSV 읽기
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")

    p_idx = v_idx = s_idx = 0
    valid_cnt = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        topic = row.get('Input', '')

        # 리스트 파싱
        try:
            places = ast.literal_eval(row.get('place', '[]'))
            verbs = ast.literal_eval(row.get('verb_obj_word', '[]'))
            scenarios = ast.literal_eval(row.get('scenario_word', '[]'))
        except (ValueError, SyntaxError):
            continue

        # 빈 리스트 처리
        places = [p for p in places if p and p.strip()]
        verbs = [v for v in verbs if v and v.strip()]
        scenarios = [s for s in scenarios if s and s.strip()]

        if not places or not verbs or not scenarios:
            continue

        # 각 place에 대해 처리
        for place in places:
            # Place 임베딩
            if place not in place_cache:
                emb = model.encode(place)
                place_cache[place] = emb  # 이미 list임
                place_to_idx[place] = p_idx
                place_embed.append(place_cache[place])
                p_idx += 1
            place_in_sentence[place].append(valid_cnt)

            # Verb 처리 및 그래프 엣지
            for verb in verbs:
                if verb not in verb_cache:
                    emb = model.encode(verb)
                    verb_cache[verb] = emb  # 이미 list임
                    verb_to_idx[verb] = v_idx
                    verb_words_embed.append(verb_cache[verb])
                    v_idx += 1
                verb_in_sentence[verb].append(valid_cnt)
                G_place_verb.add_edge(place, verb)

            # Scenario 처리 및 그래프 엣지
            for scenario in scenarios:
                if scenario not in scenario_cache:
                    emb = model.encode(scenario)
                    scenario_cache[scenario] = emb  # 이미 list임
                    scenario_to_idx[scenario] = s_idx
                    scenario_words_embed.append(scenario_cache[scenario])
                    s_idx += 1
                scenario_in_sentence[scenario].append(valid_cnt)
                G_place_scene.add_edge(place, scenario)

        valid_cnt += 1

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # JSON 저장 함수
    def save_json(data, name):
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {name}.json")

    # 데이터 저장
    save_json(place_to_idx, "place_to_idx")
    save_json(verb_to_idx, "verb_to_idx")
    save_json(scenario_to_idx, "scenario_to_idx")
    save_json(dict(place_in_sentence), "place_in_sentence")
    save_json(dict(verb_in_sentence), "verb_in_sentence")
    save_json(dict(scenario_in_sentence), "scenario_in_sentence")
    save_json(place_embed, "place_embed")
    save_json(verb_words_embed, "verb_words_embed")
    save_json(scenario_words_embed, "scenario_words_embed")

    # 통계 정보
    data_info = {
        "valid_sentences": valid_cnt,
        "num_places": p_idx,
        "num_verbs": v_idx,
        "num_scenarios": s_idx,
        "num_place_verb_edges": G_place_verb.number_of_edges(),
        "num_place_scene_edges": G_place_scene.number_of_edges()
    }
    save_json(data_info, "data_info")

    # 그래프 저장
    nx.write_graphml(G_place_verb, os.path.join(output_dir, "graph_place_verb.graphml"))
    nx.write_graphml(G_place_scene, os.path.join(output_dir, "graph_place_scene.graphml"))
    print("Saved: graph_place_verb.graphml")
    print("Saved: graph_place_scene.graphml")

    print(f"\n=== 완료 ===")
    print(f"Places: {p_idx}")
    print(f"Verbs: {v_idx}")
    print(f"Scenarios: {s_idx}")
    print(f"Valid sentences: {valid_cnt}")


if __name__ == "__main__":
    construct_graph()
