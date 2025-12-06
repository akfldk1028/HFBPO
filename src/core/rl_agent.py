"""
RAPO 조합 기반 Multi-Armed Bandit Agent
place + verb + scenario 조합별로 가중치 학습

변경사항:
- 더 이상 CSV에서 전체 조합을 로드하지 않음
- Retriever가 제공하는 후보 조합 중에서만 선택
- 새로운 조합은 동적으로 추가됨
"""
import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple


class RapoBanditAgent:
    """
    RAPO 그래프 조합 기반 Thompson Sampling Agent

    - Retriever가 제공하는 후보 조합 중에서 선택
    - 각 조합별 alpha/beta 관리 (동적 추가)
    - 성과 기반 가중치 업데이트
    """

    def __init__(self, state_file: str = "rapo_bandit_state.json"):
        self.state_file = state_file

        # 조합별 가중치: {"place|verb|scenario": {"alpha": 1, "beta": 1}}
        self.combinations: Dict[str, Dict[str, float]] = {}

        # 저장된 상태 로드
        self.load_state()

    def _make_key(self, place: str, verb: str, scenario: str) -> str:
        """조합 키 생성"""
        return f"{place}|{verb}|{scenario}"

    def _parse_key(self, key: str) -> Tuple[str, str, str]:
        """키에서 조합 추출"""
        parts = key.split('|')
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return "", "", ""

    def select_combination(self, candidates: List[Tuple[str, str, str]] = None) -> Dict[str, Any]:
        """
        Thompson Sampling으로 조합 선택

        Args:
            candidates: Retriever가 제공한 (place, verb, scenario) 튜플 리스트
                       None이면 저장된 모든 조합에서 선택 (테스트용)

        Returns:
            선택된 조합 정보
        """
        # 후보가 없으면 기본값
        if not candidates and not self.combinations:
            return {
                "combination_key": None,
                "place": "default place",
                "verb": "default verb",
                "scenario": "default scenario",
                "estimated_reward": 0.5
            }

        # 후보 조합들의 키 생성 및 등록
        if candidates:
            keys = []
            for place, verb, scenario in candidates:
                key = self._make_key(place, verb, scenario)
                keys.append(key)
                # 새로운 조합이면 초기화
                if key not in self.combinations:
                    self.combinations[key] = {"alpha": 1.0, "beta": 1.0}
        else:
            keys = list(self.combinations.keys())

        # 각 조합에서 Thompson Sampling
        sampled_values = []
        for key in keys:
            alpha = self.combinations[key]["alpha"]
            beta = self.combinations[key]["beta"]
            sampled = np.random.beta(alpha, beta)
            sampled_values.append(sampled)

        # 가장 높은 값 선택
        best_idx = np.argmax(sampled_values)
        best_key = keys[best_idx]
        place, verb, scenario = self._parse_key(best_key)

        return {
            "combination_key": best_key,
            "place": place,
            "verb": verb,
            "scenario": scenario,
            "estimated_reward": float(sampled_values[best_idx])
        }

    def update_reward(self, combination_key: str, reward: float):
        """조합에 보상 업데이트"""
        reward = max(0.0, min(1.0, reward))

        if combination_key not in self.combinations:
            print(f"[RapoBandit] Unknown combination: {combination_key}")
            return

        self.combinations[combination_key]["alpha"] += reward
        self.combinations[combination_key]["beta"] += (1.0 - reward)

        self.save_state()

        alpha = self.combinations[combination_key]["alpha"]
        beta = self.combinations[combination_key]["beta"]
        print(f"[RapoBandit] Updated '{combination_key}': alpha={alpha:.2f}, beta={beta:.2f}")

    def get_top_combinations(self, n: int = 10) -> List[Dict[str, Any]]:
        """성과 좋은 조합 Top N"""
        results = []
        for key, vals in self.combinations.items():
            mean = vals["alpha"] / (vals["alpha"] + vals["beta"])
            place, verb, scenario = self._parse_key(key)
            results.append({
                "key": key,
                "place": place,
                "verb": verb,
                "scenario": scenario,
                "mean_reward": mean,
                "alpha": vals["alpha"],
                "beta": vals["beta"]
            })

        results.sort(key=lambda x: x["mean_reward"], reverse=True)
        return results[:n]

    def save_state(self):
        """상태 저장"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.combinations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[RapoBandit] Error saving state: {e}")

    def load_state(self):
        """저장된 상태 로드"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    saved = json.load(f)

                    # 저장된 가중치 적용
                    for key, vals in saved.items():
                        if key in self.combinations:
                            self.combinations[key] = vals
                        else:
                            # 새 조합이면 추가
                            self.combinations[key] = vals

                    print(f"[RapoBandit] Loaded state from {self.state_file}")

            except Exception as e:
                print(f"[RapoBandit] Error loading state: {e}")


# 테스트
if __name__ == "__main__":
    from src.rapo.retrieve_modifiers import ModifierRetriever

    # Retriever로 후보 가져오기
    retriever = ModifierRetriever()
    topic = "디즈니 공주 신혼방 투어"
    result = retriever.retrieve(topic)

    print(f"토픽: {topic}")
    print(f"후보 조합 수: {len(result['combinations'])}개")

    # Bandit 테스트
    agent = RapoBanditAgent()
    print(f"\n저장된 조합 수: {len(agent.combinations)}")

    print("\n=== 조합 선택 테스트 (후보 중에서) ===")
    for i in range(3):
        selection = agent.select_combination(candidates=result['combinations'])
        print(f"{i+1}. {selection['place']} | {selection['verb']} | {selection['scenario']}")
        print(f"   예상 보상: {selection['estimated_reward']:.3f}")

    print("\n=== 보상 업데이트 테스트 ===")
    selection = agent.select_combination(candidates=result['combinations'])
    agent.update_reward(selection['combination_key'], 0.8)

    print("\n=== Top 5 조합 ===")
    for item in agent.get_top_combinations(5):
        print(f"- {item['place']} | {item['verb']} | {item['scenario']}")
        print(f"  mean={item['mean_reward']:.3f}, alpha={item['alpha']:.1f}, beta={item['beta']:.1f}")
