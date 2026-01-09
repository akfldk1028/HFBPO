"""
CatProject Multi-Armed Bandit Agent
hook_type | conflict_type | ending_type 조합 학습

Features:
- Thompson Sampling 기반 조합 선택
- 호환성 매트릭스 기반 필터링
- 하이브리드 탐색 전략 (Random → Factored → Exploitation)
- Informative Prior 지원
"""
import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime


class CatBanditAgent:
    """
    CatProject용 Thompson Sampling Agent

    조합: hook_type | conflict_type | ending_type
    총 조합 수: 5 × 6 × 6 = 180개
    """

    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: data 폴더 경로 (modifier JSON, config, state 파일 위치)
        """
        if data_dir is None:
            # 기본 경로: 현재 파일 기준 상위의 data 폴더
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        self.data_dir = os.path.abspath(data_dir)

        # 파일 경로
        self.state_file = os.path.join(self.data_dir, 'bandit_state.json')
        self.config_file = os.path.join(self.data_dir, 'config.json')
        self.compatibility_file = os.path.join(self.data_dir, 'compatibility_matrix.json')

        # 데이터 로드
        self.config = self._load_json(self.config_file)
        self.compatibility = self._load_json(self.compatibility_file)
        self.hook_types = self._load_json(os.path.join(self.data_dir, 'hook_type.json'))
        self.conflict_types = self._load_json(os.path.join(self.data_dir, 'conflict_type.json'))
        self.ending_types = self._load_json(os.path.join(self.data_dir, 'ending_type.json'))

        # Bandit 상태
        self.state = self._load_state()
        self.combinations: Dict[str, Dict[str, float]] = self.state.get('combinations', {})
        self.total_trials = self.state.get('_meta', {}).get('total_trials', 0)

        # 모든 가능한 조합 생성
        self.all_combinations = self._generate_all_combinations()

        print(f"[CatBandit] Loaded {len(self.combinations)} saved combinations")
        print(f"[CatBandit] Total possible combinations: {len(self.all_combinations)}")

    def _load_json(self, filepath: str) -> Dict:
        """JSON 파일 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CatBandit] Error loading {filepath}: {e}")
            return {}

    def _load_state(self) -> Dict:
        """Bandit 상태 로드"""
        if os.path.exists(self.state_file):
            return self._load_json(self.state_file)
        return {"_meta": {"total_trials": 0}, "combinations": {}}

    def _save_state(self):
        """상태 저장"""
        self.state['_meta']['total_trials'] = self.total_trials
        self.state['_meta']['last_updated'] = datetime.now().isoformat()
        self.state['combinations'] = self.combinations

        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[CatBandit] Error saving state: {e}")

    def _generate_all_combinations(self) -> List[str]:
        """모든 가능한 조합 키 생성"""
        combinations = []
        hooks = list(self.hook_types.get('modifiers', {}).keys())
        conflicts = list(self.conflict_types.get('modifiers', {}).keys())
        endings = list(self.ending_types.get('modifiers', {}).keys())

        for hook in hooks:
            for conflict in conflicts:
                for ending in endings:
                    key = f"{hook}|{conflict}|{ending}"
                    combinations.append(key)

        return combinations

    def _make_key(self, hook: str, conflict: str, ending: str) -> str:
        """조합 키 생성"""
        return f"{hook}|{conflict}|{ending}"

    def _parse_key(self, key: str) -> Tuple[str, str, str]:
        """키에서 조합 추출"""
        parts = key.split('|')
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return "", "", ""

    def _get_compatibility_score(self, hook: str, conflict: str, ending: str) -> float:
        """호환성 점수 계산 (0~1)"""
        scoring = self.compatibility.get('scoring', {'good': 1.0, 'neutral': 0.7, 'awkward': 0.3})

        # hook-conflict 호환성
        hc_compat = self.compatibility.get('hook_conflict_compatibility', {})
        hc_score = scoring.get(hc_compat.get(hook, {}).get(conflict, 'neutral'), 0.7)

        # conflict-ending 호환성
        ce_compat = self.compatibility.get('conflict_ending_compatibility', {})
        ce_score = scoring.get(ce_compat.get(conflict, {}).get(ending, 'neutral'), 0.7)

        # 블랙리스트 체크
        key = self._make_key(hook, conflict, ending)
        blacklist = self.compatibility.get('blacklist_combinations', [])
        if key in blacklist:
            return 0.0

        # 평균 반환
        return (hc_score + ce_score) / 2

    def _get_prior(self, hook: str, conflict: str, ending: str) -> Dict[str, float]:
        """Informative Prior 가져오기"""
        # 각 modifier의 기본 prior 합산
        hook_prior = self.hook_types.get('modifiers', {}).get(hook, {}).get('prior', {'alpha': 1, 'beta': 1})
        conflict_prior = self.conflict_types.get('modifiers', {}).get(conflict, {}).get('prior', {'alpha': 1, 'beta': 1})
        ending_prior = self.ending_types.get('modifiers', {}).get(ending, {}).get('prior', {'alpha': 1, 'beta': 1})

        # Golden combination 보너스
        key = self._make_key(hook, conflict, ending)
        golden = self.compatibility.get('golden_combinations', [])
        golden_keys = [g['key'] for g in golden]

        if key in golden_keys:
            bonus = 1.0
        else:
            bonus = 0.0

        # 합산된 prior
        alpha = (hook_prior['alpha'] + conflict_prior['alpha'] + ending_prior['alpha']) / 3 + bonus
        beta = (hook_prior['beta'] + conflict_prior['beta'] + ending_prior['beta']) / 3

        return {'alpha': alpha, 'beta': beta}

    def _init_combination(self, key: str) -> Dict[str, float]:
        """새 조합 초기화 (Informative Prior 적용)"""
        hook, conflict, ending = self._parse_key(key)
        prior = self._get_prior(hook, conflict, ending)
        compatibility = self._get_compatibility_score(hook, conflict, ending)

        # 호환성이 낮은 조합은 beta 증가 (부정적 prior)
        if compatibility < 0.5:
            prior['beta'] += 1.0

        return prior

    def select_combination(self, series: str = "daily_life") -> Dict[str, Any]:
        """
        Thompson Sampling으로 조합 선택

        하이브리드 탐색 전략:
        - trials < 30: 완전 랜덤 탐색
        - 30 <= trials < 60: Factored 탐색
        - trials >= 60: Thompson Sampling (10% 탐색 유지)

        Args:
            series: 시리즈 타입 (daily_life, kami_solo, special)

        Returns:
            선택된 조합 정보
        """
        exploration_config = self.config.get('exploration_strategy', {})
        initial_random = exploration_config.get('initial_random_trials', 30)
        factored_end = exploration_config.get('factored_trials_end', 60)
        explore_rate = exploration_config.get('exploration_rate_after', 0.10)

        # Phase 1: 완전 랜덤 탐색 (처음 30회)
        if self.total_trials < initial_random:
            key = np.random.choice(self.all_combinations)
            hook, conflict, ending = self._parse_key(key)

            if key not in self.combinations:
                self.combinations[key] = self._init_combination(key)

            return self._build_result(key, hook, conflict, ending, "random_exploration")

        # Phase 2: Factored 탐색 (31-60회)
        elif self.total_trials < factored_end:
            # 각 modifier별로 독립적으로 샘플링
            hook = self._sample_modifier('hook')
            conflict = self._sample_modifier('conflict')
            ending = self._sample_modifier('ending')
            key = self._make_key(hook, conflict, ending)

            if key not in self.combinations:
                self.combinations[key] = self._init_combination(key)

            return self._build_result(key, hook, conflict, ending, "factored_exploration")

        # Phase 3: Thompson Sampling with 10% exploration
        else:
            if np.random.random() < explore_rate:
                # 탐색: 랜덤 조합
                key = np.random.choice(self.all_combinations)
                hook, conflict, ending = self._parse_key(key)
                mode = "exploitation_explore"
            else:
                # 활용: Thompson Sampling
                key, hook, conflict, ending = self._thompson_sample()
                mode = "exploitation_exploit"

            if key not in self.combinations:
                self.combinations[key] = self._init_combination(key)

            return self._build_result(key, hook, conflict, ending, mode)

    def _sample_modifier(self, modifier_type: str) -> str:
        """특정 modifier 타입에서 Thompson Sampling"""
        if modifier_type == 'hook':
            modifiers = list(self.hook_types.get('modifiers', {}).keys())
            priors = self.hook_types.get('modifiers', {})
        elif modifier_type == 'conflict':
            modifiers = list(self.conflict_types.get('modifiers', {}).keys())
            priors = self.conflict_types.get('modifiers', {})
        else:
            modifiers = list(self.ending_types.get('modifiers', {}).keys())
            priors = self.ending_types.get('modifiers', {})

        # 각 modifier의 경험적 성과 계산
        modifier_stats = {m: {'alpha': 1.0, 'beta': 1.0} for m in modifiers}

        for key, vals in self.combinations.items():
            parts = self._parse_key(key)
            if modifier_type == 'hook' and parts[0] in modifier_stats:
                modifier_stats[parts[0]]['alpha'] += vals['alpha'] - 1
                modifier_stats[parts[0]]['beta'] += vals['beta'] - 1
            elif modifier_type == 'conflict' and parts[1] in modifier_stats:
                modifier_stats[parts[1]]['alpha'] += vals['alpha'] - 1
                modifier_stats[parts[1]]['beta'] += vals['beta'] - 1
            elif modifier_type == 'ending' and parts[2] in modifier_stats:
                modifier_stats[parts[2]]['alpha'] += vals['alpha'] - 1
                modifier_stats[parts[2]]['beta'] += vals['beta'] - 1

        # Thompson Sampling
        samples = []
        for m in modifiers:
            alpha = max(1, modifier_stats[m]['alpha'])
            beta = max(1, modifier_stats[m]['beta'])
            samples.append(np.random.beta(alpha, beta))

        return modifiers[np.argmax(samples)]

    def _thompson_sample(self) -> Tuple[str, str, str, str]:
        """전체 조합에서 Thompson Sampling"""
        # 호환성 필터링된 조합만 고려
        valid_combinations = []
        for key in self.all_combinations:
            hook, conflict, ending = self._parse_key(key)
            compat = self._get_compatibility_score(hook, conflict, ending)
            if compat > 0.3:  # awkward 이상만 허용
                valid_combinations.append(key)

        # 각 조합에서 샘플링
        samples = []
        for key in valid_combinations:
            if key in self.combinations:
                alpha = self.combinations[key]['alpha']
                beta = self.combinations[key]['beta']
            else:
                # 초기화
                self.combinations[key] = self._init_combination(key)
                alpha = self.combinations[key]['alpha']
                beta = self.combinations[key]['beta']

            samples.append(np.random.beta(alpha, beta))

        # 최고 샘플 선택
        best_idx = np.argmax(samples)
        best_key = valid_combinations[best_idx]
        hook, conflict, ending = self._parse_key(best_key)

        return best_key, hook, conflict, ending

    def _build_result(self, key: str, hook: str, conflict: str, ending: str, mode: str) -> Dict[str, Any]:
        """결과 딕셔너리 생성"""
        vals = self.combinations.get(key, {'alpha': 1, 'beta': 1})
        estimated_reward = vals['alpha'] / (vals['alpha'] + vals['beta'])
        compatibility = self._get_compatibility_score(hook, conflict, ending)

        return {
            "combination_key": key,
            "hook_type": hook,
            "conflict_type": conflict,
            "ending_type": ending,
            "hook_info": self.hook_types.get('modifiers', {}).get(hook, {}),
            "conflict_info": self.conflict_types.get('modifiers', {}).get(conflict, {}),
            "ending_info": self.ending_types.get('modifiers', {}).get(ending, {}),
            "estimated_reward": round(estimated_reward, 4),
            "compatibility_score": round(compatibility, 4),
            "selection_mode": mode,
            "trial_number": self.total_trials + 1
        }

    def update_reward(self, combination_key: str, reward: float):
        """
        조합에 보상 업데이트

        Args:
            combination_key: 조합 키 (hook|conflict|ending)
            reward: 정규화된 보상 (0~1)
        """
        reward = max(0.0, min(1.0, reward))

        if combination_key not in self.combinations:
            self.combinations[combination_key] = self._init_combination(combination_key)

        # Beta distribution 업데이트
        self.combinations[combination_key]['alpha'] += reward
        self.combinations[combination_key]['beta'] += (1.0 - reward)

        self.total_trials += 1
        self._save_state()

        alpha = self.combinations[combination_key]['alpha']
        beta = self.combinations[combination_key]['beta']
        mean_reward = alpha / (alpha + beta)

        print(f"[CatBandit] Updated '{combination_key}'")
        print(f"  reward={reward:.3f}, alpha={alpha:.2f}, beta={beta:.2f}, mean={mean_reward:.3f}")

    def get_top_combinations(self, n: int = 10) -> List[Dict[str, Any]]:
        """성과 좋은 조합 Top N"""
        results = []
        for key, vals in self.combinations.items():
            mean = vals['alpha'] / (vals['alpha'] + vals['beta'])
            hook, conflict, ending = self._parse_key(key)
            trials = vals['alpha'] + vals['beta'] - 2  # 초기값 제외

            results.append({
                "key": key,
                "hook_type": hook,
                "conflict_type": conflict,
                "ending_type": ending,
                "mean_reward": round(mean, 4),
                "alpha": round(vals['alpha'], 2),
                "beta": round(vals['beta'], 2),
                "trials": int(trials)
            })

        results.sort(key=lambda x: x['mean_reward'], reverse=True)
        return results[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Bandit 통계"""
        if not self.combinations:
            return {"total_trials": 0, "unique_combinations": 0}

        rewards = [v['alpha'] / (v['alpha'] + v['beta']) for v in self.combinations.values()]

        return {
            "total_trials": self.total_trials,
            "unique_combinations": len(self.combinations),
            "total_possible": len(self.all_combinations),
            "coverage": round(len(self.combinations) / len(self.all_combinations) * 100, 1),
            "mean_reward": round(np.mean(rewards), 4),
            "max_reward": round(np.max(rewards), 4),
            "min_reward": round(np.min(rewards), 4),
            "top_3": self.get_top_combinations(3)
        }


# 테스트
if __name__ == "__main__":
    agent = CatBanditAgent()

    print("\n=== CatBandit 테스트 ===")
    print(f"총 가능한 조합: {len(agent.all_combinations)}개")

    print("\n=== 조합 선택 테스트 ===")
    for i in range(5):
        result = agent.select_combination()
        print(f"\n{i+1}. {result['combination_key']}")
        print(f"   Mode: {result['selection_mode']}")
        print(f"   Estimated reward: {result['estimated_reward']}")
        print(f"   Compatibility: {result['compatibility_score']}")

    print("\n=== 보상 업데이트 테스트 ===")
    result = agent.select_combination()
    agent.update_reward(result['combination_key'], 0.75)

    print("\n=== 통계 ===")
    stats = agent.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
