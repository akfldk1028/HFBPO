"""
CatProject Episode Generator
GPT를 사용하여 hook|conflict|ending 조합 기반 에피소드 생성

Features:
- OpenAI GPT-4o 기반 에피소드 생성
- temperature/seed 제어로 출력 분산 감소
- ShortVideoMaker API 형식 출력
"""
import os
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("[EpisodeGenerator] Warning: openai package not installed")


class CatEpisodeGenerator:
    """
    CatProject 에피소드 생성기

    - MAB 조합을 받아 GPT로 에피소드 생성
    - ShortVideoMaker consistent-shorts API 형식 출력
    """

    def __init__(self, config_path: str = None, prompts_dir: str = None):
        """
        Args:
            config_path: config.json 경로
            prompts_dir: prompts 폴더 경로
        """
        base_dir = os.path.dirname(__file__)

        if config_path is None:
            config_path = os.path.join(base_dir, '..', 'data', 'config.json')
        if prompts_dir is None:
            prompts_dir = os.path.join(base_dir, '..', 'prompts')

        self.config = self._load_json(config_path)
        self.prompts_dir = os.path.abspath(prompts_dir)

        # GPT 설정
        gpt_settings = self.config.get('gpt_settings', {})
        self.model = gpt_settings.get('model', 'gpt-4o')
        self.temperature = gpt_settings.get('temperature', 0.4)
        self.use_seed = gpt_settings.get('use_seed', True)

        # OpenAI 클라이언트
        self.client = None
        if OpenAI:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)

        # 프롬프트 템플릿 로드
        self.system_prompt = self._load_prompt('system_message.md')
        self.user_template = self._load_prompt('user_message_template.md')

    def _load_json(self, path: str) -> Dict:
        """JSON 파일 로드"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[EpisodeGenerator] Error loading {path}: {e}")
            return {}

    def _load_prompt(self, filename: str) -> str:
        """프롬프트 파일 로드"""
        path = os.path.join(self.prompts_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[EpisodeGenerator] Error loading prompt {filename}: {e}")
            return ""

    def _get_seed(self, combination_key: str) -> int:
        """조합 키에서 고정 시드 생성"""
        hash_bytes = hashlib.md5(combination_key.encode()).hexdigest()
        return int(hash_bytes[:8], 16) % 2147483647

    def generate_episode(
        self,
        combination: Dict[str, Any],
        theme: str = "커플 일상",
        series: str = "daily_life"
    ) -> Dict[str, Any]:
        """
        GPT로 에피소드 생성

        Args:
            combination: MAB 에이전트가 선택한 조합 정보
            theme: 에피소드 테마
            series: 시리즈 타입

        Returns:
            ShortVideoMaker API 형식의 에피소드 딕셔너리
        """
        if not self.client:
            return self._generate_fallback(combination, theme)

        # 조합 정보 추출
        hook_type = combination.get('hook_type', 'shock_reveal')
        conflict_type = combination.get('conflict_type', 'kami_forgets')
        ending_type = combination.get('ending_type', 'both_dumb')
        combination_key = combination.get('combination_key', f"{hook_type}|{conflict_type}|{ending_type}")

        hook_info = combination.get('hook_info', {})
        conflict_info = combination.get('conflict_info', {})
        ending_info = combination.get('ending_info', {})

        # 사용자 프롬프트 구성
        user_prompt = self._build_user_prompt(
            hook_type, conflict_type, ending_type,
            hook_info, conflict_info, ending_info,
            theme, series
        )

        # GPT 호출
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"}
            }

            # Seed 설정 (출력 분산 제어)
            if self.use_seed:
                kwargs["seed"] = self._get_seed(combination_key)

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            episode = json.loads(content)

            # 메타데이터 추가
            episode['_meta'] = {
                'combination_key': combination_key,
                'hook_type': hook_type,
                'conflict_type': conflict_type,
                'ending_type': ending_type,
                'theme': theme,
                'series': series,
                'generated_at': datetime.now().isoformat(),
                'model': self.model,
                'seed': kwargs.get('seed')
            }

            return episode

        except Exception as e:
            print(f"[EpisodeGenerator] GPT Error: {e}")
            return self._generate_fallback(combination, theme)

    def _build_user_prompt(
        self,
        hook_type: str, conflict_type: str, ending_type: str,
        hook_info: Dict, conflict_info: Dict, ending_info: Dict,
        theme: str, series: str
    ) -> str:
        """사용자 프롬프트 구성"""
        if self.user_template:
            prompt = self.user_template
            prompt = prompt.replace('{{HOOK_TYPE}}', hook_type)
            prompt = prompt.replace('{{CONFLICT_TYPE}}', conflict_type)
            prompt = prompt.replace('{{ENDING_TYPE}}', ending_type)
            prompt = prompt.replace('{{HOOK_DESCRIPTION}}', hook_info.get('description', ''))
            prompt = prompt.replace('{{CONFLICT_DESCRIPTION}}', conflict_info.get('description', ''))
            prompt = prompt.replace('{{ENDING_DESCRIPTION}}', ending_info.get('description', ''))
            prompt = prompt.replace('{{KAMI_ROLE}}', conflict_info.get('kami_role', ''))
            prompt = prompt.replace('{{DALGI_ROLE}}', conflict_info.get('dalgi_role', ''))
            prompt = prompt.replace('{{THEME}}', theme)
            prompt = prompt.replace('{{SERIES}}', series)
            return prompt

        # 템플릿 없으면 기본 프롬프트
        return f"""
테마: {theme}
시리즈: {series}

조합:
- Hook: {hook_type} - {hook_info.get('description', '')}
- Conflict: {conflict_type} - {conflict_info.get('description', '')}
- Ending: {ending_type} - {ending_info.get('description', '')}

캐릭터 역할:
- Kami (까미): {conflict_info.get('kami_role', '바보, 깜빡쟁이')}
- Dalgi (딸기): {conflict_info.get('dalgi_role', '현실적, 잔소리')}

위 조합으로 3개 씬의 에피소드를 JSON 형식으로 생성해주세요.
각 씬은 8초 분량입니다.
"""

    def _generate_fallback(self, combination: Dict, theme: str) -> Dict[str, Any]:
        """GPT 실패 시 폴백 에피소드"""
        hook_type = combination.get('hook_type', 'shock_reveal')
        conflict_type = combination.get('conflict_type', 'kami_forgets')
        ending_type = combination.get('ending_type', 'both_dumb')

        return {
            "titleText": {
                "ko": "오늘도 평화로운 하루",
                "en": "Another peaceful day"
            },
            "scenes": [
                {
                    "characterIds": ["kami"],
                    "text": "Something feels off...",
                    "textEnglish": "Something feels off...",
                    "scenePrompt": "Adorable black cat with wide eyes looking confused",
                    "duration": 8
                },
                {
                    "characterIds": ["kami", "dalgi"],
                    "text": "Wait, what did I forget?",
                    "textEnglish": "Wait, what did I forget?",
                    "scenePrompt": "Two cats facing each other, black cat looking guilty",
                    "duration": 8
                },
                {
                    "characterIds": ["kami", "dalgi"],
                    "text": "We're both confused now...",
                    "textEnglish": "We're both confused now...",
                    "scenePrompt": "Both cats with confused expressions, comedic atmosphere",
                    "duration": 8
                }
            ],
            "soundEffects": [
                {"type": "preset", "value": "QUESTION", "startTime": 0.5, "volume": 0.35}
            ],
            "youtubeTitle": f"Cat couple daily life #{datetime.now().strftime('%Y%m%d')} #shorts",
            "youtubeDescription": "Another day in the life of Kami and Dalgi",
            "_meta": {
                "combination_key": f"{hook_type}|{conflict_type}|{ending_type}",
                "fallback": True
            }
        }

    def format_for_api(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """
        ShortVideoMaker consistent-shorts API 형식으로 변환

        Returns:
            API 요청 본문
        """
        return {
            "characterReference": {
                "profileId": "cat-couple",
                "characterIds": ["kami", "dalgi"]
            },
            "titleText": episode.get('titleText', {}),
            "scenes": episode.get('scenes', []),
            "config": {
                "orientation": "portrait",
                "generateVideos": True,
                "useFrameInterpolation": True,
                "skipTTS": True,
                "language": "korean"
            },
            "audio_config": {
                "backgroundMusic": {
                    "source": "calm_piano",
                    "volume": 0.15,
                    "loop": True
                },
                "soundEffects": episode.get('soundEffects', [])
            },
            "youtubeUpload": {
                "channelName": "why_cat",
                "metadata": {
                    "title": episode.get('youtubeTitle', 'Cat couple daily #shorts'),
                    "description": episode.get('youtubeDescription', ''),
                    "tags": ["고양이", "cat", "shorts", "funny", "couple"],
                    "privacyStatus": "public"
                }
            },
            "hfbpo": {
                "combinationKey": episode.get('_meta', {}).get('combination_key'),
                "estimatedReward": episode.get('_meta', {}).get('estimated_reward')
            }
        }


# 테스트
if __name__ == "__main__":
    generator = CatEpisodeGenerator()

    print("=== CatEpisodeGenerator 테스트 ===\n")

    # 테스트 조합
    test_combination = {
        "combination_key": "shock_reveal|kami_forgets|both_dumb",
        "hook_type": "shock_reveal",
        "conflict_type": "kami_forgets",
        "ending_type": "both_dumb",
        "hook_info": {"description": "충격적 표정으로 시작"},
        "conflict_info": {
            "description": "Kami가 중요한 것을 깜빡함",
            "kami_role": "실수 주인공",
            "dalgi_role": "한숨"
        },
        "ending_info": {"description": "둘 다 바보같은 상황"}
    }

    print("=== 에피소드 생성 (Fallback) ===")
    episode = generator._generate_fallback(test_combination, "커플 일상")
    print(json.dumps(episode, indent=2, ensure_ascii=False))

    print("\n=== API 형식 변환 ===")
    api_body = generator.format_for_api(episode)
    print(json.dumps(api_body, indent=2, ensure_ascii=False)[:1000] + "...")
