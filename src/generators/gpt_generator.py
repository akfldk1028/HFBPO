import os
from openai import OpenAI


class GptGenerator:
    """GPT-4o-mini로 프롬프트 생성"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("[GptGen] Warning: OPENAI_API_KEY not found.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

    def generate(self, topic: str, template: str = "Cinematic, 4k, highly detailed") -> str:
        if not self.client:
            return f"Mock GPT Prompt: {template} about {topic}"

        system_prompt = """You are an expert YouTube Shorts scriptwriter and visual director.
        Your goal is to create a highly engaging, viral video prompt based on a given style template.
        The prompt should describe the visual scene, camera movement, and mood for an AI video generator."""

        user_prompt = f"""
        Style Template: {template}
        Topic: {topic}

        Generate a detailed visual prompt for an AI video generator (like Veo/Sora).
        Keep it under 50 words. Focus on visual details.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[GptGen] Error: {e}")
            return f"Fallback: {template} about {topic}"
