from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional

from src.core.rl_agent import BanditAgent
from src.generators.gpt_generator import GptGenerator
from src.generators.rapo_generator import RapoGenerator, ImprovedRapoGenerator
from src.api.google_sheets_client import GoogleSheetsClient
from src.api.analytics_client import AnalyticsClient
from src.core.reward_calculator import RewardCalculator

# Load Environment Variables
load_dotenv()

app = FastAPI(title="HFBPO API", description="Human Feedback Bandit Policy Optimization API")

# Initialize Components
# Define Prompt Templates (Arms) - These could be loaded from config
PROMPT_TEMPLATES = [
    "Cinematic, 4k, highly detailed",
    "Anime style, vibrant colors, studio ghibli inspired",
    "3D Render, Unreal Engine 5, futuristic",
    "Oil painting, impressionist, textured",
    "Cyberpunk, neon lights, dark atmosphere"
]

# Global Instances
agent = BanditAgent(PROMPT_TEMPLATES)
gpt_gen = GptGenerator()
rapo_gen = RapoGenerator(data_dir="data/graph_output")
rapo_gen_improved = ImprovedRapoGenerator(data_dir="data/graph_output", csv_path="data/graph_test1.csv")
sheets_client = GoogleSheetsClient()
analytics_client = AnalyticsClient()
reward_calc = RewardCalculator()

# Generator Map
GENERATORS = {
    "gpt": gpt_gen,
    "rapo": rapo_gen,
    "rapo_improved": rapo_gen_improved
}

class PromptResponse(BaseModel):
    prompt: str
    arm_index: int
    template: str
    estimated_reward: float

class UpdatePolicyResponse(BaseModel):
    processed_count: int
    details: list

@app.get("/health")
def health_check():
    return {"status": "ok", "agent_arms": len(agent.arms)}

@app.get("/prompt", response_model=PromptResponse)
def get_prompt(topic: str = "random", generator_type: str = "gpt"):
    """
    Selects a template using Bandit and generates a prompt.
    generator_type: 'gpt' or 'rapo'
    """
    # 1. Select Generator
    generator = GENERATORS.get(generator_type.lower(), gpt_gen)

    # 2. Select Arm
    action = agent.select_action()
    arm_index = action['arm_index']
    template = action['prompt_template']
    
    # 3. Generate Prompt
    final_prompt = generator.generate(template, topic)
    
    return {
        "prompt": final_prompt,
        "arm_index": arm_index,
        "template": template,
        "estimated_reward": action['estimated_reward']
    }

@app.post("/update-policy", response_model=UpdatePolicyResponse)
def update_policy():
    """
    Triggers the feedback loop:
    1. Find pending videos (>6h old) from Sheets.
    2. Fetch Analytics.
    3. Calculate Reward.
    4. Update Bandit.
    """
    if not sheets_client.client:
        raise HTTPException(status_code=503, detail="Google Sheets Client not initialized (Check credentials)")

    pending_videos = sheets_client.get_pending_videos()
    processed_details = []
    
    for video in pending_videos:
        video_id = str(video.get("VideoID"))
        arm_index = int(video.get("ArmIndex", 0))
        row_index = video.get("row_index")
        
        # Fetch Analytics
        metrics = analytics_client.get_video_analytics(video_id)
        if not metrics:
            print(f"Skipping {video_id}: No metrics found")
            continue
            
        # Calculate Reward
        reward = reward_calc.calculate_reward(metrics)
        
        # Update Bandit
        agent.update_reward(arm_index, reward)
        
        # Update Sheet
        sheets_client.update_video_status(row_index, "DONE", reward)
        
        processed_details.append({
            "video_id": video_id,
            "reward": reward,
            "metrics": metrics
        })
        
    return {
        "processed_count": len(processed_details),
        "details": processed_details
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
