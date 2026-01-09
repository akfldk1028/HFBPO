"""
HFBPO CatProject - Multi-Armed Bandit for Cat Couple Shorts

Components:
- CatBanditAgent: Thompson Sampling MAB with hybrid exploration
- CatRewardCalculator: Normalized reward with multi-horizon support
- CatEpisodeGenerator: GPT-based episode generation with seed control
"""

from .cat_bandit_agent import CatBanditAgent
from .cat_reward_calculator import CatRewardCalculator, YouTubeMetrics, calculate_reward_from_dict
from .cat_episode_generator import CatEpisodeGenerator

__all__ = [
    'CatBanditAgent',
    'CatRewardCalculator',
    'CatEpisodeGenerator',
    'YouTubeMetrics',
    'calculate_reward_from_dict'
]

__version__ = '1.0.0'
