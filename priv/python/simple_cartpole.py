"""
Simple Python bridge for CartPole OpenAI Gym environment.

This module provides a simplified interface to the CartPole environment
that can be called from Elixir.
"""

import gymnasium as gym
import numpy as np

# Global environment variable to maintain state between calls
env = None

def create_env():
    """Create and return a CartPole environment."""
    global env
    env = gym.make('CartPole-v1', render_mode='human')
    return True

def reset_env():
    """Reset the environment and return the initial observation."""
    global env
    if env is None:
        create_env()
    observation, _ = env.reset()
    return observation.tolist()

def step_env(action):
    """Take a step in the environment."""
    global env
    if env is None:
        create_env()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return (observation.tolist(), reward, done)

def render_env():
    """Render the environment."""
    global env
    if env is None:
        create_env()
    env.render()
    return True

def close_env():
    """Close the environment."""
    global env
    if env is not None:
        env.close()
        env = None
    return True
