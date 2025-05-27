"""
Python bridge for CartPole OpenAI Gym environment.

This module provides the Python side of the bridge between Elixir and OpenAI Gym,
implementing the standard CartPole environment with visualization capabilities.
"""

import gymnasium as gym
import numpy as np

# Register the environment with Gym
env_id = 'CartPole-v1'

def create_env():
    """
    Create and return a CartPole environment.
    
    Returns:
        gym.Env: The created environment
    """
    env = gym.make(env_id, render_mode='human')
    return env

def reset_env(env):
    """
    Reset the environment and return the initial observation.
    
    Args:
        env (gym.Env): The environment to reset
        
    Returns:
        numpy.ndarray: Initial observation
    """
    return env.reset()

def step_env(env, action):
    """
    Take a step in the environment.
    
    Args:
        env (gym.Env): The environment to step in
        action (int): The action to take
        
    Returns:
        tuple: (observation, reward, done, info)
    """
    return env.step(action)

def render_env(env, mode='human'):
    """
    Render the environment.
    
    Args:
        env (gym.Env): The environment to render
        mode (str): The rendering mode
        
    Returns:
        None or rendered frame depending on the environment
    """
    return env.render()

def close_env(env):
    """
    Close the environment.
    
    Args:
        env (gym.Env): The environment to close
    """
    env.close()

def get_action_space_size(env):
    """
    Get the size of the action space.
    
    Args:
        env (gym.Env): The environment
        
    Returns:
        int: Size of the action space
    """
    return env.action_space.n

def get_observation_space_shape(env):
    """
    Get the shape of the observation space.
    
    Args:
        env (gym.Env): The environment
        
    Returns:
        tuple: Shape of the observation space
    """
    return env.observation_space.shape
