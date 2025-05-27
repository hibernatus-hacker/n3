"""
Python bridge for T-Maze OpenAI Gym environment.

This module provides the Python side of the bridge between Elixir and OpenAI Gym,
implementing a custom T-Maze environment that follows the Gym interface.
"""

import gymnasium as gym
import numpy as np
from gym import spaces

class TMazeEnv(gym.Env):
    """
    T-Maze environment for OpenAI Gym.
    
    This environment implements the T-Maze task as a standard OpenAI Gym environment,
    allowing it to be used with standard RL algorithms.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        """Initialize the T-Maze environment."""
        # Define action space: 0 = left, 1 = right
        self.action_space = spaces.Discrete(2)
        
        # Define observation space: [cue, position, bias]
        # cue: -1.0 to 1.0, position: 0.0 to 1.0, bias: 1.0
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Initialize state
        self.reward_location = 'left'  # 'left' or 'right'
        self.position = 'start'  # 'start', 'corridor', 'junction', 'left', 'right'
        self.trial_step = 0
        self.max_steps = 5
        self.cue_visible = True
        
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            observation (numpy.ndarray): Initial observation
        """
        # Randomly set the reward location
        self.reward_location = 'left' if np.random.random() < 0.5 else 'right'
        
        # Reset the state
        self.position = 'start'
        self.trial_step = 0
        self.cue_visible = True
        
        # Return the initial observation
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): The action to take (0 for left, 1 for right)
            
        Returns:
            observation (numpy.ndarray): The current observation
            reward (float): The reward for the current step
            done (bool): Whether the episode is done
            info (dict): Additional information
        """
        # Convert action to direction
        direction = 'left' if action == 0 else 'right'
        
        # Update position based on current position and action
        reward = 0.0
        done = False
        
        if self.position == 'start':
            # From start, always move to corridor
            self.position = 'corridor'
        elif self.position == 'corridor':
            # From corridor, always move to junction
            self.position = 'junction'
        elif self.position == 'junction':
            # From junction, move to left or right based on action
            self.position = direction
        elif self.position == direction:
            # If already at left/right, check if it matches reward location
            if direction == self.reward_location:
                reward = 1.0
            done = True
        else:
            # If at the wrong arm, no reward and done
            reward = 0.0
            done = True
        
        # Update state
        self.trial_step += 1
        self.cue_visible = (self.position == 'start')  # Cue only visible at start
        
        # Check if maximum steps reached
        if self.trial_step >= self.max_steps:
            done = True
        
        # Additional info
        info = {
            'position': self.position,
            'reward_location': self.reward_location,
            'trial_step': self.trial_step
        }
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The rendering mode
            
        Returns:
            str: ASCII representation of the maze
        """
        # Define the maze layout
        maze = [
            "  +---+---+  ",
            "  |       |  ",
            "  L       R  ",
            "  +   +---+  ",
            "  |   |      ",
            "  |   |      ",
            "  |   |      ",
            "  +---+      ",
            "              "
        ]
        
        # Mark agent position
        if self.position == 'start':
            maze[6] = maze[6][:2] + "A" + maze[6][3:]
        elif self.position == 'corridor':
            maze[4] = maze[4][:2] + "A" + maze[4][3:]
        elif self.position == 'junction':
            maze[2] = maze[2][:2] + "A" + maze[2][3:]
        elif self.position == 'left':
            maze[2] = maze[2][:1] + "A" + maze[2][2:]
        elif self.position == 'right':
            maze[2] = maze[2][:9] + "A" + maze[2][10:]
        
        # Mark reward location
        if self.reward_location == 'left':
            maze[2] = maze[2][:1] + "R" + maze[2][2:]
        elif self.reward_location == 'right':
            maze[2] = maze[2][:9] + "R" + maze[2][10:]
        
        # Join the maze rows into a single string
        maze_str = "\n".join(maze)
        
        if mode == 'human':
            print(maze_str)
        
        return maze_str
    
    def close(self):
        """Close the environment and release resources."""
        pass
    
    def _get_observation(self):
        """
        Generate an observation based on the current state.
        
        Returns:
            numpy.ndarray: The current observation
        """
        # Convert reward location to input signal
        cue_signal = 1.0 if self.reward_location == 'left' else -1.0
        
        # Position encoding
        position_map = {
            'start': 0.0,
            'corridor': 0.3,
            'junction': 0.7,
            'left': 1.0,
            'right': 1.0
        }
        position_signal = position_map.get(self.position, 0.0)
        
        # Only include the cue signal if it's visible
        cue = cue_signal if self.cue_visible else 0.0
        
        # Return the observation as a numpy array
        return np.array([cue, position_signal, 1.0], dtype=np.float32)

# Register the environment with Gym
gym.envs.registration.register(
    id='TMaze-v0',
    entry_point='t_maze_gym:TMazeEnv',
    max_episode_steps=5,
    reward_threshold=1.0,
)

# Functions to be called from Elixir via erlport

def create_env():
    """Create and return a T-Maze environment."""
    return gym.make('TMaze-v0')

def reset_env(env):
    """Reset the environment and return the initial observation."""
    return env.reset().tolist()

def step_env(env, action):
    """Take a step in the environment."""
    obs, reward, done, info = env.step(action)
    return (obs.tolist(), reward, done, info)

def render_env(env, mode='human'):
    """Render the environment."""
    return env.render(mode)

def close_env(env):
    """Close the environment."""
    env.close()
