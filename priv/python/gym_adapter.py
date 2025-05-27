"""
Generic adapter for OpenAI Gym environments.

This module provides a flexible interface to interact with various OpenAI Gym
environments, allowing N3 to be used for a wide range of reinforcement learning tasks.
"""

# Import compatibility layer for erlport to work with Python 3.11+
try:
    import erlport_compat
except ImportError:
    print("Warning: erlport_compat not found, may have issues with Python 3.11+")

# Import standard libraries
import time
import sys
import traceback

# Import numpy with safe handling
import numpy as np

# Define numpy bool type safely (for compatibility with different numpy versions)
try:
    NUMPY_BOOL = np.bool_
except AttributeError:
    try:
        NUMPY_BOOL = np.bool
    except AttributeError:
        NUMPY_BOOL = bool

# Import gymnasium with version handling
try:
    import gymnasium as gym
    # Check gymnasium version
    gym_version = gym.__version__
    print(f"Using gymnasium version: {gym_version}")
    NEW_GYM_API = True
except ImportError:
    print("Error: gymnasium not installed. Please install with: pip install gymnasium")
    sys.exit(1)

# Try to import box2d for Lunar Lander and BipedalWalker
try:
    import Box2D
    print("Box2D successfully imported")
except ImportError:
    print("Box2D not installed. To use Lunar Lander or BipedalWalker, install Box2D:")
    print("pip install box2d-py")
    print("If that fails, try: pip install Box2D")
    
# Try to import pygame for rendering
try:
    import pygame
    print("pygame successfully imported")
except ImportError:
    print("pygame not installed. For better rendering, install pygame:")
    print("pip install pygame")

# Dictionary to store environments
envs = {}

def create_env(env_name, options=None):
    """Create and store a Gym environment."""
    global envs
    
    try:
        # Convert env_name to string if it's bytes
        if isinstance(env_name, bytes):
            env_name_str = env_name.decode('utf-8')
        else:
            env_name_str = str(env_name)
        
        # Create a new mutable dictionary from the options
        opts = {}
        if options is not None:
            # Copy all items from the original options and ensure keys are strings
            for k, v in options.items():
                if isinstance(k, bytes):
                    k = k.decode('utf-8')
                opts[str(k)] = v
        
        # Set default render mode if not specified
        if 'render_mode' not in opts:
            opts['render_mode'] = 'human'
        elif opts['render_mode'] == 'none' or opts['render_mode'] == b'none':
            # For terminal mode, use rgb_array which doesn't display anything but is valid
            opts['render_mode'] = 'rgb_array'
            # Also set disable_rendering flag to skip rendering steps
            opts['disable_env_checker'] = True
        
        # Create the environment with proper error handling
        try:
            envs[env_name] = gym.make(env_name_str, **opts)
            print(f"Successfully created environment: {env_name_str}")
            return True
        except Exception as e:
            print(f"Error creating environment {env_name_str}: {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"Unexpected error in create_env: {e}")
        traceback.print_exc()
        return False

def reset_env(env_name):
    """Reset the environment and return the initial observation."""
    global envs
    
    try:
        if env_name not in envs:
            if not create_env(env_name):
                # If environment creation failed, return an empty list
                print(f"Failed to create environment {env_name} for reset")
                return []
        
        # Handle different gym API versions
        try:
            # For newer gym versions that return (obs, info)
            result = envs[env_name].reset()
            if isinstance(result, tuple) and len(result) >= 2:
                observation = result[0]  # First element is observation
            else:
                # For older gym versions that just return obs
                observation = result
            
            # Convert to list for JSON serialization
            if isinstance(observation, np.ndarray):
                observation = observation.tolist()
            
            return observation
        except Exception as e:
            print(f"Error in reset_env: {e}")
            traceback.print_exc()
            return []
    except Exception as e:
        print(f"Unexpected error in reset_env: {e}")
        traceback.print_exc()
        return []

def step_env(env_name, action):
    """Take a step in the environment."""
    global envs
    
    try:
        if env_name not in envs:
            if not create_env(env_name):
                # If environment creation failed, return empty values
                print(f"Failed to create environment {env_name} for step")
                return ([], 0.0, True)
        
        # Ensure action is in the correct format for the environment
        action_space = envs[env_name].action_space
        
        # Handle different action space types
        if hasattr(action_space, 'shape') and len(action_space.shape) > 0:  # Continuous action space
            # Convert action to a list if it's not already
            if not isinstance(action, (list, np.ndarray)):
                try:
                    # Try to convert to a list of appropriate length
                    action_list = [float(action)] * action_space.shape[0]
                    action = action_list
                except (ValueError, TypeError) as e:
                    print(f"Error converting action to continuous format: {e}")
                    # Default to zero actions
                    action = [0.0] * action_space.shape[0]
        
        # Handle different gym API versions with proper error handling
        try:
            # Try to take a step in the environment
            result = envs[env_name].step(action)
            
            # Handle different return formats based on gym version
            if isinstance(result, tuple):
                if len(result) == 5:  # Newer gym API (returns 5 values)
                    observation, reward, terminated, truncated, info = result
                    # Convert numpy boolean types to Python booleans to avoid compatibility issues
                    if hasattr(terminated, 'item'):
                        terminated = bool(terminated.item())
                    if hasattr(truncated, 'item'):
                        truncated = bool(truncated.item())
                    done = terminated or truncated
                elif len(result) == 4:  # Older gym API (returns 4 values)
                    observation, reward, done, info = result
                    # Convert numpy boolean types to Python booleans
                    if hasattr(done, 'item'):
                        done = bool(done.item())
                else:
                    print(f"Unexpected result format from gym step: {result}")
                    return ([], 0.0, True)
            else:
                print(f"Unexpected result type from gym step: {type(result)}")
                return ([], 0.0, True)
            
            # Convert numpy types to Python native types for Erlang compatibility
            if isinstance(reward, (np.float32, np.float64)):
                reward = float(reward)
            elif isinstance(reward, (np.int32, np.int64)):
                reward = int(reward)
                
            # Handle boolean type conversion safely
            if hasattr(np, 'bool_') and isinstance(done, np.bool_):
                done = bool(done)
            elif isinstance(done, bool):
                done = bool(done)
            else:
                # Fallback for any other type
                done = bool(done)
            
            # Convert observation to list for JSON serialization
            if isinstance(observation, np.ndarray):
                observation = observation.tolist()
            
            return (observation, reward, done)
        except Exception as e:
            print(f"Error in step_env: {e}")
            traceback.print_exc()
            # Return default values in case of error
            return ([], 0.0, True)
    except Exception as e:
        print(f"Unexpected error in step_env: {e}")
        traceback.print_exc()
        return ([], 0.0, True)

def render_env(env_name):
    """Render the environment."""
    global envs
    
    try:
        if env_name not in envs:
            if not create_env(env_name):
                # If environment creation failed, return False
                print(f"Failed to create environment {env_name} for rendering")
                return False
        
        try:
            # Different gym versions have different rendering APIs
            if hasattr(envs[env_name], 'render'):
                envs[env_name].render()
            return True
        except Exception as e:
            print(f"Error in render_env: {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"Unexpected error in render_env: {e}")
        traceback.print_exc()
        return False

def close_env(env_name=None):
    """Close the environment."""
    global envs
    
    try:
        if env_name is None:
            # Close all environments
            for env in envs.values():
                try:
                    env.close()
                except Exception as e:
                    print(f"Error closing environment: {e}")
            envs.clear()
        elif env_name in envs:
            # Close specific environment
            try:
                envs[env_name].close()
                del envs[env_name]
            except Exception as e:
                print(f"Error closing environment {env_name}: {e}")
        
        return True
    except Exception as e:
        print(f"Unexpected error in close_env: {e}")
        traceback.print_exc()
        return False

def get_action_space_info(env_name):
    """Get information about the action space."""
    global envs
    
    try:
        if env_name not in envs:
            if not create_env(env_name):
                # If environment creation failed, return error info
                return {'type': 'error', 'info': 'Failed to create environment'}
        
        action_space = envs[env_name].action_space
        
        try:
            if isinstance(action_space, gym.spaces.Discrete):
                return {
                    'type': 'discrete',
                    'n': int(action_space.n)  # Ensure it's a Python int, not numpy int
                }
            elif isinstance(action_space, gym.spaces.Box):
                # Convert numpy arrays to lists for JSON serialization
                return {
                    'type': 'continuous',
                    'shape': list(map(int, action_space.shape)),  # Convert shape to list of ints
                    'low': action_space.low.tolist(),
                    'high': action_space.high.tolist()
                }
            else:
                return {
                    'type': 'unknown',
                    'info': str(action_space)
                }
        except Exception as e:
            print(f"Error getting action space info: {e}")
            traceback.print_exc()
            return {'type': 'error', 'info': f'Error: {str(e)}'}
    except Exception as e:
        print(f"Unexpected error in get_action_space_info: {e}")
        traceback.print_exc()
        return {'type': 'error', 'info': f'Unexpected error: {str(e)}'}

def get_observation_space_info(env_name):
    """Get information about the observation space."""
    global envs
    
    try:
        if env_name not in envs:
            if not create_env(env_name):
                # If environment creation failed, return error info
                return {'type': 'error', 'info': 'Failed to create environment'}
        
        observation_space = envs[env_name].observation_space
        
        try:
            if isinstance(observation_space, gym.spaces.Box):
                # Convert numpy arrays to lists for JSON serialization
                return {
                    'type': 'box',
                    'shape': list(map(int, observation_space.shape)),  # Convert shape to list of ints
                    'low': observation_space.low.tolist(),
                    'high': observation_space.high.tolist()
                }
            else:
                return {
                    'type': 'unknown',
                    'info': str(observation_space)
                }
        except Exception as e:
            print(f"Error getting observation space info: {e}")
            traceback.print_exc()
            return {'type': 'error', 'info': f'Error: {str(e)}'}
    except Exception as e:
        print(f"Unexpected error in get_observation_space_info: {e}")
        traceback.print_exc()
        return {'type': 'error', 'info': f'Unexpected error: {str(e)}'}

def get_available_environments():
    """Get a list of available Gym environments."""
    try:
        # Different gym versions have different ways to access the registry
        if hasattr(gym.envs, 'registry') and hasattr(gym.envs.registry, 'keys'):
            return list(gym.envs.registry.keys())
        elif hasattr(gym, 'envs') and hasattr(gym.envs, 'registry'):
            if hasattr(gym.envs.registry, 'all'):
                return [env_spec.id for env_spec in gym.envs.registry.all()]
            elif hasattr(gym.envs.registry, 'env_specs'):
                return list(gym.envs.registry.env_specs.keys())
        
        # Fallback for other gym versions
        print("Warning: Could not determine gym registry format, returning empty list")
        return []
    except Exception as e:
        print(f"Error getting available environments: {e}")
        traceback.print_exc()
        return []
