defmodule NeuroEvolution.Environments.Gym.TMazeEnv do
  @moduledoc """
  A Gym-like interface for the T-Maze environment.
  
  This module provides a standard reinforcement learning environment interface
  similar to OpenAI Gym for the T-Maze task, allowing it to be used with
  standard RL algorithms.
  """
  
  alias NeuroEvolution.Environments.TMaze
  alias NeuroEvolution.Evaluator.BatchEvaluator
  
  @doc """
  Creates a new T-Maze environment.
  
  ## Returns
  - A map representing the environment state
  """
  def new do
    %{
      reward_location: :left,  # Default reward location
      position: :start,        # Default starting position
      trial_step: 0,           # Current step in the trial
      max_steps: 5,            # Maximum steps per trial
      cue_visible: true,       # Whether the cue is visible
      done: false,             # Whether the episode is done
      evaluator: BatchEvaluator.new(plasticity: true, device: :cpu)
    }
  end
  
  @doc """
  Resets the environment to its initial state.
  
  ## Parameters
  - env: The environment state
  
  ## Returns
  - {observation, updated_env} tuple
  """
  def reset(env) do
    # Randomly set the reward location
    reward_location = if :rand.uniform() < 0.5, do: :left, else: :right
    
    # Reset the environment state
    updated_env = %{
      env |
      reward_location: reward_location,
      position: :start,
      trial_step: 0,
      cue_visible: true,
      done: false
    }
    
    # Generate the initial observation
    observation = get_observation(updated_env)
    
    {observation, updated_env}
  end
  
  @doc """
  Takes a step in the environment based on the given action.
  
  ## Parameters
  - env: The environment state
  - action: The action to take (0 for left, 1 for right)
  
  ## Returns
  - {observation, reward, done, info, updated_env} tuple
  """
  def step(env, action) do
    # Convert action to direction
    direction = if action == 0, do: :left, else: :right
    
    # Update position based on current position and action
    {new_position, reward, done} = case env.position do
      :start -> 
        # From start, always move to corridor
        {:corridor, 0.0, false}
      :corridor -> 
        # From corridor, always move to junction
        {:junction, 0.0, false}
      :junction -> 
        # From junction, move to left or right based on action
        {direction, 0.0, false}
      ^direction ->
        # If already at left/right, check if it matches reward location
        reward = if direction == env.reward_location, do: 1.0, else: 0.0
        {direction, reward, true}
      _ ->
        # If at the wrong arm, no reward and done
        {env.position, 0.0, true}
    end
    
    # Update the environment state
    updated_env = %{
      env |
      position: new_position,
      trial_step: env.trial_step + 1,
      cue_visible: env.position == :start,  # Cue only visible at start
      done: done || env.trial_step >= env.max_steps
    }
    
    # Generate the observation
    observation = get_observation(updated_env)
    
    # Additional info
    info = %{
      "position" => new_position,
      "reward_location" => env.reward_location,
      "trial_step" => updated_env.trial_step
    }
    
    {observation, reward, updated_env.done, info, updated_env}
  end
  
  @doc """
  Renders the environment as a string.
  
  ## Parameters
  - env: The environment state
  
  ## Returns
  - A string representation of the environment
  """
  def render(env) do
    TMaze.render_maze(env.position, env.reward_location)
  end
  
  # Helper function to generate observations based on the current state
  defp get_observation(env) do
    # Convert reward location to input signal
    # Cue signal: 1.0 for left reward, -1.0 for right reward
    cue_signal = if env.reward_location == :left, do: 1.0, else: -1.0
    
    # Position encoding
    position_signal = case env.position do
      :start -> 0.0
      :corridor -> 0.3
      :junction -> 0.7
      :left -> 1.0
      :right -> 1.0
      _ -> 0.0
    end
    
    # Only include the cue signal if it's visible
    cue = if env.cue_visible, do: cue_signal, else: 0.0
    
    # Return the observation as a list
    [cue, position_signal, 1.0]  # Include bias
  end
end
