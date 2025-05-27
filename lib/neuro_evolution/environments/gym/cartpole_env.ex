defmodule NeuroEvolution.Environments.Gym.CartPoleEnv do
  @moduledoc """
  A Gym-like interface for the CartPole environment.
  
  This module provides a standard reinforcement learning environment interface
  for the CartPole task, allowing it to be used with NeuroEvolution algorithms.
  """
  
  alias NeuroEvolution.Evaluator.BatchEvaluator
  
  @doc """
  Creates a new CartPole environment.
  
  ## Returns
  - A map representing the environment state
  """
  def new do
    # Initialize Python environment
    python_env = :python.call(:python, :cartpole_gym, :create_env, [])
    
    # Get action and observation space information
    action_space_size = :python.call(:python, :cartpole_gym, :get_action_space_size, [python_env])
    
    %{
      python_env: python_env,
      action_space_size: action_space_size,
      done: false,
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
    # Reset the Python environment
    observation = :python.call(:python, :cartpole_gym, :reset_env, [env.python_env])
    
    # Convert observation to Elixir list
    observation_list = Enum.map(Tuple.to_list(observation), fn x -> x end)
    
    # Reset the environment state
    updated_env = %{
      env |
      done: false
    }
    
    {observation_list, updated_env}
  end
  
  @doc """
  Takes a step in the environment based on the given action.
  
  ## Parameters
  - env: The environment state
  - action: The action to take (0 or 1 for CartPole)
  
  ## Returns
  - {observation, reward, done, info, updated_env} tuple
  """
  def step(env, action) do
    # Take a step in the Python environment
    {observation, reward, done, _info} = :python.call(:python, :cartpole_gym, :step_env, [env.python_env, action])
    
    # Convert observation to Elixir list
    observation_list = Enum.map(Tuple.to_list(observation), fn x -> x end)
    
    # Update the environment state
    updated_env = %{
      env |
      done: done
    }
    
    # Convert info to Elixir map
    info_map = Map.new()
    
    {observation_list, reward, done, info_map, updated_env}
  end
  
  @doc """
  Renders the environment.
  
  ## Parameters
  - env: The environment state
  
  ## Returns
  - :ok
  """
  def render(env) do
    :python.call(:python, :cartpole_gym, :render_env, [env.python_env])
    :ok
  end
  
  @doc """
  Closes the environment and releases resources.
  
  ## Parameters
  - env: The environment state
  """
  def close(env) do
    :python.call(:python, :cartpole_gym, :close_env, [env.python_env])
    :ok
  end
end
