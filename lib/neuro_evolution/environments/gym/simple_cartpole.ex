defmodule NeuroEvolution.Environments.Gym.SimpleCartPole do
  @moduledoc """
  A simple interface for the CartPole environment.
  
  This module provides a direct interface to the CartPole environment
  through Python's OpenAI Gym, allowing for visualization of N3's performance.
  """
  
  alias NeuroEvolution.Evaluator.BatchEvaluator
  
  @doc """
  Initializes the Python interface.
  """
  def init do
    python_path = Path.join(:code.priv_dir(:neuro_evolution), "python")
    :python.start([{:python_path, to_charlist(python_path)}])
  end
  
  @doc """
  Creates a new CartPole environment.
  """
  def create_env do
    :python.call(:python, :simple_cartpole, :create_env, [])
  end
  
  @doc """
  Resets the environment and returns the initial observation.
  """
  def reset_env do
    :python.call(:python, :simple_cartpole, :reset_env, [])
  end
  
  @doc """
  Takes a step in the environment.
  
  ## Parameters
  - action: The action to take (0 or 1 for CartPole)
  
  ## Returns
  - {observation, reward, done} tuple
  """
  def step_env(action) do
    :python.call(:python, :simple_cartpole, :step_env, [action])
  end
  
  @doc """
  Renders the environment.
  """
  def render_env do
    :python.call(:python, :simple_cartpole, :render_env, [])
  end
  
  @doc """
  Closes the environment.
  """
  def close_env do
    :python.call(:python, :simple_cartpole, :close_env, [])
  end
  
  @doc """
  Evaluates a genome on the CartPole environment.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - num_episodes: Number of episodes to run
  - render: Whether to render the environment
  
  ## Returns
  - The total reward across all episodes
  """
  def evaluate_genome(genome, num_episodes \\ 1, render \\ true) do
    # Initialize Python
    init()
    
    # Create the environment
    create_env()
    
    # Run multiple episodes
    total_reward = Enum.reduce(1..num_episodes, 0.0, fn episode, acc_reward ->
      # Reset the environment
      observation = reset_env()
      
      # Run a single episode
      episode_reward = run_episode(genome, observation, 0.0, render)
      
      IO.puts("Episode #{episode} reward: #{episode_reward}")
      
      # Accumulate rewards
      acc_reward + episode_reward
    end)
    
    # Close the environment
    close_env()
    
    # Return average reward
    total_reward / num_episodes
  end
  
  @doc """
  Evolves a population on the CartPole environment.
  
  ## Parameters
  - population: The initial population of genomes
  - num_generations: Number of generations to evolve
  - num_episodes: Number of episodes to run per genome evaluation
  - render_best: Whether to render the best genome of each generation
  
  ## Returns
  - {evolved_population, stats} tuple with the final population and statistics
  """
  def evolve_population(population, num_generations \\ 10, num_episodes \\ 3, render_best \\ true) do
    # Define the fitness function using the CartPole environment
    fitness_fn = fn genome ->
      evaluate_genome(genome, num_episodes, false)
    end
    
    # Evolve the population
    {evolved_pop, stats} = NeuroEvolution.evolve(population, fitness_fn, num_generations)
    
    # Render the best genome if requested
    if render_best do
      best_genome = NeuroEvolution.get_best_genome(evolved_pop)
      IO.puts("\nRendering best genome with fitness: #{NeuroEvolution.get_fitness(best_genome)}")
      evaluate_genome(best_genome, 1, true)
    end
    
    {evolved_pop, stats}
  end
  
  # Helper function to run a single CartPole episode
  defp run_episode(genome, observation, accumulated_reward, render) do
    # Render the environment if requested
    if render, do: render_env()
    
    # Use the genome to select an action
    action = select_action(genome, observation)
    
    # Take a step in the environment
    {new_observation, reward, done} = step_env(action)
    
    # Update the accumulated reward
    new_accumulated_reward = accumulated_reward + reward
    
    if done do
      # If the episode is done, return the total reward
      new_accumulated_reward
    else
      # Otherwise, continue the episode
      run_episode(genome, new_observation, new_accumulated_reward, render)
    end
  end
  
  # Helper function to select an action based on the genome's output
  defp select_action(genome, observation) do
    # Create a simple evaluator for the genome
    evaluator = BatchEvaluator.new(device: :cpu)
    
    # Get the network outputs
    {outputs, _} = NeuroEvolution.activate(genome, observation, evaluator)
    
    # Choose the action with the highest activation
    Enum.with_index(outputs)
    |> Enum.max_by(fn {value, _} -> value end)
    |> elem(1)
  end
end
