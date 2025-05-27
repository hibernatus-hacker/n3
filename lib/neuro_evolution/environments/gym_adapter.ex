defmodule NeuroEvolution.Environments.GymAdapter do
  @moduledoc """
  A generic adapter for OpenAI Gym environments.
  
  This module provides a flexible interface to interact with various OpenAI Gym
  environments from Elixir, allowing N3 to be used for a wide range of
  reinforcement learning tasks.
  """
  
  # Uncomment if needed
  # alias NeuroEvolution.Evaluator.BatchEvaluator
  
  @doc """
  Evaluates a genome on a specified Gym environment.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - env_name: The name of the Gym environment (e.g., "CartPole-v1", "MountainCar-v0")
  - num_trials: Number of trials to run (default: 1)
  - options: Additional options for the environment
  
  ## Returns
  - The average fitness score across all trials
  """
  def evaluate(genome, env_name, num_trials \\ 1, options \\ %{}) do
    # Ensure Python bridge is started
    ensure_python_bridge_started()
    
    # Initialize environment
    {:ok, _} = NeuroEvolution.Environments.PythonBridge.call(:gym_adapter, :create_env, [env_name, options])
    
    # Run multiple trials
    results = Enum.map(1..num_trials, fn _ ->
      run_trial(genome, env_name)
    end)
    
    # Return the average score
    Enum.sum(results) / num_trials
  end
  
  @doc """
  Runs a single trial on the specified environment.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - env_name: The name of the Gym environment
  
  ## Returns
  - The total reward obtained during the trial
  """
  def run_trial(genome, env_name) do
    # Reset the environment
    {:ok, observation} = NeuroEvolution.Environments.PythonBridge.call(:gym_adapter, :reset_env, [env_name])
    
    # Run the episode
    run_episode(genome, observation, 0.0, env_name)
  end
  
  @doc """
  Runs a single episode in the specified environment.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - observation: The current observation
  - accumulated_reward: The accumulated reward so far
  - env_name: The name of the Gym environment
  
  ## Returns
  - The total reward obtained during the episode
  """
  def run_episode(genome, observation, accumulated_reward, env_name) do
    # Get the network outputs
    outputs = activate_genome(genome, observation)
    
    # Choose the action based on the environment type
    action = choose_action(outputs, env_name)
    
    # Take a step in the environment
    result = NeuroEvolution.Environments.PythonBridge.call(:gym_adapter, :step_env, [env_name, action])
    
    # Handle the result with proper error checking
    case result do
      {:ok, {new_observation, reward, done}} ->
        # Ensure reward is a proper Elixir float
        reward_float = case reward do
          r when is_float(r) -> r
          r when is_integer(r) -> r * 1.0
          _ -> 0.0  # Default if reward is not a number
        end
        
        # Update the accumulated reward
        new_accumulated_reward = accumulated_reward + reward_float
        
        # Render the environment (only for visualization)
        # Commenting this out for regular evaluation to improve performance
        # {:ok, _} = NeuroEvolution.Environments.PythonBridge.call(:gym_adapter, :render_env, [env_name])
        
        if done do
          # If the episode is done, return the total reward
          new_accumulated_reward
        else
          # Otherwise, continue the episode
          run_episode(genome, new_observation, new_accumulated_reward, env_name)
        end
        
      {:error, reason} ->
        IO.puts("Error in run_episode: #{inspect(reason)}")
        accumulated_reward  # Return accumulated reward so far
        
      _ ->
        IO.puts("Unexpected result from step_env")
        accumulated_reward  # Return accumulated reward so far
    end
  end
  
  @doc """
  Visualizes a genome's performance on the specified environment.
  
  ## Parameters
  - genome: The neural network genome to visualize
  - env_name: The name of the Gym environment
  - num_trials: Number of trials to run (default: 1)
  
  ## Returns
  - The average score across all trials
  """
  def visualize(genome, env_name, num_trials \\ 1) do
    evaluate(genome, env_name, num_trials)
  end
  
  @doc """
  Evolves a population on the specified environment.
  
  ## Parameters
  - population: The initial population to evolve
  - env_name: The name of the Gym environment
  - generations: Number of generations to evolve (default: 10)
  - options: Additional options for the environment and evolution process
  
  ## Returns
  - {evolved_population, stats} tuple with the final population and evolution statistics
  """
  def evolve(population, env_name, generations \\ 10, options \\ %{}) do
    # Define fitness function using the specified environment
    fitness_fn = fn genome ->
      evaluate(genome, env_name, 1, options)
    end
    
    # Evolve the population
    evolved_population = NeuroEvolution.evolve(population, fitness_fn, generations: generations)
    
    # Return evolved population and basic stats
    stats = %{
      final_generation: generations,
      best_fitness: NeuroEvolution.get_fitness(NeuroEvolution.get_best_genome(evolved_population))
    }
    
    {evolved_population, stats}
  end
  
  # Private functions
  
  # Ensure the Python bridge is started
  defp ensure_python_bridge_started do
    case Process.whereis(NeuroEvolution.Environments.PythonBridge) do
      nil ->
        {:ok, _pid} = NeuroEvolution.Environments.PythonBridge.start_link()
      _pid ->
        :ok
    end
  end
  
  # Choose action based on environment type
  defp choose_action(outputs, env_name) do
    case env_name do
      "CartPole-v1" ->
        # Binary action (left or right)
        Enum.with_index(outputs)
        |> Enum.max_by(fn {value, _} -> value end)
        |> elem(1)
      
      "MountainCar-v0" ->
        # Discrete action with 3 possibilities (left, nothing, right)
        Enum.with_index(outputs)
        |> Enum.max_by(fn {value, _} -> value end)
        |> elem(1)
      
      "Acrobot-v1" ->
        # Discrete action with 3 possibilities
        Enum.with_index(outputs)
        |> Enum.max_by(fn {value, _} -> value end)
        |> elem(1)
      
      "LunarLander-v2" ->
        # Discrete action with 4 possibilities
        Enum.with_index(outputs)
        |> Enum.max_by(fn {value, _} -> value end)
        |> elem(1)
      
      # For continuous action spaces, we would need to scale the outputs
      "Pendulum-v1" ->
        # Continuous action space
        # Scale output from [0,1] to [-2,2]
        output = List.first(outputs)
        output * 4.0 - 2.0
      
      # Default case
      _ ->
        # Assume discrete action space, take argmax
        Enum.with_index(outputs)
        |> Enum.max_by(fn {value, _} -> value end)
        |> elem(1)
    end
  end
  
  # Helper function to manually activate a genome
  defp activate_genome(genome, inputs) do
    # Initialize activations for all nodes
    activations = %{}
    
    # Set input activations
    activations = Enum.with_index(inputs, 1)
      |> Enum.reduce(activations, fn {value, idx}, acc ->
        Map.put(acc, idx, value)
      end)
    
    # Process hidden and output nodes in topological order
    # This is a simplified approach that assumes no cycles
    sorted_nodes = genome.inputs ++ (Map.keys(genome.nodes) -- genome.inputs -- genome.outputs) ++ genome.outputs
    
    # Propagate signals through the network
    final_activations = Enum.reduce(sorted_nodes, activations, fn node_id, acc ->
      # Skip input nodes as they already have activations
      if node_id in genome.inputs do
        acc
      else
        # Get all incoming connections to this node
        incoming = Enum.filter(genome.connections, fn {_id, conn} -> 
          conn.to == node_id && conn.enabled
        end)
        
        # Sum weighted inputs
        weighted_sum = Enum.reduce(incoming, 0.0, fn {_id, conn}, sum ->
          from_activation = Map.get(acc, conn.from, 0.0)
          sum + from_activation * conn.weight
        end)
        
        # Apply activation function (sigmoid)
        activation = 1.0 / (1.0 + :math.exp(-weighted_sum))
        
        # Store the activation
        Map.put(acc, node_id, activation)
      end
    end)
    
    # Extract output activations
    Enum.map(genome.outputs, fn output_id ->
      Map.get(final_activations, output_id, 0.0)
    end)
  end
end
