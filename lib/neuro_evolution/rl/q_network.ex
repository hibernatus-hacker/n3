defmodule NeuroEvolution.RL.QNetwork do
  @moduledoc """
  A Q-Network implementation that leverages the TWEANN (Topology and Weight Evolving Artificial Neural Network)
  infrastructure for reinforcement learning tasks.
  
  This module combines the strengths of Q-learning with neuroevolution:
  1. Uses TWEANN as the function approximator for Q-values
  2. Leverages Hebbian plasticity for online adaptation
  3. Supports both direct Q-learning updates and evolutionary optimization
  """
  
  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Evaluator.BatchEvaluator
  
  @doc """
  Creates a new Q-Network from a TWEANN genome.
  
  ## Parameters
  - genome: The TWEANN genome to use as the Q-network
  - opts: Options for the Q-network
    - learning_rate: Learning rate for Q-learning updates (default: 0.1)
    - discount_factor: Discount factor for future rewards (default: 0.99)
    - exploration_rate: Initial exploration rate (epsilon) (default: 1.0)
    - min_exploration_rate: Minimum exploration rate (default: 0.01)
    - exploration_decay: Decay rate for exploration (default: 0.995)
    - use_target_network: Whether to use a target network for stability (default: true)
    - target_update_frequency: How often to update the target network (default: 100)
    - experience_replay: Whether to use experience replay (default: true)
    - replay_buffer_size: Size of the experience replay buffer (default: 10000)
    - batch_size: Batch size for experience replay updates (default: 32)
    - device: Device to use for computation (:cpu or :cuda) (default: :cpu)
  
  ## Returns
  - A new Q-network struct
  """
  def new(genome, opts \\ []) do
    # Default options
    opts = Keyword.merge([
      learning_rate: 0.1,
      discount_factor: 0.99,
      exploration_rate: 1.0,
      min_exploration_rate: 0.01,
      exploration_decay: 0.995,
      use_target_network: true,
      target_update_frequency: 100,
      experience_replay: true,
      replay_buffer_size: 10000,
      batch_size: 32,
      device: :cpu
    ], opts)
    
    # Create the Q-network struct
    %{
      genome: genome,
      target_genome: if(opts[:use_target_network], do: genome, else: nil),
      learning_rate: opts[:learning_rate],
      discount_factor: opts[:discount_factor],
      exploration_rate: opts[:exploration_rate],
      min_exploration_rate: opts[:min_exploration_rate],
      exploration_decay: opts[:exploration_decay],
      use_target_network: opts[:use_target_network],
      target_update_frequency: opts[:target_update_frequency],
      update_counter: 0,
      experience_replay: opts[:experience_replay],
      replay_buffer: if(opts[:experience_replay], do: [], else: nil),
      replay_buffer_size: opts[:replay_buffer_size],
      batch_size: opts[:batch_size],
      device: opts[:device],
      evaluator: BatchEvaluator.new(plasticity: true, device: opts[:device]),
      total_reward: 0,
      episode_count: 0
    }
  end
  
  @doc """
  Selects an action based on the current state using an epsilon-greedy policy.
  
  ## Parameters
  - q_network: The Q-network
  - state: The current state
  
  ## Returns
  - {action, updated_q_network} tuple with the selected action and updated Q-network
  """
  def select_action(q_network, state) do
    # Epsilon-greedy action selection
    if :rand.uniform() < q_network.exploration_rate do
      # Explore: select a random action
      action = :rand.uniform(length(q_network.genome.outputs)) - 1
      {action, q_network}
    else
      # Exploit: select the best action based on Q-values
      {q_values, updated_genome} = predict(q_network.genome, state, q_network.evaluator)
      
      # Find the action with the highest Q-value
      {_, action} = Enum.with_index(q_values)
        |> Enum.max_by(fn {value, _} -> value end)
      
      # Update the Q-network with the updated genome
      updated_q_network = %{q_network | genome: updated_genome}
      
      {action, updated_q_network}
    end
  end
  
  @doc """
  Updates the Q-network based on a transition (state, action, reward, next_state, done).
  
  ## Parameters
  - q_network: The Q-network
  - transition: A tuple {state, action, reward, next_state, done}
  
  ## Returns
  - Updated Q-network
  """
  def update(q_network, {state, action, reward, next_state, done}) do
    # Update total reward
    q_network = %{q_network | 
      total_reward: q_network.total_reward + reward,
      episode_count: q_network.episode_count + (if done, do: 1, else: 0)
    }
    
    # Add transition to replay buffer if using experience replay
    q_network = if q_network.experience_replay do
      # Add the transition to the replay buffer
      updated_buffer = [
        {state, action, reward, next_state, done} | q_network.replay_buffer
      ] |> Enum.take(q_network.replay_buffer_size)
      
      %{q_network | replay_buffer: updated_buffer}
    else
      q_network
    end
    
    # Perform Q-learning update
    updated_q_network = if q_network.experience_replay do
      # If using experience replay, only update if we have enough samples
      if length(q_network.replay_buffer) >= q_network.batch_size do
        # Sample a batch from the replay buffer
        batch = Enum.take_random(q_network.replay_buffer, q_network.batch_size)
        
        # Update the Q-network with the batch
        update_with_batch(q_network, batch)
      else
        q_network
      end
    else
      # If not using experience replay, update directly with the current transition
      update_with_batch(q_network, [{state, action, reward, next_state, done}])
    end
    
    # Decay exploration rate
    updated_q_network = %{updated_q_network | 
      exploration_rate: max(
        updated_q_network.min_exploration_rate,
        updated_q_network.exploration_rate * updated_q_network.exploration_decay
      )
    }
    
    # Update target network if needed
    if updated_q_network.use_target_network do
      update_counter = updated_q_network.update_counter + 1
      
      if rem(update_counter, updated_q_network.target_update_frequency) == 0 do
        # Update target network
        %{updated_q_network | 
          target_genome: updated_q_network.genome,
          update_counter: update_counter
        }
      else
        %{updated_q_network | update_counter: update_counter}
      end
    else
      updated_q_network
    end
  end
  
  # Updates the Q-network with a batch of transitions
  defp update_with_batch(q_network, batch) do
    # Process each transition in the batch
    Enum.reduce(batch, q_network, fn {state, action, reward, next_state, done}, acc_q_network ->
      # Get current Q-values for the state
      {q_values, updated_genome} = predict(acc_q_network.genome, state, acc_q_network.evaluator)
      
      # Get next Q-values using the target network if available, otherwise use the current network
      target_genome = if acc_q_network.use_target_network, do: acc_q_network.target_genome, else: acc_q_network.genome
      {next_q_values, _} = predict(target_genome, next_state, acc_q_network.evaluator)
      
      # Calculate target Q-value for the action
      target_q = if done do
        reward
      else
        reward + acc_q_network.discount_factor * Enum.max(next_q_values)
      end
      
      # Calculate the TD error
      current_q = Enum.at(q_values, action)
      td_error = target_q - current_q
      
      # Update the Q-value for the action using the TD error
      updated_q_values = List.update_at(q_values, action, fn q -> 
        q + acc_q_network.learning_rate * td_error
      end)
      
      # Update the genome with the new Q-values
      # This is a simplified approach - in a real implementation, we would use backpropagation
      # Here we're using a direct update to demonstrate the concept
      updated_genome = update_genome_with_target(updated_genome, state, updated_q_values, acc_q_network.learning_rate)
      
      # Return the updated Q-network
      %{acc_q_network | genome: updated_genome}
    end)
  end
  
  @doc """
  Predicts Q-values for a given state using the genome.
  
  ## Parameters
  - genome: The genome to use for prediction
  - state: The state to predict Q-values for
  - evaluator: The batch evaluator to use
  
  ## Returns
  - {q_values, updated_genome} tuple with the predicted Q-values and updated genome
  """
  def predict(genome, state, _evaluator) do
    # Forward pass through the network
    # In a real implementation, this would use the BatchEvaluator
    # Here we're using a simplified approach for demonstration
    
    # Initialize activations for all nodes
    activations = %{}
    
    # Set input activations
    activations = Enum.with_index(state, 1)
      |> Enum.reduce(activations, fn {value, idx}, acc ->
        Map.put(acc, Integer.to_string(idx), value)
      end)
    
    # Process hidden and output nodes in topological order
    sorted_nodes = (Map.keys(genome.nodes) -- genome.inputs) ++ genome.outputs
    
    # Propagate signals through the network
    final_activations = Enum.reduce(sorted_nodes, activations, fn node_id, acc ->
      # Skip input nodes as they already have activations
      if node_id in genome.inputs do
        acc
      else
        # Get all incoming connections to this node
        incoming = Enum.filter(Map.to_list(genome.connections), fn {_id, conn} -> 
          conn.to == node_id && conn.enabled
        end)
        
        # Sum weighted inputs
        weighted_sum = Enum.reduce(incoming, 0.0, fn {_id, conn}, sum ->
          from_activation = Map.get(acc, conn.from, 0.0)
          sum + from_activation * conn.weight
        end)
        
        # Apply activation function (tanh for hidden, identity for output)
        activation = if node_id in genome.outputs do
          weighted_sum  # Identity for Q-values
        else
          :math.tanh(weighted_sum)  # Tanh for hidden nodes
        end
        
        # Store the activation
        Map.put(acc, node_id, activation)
      end
    end)
    
    # Extract output activations (Q-values)
    q_values = Enum.map(genome.outputs, fn output_id ->
      Map.get(final_activations, output_id, 0.0)
    end)
    
    # Apply Hebbian learning if plasticity is enabled
    updated_genome = if genome.plasticity_config != nil do
      apply_hebbian_learning(genome, activations, final_activations)
    else
      genome
    end
    
    {q_values, updated_genome}
  end
  
  # Applies Hebbian learning to the genome
  defp apply_hebbian_learning(genome, pre_activations, post_activations) do
    # Get learning rate from plasticity config
    learning_rate = Map.get(genome.plasticity_config, :learning_rate, 0.1)
    
    # Update connections based on Hebbian rule
    updated_connections = Enum.reduce(genome.connections, %{}, fn {conn_id, conn}, acc ->
      # Skip if connection is disabled
      if !conn.enabled do
        Map.put(acc, conn_id, conn)
      else
        # Get pre and post synaptic activations
        pre_activation = Map.get(pre_activations, conn.from, 0.0)
        post_activation = Map.get(post_activations, conn.to, 0.0)
        
        # Apply Hebbian rule
        weight_change = learning_rate * pre_activation * post_activation
        
        # Update weight
        new_weight = conn.weight + weight_change
        
        # Add some bounds to prevent extreme weights
        bounded_weight = max(-2.0, min(2.0, new_weight))
        
        # Create updated connection
        updated_conn = %{conn | weight: bounded_weight}
        Map.put(acc, conn_id, updated_conn)
      end
    end)
    
    # Return updated genome
    %{genome | connections: updated_connections}
  end
  
  # Updates the genome with target Q-values
  defp update_genome_with_target(genome, state, target_q_values, learning_rate) do
    # This is a simplified direct update approach
    # In a real implementation, we would use backpropagation
    
    # Get current Q-values
    {current_q_values, _} = predict(genome, state, nil)
    
    # Calculate errors
    errors = Enum.zip(current_q_values, target_q_values)
      |> Enum.map(fn {current, target} -> target - current end)
    
    # Update output layer weights directly
    # This is a very simplified approach - in a real implementation, we would use backpropagation
    updated_connections = Enum.reduce(genome.connections, %{}, fn {conn_id, conn}, acc ->
      if conn.to in genome.outputs do
        # Get the output index
        output_idx = Enum.find_index(genome.outputs, fn id -> id == conn.to end)
        
        # Get the error for this output
        error = Enum.at(errors, output_idx)
        
        # Update weight based on error and input activation
        # This is a simplified direct update
        weight_change = learning_rate * error
        new_weight = conn.weight + weight_change
        
        # Bound the weight
        bounded_weight = max(-2.0, min(2.0, new_weight))
        
        # Create updated connection
        updated_conn = %{conn | weight: bounded_weight}
        Map.put(acc, conn_id, updated_conn)
      else
        # Keep other connections unchanged
        Map.put(acc, conn_id, conn)
      end
    end)
    
    # Return updated genome
    %{genome | connections: updated_connections}
  end
  
  @doc """
  Creates a specialized genome structure for Q-learning on the CartPole task.
  
  ## Parameters
  - opts: Options for the genome
    - plasticity: Whether to enable plasticity (default: true)
    - learning_rate: Learning rate for plasticity (default: 0.05)
  
  ## Returns
  - A genome optimized for CartPole Q-learning
  """
  def create_cartpole_genome(opts \\ []) do
    # Default options
    opts = Keyword.merge([
      plasticity: true,
      learning_rate: 0.05
    ], opts)
    
    # Create a basic genome with 4 inputs and 2 outputs
    # CartPole has 4 inputs: position, velocity, angle, angular velocity
    # and 2 outputs: left and right force (Q-values)
    plasticity_config = if opts[:plasticity] do
      %{
        plasticity_type: :hebbian,
        learning_rate: opts[:learning_rate],
        modulation_enabled: true
      }
    else
      nil
    end
    
    genome = Genome.new(4, 2, plasticity: plasticity_config)
    
    # Add hidden nodes to create a more robust control system
    add_node = fn genome, id, type ->
      node_id = if is_binary(id), do: id, else: Integer.to_string(id)
      nodes = Map.put(genome.nodes, node_id, %{type: type, bias: 0.0, activation_fn: :tanh})
      %{genome | nodes: nodes}
    end
    
    add_connection = fn genome, from, to, weight ->
      from_id = if is_binary(from), do: from, else: Integer.to_string(from)
      to_id = if is_binary(to), do: to, else: Integer.to_string(to)
      
      connection_id = "#{from_id}_#{to_id}"
      connection = %{from: from_id, to: to_id, weight: weight, enabled: true}
      connections = Map.put(genome.connections, connection_id, connection)
      
      %{genome | connections: connections}
    end
    
    # Add hidden nodes - one for balance control, one for position control, and one for memory
    genome = add_node.(genome, "balance", :hidden)
    genome = add_node.(genome, "position", :hidden)
    genome = add_node.(genome, "memory", :hidden)
    
    # Add recurrent connections for memory
    genome = add_connection.(genome, "memory", "memory", 0.9)  # Recurrent connection
    
    # Add connections from inputs to hidden nodes
    genome = add_connection.(genome, 1, "position", 1.0)  # Position to position control
    genome = add_connection.(genome, 2, "position", 0.5)  # Velocity to position control
    genome = add_connection.(genome, 3, "balance", 1.0)   # Angle to balance control
    genome = add_connection.(genome, 4, "balance", 0.5)   # Angular velocity to balance control
    
    # Add connections from hidden nodes to outputs
    genome = add_connection.(genome, "balance", 5, 0.8)   # Balance to left output
    genome = add_connection.(genome, "balance", 6, -0.8)  # Balance to right output
    genome = add_connection.(genome, "position", 5, 0.6)  # Position to left output
    genome = add_connection.(genome, "position", 6, -0.6) # Position to right output
    genome = add_connection.(genome, "memory", 5, 0.3)    # Memory to left output
    genome = add_connection.(genome, "memory", 6, -0.3)   # Memory to right output
    
    # Add connections from hidden nodes to memory
    genome = add_connection.(genome, "balance", "memory", 0.5)
    genome = add_connection.(genome, "position", "memory", 0.5)
    
    # Add direct connections from inputs to outputs
    genome = add_connection.(genome, 1, 5, 0.3)  # Position to left output
    genome = add_connection.(genome, 1, 6, -0.3) # Position to right output
    genome = add_connection.(genome, 3, 5, 0.7)  # Angle to left output
    genome = add_connection.(genome, 3, 6, -0.7) # Angle to right output
    
    genome
  end
  
  @doc """
  Trains a Q-network on the CartPole task.
  
  ## Parameters
  - q_network: The Q-network to train
  - num_episodes: Number of episodes to train for
  - max_steps: Maximum number of steps per episode
  - env_module: The environment module to use
  
  ## Returns
  - {trained_q_network, training_stats} tuple with the trained Q-network and training statistics
  """
  def train(q_network, num_episodes, max_steps, env_module) do
    # Initialize training statistics
    stats = %{
      episode_rewards: [],
      episode_lengths: [],
      exploration_rates: []
    }
    
    # Train for the specified number of episodes
    Enum.reduce(1..num_episodes, {q_network, stats}, fn episode, {current_q_network, current_stats} ->
      # Initialize the environment
      {:ok, env} = env_module.init()
      # Reset the environment
      {:ok, state, _info} = env_module.reset(env)
      
      # Run the episode
      {updated_q_network, episode_reward, episode_length} = run_episode(
        current_q_network, 
        env,
        state, 
        0, 
        0, 
        max_steps, 
        env_module
      )
      
      # Update statistics
      updated_stats = %{
        episode_rewards: current_stats.episode_rewards ++ [episode_reward],
        episode_lengths: current_stats.episode_lengths ++ [episode_length],
        exploration_rates: current_stats.exploration_rates ++ [updated_q_network.exploration_rate]
      }
      
      # Print progress
      if rem(episode, max(1, div(num_episodes, 20))) == 0 do
        avg_reward = Enum.sum(Enum.take(updated_stats.episode_rewards, -10)) / min(10, episode)
        IO.puts("Episode #{episode}/#{num_episodes}: Reward=#{episode_reward}, Avg(10)=#{avg_reward}, Epsilon=#{updated_q_network.exploration_rate}")
      end
      
      {updated_q_network, updated_stats}
    end)
  end
  
  # Runs a single episode
  defp run_episode(q_network, env, state, total_reward, step, max_steps, env_module) do
    # Select an action
    {action, updated_q_network} = select_action(q_network, state)
    
    # Take a step in the environment
    {:ok, next_state, reward, done, _info} = env_module.step(env, action)
    
    # Update the Q-network
    updated_q_network = update(updated_q_network, {state, action, reward, next_state, done})
    
    # Update total reward
    new_total_reward = total_reward + reward
    
    # Check if episode is done
    if done || step >= max_steps - 1 do
      {updated_q_network, new_total_reward, step + 1}
    else
      # Continue the episode
      run_episode(updated_q_network, env, next_state, new_total_reward, step + 1, max_steps, env_module)
    end
  end
  
  @doc """
  Evaluates a trained Q-network on the CartPole task.
  
  ## Parameters
  - q_network: The trained Q-network
  - num_episodes: Number of episodes to evaluate for
  - max_steps: Maximum number of steps per episode
  - env_module: The environment module to use
  - render: Whether to render the environment
  
  ## Returns
  - {evaluation_stats, final_q_network} tuple with evaluation statistics and the final Q-network
  """
  def evaluate(q_network, num_episodes, max_steps, env_module, render \\ false) do
    # Set exploration rate to 0 for evaluation
    q_network = %{q_network | exploration_rate: 0.0}
    
    # Initialize evaluation statistics
    stats = %{
      episode_rewards: [],
      episode_lengths: []
    }
    
    # Evaluate for the specified number of episodes
    Enum.reduce(1..num_episodes, {q_network, stats}, fn episode, {current_q_network, current_stats} ->
      # Initialize the environment
      {:ok, env} = env_module.init()
      # Reset the environment
      {:ok, state, _info} = env_module.reset(env)
      
      # Run the episode
      {updated_q_network, episode_reward, episode_length} = evaluate_episode(
        current_q_network,
        env, 
        state, 
        0, 
        0, 
        max_steps, 
        env_module,
        render
      )
      
      # Update statistics
      updated_stats = %{
        episode_rewards: current_stats.episode_rewards ++ [episode_reward],
        episode_lengths: current_stats.episode_lengths ++ [episode_length]
      }
      
      # Print progress
      if render || episode == num_episodes do
        avg_reward = Enum.sum(updated_stats.episode_rewards) / length(updated_stats.episode_rewards)
        IO.puts("Evaluation: Episode #{episode}/#{num_episodes}: Reward=#{episode_reward}, Avg=#{avg_reward}")
      end
      
      {updated_stats, updated_q_network}
    end)
  end
  
  # Runs a single evaluation episode
  defp evaluate_episode(q_network, env, state, total_reward, step, max_steps, env_module, render) do
    # Render the environment if requested
    if render do
      env_module.render(env)
    end
    
    # Select an action (no exploration during evaluation)
    {action, updated_q_network} = select_action(q_network, state)
    
    # Take a step in the environment
    {:ok, next_state, reward, done, _info} = env_module.step(env, action)
    
    # Update total reward
    new_total_reward = total_reward + reward
    
    # Check if episode is done
    if done || step >= max_steps - 1 do
      {updated_q_network, new_total_reward, step + 1}
    else
      # Continue the episode
      evaluate_episode(updated_q_network, env, next_state, new_total_reward, step + 1, max_steps, env_module, render)
    end
  end
end
