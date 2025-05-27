defmodule NeuroEvolution.Environments.CartPole do
  @moduledoc """
  CartPole environment for testing control capabilities of neural networks.
  
  The CartPole is a classic reinforcement learning task that tests an agent's ability
  to balance a pole on a moving cart by applying forces to the cart.
  """
  
  alias NeuroEvolution.Evaluator.BatchEvaluator
  
  @doc """
  Evaluates a genome's performance on the CartPole task.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - num_trials: Number of trials to run (default: 1)
  - evaluator: The batch evaluator to use (default: creates a new one)
  
  ## Returns
  - The total fitness score across all trials
  """
  def evaluate(genome, num_trials \\ 1, evaluator \\ nil) do
    # Initialize the evaluator if not provided
    evaluator = evaluator || BatchEvaluator.new(device: :cpu)
    
    # Start the Python bridge if not already started
    ensure_python_bridge_started()
    
    # Run multiple trials
    results = Enum.map(1..num_trials, fn _ ->
      # Run a single CartPole trial
      run_trial(genome, evaluator)
    end)
    
    # Return the average score
    Enum.sum(results) / num_trials
  end
  
  # Ensure the Python bridge is started
  defp ensure_python_bridge_started do
    case Process.whereis(NeuroEvolution.Environments.PythonBridge) do
      nil ->
        {:ok, _pid} = NeuroEvolution.Environments.PythonBridge.start_link()
      _pid ->
        :ok
    end
  end
  
  @doc """
  Initializes the CartPole environment.
  
  ## Returns
  - {:ok, env} tuple with the environment handle
  """
  def init do
    # Ensure Python bridge is started
    ensure_python_bridge_started()
    
    # Initialize the CartPole environment
    {:ok, env} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :create_env, [])
    
    {:ok, env}
  end
  
  @doc """
  Resets the CartPole environment.
  
  ## Parameters
  - env: The environment handle
  
  ## Returns
  - {:ok, state, info} tuple with the initial state and info
  """
  def reset(_env) do
    # Reset the environment
    {:ok, observation} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :reset_env, [])
    
    # Return the initial state with empty info
    {:ok, observation, %{}}
  end
  
  @doc """
  Takes a step in the CartPole environment.
  
  ## Parameters
  - env: The environment handle
  - action: The action to take (0 or 1)
  
  ## Returns
  - {:ok, next_state, reward, done, info} tuple with the next state, reward, done flag, and info
  """
  def step(_env, action) do
    # Take a step in the environment
    {:ok, {new_observation, reward, done}} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :step_env, [action])
    
    # Return the result with empty info
    {:ok, new_observation, reward, done, %{}}
  end
  
  @doc """
  Renders the CartPole environment.
  
  ## Parameters
  - env: The environment handle
  
  ## Returns
  - :ok
  """
  def render(_env) do
    # Render the environment
    {:ok, _} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :render_env, [])
    
    :ok
  end
  
  @doc """
  Runs a single CartPole trial with the given genome.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - evaluator: The batch evaluator to use
  
  ## Returns
  - The total reward obtained during the trial
  """
  def run_trial(genome, evaluator) do
    # Initialize the CartPole environment
    {:ok, _} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :create_env, [])
    
    # Reset the environment
    {:ok, observation} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :reset_env, [])
    
    # Run the episode
    run_episode(genome, evaluator, observation, 0.0)
  end
  
  @doc """
  Runs a single episode in the CartPole environment.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - evaluator: The batch evaluator to use
  - observation: The current observation
  - accumulated_reward: The accumulated reward so far
  
  ## Returns
  - The total reward obtained during the episode
  """
  def run_episode(genome, evaluator, observation, accumulated_reward) do
    # Render the environment
    {:ok, _} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :render_env, [])
    
    # Get the network outputs
    # Manually activate the genome since we don't have direct access to the activate function
    outputs = activate_genome(genome, observation)
    
    # Choose the action with the highest activation
    action = Enum.with_index(outputs)
    |> Enum.max_by(fn {value, _} -> value end)
    |> elem(1)
    
    # Take a step in the environment
    {:ok, {new_observation, reward, done}} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :step_env, [action])
    
    # Update the accumulated reward
    new_accumulated_reward = accumulated_reward + reward
    
    if done do
      # If the episode is done, return the total reward
      new_accumulated_reward
    else
      # Otherwise, continue the episode
      run_episode(genome, evaluator, new_observation, new_accumulated_reward)
    end
  end
  
  @doc """
  Visualizes a genome's performance on the CartPole task.
  
  ## Parameters
  - genome: The neural network genome to visualize
  - num_trials: Number of trials to run (default: 1)
  
  ## Returns
  - The average score across all trials
  """
  def visualize(genome, num_trials \\ 1) do
    evaluate(genome, num_trials)
  end
  
  # Helper function to manually activate a genome with improved topological sorting and activation
  defp activate_genome(genome, inputs) do
    # Initialize activations for all nodes
    activations = %{}
    
    # Set input activations (normalize inputs to range [-1, 1] for better performance)
    normalized_inputs = normalize_inputs(inputs)
    activations = Enum.with_index(normalized_inputs, 1)
      |> Enum.reduce(activations, fn {value, idx}, acc ->
        Map.put(acc, idx, value)
      end)
    
    # Create a proper dependency graph for topological sorting
    dependency_graph = create_dependency_graph(genome)
    
    # Get a proper topological sort of nodes (handles recurrent connections properly)
    sorted_nodes = topological_sort(dependency_graph, genome.inputs, genome.outputs)
    
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
        
        # Get node activation function (default to tanh for better performance)
        node = Map.get(genome.nodes, node_id)
        activation_fn = if node, do: Map.get(node, :activation_fn, :tanh), else: :tanh
        
        # Apply activation function
        activation = apply_activation(activation_fn, weighted_sum)
        
        # Store the activation
        Map.put(acc, node_id, activation)
      end
    end)
    
    # Extract output activations
    Enum.map(genome.outputs, fn output_id ->
      Map.get(final_activations, output_id, 0.0)
    end)
  end
  
  # Normalize CartPole inputs to range [-1, 1] for better neural network performance
  defp normalize_inputs([cart_pos, cart_vel, pole_angle, pole_vel]) do
    # Typical ranges for CartPole values
    # Position: [-2.4, 2.4], Velocity: [-inf, inf] but typically [-10, 10]
    # Angle: [-0.21, 0.21] radians, Angular velocity: [-inf, inf] but typically [-10, 10]
    
    [
      cart_pos / 2.4,  # Normalize position
      :math.tanh(cart_vel / 5.0),  # Normalize velocity using tanh
      pole_angle / 0.21,  # Normalize angle
      :math.tanh(pole_vel / 5.0)  # Normalize angular velocity using tanh
    ]
  end
  
  # Create a dependency graph for topological sorting
  defp create_dependency_graph(genome) do
    # Initialize graph with all nodes
    graph = Enum.reduce(Map.keys(genome.nodes), %{}, fn node_id, acc ->
      Map.put(acc, node_id, [])
    end)
    
    # Add input nodes if not already in the graph
    graph = Enum.reduce(genome.inputs, graph, fn node_id, acc ->
      if Map.has_key?(acc, node_id), do: acc, else: Map.put(acc, node_id, [])
    end)
    
    # Add output nodes if not already in the graph
    graph = Enum.reduce(genome.outputs, graph, fn node_id, acc ->
      if Map.has_key?(acc, node_id), do: acc, else: Map.put(acc, node_id, [])
    end)
    
    # Add edges (dependencies) from the connections
    Enum.reduce(genome.connections, graph, fn {_id, conn}, acc ->
      if conn.enabled do
        # Add the 'from' node as a dependency of the 'to' node
        deps = Map.get(acc, conn.to, [])
        Map.put(acc, conn.to, [conn.from | deps])
      else
        acc
      end
    end)
  end
  
  # Perform topological sort on the dependency graph
  defp topological_sort(graph, inputs, outputs) do
    # Start with input nodes
    visited = MapSet.new(inputs)
    sorted = inputs
    
    # Process all other nodes (hidden and output)
    remaining_nodes = (Map.keys(graph) -- inputs) |> Enum.sort()
    
    # Prioritize output nodes to ensure they're processed last
    remaining_nodes = (remaining_nodes -- outputs) ++ outputs
    
    # Process nodes in order
    {sorted, _} = Enum.reduce(remaining_nodes, {sorted, visited}, fn node, {sorted_acc, visited_acc} ->
      visit_node(node, graph, sorted_acc, visited_acc)
    end)
    
    sorted
  end
  
  # Visit a node in the topological sort
  defp visit_node(node, graph, sorted, visited) do
    if MapSet.member?(visited, node) do
      {sorted, visited}
    else
      # Mark node as visited
      visited = MapSet.put(visited, node)
      
      # Visit dependencies first
      deps = Map.get(graph, node, [])
      {sorted, visited} = Enum.reduce(deps, {sorted, visited}, fn dep, {s_acc, v_acc} ->
        # Skip if it would create a cycle
        if MapSet.member?(v_acc, dep), do: {s_acc, v_acc}, else: visit_node(dep, graph, s_acc, v_acc)
      end)
      
      # Add node to sorted list after its dependencies
      {sorted ++ [node], visited}
    end
  end
  
  # Apply the specified activation function
  defp apply_activation(:tanh, x), do: :math.tanh(x)
  defp apply_activation(:sigmoid, x), do: 1.0 / (1.0 + :math.exp(-x))
  defp apply_activation(:relu, x), do: max(0.0, x)
  defp apply_activation(:leaky_relu, x), do: if(x > 0, do: x, else: 0.01 * x)
  defp apply_activation(_, x), do: :math.tanh(x)  # Default to tanh
end
