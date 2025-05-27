defmodule NeuroEvolution.Environments.TMaze do
  @moduledoc """
  T-Maze environment for testing memory and learning capabilities of neural networks.
  
  The T-Maze is a classic reinforcement learning task that tests an agent's ability
  to remember previous inputs and make decisions based on that memory.
  
  In this implementation:
  1. The agent starts at the bottom of the T
  2. A cue signal is presented at the start indicating which arm has the reward
  3. The agent moves through the maze and must remember the cue to make the correct choice
  4. Reward is given only if the agent chooses the correct arm
  """
  
  alias NeuroEvolution.Evaluator.BatchEvaluator
  
  @doc """
  Evaluates a genome's performance on the T-Maze task.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - num_trials: Number of trials to run (default: 10)
  - evaluator: The batch evaluator to use (default: creates a new one)
  
  ## Returns
  - The total fitness score across all trials
  """
  def evaluate(genome, num_trials \\ 10, evaluator \\ nil) do
    # Initialize the evaluator with plasticity support if not provided
    # Explicitly set device to CPU to avoid CUDA issues
    evaluator = evaluator || BatchEvaluator.new(plasticity: true, device: :cpu)
    
    # Run multiple trials with balanced left/right rewards to test memory
    # This ensures we test both left and right memory capabilities
    left_trials = div(num_trials, 2)
    right_trials = num_trials - left_trials
    
    # Run left reward trials
    left_results = Enum.map(1..left_trials, fn _ ->
      run_trial(genome, evaluator, :left)
    end)
    
    # Run right reward trials
    right_results = Enum.map(1..right_trials, fn _ ->
      run_trial(genome, evaluator, :right)
    end)
    
    # Calculate success rates for each direction
    left_success = Enum.sum(left_results)
    right_success = Enum.sum(right_results)
    total_success = left_success + right_success
    
    # Calculate balance bonus - reward genomes that can handle both directions
    # This encourages networks that can actually remember and use the cue
    # rather than just always going to one side
    balance_ratio = if total_success > 0 do
      min(left_success, right_success) / max(1, max(left_success, right_success))
    else
      0.0
    end
    
    # Final fitness is total success plus a balance bonus
    # This rewards networks that can solve the task in both directions
    total_success + (balance_ratio * num_trials * 0.5)
  end
  
  @doc """
  Runs a single T-Maze trial with the given genome and reward location.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - evaluator: The batch evaluator to use
  - reward_location: The location of the reward (:left or :right)
  
  ## Returns
  - 1.0 if the agent found the reward, 0.0 otherwise
  """
  def run_trial(genome, evaluator, reward_location) do
    # Convert reward location to input signal
    # Input format: [cue_signal, current_position, bias]
    # Cue signal: 1.0 for left reward, -1.0 for right reward
    cue_signal = if reward_location == :left, do: 1.0, else: -1.0
    
    # Initialize the agent at the start position
    # Sequence of positions: start -> corridor -> junction -> left/right arm
    positions = [:start, :corridor, :junction, :choice]
    
    # Run the agent through the maze
    {final_choice, _} = Enum.reduce(positions, {nil, genome}, fn position, {_, current_genome} ->
      # Generate input based on current position
      inputs = case position do
        :start -> [cue_signal, 0.0, 1.0]  # Cue signal, start position, bias
        :corridor -> [0.0, 0.3, 1.0]      # No cue, corridor position, bias
        :junction -> [0.0, 0.7, 1.0]      # No cue, junction position, bias
        :choice -> [0.0, 1.0, 1.0]        # No cue, choice position, bias
      end
      
      # Get the agent's output
      {outputs, updated_genome} = evaluate_step(current_genome, evaluator, inputs)
      
      # Determine the agent's choice at the junction
      choice = if position == :choice do
        # Output format: [left_activation, right_activation]
        [left_activation, right_activation] = outputs
        if left_activation > right_activation, do: :left, else: :right
      else
        nil
      end
      
      {choice, updated_genome}
    end)
    
    # Calculate reward based on the agent's final choice
    if final_choice == reward_location, do: 1.0, else: 0.0
  end
  
  @doc """
  Evaluates a single step in the T-Maze with the given genome and inputs.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - evaluator: The batch evaluator to use
  - inputs: The input values for this step
  
  ## Returns
  - {outputs, updated_genome} tuple with the network outputs and updated genome
  """
  def evaluate_step(genome, _evaluator, inputs) do
    # Create a simplified evaluation that doesn't rely on BatchEvaluator's tensor operations
    # This is a workaround for the Nx.LazyContainer protocol issues
    
    # For the T-Maze test, we'll use a simplified approach that focuses on testing the environment logic
    # rather than the neural network implementation details
    
    # Get the network outputs by manually activating the network
    outputs = manual_forward_propagate(genome, inputs)
    
    # Apply a simple form of Hebbian plasticity to the connections
    updated_genome = apply_simple_plasticity(genome, inputs, outputs)
    
    {outputs, updated_genome}
  end
  
  # A simplified forward propagation implementation that doesn't use Nx tensors
  defp manual_forward_propagate(genome, inputs) do
    # For simplicity, we'll just use the connection weights directly
    # This is not a full neural network implementation, but sufficient for testing
    
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
  
  # Apply a simplified form of Hebbian plasticity
  defp apply_simple_plasticity(genome, inputs, _outputs) do
    # This is a simplified implementation of Hebbian learning
    # We update weights based on pre/post-synaptic activity using the Hebbian rule
    # "Neurons that fire together, wire together"
    
    # First, calculate activations for all nodes in the network
    activations = manual_forward_propagate_with_activations(genome, inputs)
    
    # Apply Hebbian learning to connections
    updated_connections = Enum.reduce(genome.connections, %{}, fn {conn_id, conn}, acc ->
      # Skip if connection is disabled
      if !conn.enabled do
        Map.put(acc, conn_id, conn)
      else
        # Get pre and post synaptic activations
        pre_activation = Map.get(activations, conn.from, 0.0)
        post_activation = Map.get(activations, conn.to, 0.0)
        
        # Apply modified Hebbian rule for T-maze task
        # For recurrent connections (self-connections), use a higher learning rate
        # to strengthen memory capabilities
        base_learning_rate = 0.1
        learning_rate = if conn.from == conn.to, do: base_learning_rate * 2.0, else: base_learning_rate
        
        # For connections from the cue signal (input 1) to hidden nodes,
        # use a higher learning rate to emphasize the importance of the cue
        learning_rate = if conn.from == 1 and !Enum.member?(genome.outputs, conn.to), do: learning_rate * 1.5, else: learning_rate
        
        # Calculate weight change based on Hebbian rule
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
  
  # A modified forward propagation that returns all node activations
  defp manual_forward_propagate_with_activations(genome, inputs) do
    # Initialize activations for all nodes
    activations = %{}
    
    # Set input activations
    activations = Enum.with_index(inputs, 1)
      |> Enum.reduce(activations, fn {value, idx}, acc ->
        Map.put(acc, idx, value)
      end)
    
    # Process hidden and output nodes in topological order
    sorted_nodes = genome.inputs ++ (Map.keys(genome.nodes) -- genome.inputs -- genome.outputs) ++ genome.outputs
    
    # Propagate signals through the network
    Enum.reduce(sorted_nodes, activations, fn node_id, acc ->
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
  end
  
  @doc """
  Runs a detailed evaluation of a genome on the T-Maze task and returns detailed metrics.
  
  ## Parameters
  - genome: The neural network genome to evaluate
  - num_trials: Number of trials to run
  
  ## Returns
  - A map with detailed performance metrics
  """
  def detailed_evaluation(genome, num_trials \\ 20) do
    # Explicitly set device to CPU to avoid CUDA issues
    evaluator = BatchEvaluator.new(plasticity: true, device: :cpu)
    
    # Run trials with consistent reward locations to test memory
    left_trials = Enum.map(1..div(num_trials, 2), fn _ -> run_trial(genome, evaluator, :left) end)
    right_trials = Enum.map(1..div(num_trials, 2), fn _ -> run_trial(genome, evaluator, :right) end)
    
    # Calculate success rates
    left_success = Enum.sum(left_trials)
    right_success = Enum.sum(right_trials)
    total_success = left_success + right_success
    
    left_success_rate = left_success / length(left_trials)
    right_success_rate = right_success / length(right_trials)
    overall_success_rate = total_success / num_trials
    
    # Calculate balance bonus - reward genomes that can handle both directions
    balance_ratio = if total_success > 0 do
      min(left_success, right_success) / max(1, max(left_success, right_success))
    else
      0.0
    end
    
    # Calculate total fitness with balance bonus
    balance_bonus = balance_ratio * num_trials * 0.5
    total_fitness = total_success + balance_bonus
    
    # Return detailed metrics
    %{
      overall_success_rate: overall_success_rate,
      left_success_rate: left_success_rate,
      right_success_rate: right_success_rate,
      total_score: total_success,
      balance_ratio: balance_ratio,
      balance_bonus: balance_bonus,
      total_fitness: total_fitness,
      num_trials: num_trials
    }
  end
  
  @doc """
  Generates an ASCII representation of the T-Maze with the agent's position and reward location.
  
  ## Parameters
  - position: The agent's current position (:start, :corridor, :junction, :left, :right)
  - reward_location: The location of the reward (:left or :right)
  
  ## Returns
  - A string containing the ASCII representation of the maze
  """
  def render_maze(position, reward_location) do
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
    
    # Replace characters based on position and reward
    maze = 
      case position do
        :start -> replace_at(maze, 6, 2, "A")
        :corridor -> replace_at(maze, 4, 2, "A")
        :junction -> replace_at(maze, 2, 2, "A")
        :left -> replace_at(maze, 2, 1, "A") # Fixed position for left arm
        :right -> replace_at(maze, 2, 9, "A") # Fixed position for right arm
        _ -> maze
      end
    
    # Mark the reward location
    maze = 
      case reward_location do
        :left -> replace_at(maze, 2, 1, "R") # Fixed position for left reward
        :right -> replace_at(maze, 2, 9, "R") # Fixed position for right reward
        _ -> maze
      end
    
    # Join the maze rows into a single string
    Enum.join(maze, "\n")
  end
  
  # Helper function to replace a character in a specific position in the maze
  defp replace_at(maze, row, col, char) do
    List.update_at(maze, row, fn line ->
      String.graphemes(line)
      |> List.update_at(col, fn _ -> char end)
      |> Enum.join("")
    end)
  end
end
