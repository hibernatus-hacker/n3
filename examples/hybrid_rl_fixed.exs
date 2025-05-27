#!/usr/bin/env elixir

# Hybrid Reinforcement Learning Example (Fixed) - Demonstrates combined Q-learning and neuroevolution
#
# This example shows how to leverage both traditional reinforcement learning (Q-learning)
# and neuroevolution to solve the CartPole control task, demonstrating the full pipeline:
# 1. Initialize specialized genomes with recurrent connections and Hebbian plasticity
# 2. Train with Q-learning
# 3. Convert the specialized genome to a standard numeric-ID format for evolution
# 4. Further optimize with evolutionary algorithms
# 5. Fine-tune the evolved network
# 6. Visualize the results

defmodule HybridRLFixed do
  alias NeuroEvolution.RL.QNetwork
  alias NeuroEvolution.Environments.CartPole
  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.TWEANN.GenomeMutations
  alias NeuroEvolution.Population.Population
  
  def run do
    IO.puts("🧠 NeuroEvolution - Hybrid Reinforcement Learning Example (Fixed)")
    IO.puts("======================================================\n")
    
    # Start Python bridge for gym environment
    IO.puts("🔗 Starting Python bridge...")
    {:ok, _bridge} = NeuroEvolution.Environments.PythonBridge.start_link()
    
    # Step 1: Initialize specialized genome with recurrent connections and plasticity
    IO.puts("\n🧬 Step 1: Creating specialized genome with plasticity...")
    
    plasticity_config = %{
      plasticity_type: :hebbian,
      learning_rate: 0.05,
      modulation_strength: 0.8
    }
    
    genome = QNetwork.create_cartpole_genome(
      plasticity: plasticity_config,
      recurrent: true,
      hidden_layers: [8, 8]
    )
    
    IO.puts("   📊 Genome structure:")
    IO.puts("     • Nodes: #{map_size(genome.nodes)}")
    IO.puts("     • Connections: #{map_size(genome.connections)}")
    IO.puts("     • Plasticity: #{inspect(plasticity_config.plasticity_type)}")
    
    # Step 2: Train with Q-learning
    IO.puts("\n📚 Step 2: Training with Q-learning...")
    
    q_network = QNetwork.new(genome,
      learning_rate: 0.1,
      discount_factor: 0.99,
      exploration_rate: 1.0,
      exploration_decay: 0.995,
      min_exploration_rate: 0.01,
      experience_replay: true,
      batch_size: 32
    )
    
    # Train for a small number of episodes (reduced for testing)
    {trained_q_network, q_stats} = QNetwork.train(q_network, 5, 100, CartPole)
    
    # Extract the trained genome
    trained_genome = trained_q_network.genome
    
    IO.puts("   📈 Q-learning results:")
    IO.puts("     • Initial exploration rate: #{q_network.exploration_rate}")
    IO.puts("     • Final exploration rate: #{Float.round(trained_q_network.exploration_rate, 3)}")
    IO.puts("     • Average reward (last 10 episodes): #{Float.round(average_last_n(q_stats.episode_rewards, 10), 1)}")
    
    # Step 3: Convert the specialized genome to a standard format for evolution
    IO.puts("\n🔄 Step 3: Converting specialized genome to standard format...")
    
    standard_genome = convert_to_standard_genome(trained_genome)
    
    IO.puts("   📊 Converted genome structure:")
    IO.puts("     • Nodes: #{map_size(standard_genome.nodes)}")
    IO.puts("     • Connections: #{map_size(standard_genome.connections)}")
    
    # Step 4: Evolve the trained network
    IO.puts("\n🧬 Step 4: Evolving the Q-trained network...")
    
    # Create a population from the trained network for evolution
    population = create_population_from_genome(standard_genome, 20)
    
    IO.puts("   🧬 Created population with #{length(population.genomes)} genomes")
    
    # Define fitness function for CartPole
    fitness_fn = fn genome ->
      try do
        # Evaluate genome on CartPole for 3 trials, take average
        scores = for _trial <- 1..3 do
          CartPole.evaluate(genome, 1)
        end
        
        Enum.sum(scores) / length(scores)
      rescue
        _ -> 0.0  # Return 0 fitness if evaluation fails
      end
    end
    
    # Evolve the population
    evolved_population = NeuroEvolution.evolve(population, fitness_fn,
      generations: 10,
      selection_strategy: :tournament,
      tournament_size: 3,
      elitism: 2,
      mutation_rate: 0.3,
      crossover_rate: 0.7
    )
    
    # Get best evolved genome
    best_evolved_genome = evolved_population.best_genome
    best_evolved_fitness = evolved_population.best_fitness
    
    IO.puts("   🏆 Evolution results:")
    IO.puts("     • Best fitness: #{Float.round(best_evolved_fitness || 0.0, 1)} steps")
    IO.puts("     • Number of generations: #{evolved_population.generation}")
    IO.puts("     • Number of species: #{length(evolved_population.species)}")
    
    # Step 5: Fine-tune the evolved network with Q-learning
    IO.puts("\n📝 Step 5: Fine-tuning the evolved network...")
    
    # Create a new Q-network with the best evolved genome
    fine_tuning_q_network = QNetwork.new(best_evolved_genome,
      learning_rate: 0.05,  # Lower learning rate for fine-tuning
      discount_factor: 0.99,
      exploration_rate: 0.3,  # Start with lower exploration
      exploration_decay: 0.99,
      min_exploration_rate: 0.01,
      experience_replay: true,
      batch_size: 32
    )
    
    # Fine-tune for a few more episodes
    {fine_tuned_q_network, fine_tuning_stats} = QNetwork.train(fine_tuning_q_network, 5, 100, CartPole)
    
    # Evaluate the fine-tuned network
    final_fitness = CartPole.evaluate(fine_tuned_q_network.genome, 5)
    
    IO.puts("   📈 Fine-tuning results:")
    IO.puts("     • Initial fitness: #{Float.round(best_evolved_fitness || 0.0, 1)} steps")
    IO.puts("     • Final fitness: #{Float.round(final_fitness, 1)} steps")
    IO.puts("     • Improvement: #{Float.round(final_fitness - (best_evolved_fitness || 0.0), 1)} steps")
    
    # Step 6: Visualize learning progress
    IO.puts("\n📊 Step 6: Visualizing learning progress...")
    plot_learning_curves(q_stats, fine_tuning_stats)
    
    # Compare performance of different approaches
    IO.puts("\n🔍 Performance comparison:")
    IO.puts("   📊 Q-learning only: #{Float.round(average_last_n(q_stats.episode_rewards, 10), 1)} steps")
    IO.puts("   🧬 Evolution after Q-learning: #{Float.round(best_evolved_fitness || 0.0, 1)} steps")
    IO.puts("   🔄 Hybrid approach (Q-learning + Evolution + Fine-tuning): #{Float.round(final_fitness, 1)} steps")
    
    # Clean up
    NeuroEvolution.Environments.PythonBridge.stop()
    
    IO.puts("\n✅ Hybrid RL example complete!")
    IO.puts("🎯 Successfully demonstrated the power of combining Q-learning with neuroevolution.")
  end
  
  # Convert a specialized genome with string IDs to a standard genome with numeric IDs
  defp convert_to_standard_genome(genome) do
    # Create a mapping from string IDs to numeric IDs
    string_nodes = Enum.filter(genome.nodes, fn {id, _node} -> is_binary(id) end)
    next_id = Enum.max(Enum.filter(Map.keys(genome.nodes), &is_integer/1)) + 1
    
    id_mapping = string_nodes
    |> Enum.with_index(next_id)
    |> Map.new(fn {{string_id, _}, numeric_id} -> {string_id, numeric_id} end)
    
    # Convert nodes with string IDs to numeric IDs
    nodes = Enum.reduce(genome.nodes, %{}, fn
      {id, node}, acc when is_binary(id) ->
        numeric_id = Map.get(id_mapping, id)
        Map.put(acc, numeric_id, node)
      {id, node}, acc ->
        Map.put(acc, id, node)
    end)
    
    # Convert connections to use numeric IDs
    connections = Enum.reduce(genome.connections, %{}, fn
      {conn_id, conn}, acc ->
        # Parse the connection ID to get source and target
        [from_str, to_str] = String.split(conn_id, "_")
        
        # Convert source and target IDs if they're strings
        from = if String.match?(from_str, ~r/^\d+$/) do
          String.to_integer(from_str)
        else
          Map.get(id_mapping, from_str)
        end
        
        to = if String.match?(to_str, ~r/^\d+$/) do
          String.to_integer(to_str)
        else
          Map.get(id_mapping, to_str)
        end
        
        # Create new connection with numeric IDs
        new_conn_id = "#{from}_#{to}"
        new_conn = %{
          from: from,
          to: to,
          weight: conn.weight,
          enabled: conn.enabled
        }
        
        Map.put(acc, new_conn_id, new_conn)
    end)
    
    # Return the converted genome
    %{genome | 
      nodes: nodes, 
      connections: connections
    }
  end
  
  # Create a population from a seed genome with variations
  defp create_population_from_genome(seed_genome, size) do
    # Create variations of the seed genome
    genomes = for i <- 1..size do
      if i == 1 do
        # Keep one copy of the original
        seed_genome
      else
        # Create variations with different mutation rates
        mutation_rate = 0.1 + :rand.uniform() * 0.4
        
        # Apply mutations
        GenomeMutations.mutate(seed_genome, [
          weight_mutation_rate: mutation_rate,
          weight_mutation_power: 0.2,
          add_node_rate: 0.1,
          add_connection_rate: 0.2,
          enable_connection_rate: 0.2,
          disable_connection_rate: 0.1
        ])
      end
    end
    
    # Use the NeuroEvolution module to create a proper Population struct
    # This ensures we create a population that's compatible with NeuroEvolution.evolve/3
    input_count = length(seed_genome.inputs)
    output_count = length(seed_genome.outputs)
    
    population = NeuroEvolution.new_population(size, input_count, output_count, [
      compatibility_threshold: 3.0,
      compatibility_disjoint_coefficient: 1.0,
      compatibility_weight_coefficient: 0.4,
      species_target: 5,
      species_elitism: 2,
      survival_threshold: 0.2,
      weight_mutation_rate: 0.8,
      weight_mutation_power: 0.5,
      add_node_rate: 0.03,
      add_connection_rate: 0.05,
      enable_connection_rate: 0.01,
      disable_connection_rate: 0.01
    ])
    
    # Replace the randomly generated genomes with our variations
    %{population | genomes: genomes}
  end
  
  # Calculate average of last n elements in a list
  defp average_last_n(list, n) do
    list
    |> Enum.take(-min(n, length(list)))
    |> then(fn elements -> Enum.sum(elements) / max(length(elements), 1) end)
  end
  
  # Plot learning curves (simplified visualization)
  defp plot_learning_curves(q_stats, fine_tuning_stats) do
    q_rewards = q_stats.episode_rewards
    fine_tuning_rewards = fine_tuning_stats.episode_rewards
    
    # Calculate moving averages for smoother curves
    q_moving_avg = moving_average(q_rewards, 5)
    fine_tuning_moving_avg = moving_average(fine_tuning_rewards, 3)
    
    # Print simplified ASCII plot
    IO.puts("   📈 Learning curves (moving average):")
    IO.puts("     Q-learning phase: #{format_curve(q_moving_avg)}")
    IO.puts("     Fine-tuning phase: #{format_curve(fine_tuning_moving_avg)}")
  end
  
  # Calculate moving average of a list
  defp moving_average(list, window_size) do
    list
    |> Enum.chunk_every(window_size, 1, :discard)
    |> Enum.map(fn window -> Enum.sum(window) / length(window) end)
  end
  
  # Format curve for ASCII visualization
  defp format_curve(values) do
    # Sample points for visualization
    samples = sample_points(values, 10)
    
    # Map to simple ASCII representation
    samples
    |> Enum.map(&Float.round(&1, 0))
    |> Enum.map(&trunc(&1 / 50))  # Scale down for ASCII display
    |> Enum.map(fn v -> String.duplicate("█", max(v, 1)) end)
    |> Enum.join(" ")
  end
  
  # Sample n evenly spaced points from a list
  defp sample_points(list, n) do
    len = length(list)
    
    if len <= n do
      list
    else
      indices = for i <- 0..(n-1), do: trunc(i * (len - 1) / (n - 1))
      Enum.map(indices, &Enum.at(list, &1))
    end
  end
end

# Run the example
HybridRLFixed.run()
