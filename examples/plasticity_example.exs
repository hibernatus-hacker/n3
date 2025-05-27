#!/usr/bin/env elixir

# Neural Plasticity Example - Demonstrates adaptive learning mechanisms
#
# This example shows how to use different plasticity rules (Hebbian, STDP, BCM, Oja)
# to create networks that adapt their connectivity during their lifetime.

defmodule PlasticityExample do
  def run do
    IO.puts("🧠 NeuroEvolution - Neural Plasticity Example")
    IO.puts("===========================================\n")

    # Test different plasticity rules on a simple pattern association task
    plasticity_rules = [
      {:hebbian, %{learning_rate: 0.05, threshold: 0.1}},
      {:stdp, %{a_plus: 0.1, a_minus: 0.12, tau_plus: 20.0}},
      {:bcm, %{learning_rate: 0.03, threshold_rate: 0.001}},
      {:oja, %{learning_rate: 0.04}}
    ]

    # Pattern association task: learn to associate cue with delayed response
    pattern_data = [
      # {cue_input, context_input, expected_output}
      {[1.0, 0.0], [0.5], [1.0]},  # Cue A + context → response 1
      {[0.0, 1.0], [0.5], [0.0]},  # Cue B + context → response 0
      {[1.0, 0.0], [0.0], [0.5]},  # Cue A alone → weak response
      {[0.0, 1.0], [0.0], [0.5]}   # Cue B alone → weak response
    ]

    Enum.each(plasticity_rules, fn {rule_type, params} ->
      IO.puts("🔬 Testing #{rule_type} plasticity...")
      
      # Create genome with specific plasticity rule
      plasticity_config = Map.merge(params, %{plasticity_type: rule_type})
      genome = NeuroEvolution.new_genome(3, 1, plasticity: plasticity_config)

      # Add plastic connections
      enhanced_genome = add_plastic_connections(genome, rule_type)

      # Test learning performance
      {final_genome, learning_curve} = simulate_learning(enhanced_genome, pattern_data)
      
      # Calculate final performance
      final_error = test_performance(final_genome, pattern_data)
      learning_improvement = List.first(learning_curve) - List.last(learning_curve)

      IO.puts("   📊 Results:")
      IO.puts("     • Initial error: #{Float.round(List.first(learning_curve), 4)}")
      IO.puts("     • Final error: #{Float.round(final_error, 4)}")
      IO.puts("     • Learning improvement: #{Float.round(learning_improvement, 4)}")
      IO.puts("")
    end)

    # Demonstrate plasticity evolution
    IO.puts("🧬 Evolving plastic networks for rapid adaptation...")
    
    # Create population with mixed plasticity
    population = create_plastic_population()
    
    # Fitness function favors networks that can quickly adapt to new patterns
    fitness_fn = fn genome ->
      # Test on multiple pattern sets to reward generalization
      test_sets = [
        pattern_data,
        # Add noise to test robustness
        add_noise_to_patterns(pattern_data, 0.1)
      ]
      
      total_fitness = Enum.reduce(test_sets, 0.0, fn test_patterns, acc ->
        {_final, curve} = simulate_learning(genome, test_patterns, trials: 10)
        learning_speed = calculate_learning_speed(curve)
        acc + learning_speed
      end)
      
      total_fitness / length(test_sets)
    end

    # Evolve for adaptation capability
    evolved_population = NeuroEvolution.evolve(population, fitness_fn, generations: 15)
    best_plastic_genome = NeuroEvolution.get_best_genome(evolved_population)
    
    IO.puts("   🏆 Best plastic network fitness: #{Float.round(NeuroEvolution.get_fitness(best_plastic_genome), 3)}")
    
    # Demonstrate the evolved plastic network
    IO.puts("\n🎯 Testing evolved plastic network:")
    {_final, evolved_curve} = simulate_learning(best_plastic_genome, pattern_data, trials: 20)
    
    IO.puts("   📈 Learning curve: #{inspect(Enum.take(evolved_curve, 5))} ... #{inspect(Enum.take(evolved_curve, -3))}")
    IO.puts("   ⚡ Learning speed: #{Float.round(calculate_learning_speed(evolved_curve), 3)}")

    IO.puts("\n✅ Plasticity example complete!")
    IO.puts("🎯 Networks successfully demonstrated adaptive learning capabilities.")
  end

  # Add plastic connections with specific plasticity rule
  defp add_plastic_connections(genome, plasticity_type) do
    # Add a few connections and make them plastic
    with_connections = NeuroEvolution.TWEANN.Genome.add_connection(genome, 1, 4)  # Input to output
    
    # Enable plasticity on connections
    plastic_connections = Enum.reduce(with_connections.connections, %{}, fn {id, conn}, acc ->
      plastic_conn = NeuroEvolution.TWEANN.Connection.with_plasticity(conn, plasticity_type)
      Map.put(acc, id, plastic_conn)
    end)
    
    %{with_connections | connections: plastic_connections}
  end

  # Simulate learning over multiple trials
  defp simulate_learning(genome, patterns, opts \\ []) do
    trials = Keyword.get(opts, :trials, 15)
    
    {final_genome, curve} = Enum.reduce(1..trials, {genome, []}, fn _trial, {current_genome, errors} ->
      # Test performance before plasticity update
      error = test_performance(current_genome, patterns)
      
      # Apply plasticity updates based on patterns
      updated_genome = Enum.reduce(patterns, current_genome, fn {inputs, expected}, acc ->
        outputs = NeuroEvolution.activate(acc, inputs)
        apply_plasticity_update(acc, inputs, outputs, expected)
      end)
      
      {updated_genome, [error | errors]}
    end)
    
    {final_genome, Enum.reverse(curve)}
  end

  # Test network performance on patterns
  defp test_performance(genome, patterns) do
    total_error = Enum.reduce(patterns, 0.0, fn {inputs, expected}, acc ->
      outputs = NeuroEvolution.activate(genome, inputs)
      error = :math.pow(List.first(outputs, 0.0) - List.first(expected), 2)
      acc + error
    end)
    
    total_error / length(patterns)
  end

  # Apply plasticity updates based on activity patterns
  defp apply_plasticity_update(genome, inputs, outputs, expected) do
    # Extract connection information
    connections = genome.connections
    
    # Update each plastic connection based on pre/post-synaptic activity
    updated_connections = Enum.reduce(connections, %{}, fn {id, connection}, acc ->
      # Skip non-plastic connections
      if connection.plasticity_type == nil do
        Map.put(acc, id, connection)
      else
        # Find pre and post nodes
        pre_node_id = connection.source_id
        post_node_id = connection.target_id
        
        # Get activities (simplified for demonstration)
        pre_activity = if pre_node_id <= length(inputs), do: Enum.at(inputs, pre_node_id - 1, 0.0), else: 0.5
        post_activity = if post_node_id > length(genome.inputs), do: Enum.at(outputs, post_node_id - length(genome.inputs) - 1, 0.0), else: 0.5
        
        # Create plasticity context with error signal
        error = if post_node_id > length(genome.inputs) + length(genome.hidden) do
          output_idx = post_node_id - length(genome.inputs) - length(genome.hidden) - 1
          expected_output = Enum.at(expected, output_idx, 0.0)
          actual_output = Enum.at(outputs, output_idx, 0.0)
          expected_output - actual_output
        else
          0.0
        end
        
        context = %{
          error_signal: error,
          gating_signal: abs(error),
          neuromodulation: 1.0 + abs(error) * 2.0  # Modulate based on error magnitude
        }
        
        # Apply plasticity rule based on connection's plasticity type
        updated_connection = case connection.plasticity_type do
          :hebbian ->
            params = %{learning_rate: 0.05, threshold: 0.1}
            NeuroEvolution.Plasticity.HebbianRule.update_weight(connection, pre_activity, post_activity, params, context)
          
          :stdp ->
            # For STDP, we need timing information
            timing_context = Map.merge(context, %{
              current_time: :os.system_time(:millisecond) / 1000.0,
              pre_spike_time: :os.system_time(:millisecond) / 1000.0 - 0.01,
              post_spike_time: :os.system_time(:millisecond) / 1000.0
            })
            params = %{a_plus: 0.1, a_minus: 0.12, tau_plus: 20.0}
            NeuroEvolution.Plasticity.STDPRule.update_weight(connection, pre_activity, post_activity, params, timing_context)
          
          :bcm ->
            params = %{learning_rate: 0.03, threshold_rate: 0.001}
            NeuroEvolution.Plasticity.BCMRule.update_weight(connection, pre_activity, post_activity, params, context)
          
          :oja ->
            params = %{learning_rate: 0.04}
            NeuroEvolution.Plasticity.OjaRule.update_weight(connection, pre_activity, post_activity, params, context)
          
          _ ->
            connection
        end
        
        Map.put(acc, id, updated_connection)
      end
    end)
    
    # Return updated genome with modified connections
    %{genome | connections: updated_connections}
  end

  # Create population with different plasticity configurations
  defp create_plastic_population do
    plasticity_types = [:hebbian, :stdp, :bcm, :oja]
    
    genomes = for _i <- 1..20 do
      plasticity_type = Enum.random(plasticity_types)
      plasticity_config = %{
        plasticity_type: plasticity_type,
        learning_rate: 0.02 + :rand.uniform() * 0.08
      }
      
      NeuroEvolution.new_genome(3, 1, plasticity: plasticity_config)
    end
    
    %{genomes: Enum.with_index(genomes) |> Map.new(fn {g, i} -> {i, g} end)}
  end

  # Add noise to patterns for robustness testing
  defp add_noise_to_patterns(patterns, noise_level) do
    Enum.map(patterns, fn {inputs, outputs} ->
      noisy_inputs = Enum.map(inputs, fn x -> 
        x + ((:rand.uniform() - 0.5) * 2 * noise_level)
      end)
      {noisy_inputs, outputs}
    end)
  end

  # Calculate learning speed from error curve
  defp calculate_learning_speed(curve) do
    if length(curve) < 2 do
      0.0
    else
      initial_error = List.first(curve)
      final_error = List.last(curve)
      improvement = initial_error - final_error
      # Normalize by number of trials
      improvement / length(curve)
    end
  end
end

# Run the example
PlasticityExample.run()