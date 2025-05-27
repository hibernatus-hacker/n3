defmodule PlasticityExampleTest do
  use ExUnit.Case
  
  alias NeuroEvolution.TWEANN.{Genome, Connection}
  alias NeuroEvolution.Plasticity.NeuralPlasticity
  
  describe "pattern association learning" do
    test "hebbian plasticity improves pattern association performance" do
      # Pattern association task data
      pattern_data = [
        # {cue_input, context_input, expected_output}
        {[1.0, 0.0], [0.5], [1.0]},  # Cue A + context → response 1
        {[0.0, 1.0], [0.5], [0.0]},  # Cue B + context → response 0
        {[1.0, 0.0], [0.0], [0.5]},  # Cue A alone → weak response
        {[0.0, 1.0], [0.0], [0.5]}   # Cue B alone → weak response
      ]
      
      # Create genome with hebbian plasticity
      plasticity_config = %{plasticity_type: :hebbian, learning_rate: 0.05, threshold: 0.1}
      genome = NeuroEvolution.new_genome(3, 1, plasticity: plasticity_config)
      
      # Add plastic connections
      enhanced_genome = add_plastic_connections(genome, :hebbian)
      
      # Test learning performance
      {final_genome, learning_curve} = simulate_learning(enhanced_genome, pattern_data, trials: 15)
      
      # Verify learning occurred
      assert length(learning_curve) == 15
      assert List.first(learning_curve) > List.last(learning_curve)
      
      # Test final performance
      final_error = test_performance(final_genome, pattern_data)
      assert final_error < List.first(learning_curve)
    end
    
    test "different plasticity rules have different learning characteristics" do
      # Pattern association task data
      pattern_data = [
        {[1.0, 0.0], [0.5], [1.0]},
        {[0.0, 1.0], [0.5], [0.0]}
      ]
      
      # Test different plasticity rules
      plasticity_rules = [:hebbian, :stdp, :bcm, :oja]
      
      results = Enum.map(plasticity_rules, fn rule_type ->
        # Create genome with specific plasticity rule
        plasticity_config = %{plasticity_type: rule_type}
        genome = NeuroEvolution.new_genome(3, 1, plasticity: plasticity_config)
        
        # Add plastic connections
        enhanced_genome = add_plastic_connections(genome, rule_type)
        
        # Test learning performance
        {final_genome, learning_curve} = simulate_learning(enhanced_genome, pattern_data, trials: 10)
        
        # Calculate improvement
        initial_error = List.first(learning_curve)
        final_error = List.last(learning_curve)
        improvement = initial_error - final_error
        
        {rule_type, improvement}
      end)
      
      # All rules should show some improvement
      Enum.each(results, fn {_rule, improvement} ->
        assert improvement > 0
      end)
      
      # Rules should have different learning characteristics
      improvements = Enum.map(results, fn {_rule, improvement} -> improvement end)
      assert length(Enum.uniq(improvements)) > 1
    end
    
    test "plasticity evolution improves adaptation capability" do
      # Pattern association task data
      pattern_data = [
        {[1.0, 0.0], [0.5], [1.0]},
        {[0.0, 1.0], [0.5], [0.0]}
      ]
      
      # Create population with mixed plasticity
      population = create_plastic_population()
      
      # Fitness function favors networks that can quickly adapt to new patterns
      fitness_fn = fn genome ->
        {_final, curve} = simulate_learning(genome, pattern_data, trials: 5)
        learning_speed = calculate_learning_speed(curve)
        learning_speed
      end
      
      # Evolve for adaptation capability (just a few generations for test)
      evolved_population = NeuroEvolution.evolve(population, fitness_fn, generations: 3)
      best_plastic_genome = NeuroEvolution.get_best_genome(evolved_population)
      
      # Best genome should have positive fitness
      assert NeuroEvolution.get_fitness(best_plastic_genome) > 0
    end
  end
  
  # Helper functions (copied from plasticity_example.exs for testing)
  
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
  
  defp simulate_learning(genome, patterns, opts \\ []) do
    trials = Keyword.get(opts, :trials, 15)
    
    {final_genome, curve} = Enum.reduce(1..trials, {genome, []}, fn _trial, {current_genome, errors} ->
      # Test performance before plasticity update
      error = test_performance(current_genome, patterns)
      
      # Apply plasticity updates based on patterns
      updated_genome = Enum.reduce(patterns, current_genome, fn {inputs, context, expected}, acc ->
        combined_inputs = inputs ++ context
        outputs = NeuroEvolution.activate(acc, combined_inputs)
        apply_plasticity_update(acc, combined_inputs, outputs, expected)
      end)
      
      {updated_genome, [error | errors]}
    end)
    
    {final_genome, Enum.reverse(curve)}
  end
  
  defp test_performance(genome, patterns) do
    total_error = Enum.reduce(patterns, 0.0, fn {inputs, context, expected}, acc ->
      combined_inputs = inputs ++ context
      outputs = NeuroEvolution.activate(genome, combined_inputs)
      error = :math.pow(List.first(outputs, 0.0) - List.first(expected), 2)
      acc + error
    end)
    
    total_error / length(patterns)
  end
  
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
  
  defp create_plastic_population do
    plasticity_types = [:hebbian, :stdp, :bcm, :oja]
    
    genomes = for _i <- 1..10 do
      plasticity_type = Enum.random(plasticity_types)
      plasticity_config = %{
        plasticity_type: plasticity_type,
        learning_rate: 0.02 + :rand.uniform() * 0.08
      }
      
      NeuroEvolution.new_genome(3, 1, plasticity: plasticity_config)
    end
    
    %{genomes: Enum.with_index(genomes) |> Map.new(fn {g, i} -> {i, g} end)}
  end
  
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
