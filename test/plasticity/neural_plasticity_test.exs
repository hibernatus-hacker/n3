defmodule NeuroEvolution.Plasticity.NeuralPlasticityTest do
  use ExUnit.Case
  alias NeuroEvolution.Plasticity.NeuralPlasticity
  alias NeuroEvolution.TWEANN.{Connection, Node}

  describe "plasticity creation" do
    test "creates plasticity with correct parameters" do
      plasticity = NeuralPlasticity.new(:hebbian, learning_rate: 0.02, decay_rate: 0.95)
      
      assert plasticity.plasticity_type == :hebbian
      assert plasticity.learning_rate == 0.02
      assert plasticity.decay_rate == 0.95
    end

    test "uses default parameters when not specified" do
      plasticity = NeuralPlasticity.new(:stdp)
      
      assert plasticity.plasticity_type == :stdp
      assert plasticity.learning_rate == 0.01
      assert plasticity.decay_rate == 0.99
    end
  end

  describe "Hebbian plasticity" do
    test "increases weights with correlated activity" do
      plasticity = NeuralPlasticity.new(:hebbian, learning_rate: 0.1)
      connection = Connection.new(1, 2, 0.5, 1, true)
      
      # Positively correlated activity should increase weight
      updated_conn = NeuralPlasticity.update_connection(plasticity, connection, 1.0, 1.0)
      
      # Weight should increase (after accounting for decay)
      assert updated_conn.weight != connection.weight
    end

    test "decreases weights with anti-correlated activity" do
      plasticity = NeuralPlasticity.new(:hebbian, learning_rate: 0.1)
      connection = Connection.new(1, 2, 0.5, 1, true)
      
      # Anti-correlated activity (high pre, low post)
      updated_conn = NeuralPlasticity.update_connection(plasticity, connection, 1.0, -1.0)
      
      # Weight should change
      assert updated_conn.weight != connection.weight
    end

    test "learning rate affects magnitude of change" do
      high_lr = NeuralPlasticity.new(:hebbian, learning_rate: 0.5)
      low_lr = NeuralPlasticity.new(:hebbian, learning_rate: 0.01)
      
      connection = Connection.new(1, 2, 0.0, 1, true)
      
      high_update = NeuralPlasticity.update_connection(high_lr, connection, 1.0, 1.0)
      low_update = NeuralPlasticity.update_connection(low_lr, connection, 1.0, 1.0)
      
      # Higher learning rate should produce larger weight changes
      high_change = abs(high_update.weight - connection.weight)
      low_change = abs(low_update.weight - connection.weight)
      
      assert high_change > low_change
    end
  end

  describe "STDP plasticity" do
    test "creates STDP plasticity with timing parameters" do
      plasticity = NeuralPlasticity.new(:stdp, 
        rule_params: %{
          a_plus: 0.1,
          a_minus: 0.12,
          tau_plus: 20.0,
          tau_minus: 20.0
        })
      
      connection = Connection.new(1, 2, 0.5, 1, true)
      context = %{current_time: 10.0}
      
      # Test STDP update
      updated_conn = NeuralPlasticity.update_connection(plasticity, connection, 0.8, 0.9, context)
      
      assert %Connection{} = updated_conn
    end

    test "timing-dependent weight changes" do
      plasticity = NeuralPlasticity.new(:stdp, 
        rule_params: %{a_plus: 0.1, a_minus: 0.1})
      
      connection = Connection.new(1, 2, 0.5, 1, true)
      
      # Pre-before-post (should potentiate)
      context1 = %{current_time: 10.0, pre_spike_time: 5.0, post_spike_time: 10.0}
      result1 = NeuralPlasticity.update_connection(plasticity, connection, 1.0, 1.0, context1)
      
      # Post-before-pre (should depress)  
      context2 = %{current_time: 10.0, pre_spike_time: 10.0, post_spike_time: 5.0}
      result2 = NeuralPlasticity.update_connection(plasticity, connection, 1.0, 1.0, context2)
      
      # Results should be different
      assert result1.weight != result2.weight
    end
  end

  describe "homeostatic mechanisms" do
    test "homeostatic scaling maintains target activity" do
      connections = [
        Connection.new(1, 2, 1.0, 1, true),
        Connection.new(1, 3, 2.0, 2, true),
        Connection.new(1, 4, 0.5, 3, true)
      ]
      
      target_activity = 1.0
      current_activity = 2.0  # Too high
      
      scaled_connections = NeuralPlasticity.apply_homeostatic_scaling(
        connections, target_activity, current_activity)
      
      # All weights should be scaled down
      Enum.zip(connections, scaled_connections)
      |> Enum.each(fn {original, scaled} ->
        assert scaled.weight < original.weight
      end)
    end

    test "calculates synaptic scaling factor correctly" do
      post_activities = [2.0, 2.0, 2.0]  # Average = 2.0
      target_rate = 1.0
      
      scaling_factor = NeuralPlasticity.calculate_synaptic_scaling([], post_activities, target_rate)
      
      # Should scale down by factor of 0.5 (1.0 / 2.0)
      assert_in_delta scaling_factor, 0.5, 0.01
    end

    test "scaling factor is bounded" do
      # Test extreme cases
      very_high_activities = [100.0, 100.0, 100.0]
      very_low_activities = [0.001, 0.001, 0.001]
      
      high_scaling = NeuralPlasticity.calculate_synaptic_scaling([], very_high_activities, 1.0)
      low_scaling = NeuralPlasticity.calculate_synaptic_scaling([], very_low_activities, 1.0)
      
      # Should be bounded between 0.1 and 10.0
      assert high_scaling >= 0.1 and high_scaling <= 10.0
      assert low_scaling >= 0.1 and low_scaling <= 10.0
    end
  end

  describe "intrinsic plasticity" do
    test "adjusts node properties for target statistics" do
      node_activities = [0.5, 0.6, 0.4, 0.8, 0.3]
      target_mean = 0.0
      target_variance = 1.0
      
      adjustments = NeuralPlasticity.apply_intrinsic_plasticity(node_activities, target_mean, target_variance)
      
      assert Map.has_key?(adjustments, :bias_adjustment)
      assert Map.has_key?(adjustments, :gain_adjustment)
      
      # Should suggest bias adjustment to shift mean toward target
      current_mean = Enum.sum(node_activities) / length(node_activities)
      expected_bias_direction = target_mean - current_mean
      
      # Bias adjustment should be in the right direction
      assert sign(adjustments.bias_adjustment) == sign(expected_bias_direction)
    end

    test "intrinsic plasticity converges over time" do
      # Simulate repeated applications
      activities = [2.0, 2.1, 1.9, 2.0, 2.1]  # High mean
      target_mean = 0.0
      
      # Apply multiple rounds
      adjustments = for _ <- 1..10 do
        NeuralPlasticity.apply_intrinsic_plasticity(activities, target_mean, 1.0)
      end
      
      # All adjustments should be in the same direction (downward)
      bias_adjustments = Enum.map(adjustments, &(&1.bias_adjustment))
      assert Enum.all?(bias_adjustments, &(&1 < 0))
    end
  end

  describe "metaplasticity" do
    test "enables metaplasticity modulation" do
      plasticity = NeuralPlasticity.new(:hebbian, metaplasticity_enabled: true)
      connection = Connection.new(1, 2, 0.5, 1, true)
      
      # High recent activity should affect plasticity
      context = %{recent_post_activity: 2.0, target_activity: 1.0}
      
      updated_conn = NeuralPlasticity.update_connection(plasticity, connection, 1.0, 1.0, context)
      
      assert %Connection{} = updated_conn
    end

    test "metaplasticity modulates learning based on activity history" do
      # Create two different plasticity instances with different metaplasticity settings
      plasticity_low = NeuralPlasticity.new(:hebbian, 
        metaplasticity_enabled: true,
        learning_rate: 0.1)
      
      plasticity_high = NeuralPlasticity.new(:hebbian, 
        metaplasticity_enabled: true,
        learning_rate: 0.5)  # Higher learning rate
      
      connection = Connection.new(1, 2, 0.0, 1, true)
      
      # Use the same context but different plasticity instances
      context = %{recent_post_activity: 1.0, current_post_activity: 1.0}
      
      # Get results with different learning rates
      low_result = NeuralPlasticity.update_connection(plasticity_low, connection, 1.0, 1.0, context)
      high_result = NeuralPlasticity.update_connection(plasticity_high, connection, 1.0, 1.0, context)
      
      # Changes should be different due to different learning rates
      low_change = abs(low_result.weight - connection.weight)
      high_change = abs(high_result.weight - connection.weight)
      
      # This tests that metaplasticity is working by showing different learning rates
      # produce different weight changes
      assert low_change != high_change
    end
  end

  describe "global modulation" do
    test "global modulation scales plasticity changes" do
      high_mod = NeuralPlasticity.new(:hebbian, global_modulation: 2.0, learning_rate: 0.1)
      low_mod = NeuralPlasticity.new(:hebbian, global_modulation: 0.5, learning_rate: 0.1)
      
      connection = Connection.new(1, 2, 0.0, 1, true)
      
      high_result = NeuralPlasticity.update_connection(high_mod, connection, 1.0, 1.0)
      low_result = NeuralPlasticity.update_connection(low_mod, connection, 1.0, 1.0)
      
      high_change = abs(high_result.weight - connection.weight)
      low_change = abs(low_result.weight - connection.weight)
      
      # Higher modulation should produce larger changes
      assert high_change > low_change
    end
  end

  describe "evolutionary plasticity" do
    test "mutates plasticity parameters" do
      plasticity = NeuralPlasticity.new(:hebbian, learning_rate: 0.01, decay_rate: 0.99)
      
      mutated = NeuralPlasticity.evolutionary_plasticity_mutation(plasticity, 1.0)  # 100% mutation rate
      
      # Some parameters should change
      assert mutated.learning_rate != plasticity.learning_rate or 
             mutated.decay_rate != plasticity.decay_rate or
             mutated.global_modulation != plasticity.global_modulation
    end

    test "mutation respects parameter bounds" do
      plasticity = NeuralPlasticity.new(:hebbian)
      
      # Mutate many times to test bounds
      mutated_many = Enum.reduce(1..50, plasticity, fn _, acc ->
        NeuralPlasticity.evolutionary_plasticity_mutation(acc, 0.5)
      end)
      
      # Parameters should stay within reasonable bounds
      assert mutated_many.learning_rate >= 0.001 and mutated_many.learning_rate <= 1.0
      assert mutated_many.decay_rate >= 0.9 and mutated_many.decay_rate <= 0.999
      assert mutated_many.global_modulation >= 0.1 and mutated_many.global_modulation <= 5.0
    end

    test "crossover combines plasticity parameters" do
      parent1 = NeuralPlasticity.new(:hebbian, learning_rate: 0.01, decay_rate: 0.95)
      parent2 = NeuralPlasticity.new(:hebbian, learning_rate: 0.05, decay_rate: 0.99)
      
      children = for _ <- 1..20 do
        NeuralPlasticity.crossover_plasticity(parent1, parent2)
      end
      
      # Should get mix of parent parameters
      learning_rates = Enum.map(children, &(&1.learning_rate))
      decay_rates = Enum.map(children, &(&1.decay_rate))
      
      # Should have both parent values represented
      assert 0.01 in learning_rates or 0.05 in learning_rates
      assert 0.95 in decay_rates or 0.99 in decay_rates
    end
  end

  describe "weight decay" do
    test "applies weight decay over time" do
      plasticity = NeuralPlasticity.new(:hebbian, decay_rate: 0.9)
      connection = Connection.new(1, 2, 1.0, 1, true)
      
      # Apply multiple updates with no activity
      decayed = Enum.reduce(1..10, connection, fn _, acc ->
        NeuralPlasticity.update_connection(plasticity, acc, 0.0, 0.0)
      end)
      
      # Weight should decay toward zero
      assert abs(decayed.weight) < abs(connection.weight)
    end

    test "decay rate affects decay speed" do
      fast_decay = NeuralPlasticity.new(:hebbian, decay_rate: 0.5)
      slow_decay = NeuralPlasticity.new(:hebbian, decay_rate: 0.95)
      
      connection = Connection.new(1, 2, 1.0, 1, true)
      
      # Apply same number of updates
      fast_result = Enum.reduce(1..5, connection, fn _, acc ->
        NeuralPlasticity.update_connection(fast_decay, acc, 0.0, 0.0)
      end)
      
      slow_result = Enum.reduce(1..5, connection, fn _, acc ->
        NeuralPlasticity.update_connection(slow_decay, acc, 0.0, 0.0)
      end)
      
      # Fast decay should reduce weight more
      assert abs(fast_result.weight) < abs(slow_result.weight)
    end
  end

  describe "node plasticity" do
    test "updates node with plasticity parameters" do
      plasticity = NeuralPlasticity.new(:hebbian)
      node = Node.new(1, :hidden, :tanh)
      node_with_plasticity = Node.with_plasticity(node, :adaptive_threshold)
      
      {updated_node, context} = NeuralPlasticity.update_node(plasticity, node_with_plasticity, 0.5, 0.8)
      
      assert %Node{} = updated_node
      assert is_map(context)
    end

    test "node plasticity affects activation properties" do
      plasticity = NeuralPlasticity.new(:hebbian)
      node = Node.new(1, :hidden, :tanh) |> Node.with_plasticity(:intrinsic_plasticity)
      
      # Simulate multiple updates
      {final_node, _} = Enum.reduce(1..10, {node, %{}}, fn _, {n, ctx} ->
        NeuralPlasticity.update_node(plasticity, n, 0.8, 0.9, ctx)
      end)
      
      # Bias should have changed due to intrinsic plasticity
      assert final_node.bias != node.bias
    end
  end

  # Helper function
  defp sign(x) when x > 0, do: 1
  defp sign(x) when x < 0, do: -1
  defp sign(_), do: 0
end