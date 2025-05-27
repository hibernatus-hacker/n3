#!/usr/bin/env elixir

# Gym Integration Example - Demonstrates multi-environment evolution
#
# This example shows how to use the NeuroEvolution library with multiple
# OpenAI Gym environments through the secure Python bridge.

defmodule GymIntegrationExample do
  def run do
    IO.puts("🏋️  NeuroEvolution - Gym Integration Example")
    IO.puts("==========================================\n")

    # Start secure Python bridge
    IO.puts("🔗 Starting secure Python bridge...")
    case NeuroEvolution.Environments.PythonBridge.start_link() do
      {:ok, _bridge} -> 
        IO.puts("   ✅ Python bridge started successfully")
      {:error, reason} -> 
        IO.puts("   ❌ Failed to start Python bridge: #{inspect(reason)}")
        IO.puts("   💡 Make sure Python and gym are installed")
        return
    end

    # Test environments with their configurations
    environments = [
      %{
        name: "CartPole-v1",
        inputs: 4,
        outputs: 2,
        target_score: 400,
        description: "Balance pole on cart"
      },
      %{
        name: "MountainCar-v0", 
        inputs: 2,
        outputs: 3,
        target_score: -120,
        description: "Drive car up mountain"
      },
      %{
        name: "Acrobot-v1",
        inputs: 6,
        outputs: 3, 
        target_score: -100,
        description: "Swing pendulum upward"
      }
    ]

    # Test each environment
    results = Enum.map(environments, fn env ->
      test_environment(env)
    end)

    # Summary
    IO.puts("\n📊 Environment Test Summary:")
    IO.puts("=" <> String.duplicate("=", 28))
    
    Enum.zip(environments, results) 
    |> Enum.each(fn {env, result} ->
      status = if result.success, do: "✅", else: "⚠️"
      IO.puts("#{status} #{env.name}: Best score #{Float.round(result.best_score, 1)} (target: #{env.target_score})")
    end)

    # Demonstrate cross-environment transfer learning
    IO.puts("\n🔄 Testing cross-environment transfer...")
    transfer_learning_demo(environments)

    # Stop Python bridge
    NeuroEvolution.Environments.PythonBridge.stop()
    IO.puts("\n🎯 Gym integration example complete!")
  end

  defp test_environment(env) do
    IO.puts("\n🎮 Testing #{env.name} - #{env.description}")
    IO.puts("   📊 Input dims: #{env.inputs}, Output dims: #{env.outputs}")

    # Create fitness function for this environment
    fitness_fn = fn genome ->
      try do
        # Evaluate on environment with error handling
        score = NeuroEvolution.Environments.GymAdapter.evaluate(
          genome, 
          env.name, 
          episodes: 3,
          max_steps: 500
        )
        
        # Handle different scoring systems (some envs use negative scores)
        if env.target_score < 0 do
          score  # Keep negative scores as-is for environments like MountainCar
        else
          max(score, 0.0)  # Ensure positive for environments like CartPole
        end
      rescue
        error -> 
          IO.puts("     ⚠️  Evaluation error: #{inspect(error)}")
          if env.target_score < 0, do: -1000.0, else: 0.0
      end
    end

    # Create small population for quick testing
    IO.puts("   🧬 Creating population (15 individuals)...")
    population = NeuroEvolution.new_population(15, env.inputs, env.outputs)

    # Quick evolution test
    IO.puts("   🚀 Running evolution (10 generations)...")
    try do
      evolved_population = NeuroEvolution.evolve(population, fitness_fn, 
        generations: 10,
        target_fitness: env.target_score * 0.8  # Stop at 80% of target
      )

      best_genome = NeuroEvolution.get_best_genome(evolved_population)
      best_score = NeuroEvolution.get_fitness(best_genome) || 0.0
      
      success = if env.target_score < 0 do
        best_score >= env.target_score * 0.8  # For negative scoring
      else
        best_score >= env.target_score * 0.8  # For positive scoring
      end

      IO.puts("   🏆 Best score: #{Float.round(best_score, 1)}")
      
      if success do
        IO.puts("   🎉 Success! Reached target performance.")
      else
        IO.puts("   🔄 Needs more training to reach target.")
      end

      %{success: success, best_score: best_score, best_genome: best_genome}

    rescue
      error ->
        IO.puts("   ❌ Evolution failed: #{inspect(error)}")
        %{success: false, best_score: 0.0, best_genome: nil}
    end
  end

  defp transfer_learning_demo(environments) do
    IO.puts("🔄 Demonstrating transfer learning between environments...")
    
    # Train on CartPole first
    cartpole_env = Enum.find(environments, fn env -> env.name == "CartPole-v1" end)
    
    if cartpole_env do
      IO.puts("   🎯 Training base network on #{cartpole_env.name}...")
      
      # Create and train base network
      base_population = NeuroEvolution.new_population(10, cartpole_env.inputs, cartpole_env.outputs)
      
      cartpole_fitness_fn = fn genome ->
        try do
          NeuroEvolution.Environments.GymAdapter.evaluate(genome, cartpole_env.name, episodes: 2)
        rescue
          _ -> 0.0
        end
      end
      
      trained_population = NeuroEvolution.evolve(base_population, cartpole_fitness_fn, generations: 5)
      base_network = NeuroEvolution.get_best_genome(trained_population)
      
      IO.puts("   📊 Base network performance: #{Float.round(NeuroEvolution.get_fitness(base_network) || 0.0, 1)}")
      
      # Test transfer to other environments
      other_envs = Enum.filter(environments, fn env -> env.name != "CartPole-v1" end)
      
      Enum.each(other_envs, fn target_env ->
        IO.puts("   🎯 Transferring to #{target_env.name}...")
        
        # Adapt network architecture if needed
        adapted_network = adapt_network_architecture(base_network, target_env.inputs, target_env.outputs)
        
        # Test performance
        transfer_score = try do
          NeuroEvolution.Environments.GymAdapter.evaluate(adapted_network, target_env.name, episodes: 2)
        rescue
          _ -> if target_env.target_score < 0, do: -500.0, else: 0.0
        end
        
        IO.puts("     📈 Transfer performance: #{Float.round(transfer_score, 1)}")
      end)
    else
      IO.puts("   ⚠️  CartPole environment not available for transfer learning demo")
    end
  end

  defp adapt_network_architecture(base_genome, target_inputs, target_outputs) do
    # Simple adaptation: adjust input/output dimensions
    # In practice, this could be more sophisticated
    if length(base_genome.inputs) == target_inputs and length(base_genome.outputs) == target_outputs do
      base_genome
    else
      # Create new genome with target dimensions but transfer some weights
      NeuroEvolution.new_genome(target_inputs, target_outputs)
    end
  end
end

# Run the example
GymIntegrationExample.run()