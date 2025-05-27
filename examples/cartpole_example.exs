#!/usr/bin/env elixir

# CartPole Example - Demonstrates control task with gym environment
#
# This example shows how to evolve controllers for the CartPole balancing task
# using the NeuroEvolution library with neural plasticity for adaptation.


defmodule CartPoleExample do
  def run do
    IO.puts("🎯 NeuroEvolution - CartPole Control Example")
    IO.puts("==========================================\n")

    # Start Python bridge for gym environment
    IO.puts("🔗 Starting Python bridge...")
    {:ok, _bridge} = NeuroEvolution.Environments.PythonBridge.start_link()

    # Create fitness function for CartPole
    fitness_fn = fn genome ->
      try do
        # Evaluate genome on CartPole for 3 trials, take average
        scores = for _trial <- 1..3 do
          NeuroEvolution.Environments.CartPole.evaluate(genome, max_steps: 500)
        end

        Enum.sum(scores) / length(scores)
      rescue
        _ -> 0.0  # Return 0 fitness if evaluation fails
      end
    end

    # Create population with plasticity for adaptive control
    IO.puts("🧠 Creating population with neural plasticity...")
    plasticity_config = %{
      plasticity_type: :hebbian,
      learning_rate: 0.05,
      modulation_strength: 0.8
    }

    population = NeuroEvolution.new_population(30, 4, 2,
      plasticity: plasticity_config
    )

    # Evolve the population
    IO.puts("🚀 Evolving CartPole controllers (20 generations)...")

    evolved_population = NeuroEvolution.evolve(population, fitness_fn,
      generations: 20,
      target_fitness: 450.0  # Stop early if we achieve good performance
    )

    # Test best controller
    best_genome = NeuroEvolution.get_best_genome(evolved_population)
    best_fitness = NeuroEvolution.get_fitness(best_genome)

    IO.puts("\n✅ Evolution Complete!")
    IO.puts("🏆 Best fitness: #{Float.round(best_fitness || 0.0, 1)} steps")

    if best_fitness && best_fitness > 200 do
      IO.puts("🎉 Success! Controller learned to balance CartPole effectively.")

      IO.puts("\n🎬 Running visualization...")
      try do
        NeuroEvolution.Environments.CartPole.visualize(best_genome)
        IO.puts("✨ Visualization saved as cartpole_animation.gif")
      rescue
        e -> IO.puts("⚠️  Visualization failed: #{inspect(e)}")
      end
    else
      IO.puts("🔄 Controller needs more training. Try increasing generations or population size.")
    end

    # Stop Python bridge
    NeuroEvolution.Environments.PythonBridge.stop()

    IO.puts("\n🎯 Example complete!")
  end
end

# Run the example
CartPoleExample.run()
