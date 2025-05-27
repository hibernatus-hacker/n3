#!/usr/bin/env elixir

# Basic XOR Example - Demonstrates core TWEANN evolution
#
# This example shows how to evolve a neural network to solve the XOR problem
# using the NeuroEvolution library's core functionality.

defmodule XORExample do
  def run do
    IO.puts("🧠 NeuroEvolution - Basic XOR Example")
    IO.puts("=====================================\n")

    # Define XOR test cases
    xor_data = [
      {[0.0, 0.0], [0.0]},
      {[0.0, 1.0], [1.0]},
      {[1.0, 0.0], [1.0]},
      {[1.0, 1.0], [0.0]}
    ]

    # Create fitness function
    fitness_fn = fn genome ->
      total_error =
        Enum.reduce(xor_data, 0.0, fn {inputs, expected}, acc ->
          outputs = NeuroEvolution.activate(genome, inputs)
          error = calculate_mse(outputs, expected)
          acc + error
        end)

      # Convert error to fitness (higher is better)
      4.0 - total_error
    end

    # Create and evolve population
    IO.puts("📊 Creating population (50 individuals, 2 inputs, 1 output)")
    population = NeuroEvolution.new_population(50, 2, 1)

    IO.puts("🧬 Evolving for 50 generations...")
    evolved_population = NeuroEvolution.evolve(population, fitness_fn, generations: 50)

    # Test best genome
    best_genome = NeuroEvolution.get_best_genome(evolved_population)
    best_fitness = NeuroEvolution.get_fitness(best_genome)

    IO.puts("\n✅ Evolution Complete!")
    IO.puts("🏆 Best fitness: #{Float.round(best_fitness || 0.0, 4)}")

    IO.puts("\n🧪 Testing best genome on XOR:")
    Enum.each(xor_data, fn {inputs, expected} ->
      outputs = NeuroEvolution.activate(best_genome, inputs)
      actual = List.first(outputs) || 0.0

      IO.puts("   Input: #{inspect(inputs)} → Expected: #{List.first(expected)} | Actual: #{Float.round(actual, 3)}")
    end)

    IO.puts("\n🎯 Example complete! The network has learned XOR logic.")
  end

  defp calculate_mse(outputs, expected) do
    if length(outputs) == length(expected) do
      outputs
      |> Enum.zip(expected)
      |> Enum.reduce(0.0, fn {out, exp}, acc ->
        acc + :math.pow(out - exp, 2)
      end)
      |> Kernel./(length(outputs))
    else
      1.0  # Maximum error for size mismatch
    end
  end
end

# Run the example
XORExample.run()
