# Simple CartPole Example
#
# This script demonstrates how to use the NeuroEvolution library with the CartPole
# environment using a direct Python interface for visualization.

alias NeuroEvolution.Environments.Gym.SimpleCartPole

IO.puts("Simple CartPole Example")
IO.puts("======================")

# Create a test genome with appropriate inputs/outputs for CartPole
# CartPole has 4 inputs (cart position, cart velocity, pole angle, pole velocity) and 2 outputs (left, right)
IO.puts("\nCreating test genome...")
genome = NeuroEvolution.new_genome(4, 2)

# Evaluate the genome on the CartPole environment
IO.puts("\nEvaluating random genome on CartPole...")
score = SimpleCartPole.evaluate_genome(genome, 1)
IO.puts("Random genome score: #{score}")

# Create a small population for evolution
IO.puts("\nCreating population for evolution...")
population = NeuroEvolution.new_population(20, 4, 2, 
  %{
    speciation: %{enabled: true, compatibility_threshold: 1.0},
    mutation: %{
      weight_mutation_rate: 0.8,
      weight_perturbation: 0.5,
      add_node_rate: 0.03,
      add_connection_rate: 0.05
    },
    plasticity: %{
      enabled: true,
      plasticity_type: :hebbian,
      learning_rate: 0.1,
      modulation_enabled: true
    }
  }
)

# Evolve the population on the CartPole environment
# Using a small number of generations for demonstration
IO.puts("\nEvolving population on CartPole (3 generations)...")
{evolved_pop, stats} = SimpleCartPole.evolve_population(population, 3, 2, true)

# Get the best genome from the evolved population
best_genome = NeuroEvolution.get_best_genome(evolved_pop)
best_fitness = NeuroEvolution.get_fitness(best_genome)

IO.puts("\nEvolution complete!")
IO.puts("Best genome fitness: #{best_fitness}")

# Print evolution statistics
IO.puts("\nEvolution Statistics:")
IO.puts("====================")
Enum.with_index(stats, 1) |> Enum.each(fn {gen_stats, gen} ->
  IO.puts("Generation #{gen}:")
  IO.puts("  Best Fitness: #{gen_stats.best_fitness}")
  IO.puts("  Average Fitness: #{gen_stats.avg_fitness}")
  IO.puts("  Number of Species: #{length(gen_stats.species)}")
end)

# Visualize the best genome on CartPole for a longer episode
IO.puts("\nVisualizing best genome on CartPole...")
SimpleCartPole.evaluate_genome(best_genome, 1, true)

IO.puts("\nExample complete!")
