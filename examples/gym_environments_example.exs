# Gym Environments Example
#
# This script demonstrates how to use the NeuroEvolution library with various
# OpenAI Gym environments using the generic GymAdapter.

alias NeuroEvolution.TWEANN.Genome
alias NeuroEvolution.Environments.GymAdapter

IO.puts("Gym Environments Example")
IO.puts("=======================")

# List of environments to test
environments = [
  {"CartPole-v1", 4, 2},  # 4 inputs, 2 outputs
  {"MountainCar-v0", 2, 3},  # 2 inputs, 3 outputs
  {"Acrobot-v1", 6, 3}  # 6 inputs, 3 outputs
]

# Test each environment
Enum.each(environments, fn {env_name, inputs, outputs} ->
  IO.puts("\n\n=== Testing #{env_name} ===")
  
  # Create a test genome with appropriate inputs/outputs
  IO.puts("\nCreating test genome...")
  genome = NeuroEvolution.TWEANN.Genome.new(inputs, outputs)
  
  # Evaluate the random genome on the environment
  IO.puts("\nEvaluating random genome on #{env_name}...")
  score = GymAdapter.evaluate(genome, env_name, 1)
  IO.puts("Random genome score: #{score}")
  
  # Create a small population for evolution
  IO.puts("\nCreating population for evolution...")
  population = NeuroEvolution.Population.Population.new(20, inputs, outputs, 
    %{
      speciation: %{enabled: true, compatibility_threshold: 1.0},
      mutation: %{
        weight_mutation_rate: 0.8,
        weight_perturbation: 0.5,
        add_node_rate: 0.03,
        add_connection_rate: 0.05
      }
    }
  )
  
  # Evolve the population on the environment (just 2 generations for demonstration)
  IO.puts("\nEvolving population on #{env_name} (2 generations)...")
  {evolved_pop, stats} = GymAdapter.evolve(population, env_name, 2)
  
  # Get the best genome from the evolved population
  best_genome = Enum.max_by(evolved_pop.genomes, fn genome -> genome.fitness || 0 end)
  best_fitness = best_genome.fitness
  
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
  
  # Visualize the best genome on the environment
  IO.puts("\nVisualizing best genome on #{env_name}...")
  GymAdapter.visualize(best_genome, env_name, 1)
end)

IO.puts("\nExample complete!")
