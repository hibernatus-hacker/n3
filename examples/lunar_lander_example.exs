# Lunar Lander Example
#
# This script demonstrates how to use the NeuroEvolution library with the
# Lunar Lander environment from OpenAI Gym, which is a more complex control
# task than CartPole.

alias NeuroEvolution.TWEANN.Genome
alias NeuroEvolution.Environments.GymAdapter

IO.puts("Lunar Lander Example")
IO.puts("===================")

# Environment configuration
env_name = "LunarLander-v2"  # Discrete action space version
input_size = 8               # 8 state variables (position, velocity, angle, etc.)
output_size = 4              # 4 actions (do nothing, fire left, fire main, fire right)
generations = 20             # Number of generations to evolve
population_size = 50         # Larger population for more complex task
num_trials = 3               # Average over multiple trials for more stable fitness

# Create a test genome with appropriate inputs/outputs for Lunar Lander
IO.puts("\nCreating test genome...")
genome = NeuroEvolution.TWEANN.Genome.new(input_size, output_size)

# Evaluate the random genome on the Lunar Lander environment
IO.puts("\nEvaluating random genome on Lunar Lander...")
score = GymAdapter.evaluate(genome, env_name, 1)
IO.puts("Random genome score: #{score}")

# Create a population for evolution with appropriate parameters for Lunar Lander
IO.puts("\nCreating population for evolution...")
population = NeuroEvolution.Population.Population.new(
  population_size, 
  input_size, 
  output_size, 
  %{
    # Use speciation to maintain diversity
    speciation: %{
      enabled: true, 
      compatibility_threshold: 3.0,  # Higher threshold for more complex networks
      target_species_count: 10       # Aim for 10 species
    },
    # Mutation parameters tuned for Lunar Lander
    mutation: %{
      weight_mutation_rate: 0.8,     # High rate of weight mutation
      weight_perturbation: 0.3,      # Smaller perturbations for fine control
      add_node_rate: 0.03,           # Occasionally add nodes
      add_connection_rate: 0.05,     # Occasionally add connections
      disable_connection_rate: 0.01  # Rarely disable connections
    },
    # Enable neural plasticity for adaptive learning
    plasticity: %{
      enabled: true,
      plasticity_type: :hebbian,     # Use Hebbian learning
      learning_rate: 0.01            # Low learning rate for stability
    }
  }
)

# Define a fitness function that evaluates each genome on multiple trials
IO.puts("\nDefining fitness function...")
fitness_fn = fn genome ->
  # Average over multiple trials for more stable fitness evaluation
  GymAdapter.evaluate(genome, env_name, num_trials)
end

# Evolve the population on the Lunar Lander environment
IO.puts("\nEvolving population on Lunar Lander (#{generations} generations)...")
IO.puts("This may take some time...")

# Track progress with a progress bar
total_evaluations = generations * population_size
evaluated = 0

# Evolution loop with progress tracking
{evolved_pop, stats} = Enum.reduce(1..generations, {population, []}, fn gen, {pop, stats_acc} ->
  # Evolve for one generation
  {new_pop, gen_stats} = NeuroEvolution.evolve(pop, fitness_fn, 1)
  
  # Update progress
  evaluated = evaluated + population_size
  progress_percent = Float.round(evaluated / total_evaluations * 100, 1)
  
  # Print generation statistics
  IO.puts("Generation #{gen}/#{generations} (#{progress_percent}%): " <>
          "Best=#{Float.round(gen_stats.best_fitness, 1)}, " <>
          "Avg=#{Float.round(gen_stats.avg_fitness, 1)}, " <>
          "Species=#{length(gen_stats.species)}")
  
  # Return updated population and stats
  {new_pop, stats_acc ++ [gen_stats]}
end)

# Get the best genome from the evolved population
best_genome = Enum.max_by(evolved_pop.genomes, fn genome -> genome.fitness || 0 end)
best_fitness = best_genome.fitness

IO.puts("\nEvolution complete!")
IO.puts("Best genome fitness: #{best_fitness}")

# Print evolution statistics
IO.puts("\nEvolution Statistics:")
IO.puts("====================")
IO.puts("Initial best fitness: #{List.first(stats).best_fitness}")
IO.puts("Final best fitness: #{List.last(stats).best_fitness}")
IO.puts("Improvement: #{List.last(stats).best_fitness - List.first(stats).best_fitness}")

# Plot fitness over generations
IO.puts("\nFitness progression over generations:")
IO.puts("----------------------------------")
Enum.with_index(stats, 1) |> Enum.each(fn {gen_stats, gen} ->
  bar_length = trunc(gen_stats.best_fitness * 2)
  bar = String.duplicate("█", max(0, bar_length))
  IO.puts("Gen #{String.pad_leading(Integer.to_string(gen), 2)}: #{String.pad_leading(Float.to_string(Float.round(gen_stats.best_fitness, 1)), 6)} #{bar}")
end)

# Visualize the best genome on Lunar Lander
IO.puts("\nVisualizing best genome on Lunar Lander...")
IO.puts("(Watch the spacecraft try to land between the flags)")
GymAdapter.visualize(best_genome, env_name, 3)

IO.puts("\nExample complete!")
