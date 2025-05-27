# BipedalWalker Example
#
# This script demonstrates how to use the NeuroEvolution library with the
# BipedalWalker environment from OpenAI Gym, which is a challenging continuous
# control task that requires coordinated movement.

alias NeuroEvolution.TWEANN.Genome
alias NeuroEvolution.Environments.GymAdapter

IO.puts("BipedalWalker Example")
IO.puts("====================")

# Environment configuration
env_name = "BipedalWalker-v3"  # Continuous action space environment
input_size = 24                # 24 state variables (lidar, joints, etc.)
output_size = 4                # 4 continuous actions (hip and knee torques)
generations = 30               # Number of generations to evolve
population_size = 100          # Larger population for complex continuous control
num_trials = 3                 # Average over multiple trials for more stable fitness

# Create a test genome with appropriate inputs/outputs for BipedalWalker
IO.puts("\nCreating test genome...")
genome = NeuroEvolution.TWEANN.Genome.new(
  input_size, 
  output_size,
  %{
    # Use tanh activation for outputs to get values in [-1, 1] range
    # which is what BipedalWalker expects
    activation_function: :tanh,
    # Add some hidden nodes for more complex processing
    hidden_nodes: 16
  }
)

# Evaluate the random genome on the BipedalWalker environment
IO.puts("\nEvaluating random genome on BipedalWalker...")
score = GymAdapter.evaluate(genome, env_name, 1)
IO.puts("Random genome score: #{score}")

# Create a population for evolution with appropriate parameters for BipedalWalker
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
      target_species_count: 15       # Aim for 15 species
    },
    # Mutation parameters tuned for continuous control
    mutation: %{
      weight_mutation_rate: 0.9,     # High rate of weight mutation
      weight_perturbation: 0.2,      # Smaller perturbations for fine control
      add_node_rate: 0.05,           # More topology mutation for complex behavior
      add_connection_rate: 0.1,      # More connections for complex behavior
      disable_connection_rate: 0.01  # Rarely disable connections
    },
    # Enable neural plasticity for adaptive learning
    plasticity: %{
      enabled: true,
      plasticity_type: :hebbian,     # Use Hebbian learning
      learning_rate: 0.01            # Low learning rate for stability
    },
    # Initial network topology
    initial_topology: %{
      hidden_layers: [16, 8],        # Two hidden layers
      activation_function: :tanh,    # Tanh activation for continuous outputs
      fully_connected: true,         # Start with fully connected network
      recurrent: true                # Allow recurrent connections for memory
    }
  }
)

# Define a fitness function that evaluates each genome on multiple trials
IO.puts("\nDefining fitness function...")
fitness_fn = fn genome ->
  # Average over multiple trials for more stable fitness evaluation
  GymAdapter.evaluate(genome, env_name, num_trials)
end

# Evolve the population on the BipedalWalker environment
IO.puts("\nEvolving population on BipedalWalker (#{generations} generations)...")
IO.puts("This will take some time due to the complexity of the environment...")

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
  
  # Save the best genome every 5 generations
  if rem(gen, 5) == 0 do
    best_genome = Enum.max_by(new_pop.genomes, fn g -> g.fitness || 0 end)
    save_path = "bipedal_walker_gen_#{gen}.json"
    genome_json = NeuroEvolution.serialize(best_genome)
    File.write!(save_path, genome_json)
    IO.puts("Saved best genome to #{save_path}")
  end
  
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
  # Scale the bar based on fitness (may need adjustment for negative values)
  normalized_fitness = max(0, gen_stats.best_fitness + 300) / 400 * 50
  bar_length = trunc(normalized_fitness)
  bar = String.duplicate("█", max(0, bar_length))
  IO.puts("Gen #{String.pad_leading(Integer.to_string(gen), 2)}: #{String.pad_leading(Float.to_string(Float.round(gen_stats.best_fitness, 1)), 6)} #{bar}")
end)

# Save the best genome
final_save_path = "bipedal_walker_best.json"
final_genome_json = NeuroEvolution.serialize(best_genome)
File.write!(final_save_path, final_genome_json)
IO.puts("\nSaved best genome to #{final_save_path}")

# Visualize the best genome on BipedalWalker
IO.puts("\nVisualizing best genome on BipedalWalker...")
IO.puts("(Watch the robot try to walk across the terrain)")
GymAdapter.visualize(best_genome, env_name, 3)

IO.puts("\nExample complete!")
