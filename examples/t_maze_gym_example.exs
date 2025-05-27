# T-Maze Gym-like Interface Example
#
# This script demonstrates how to use the NeuroEvolution library with the T-Maze
# environment through a Gym-like interface.

alias NeuroEvolution.TWEANN.Genome
alias NeuroEvolution.Environments.Gym.GymAdapter
alias NeuroEvolution.Environments.Gym.TMazeEnv
alias NeuroEvolution.Environments.TMaze

IO.puts("T-Maze Gym-like Interface Example")
IO.puts("=================================")

# Create a test genome with appropriate inputs/outputs for T-Maze
IO.puts("\nCreating test genome...")
genome = NeuroEvolution.new_genome(3, 2)

# Evaluate the genome on the T-Maze environment
IO.puts("\nEvaluating random genome on T-Maze...")
score = GymAdapter.evaluate_genome_tmaze(genome, 5)
IO.puts("Random genome score: #{score}")

# Demonstrate the Gym-like interface directly
IO.puts("\nDemonstrating Gym-like interface...")
env = TMazeEnv.new()
{observation, env} = TMazeEnv.reset(env)
IO.puts("Initial observation: #{inspect(observation)}")
IO.puts("\nInitial maze state:")
IO.puts(TMaze.render_maze(env.position, env.reward_location))

# Take a few steps in the environment
IO.puts("\nTaking steps in the environment...")
{_, _, _, _, env} = TMazeEnv.step(env, 0) # Move forward
IO.puts("\nAfter step 1:")
IO.puts(TMaze.render_maze(env.position, env.reward_location))

{_, _, _, _, env} = TMazeEnv.step(env, 0) # Move forward
IO.puts("\nAfter step 2:")
IO.puts(TMaze.render_maze(env.position, env.reward_location))

{_, reward, done, info, _} = TMazeEnv.step(env, 0) # Choose left
IO.puts("\nAfter final step (choosing left):")
IO.puts("Reward: #{reward}, Done: #{done}")
IO.puts("Info: #{inspect(info)}")

# Create a small population for evolution
IO.puts("\nCreating population for evolution...")
population = NeuroEvolution.new_population(20, 3, 2, 
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

# Evolve the population on the T-Maze environment
IO.puts("\nEvolving population on T-Maze (5 generations)...")
{evolved_pop, stats} = GymAdapter.evolve_on_tmaze(population, 5, 10)

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

IO.puts("\nExample complete!")
