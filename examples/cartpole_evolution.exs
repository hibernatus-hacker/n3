#!/usr/bin/env elixir

# CartPole Evolution Example - Demonstrates neuroevolution for the CartPole task
#
# This example shows how to evolve neural networks to solve the CartPole control problem
# using the NeuroEvolution library.

defmodule CartPoleEvolution do
  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Environments.CartPole
  
  def run do
    IO.puts("🧠 NeuroEvolution - CartPole Evolution Example")
    IO.puts("======================================================\n")
    
    # Start Python bridge for gym environment
    IO.puts("🔗 Starting Python bridge...")
    {:ok, _bridge} = NeuroEvolution.Environments.PythonBridge.start_link()
    
    # Step 1: Create initial population
    population_size = 50
    IO.puts("\n🧬 Step 1: Creating initial population (size: #{population_size})...")
    
    # Create a population with 4 inputs (cart position, velocity, pole angle, pole velocity)
    # and 2 outputs (left, right)
    population = NeuroEvolution.new_population(population_size, 4, 2, [
      # Network topology options
      add_node_rate: 0.03,
      add_connection_rate: 0.05,
      
      # Speciation options
      compatibility_threshold: 3.0,
      species_target: 5,
      
      # Optional: Add plasticity for adaptive learning
      plasticity: %{
        plasticity_type: :hebbian,
        learning_rate: 0.01
      }
    ])
    
    IO.puts("   📊 Population structure:")
    IO.puts("     • Size: #{population_size} genomes")
    IO.puts("     • Inputs: 4 (cart position, cart velocity, pole angle, pole velocity)")
    IO.puts("     • Outputs: 2 (left, right)")
    
    # Step 2: Define fitness function
    IO.puts("\n🎯 Step 2: Defining fitness function...")
    
    fitness_fn = fn genome ->
      try do
        # Evaluate genome on CartPole for 3 trials, take average
        scores = for _trial <- 1..3 do
          CartPole.evaluate(genome, 1)
        end
        
        Enum.sum(scores) / length(scores)
      rescue
        _ -> 0.0  # Return 0 fitness if evaluation fails
      end
    end
    
    # Step 3: Evolve the population
    num_generations = 20
    IO.puts("\n🧬 Step 3: Evolving population for #{num_generations} generations...")
    
    # Track progress over generations
    generation_stats = []
    
    # Evolve the population
    evolved_population = NeuroEvolution.evolve(population, fitness_fn, [
      generations: num_generations,
      selection_strategy: :tournament,
      tournament_size: 3,
      elitism: 2,
      mutation_rate: 0.3,
      crossover_rate: 0.7,
      target_fitness: 195.0  # Stop early if we reach this fitness
    ])
    
    # Step 4: Evaluate the best genome
    IO.puts("\n🏆 Step 4: Evaluating the best evolved genome...")
    
    best_genome = evolved_population.best_genome
    best_fitness = evolved_population.best_fitness
    
    # Run 5 evaluation trials with the best genome
    evaluation_scores = for trial <- 1..5 do
      score = CartPole.evaluate(best_genome, 1)
      IO.puts("   Trial #{trial}: #{score} steps")
      score
    end
    
    avg_score = Enum.sum(evaluation_scores) / length(evaluation_scores)
    
    # Step 5: Display results
    IO.puts("\n📊 Step 5: Evolution results:")
    IO.puts("   • Best fitness: #{Float.round(best_fitness || 0.0, 1)} steps")
    IO.puts("   • Average evaluation score: #{Float.round(avg_score, 1)} steps")
    IO.puts("   • Number of generations: #{evolved_population.generation}")
    IO.puts("   • Number of species: #{length(evolved_population.species)}")
    IO.puts("   • Network complexity: #{map_size(best_genome.nodes)} nodes, #{map_size(best_genome.connections)} connections")
    
    # Step 6: Analyze the best genome
    IO.puts("\n🔍 Step 6: Analyzing the best genome:")
    
    # Count hidden nodes
    hidden_nodes = Enum.count(best_genome.nodes, fn {_id, node} -> 
      node.type == :hidden
    end)
    
    # Count enabled connections
    enabled_connections = Enum.count(best_genome.connections, fn {_id, conn} -> 
      conn.enabled
    end)
    
    IO.puts("   • Hidden nodes: #{hidden_nodes}")
    IO.puts("   • Enabled connections: #{enabled_connections}")
    IO.puts("   • Disabled connections: #{map_size(best_genome.connections) - enabled_connections}")
    
    # Clean up
    NeuroEvolution.Environments.PythonBridge.stop()
    
    IO.puts("\n✅ CartPole evolution example complete!")
    
    if best_fitness && best_fitness > 195 do
      IO.puts("🎉 Successfully evolved a controller that solves the CartPole task!")
    else
      IO.puts("🔄 Evolution made progress but didn't fully solve the task. Try increasing generations or population size.")
    end
  end
end

# Run the example
CartPoleEvolution.run()
