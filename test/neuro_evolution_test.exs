defmodule NeuroEvolutionTest do
  use ExUnit.Case
  doctest NeuroEvolution

  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Population.Population

  test "creates basic population" do
    population = NeuroEvolution.new_population(10, 2, 1)
    
    assert population.population_size == 10
    assert length(population.genomes) == 10
    assert population.generation == 0
    
    # Check first genome structure
    genome = List.first(population.genomes)
    assert length(genome.inputs) == 2
    assert length(genome.outputs) == 1
    assert map_size(genome.nodes) >= 3  # At least inputs + outputs
  end

  test "creates genome with substrate" do
    substrate_config = %{
      geometry_type: :grid,
      dimensions: 2,
      resolution: {3, 3}
    }
    
    genome = Genome.new(2, 1, substrate: substrate_config)
    
    assert genome.substrate_config == substrate_config
    assert length(genome.inputs) == 2
    assert length(genome.outputs) == 1
  end

  test "creates genome with plasticity" do
    plasticity_config = %{
      plasticity_type: :hebbian,
      learning_rate: 0.01
    }
    
    genome = Genome.new(2, 1, plasticity: plasticity_config)
    
    assert genome.plasticity_config == plasticity_config
  end

  test "evaluates single genome" do
    genome = Genome.new(2, 1)
    inputs = [0.5, 0.8]
    
    outputs = NeuroEvolution.evaluate_genome(genome, inputs)
    
    assert is_list(outputs)
    assert length(outputs) >= 1
    assert Enum.all?(outputs, &is_number/1)
  end

  test "calculates genome distance" do
    genome1 = Genome.new(2, 1)
    genome2 = Genome.new(2, 1)
    
    distance = Genome.distance(genome1, genome2)
    
    assert is_float(distance)
    assert distance >= 0.0
  end

  test "performs genome crossover" do
    parent1 = Genome.new(2, 1)
    parent2 = Genome.new(2, 1)
    
    child = Genome.crossover(parent1, parent2)
    
    assert child.inputs == parent1.inputs
    assert child.outputs == parent1.outputs
    assert child.generation >= max(parent1.generation, parent2.generation)
  end

  test "mutates genome weights" do
    original_genome = Genome.new(2, 1)
    
    # Add some connections first
    genome_with_connections = Genome.add_connection(original_genome, 1, 3)
    
    mutated_genome = Genome.mutate_weights(genome_with_connections)
    
    # Should still have same structure
    assert map_size(mutated_genome.nodes) == map_size(genome_with_connections.nodes)
    assert mutated_genome.inputs == genome_with_connections.inputs
    assert mutated_genome.outputs == genome_with_connections.outputs
  end

  test "creates HyperNEAT system" do
    hyperneat = NeuroEvolution.new_hyperneat([3, 3], [2, 2], [1, 1])
    
    assert hyperneat.substrate
    assert hyperneat.cppn
    assert hyperneat.connection_threshold > 0
  end

  test "creates substrate with different geometries" do
    grid_substrate = NeuroEvolution.Substrate.Substrate.grid_2d(5, 5)
    
    assert grid_substrate.geometry_type == :grid
    assert grid_substrate.dimensions == 2
    assert length(grid_substrate.input_positions) > 0
    assert length(grid_substrate.output_positions) > 0
  end

  test "creates neural plasticity" do
    plasticity = NeuroEvolution.new_plasticity(:hebbian, learning_rate: 0.02)
    
    assert plasticity.plasticity_type == :hebbian
    assert plasticity.learning_rate == 0.02
  end

  test "gets population statistics" do
    population = NeuroEvolution.new_population(5, 2, 1)
    
    stats = NeuroEvolution.get_population_stats(population)
    
    assert stats.generation == 0
    assert stats.population_size == 5
    assert stats.species_count == 0  # No species assigned yet
    assert is_map(stats.diversity_metrics)
  end

  test "XOR fitness function" do
    fitness_fn = NeuroEvolution.xor_fitness()
    genome = Genome.new(2, 1)
    
    fitness = fitness_fn.(genome)
    
    assert is_number(fitness)
    assert fitness >= 0.0
  end
end