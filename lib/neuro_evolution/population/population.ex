defmodule NeuroEvolution.Population.Population do
  @moduledoc """
  Population management for TWEANN evolution with speciation, fitness sharing,
  and adaptive population dynamics.
  """

  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Population.{Species, Selection}

  defstruct [
    :genomes,
    :species,
    :generation,
    :population_size,
    :best_genome,
    :best_fitness,
    :avg_fitness,
    :diversity_metrics,
    :stagnation_counter,
    :config
  ]

  @type t :: %__MODULE__{
    genomes: [Genome.t()],
    species: [Species.t()],
    generation: integer(),
    population_size: integer(),
    best_genome: Genome.t() | nil,
    best_fitness: float() | nil,
    avg_fitness: float(),
    diversity_metrics: map(),
    stagnation_counter: integer(),
    config: map()
  }

  def new(population_size, input_count, output_count, opts \\ []) do
    config = Map.merge(default_config(), Map.new(opts))
    
    # Create initial population
    genomes = create_initial_population(population_size, input_count, output_count, config)
    
    %__MODULE__{
      genomes: genomes,
      species: [],
      generation: 0,
      population_size: population_size,
      best_genome: nil,
      best_fitness: nil,
      avg_fitness: 0.0,
      diversity_metrics: %{},
      stagnation_counter: 0,
      config: config
    }
  end

  def evolve(%__MODULE__{} = population, fitness_fn) do
    population
    |> evaluate_fitness(fitness_fn)
    |> update_statistics()
    |> speciate()
    |> calculate_fitness_sharing()
    |> select_and_reproduce()
    |> mutate_population()
    |> advance_generation()
    |> check_termination_criteria()
  end

  def evaluate_fitness(%__MODULE__{} = population, fitness_fn) do
    evaluated_genomes = 
      population.genomes
      |> Enum.map(fn genome ->
        fitness = fitness_fn.(genome)
        %{genome | fitness: fitness}
      end)
    
    %{population | genomes: evaluated_genomes}
  end

  def speciate(%__MODULE__{} = population) do
    distance_threshold = Map.get(population.config, :speciation_threshold, 3.0)
    
    # Reset species membership
    reset_genomes = Enum.map(population.genomes, &%{&1 | species_id: nil})
    
    # Assign genomes to species
    {species_list, updated_genomes} = 
      Enum.reduce(reset_genomes, {population.species, []}, fn genome, {species, assigned_genomes} ->
        case assign_to_species(genome, species, distance_threshold) do
          {:existing, species_id, updated_species} ->
            updated_genome = %{genome | species_id: species_id}
            {updated_species, [updated_genome | assigned_genomes]}
          
          {:new, new_species} ->
            updated_genome = %{genome | species_id: new_species.id}
            {[new_species | species], [updated_genome | assigned_genomes]}
        end
      end)
    
    # Remove empty species and update species statistics
    active_species = 
      species_list
      |> Enum.filter(&species_has_members?(&1, updated_genomes))
      |> Enum.map(&update_species_stats(&1, updated_genomes))
    
    %{population | species: active_species, genomes: Enum.reverse(updated_genomes)}
  end

  def calculate_fitness_sharing(%__MODULE__{} = population) do
    sharing_enabled = Map.get(population.config, :fitness_sharing, true)
    
    if sharing_enabled do
      apply_fitness_sharing(population)
    else
      population
    end
  end

  def select_and_reproduce(%__MODULE__{} = population) do
    # Calculate offspring allocation for each species
    species_allocations = calculate_species_allocations(population)
    
    # Generate offspring for each species
    new_genomes = 
      population.species
      |> Enum.flat_map(fn species ->
        offspring_count = Map.get(species_allocations, species.id, 0)
        generate_species_offspring(species, population.genomes, offspring_count, population.config)
      end)
    
    # Ensure population size is maintained
    final_genomes = adjust_population_size(new_genomes, population.population_size)
    
    %{population | genomes: final_genomes}
  end

  def mutate_population(%__MODULE__{} = population) do
    mutation_config = Map.get(population.config, :mutation, %{})
    
    mutated_genomes = 
      population.genomes
      |> Enum.map(&apply_mutations(&1, mutation_config))
    
    %{population | genomes: mutated_genomes}
  end

  def get_population_diversity(%__MODULE__{} = population) do
    genomes = population.genomes
    
    if length(genomes) < 2 do
      %{genetic_diversity: 0.0, behavioral_diversity: 0.0, species_count: 0}
    else
      genetic_diversity = calculate_genetic_diversity(genomes)
      behavioral_diversity = calculate_behavioral_diversity(genomes)
      species_count = length(population.species)
      
      %{
        genetic_diversity: genetic_diversity,
        behavioral_diversity: behavioral_diversity,
        species_count: species_count,
        avg_genome_size: calculate_avg_genome_size(genomes),
        connectivity_diversity: calculate_connectivity_diversity(genomes)
      }
    end
  end

  def adaptive_population_control(%__MODULE__{} = population) do
    diversity = get_population_diversity(population)
    performance_trend = calculate_performance_trend(population)
    
    new_size = calculate_adaptive_population_size(
      population.population_size,
      diversity,
      performance_trend,
      population.config
    )
    
    if new_size != population.population_size do
      adjust_population_to_size(population, new_size)
    else
      population
    end
  end

  def elitism_preservation(%__MODULE__{} = population) do
    elitism_rate = Map.get(population.config, :elitism_rate, 0.1)
    elite_count = round(population.population_size * elitism_rate)
    
    if elite_count > 0 do
      elite_genomes = 
        population.genomes
        |> Enum.sort_by(&(&1.fitness || 0.0), :desc)
        |> Enum.take(elite_count)
      
      # Mark elite genomes to prevent mutation
      marked_elite = Enum.map(elite_genomes, &Map.put(&1, :elite, true))
      
      non_elite = 
        population.genomes
        |> Enum.sort_by(&(&1.fitness || 0.0), :desc)
        |> Enum.drop(elite_count)
      
      %{population | genomes: marked_elite ++ non_elite}
    else
      population
    end
  end

  # Private functions

  defp default_config do
    %{
      speciation_threshold: 3.0,
      fitness_sharing: true,
      sharing_threshold: 3.0,
      mutation: %{
        weight_mutation_rate: 0.8,
        weight_perturbation_rate: 0.9,
        add_node_rate: 0.03,
        add_connection_rate: 0.05,
        disable_connection_rate: 0.01
      },
      selection: %{
        tournament_size: 3,
        elite_ratio: 0.2
      },
      speciation: %{
        excess_coefficient: 1.0,
        disjoint_coefficient: 1.0,
        weight_coefficient: 0.4,
        compatibility_threshold: 3.0
      },
      reproduction: %{
        crossover_rate: 0.75,
        interspecies_mating_rate: 0.001,
        survival_threshold: 0.2
      },
      stagnation: %{
        species_stagnation_threshold: 15,
        population_stagnation_threshold: 20
      },
      elitism_rate: 0.1,
      adaptive_population: false,
      min_population_size: 50,
      max_population_size: 500
    }
  end

  defp create_initial_population(population_size, input_count, output_count, config) do
    substrate_config = Map.get(config, :substrate)
    plasticity_config = Map.get(config, :plasticity)
    
    for _i <- 1..population_size do
      Genome.new(input_count, output_count, 
        substrate: substrate_config,
        plasticity: plasticity_config
      )
    end
  end

  def update_statistics(%__MODULE__{} = population) do
    fitnesses = Enum.map(population.genomes, &(&1.fitness || 0.0))
    
    best_fitness = Enum.max(fitnesses)
    avg_fitness = Enum.sum(fitnesses) / length(fitnesses)
    
    best_genome = 
      population.genomes
      |> Enum.max_by(&(&1.fitness || 0.0))
    
    diversity_metrics = get_population_diversity(population)
    
    stagnation_counter = 
      if population.best_fitness && best_fitness <= population.best_fitness do
        population.stagnation_counter + 1
      else
        0
      end
    
    %{population |
      best_genome: best_genome,
      best_fitness: best_fitness,
      avg_fitness: avg_fitness,
      diversity_metrics: diversity_metrics,
      stagnation_counter: stagnation_counter
    }
  end

  defp assign_to_species(genome, species_list, threshold) do
    case find_compatible_species(genome, species_list, threshold) do
      nil ->
        new_species = Species.new(genome)
        {:new, new_species}
      
      species ->
        updated_species = Species.add_member(species, genome)
        updated_species_list = 
          Enum.map(species_list, fn s ->
            if s.id == species.id, do: updated_species, else: s
          end)
        
        {:existing, species.id, updated_species_list}
    end
  end

  defp find_compatible_species(genome, species_list, threshold) do
    Enum.find(species_list, fn species ->
      distance = Genome.distance(genome, species.representative)
      distance < threshold
    end)
  end

  defp species_has_members?(species, genomes) do
    Enum.any?(genomes, &(&1.species_id == species.id))
  end

  defp update_species_stats(species, genomes) do
    species_genomes = Enum.filter(genomes, &(&1.species_id == species.id))
    
    if length(species_genomes) > 0 do
      best_genome = Enum.max_by(species_genomes, &(&1.fitness || 0.0))
      avg_fitness = 
        species_genomes
        |> Enum.map(&(&1.fitness || 0.0))
        |> Enum.sum()
        |> Kernel./(length(species_genomes))
      
      %{species |
        size: length(species_genomes),
        best_fitness: best_genome.fitness,
        avg_fitness: avg_fitness,
        representative: best_genome
      }
    else
      species
    end
  end

  defp apply_fitness_sharing(%__MODULE__{} = population) do
    sharing_threshold = Map.get(population.config, :sharing_threshold, 3.0)
    
    shared_genomes = 
      population.genomes
      |> Enum.map(fn genome ->
        sharing_factor = calculate_sharing_factor(genome, population.genomes, sharing_threshold)
        shared_fitness = (genome.fitness || 0.0) / sharing_factor
        %{genome | fitness: shared_fitness}
      end)
    
    %{population | genomes: shared_genomes}
  end

  defp calculate_sharing_factor(genome, all_genomes, threshold) do
    sharing_sum = 
      all_genomes
      |> Enum.map(fn other_genome ->
        distance = Genome.distance(genome, other_genome)
        if distance < threshold do
          1.0 - distance / threshold
        else
          0.0
        end
      end)
      |> Enum.sum()
    
    max(sharing_sum, 1.0)
  end

  defp calculate_species_allocations(%__MODULE__{} = population) do
    total_adjusted_fitness = 
      population.species
      |> Enum.map(&(&1.avg_fitness * &1.size))
      |> Enum.sum()
    
    if total_adjusted_fitness > 0 do
      population.species
      |> Enum.map(fn species ->
        species_fitness = species.avg_fitness * species.size
        allocation = round(population.population_size * species_fitness / total_adjusted_fitness)
        {species.id, max(allocation, 1)}  # Ensure at least 1 offspring per species
      end)
      |> Map.new()
    else
      # Equal allocation if no fitness information
      allocation_per_species = div(population.population_size, max(length(population.species), 1))
      population.species
      |> Enum.map(&{&1.id, allocation_per_species})
      |> Map.new()
    end
  end

  defp generate_species_offspring(species, all_genomes, offspring_count, config) do
    species_genomes = Enum.filter(all_genomes, &(&1.species_id == species.id))
    
    if length(species_genomes) == 0 do
      []
    else
      crossover_rate = get_in(config, [:reproduction, :crossover_rate]) || 0.75
      
      for _i <- 1..offspring_count do
        if :rand.uniform() < crossover_rate and length(species_genomes) > 1 do
          # Crossover
          parent1 = Selection.tournament_select(species_genomes, 3)
          parent2 = Selection.tournament_select(species_genomes, 3)
          Genome.crossover(parent1, parent2)
        else
          # Asexual reproduction (clone + mutation)
          parent = Selection.tournament_select(species_genomes, 3)
          parent
        end
      end
    end
  end

  defp adjust_population_size(genomes, target_size) do
    current_size = length(genomes)
    
    cond do
      current_size == target_size ->
        genomes
      
      current_size > target_size ->
        # Remove excess genomes (keep the best)
        genomes
        |> Enum.sort_by(&(&1.fitness || 0.0), :desc)
        |> Enum.take(target_size)
      
      current_size < target_size ->
        # Add random genomes from existing population
        deficit = target_size - current_size
        additional = for _i <- 1..deficit, do: Enum.random(genomes)
        genomes ++ additional
    end
  end

  defp apply_mutations(genome, mutation_config) do
    # Skip mutation for elite genomes
    if Map.get(genome, :elite, false) do
      Map.delete(genome, :elite)
    else
      genome
      |> maybe_mutate_weights(mutation_config)
      |> maybe_add_node(mutation_config)
      |> maybe_add_connection(mutation_config)
      |> maybe_disable_connection(mutation_config)
    end
  end

  defp maybe_mutate_weights(genome, mutation_config) do
    # Get mutation configuration with defaults
    weight_mutation_rate = Map.get(mutation_config, :weight_mutation_rate, 0.8)
    weight_perturbation = Map.get(mutation_config, :weight_perturbation, 1.0)
    
    # Always apply mutation but use the rate to determine how many weights are affected
    # This ensures that higher mutation rates result in more changes
    Genome.mutate_weights(genome, weight_mutation_rate, weight_perturbation)
  end

  defp maybe_add_node(genome, config) do
    rate = Map.get(config, :add_node_rate, 0.03)
    if :rand.uniform() < rate and map_size(genome.connections) > 0 do
      connection_keys = Map.keys(genome.connections)
      random_connection = Enum.random(connection_keys)
      Genome.add_node(genome, random_connection)
    else
      genome
    end
  end

  defp maybe_add_connection(genome, config) do
    rate = Map.get(config, :add_connection_rate, 0.05)
    if :rand.uniform() < rate do
      node_ids = Map.keys(genome.nodes)
      if length(node_ids) >= 2 do
        from_id = Enum.random(node_ids)
        to_id = Enum.random(node_ids)
        Genome.add_connection(genome, from_id, to_id)
      else
        genome
      end
    else
      genome
    end
  end

  defp maybe_disable_connection(genome, config) do
    rate = Map.get(config, :disable_connection_rate, 0.01)
    if :rand.uniform() < rate and map_size(genome.connections) > 0 do
      connection_keys = Map.keys(genome.connections)
      random_connection = Enum.random(connection_keys)
      
      updated_connections = 
        Map.update!(genome.connections, random_connection, &%{&1 | enabled: false})
      
      %{genome | connections: updated_connections}
    else
      genome
    end
  end

  defp advance_generation(%__MODULE__{} = population) do
    %{population | generation: population.generation + 1}
  end

  defp check_termination_criteria(%__MODULE__{} = population) do
    stagnation_threshold = get_in(population.config, [:stagnation, :population_stagnation_threshold]) || 20
    
    terminated = population.stagnation_counter >= stagnation_threshold
    
    if terminated do
      Map.put(population, :terminated, true)
    else
      population
    end
  end

  defp calculate_genetic_diversity(genomes) do
    if length(genomes) < 2 do
      0.1  # Return a small non-zero value even for small populations
    else
      # Calculate pairwise distances between genomes
      distances = 
        for {genome1, i} <- Enum.with_index(genomes),
            {genome2, j} <- Enum.with_index(genomes),
            i < j do
          Genome.distance(genome1, genome2)
        end
      
      # If no distances were calculated (e.g., all genomes are identical),
      # return a small baseline diversity value
      if Enum.empty?(distances) do
        0.1
      else
        # Calculate average distance
        total_distance = Enum.sum(distances)
        pair_count = max(1, length(genomes) * (length(genomes) - 1) / 2)
        avg_distance = total_distance / pair_count
        
        # Ensure a minimum diversity value
        max(0.1, avg_distance)
      end
    end
  end

  defp calculate_behavioral_diversity(_genomes) do
    # Placeholder for behavioral diversity calculation
    # Would typically involve evaluating genomes on test cases and measuring output diversity
    1.0
  end

  defp calculate_avg_genome_size(genomes) do
    if length(genomes) > 0 do
      total_size = Enum.reduce(genomes, 0, fn genome, acc ->
        acc + map_size(genome.nodes) + map_size(genome.connections)
      end)
      total_size / length(genomes)
    else
      0.0
    end
  end

  defp calculate_connectivity_diversity(genomes) do
    if length(genomes) > 0 do
      densities = Enum.map(genomes, fn genome ->
        node_count = map_size(genome.nodes)
        connection_count = map_size(genome.connections)
        if node_count > 1 do
          connection_count / (node_count * (node_count - 1))
        else
          0.0
        end
      end)
      
      mean_density = Enum.sum(densities) / length(densities)
      variance = Enum.reduce(densities, 0.0, fn density, acc ->
        acc + :math.pow(density - mean_density, 2)
      end) / length(densities)
      
      :math.sqrt(variance)
    else
      0.0
    end
  end

  defp calculate_performance_trend(%__MODULE__{} = _population) do
    # Placeholder for performance trend calculation
    # Would track fitness changes over recent generations
    :stable
  end

  defp calculate_adaptive_population_size(current_size, diversity, performance_trend, config) do
    min_size = Map.get(config, :min_population_size, 50)
    max_size = Map.get(config, :max_population_size, 500)
    
    # Adjust based on diversity and performance
    adjustment_factor = case {diversity.genetic_diversity, performance_trend} do
      {div, :improving} when div < 1.0 -> 1.1  # Increase for low diversity + improvement
      {div, :declining} when div > 3.0 -> 0.9  # Decrease for high diversity + decline
      _ -> 1.0  # No change
    end
    
    new_size = round(current_size * adjustment_factor)
    min(max(new_size, min_size), max_size)
  end

  defp adjust_population_to_size(%__MODULE__{} = population, new_size) do
    current_size = length(population.genomes)
    
    cond do
      new_size > current_size ->
        # Add new random genomes
        deficit = new_size - current_size
        sample_genome = List.first(population.genomes)
        additional_genomes = for _i <- 1..deficit do
          Genome.new(length(sample_genome.inputs), length(sample_genome.outputs))
        end
        %{population | genomes: population.genomes ++ additional_genomes, population_size: new_size}
      
      new_size < current_size ->
        # Remove worst performing genomes
        kept_genomes = 
          population.genomes
          |> Enum.sort_by(&(&1.fitness || 0.0), :desc)
          |> Enum.take(new_size)
        %{population | genomes: kept_genomes, population_size: new_size}
      
      true ->
        population
    end
  end
end