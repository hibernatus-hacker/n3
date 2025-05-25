defmodule NeuroEvolution.Population.Species do
  @moduledoc """
  Species management for TWEANN populations with speciation based on 
  topological and weight similarity.
  """

  alias NeuroEvolution.TWEANN.Genome

  defstruct [
    :id,
    :representative,
    :members,
    :size,
    :best_fitness,
    :avg_fitness,
    :age,
    :stagnation_counter,
    :last_improvement,
    :offspring_allocation
  ]

  @type t :: %__MODULE__{
    id: integer(),
    representative: Genome.t(),
    members: [Genome.t()],
    size: integer(),
    best_fitness: float() | nil,
    avg_fitness: float(),
    age: integer(),
    stagnation_counter: integer(),
    last_improvement: integer(),
    offspring_allocation: integer()
  }

  def new(%Genome{} = representative_genome) do
    %__MODULE__{
      id: generate_species_id(),
      representative: representative_genome,
      members: [representative_genome],
      size: 1,
      best_fitness: representative_genome.fitness,
      avg_fitness: representative_genome.fitness || 0.0,
      age: 0,
      stagnation_counter: 0,
      last_improvement: 0,
      offspring_allocation: 0
    }
  end

  def add_member(%__MODULE__{} = species, %Genome{} = genome) do
    updated_members = [genome | species.members]
    
    %{species |
      members: updated_members,
      size: length(updated_members)
    }
  end

  def update_statistics(%__MODULE__{} = species, generation) do
    if species.size > 0 do
      fitnesses = Enum.map(species.members, &(&1.fitness || 0.0))
      
      best_fitness = Enum.max(fitnesses)
      avg_fitness = Enum.sum(fitnesses) / length(fitnesses)
      
      # Update stagnation counter
      {stagnation_counter, last_improvement} = 
        if species.best_fitness && best_fitness > species.best_fitness do
          {0, generation}
        else
          {species.stagnation_counter + 1, species.last_improvement}
        end
      
      # Update representative (best genome)
      new_representative = Enum.max_by(species.members, &(&1.fitness || 0.0))
      
      %{species |
        best_fitness: best_fitness,
        avg_fitness: avg_fitness,
        representative: new_representative,
        stagnation_counter: stagnation_counter,
        last_improvement: last_improvement,
        age: species.age + 1
      }
    else
      species
    end
  end

  def is_compatible?(%__MODULE__{} = species, %Genome{} = genome, threshold) do
    distance = Genome.distance(species.representative, genome)
    distance < threshold
  end

  def adjust_fitness_for_sharing(%__MODULE__{} = species) do
    sharing_penalty = calculate_sharing_penalty(species.size)
    
    adjusted_members = 
      Enum.map(species.members, fn genome ->
        adjusted_fitness = (genome.fitness || 0.0) / sharing_penalty
        %{genome | fitness: adjusted_fitness}
      end)
    
    %{species | members: adjusted_members}
  end

  def calculate_offspring_allocation(%__MODULE__{} = species, total_population, total_adjusted_fitness) do
    if total_adjusted_fitness > 0 do
      species_contribution = species.avg_fitness * species.size
      allocation = round(total_population * species_contribution / total_adjusted_fitness)
      max(allocation, 1)  # Ensure at least 1 offspring
    else
      1
    end
  end

  def remove_stagnant_members(%__MODULE__{} = species, survival_rate \\ 0.2) do
    if species.size > 1 do
      survivors_count = max(1, round(species.size * survival_rate))
      
      survivors = 
        species.members
        |> Enum.sort_by(&(&1.fitness || 0.0), :desc)
        |> Enum.take(survivors_count)
      
      %{species | members: survivors, size: length(survivors)}
    else
      species
    end
  end

  def get_species_diversity(%__MODULE__{} = species) do
    if species.size < 2 do
      %{genetic_diversity: 0.0, fitness_variance: 0.0}
    else
      genetic_diversity = calculate_genetic_diversity(species.members)
      fitness_variance = calculate_fitness_variance(species.members)
      
      %{
        genetic_diversity: genetic_diversity,
        fitness_variance: fitness_variance,
        size_diversity: calculate_size_diversity(species.members),
        connectivity_diversity: calculate_connectivity_diversity(species.members)
      }
    end
  end

  def select_parents(%__MODULE__{} = species, selection_method \\ :tournament) do
    case selection_method do
      :tournament -> tournament_selection(species.members)
      :roulette -> roulette_selection(species.members)
      :rank -> rank_selection(species.members)
      :elite -> elite_selection(species.members)
    end
  end

  def crossover_within_species(%__MODULE__{} = species, crossover_rate \\ 0.75) do
    if species.size > 1 and :rand.uniform() < crossover_rate do
      parent1 = select_parents(species, :tournament)
      parent2 = select_parents(species, :tournament)
      
      if parent1.id != parent2.id do
        Genome.crossover(parent1, parent2)
      else
        parent1
      end
    else
      # Asexual reproduction
      select_parents(species, :tournament)
    end
  end

  def apply_species_mutations(%__MODULE__{} = species, mutation_config) do
    mutated_members = 
      Enum.map(species.members, fn genome ->
        apply_genome_mutations(genome, mutation_config, species)
      end)
    
    %{species | members: mutated_members}
  end

  def age_species(%__MODULE__{} = species) do
    %{species | age: species.age + 1}
  end

  def reset_members(%__MODULE__{} = species) do
    %{species | members: [], size: 0}
  end

  def is_stagnant?(%__MODULE__{} = species, stagnation_threshold \\ 15) do
    species.stagnation_counter >= stagnation_threshold
  end

  def should_penalize_age?(%__MODULE__{} = species, age_threshold \\ 10) do
    species.age > age_threshold and species.stagnation_counter > 5
  end

  def calculate_age_penalty(%__MODULE__{} = species, max_penalty \\ 0.5) do
    if should_penalize_age?(species) do
      penalty_factor = min(species.age / 20.0, max_penalty)
      1.0 - penalty_factor
    else
      1.0
    end
  end

  def boost_young_species?(%__MODULE__{} = species, young_age_threshold \\ 10, boost_factor \\ 1.3) do
    if species.age < young_age_threshold do
      boost_factor
    else
      1.0
    end
  end

  # Private functions

  defp generate_species_id do
    System.unique_integer([:positive])
  end

  defp calculate_sharing_penalty(species_size) do
    max(species_size, 1.0)
  end

  defp calculate_genetic_diversity(members) do
    if length(members) < 2 do
      0.0
    else
      distances = 
        for {genome1, i} <- Enum.with_index(members),
            {genome2, j} <- Enum.with_index(members),
            i < j do
          Genome.distance(genome1, genome2)
        end
      
      if length(distances) > 0 do
        Enum.sum(distances) / length(distances)
      else
        0.0
      end
    end
  end

  defp calculate_fitness_variance(members) do
    fitnesses = Enum.map(members, &(&1.fitness || 0.0))
    
    if length(fitnesses) > 1 do
      mean_fitness = Enum.sum(fitnesses) / length(fitnesses)
      
      variance = 
        fitnesses
        |> Enum.map(fn fitness -> :math.pow(fitness - mean_fitness, 2) end)
        |> Enum.sum()
        |> Kernel./(length(fitnesses) - 1)
      
      variance
    else
      0.0
    end
  end

  defp calculate_size_diversity(members) do
    sizes = Enum.map(members, fn genome ->
      map_size(genome.nodes) + map_size(genome.connections)
    end)
    
    if length(sizes) > 1 do
      mean_size = Enum.sum(sizes) / length(sizes)
      variance = Enum.reduce(sizes, 0.0, fn size, acc ->
        acc + :math.pow(size - mean_size, 2)
      end) / length(sizes)
      :math.sqrt(variance)
    else
      0.0
    end
  end

  defp calculate_connectivity_diversity(members) do
    densities = Enum.map(members, fn genome ->
      node_count = map_size(genome.nodes)
      connection_count = map_size(genome.connections)
      if node_count > 1 do
        connection_count / (node_count * (node_count - 1))
      else
        0.0
      end
    end)
    
    if length(densities) > 1 do
      mean_density = Enum.sum(densities) / length(densities)
      variance = Enum.reduce(densities, 0.0, fn density, acc ->
        acc + :math.pow(density - mean_density, 2)
      end) / length(densities)
      :math.sqrt(variance)
    else
      0.0
    end
  end

  defp tournament_selection(members, tournament_size \\ 3) do
    tournament_size = min(tournament_size, length(members))
    
    tournament = Enum.take_random(members, tournament_size)
    Enum.max_by(tournament, &(&1.fitness || 0.0))
  end

  defp roulette_selection(members) do
    total_fitness = 
      members
      |> Enum.map(&(&1.fitness || 0.0))
      |> Enum.sum()
    
    if total_fitness > 0 do
      selection_point = :rand.uniform() * total_fitness
      
      {selected, _} = 
        Enum.reduce_while(members, {nil, 0.0}, fn genome, {_, cumulative} ->
          new_cumulative = cumulative + (genome.fitness || 0.0)
          if new_cumulative >= selection_point do
            {:halt, {genome, new_cumulative}}
          else
            {:cont, {nil, new_cumulative}}
          end
        end)
      
      selected || List.first(members)
    else
      Enum.random(members)
    end
  end

  defp rank_selection(members) do
    ranked_members = 
      members
      |> Enum.with_index()
      |> Enum.sort_by(fn {genome, _} -> genome.fitness || 0.0 end, :desc)
    
    # Linear ranking: best gets rank n, worst gets rank 1
    total_rank = length(members) * (length(members) + 1) / 2
    selection_point = :rand.uniform() * total_rank
    
    {selected, _} = 
      Enum.reduce_while(ranked_members, {nil, 0}, fn {{genome, _}, rank}, {_, cumulative} ->
        new_cumulative = cumulative + (length(members) - rank)
        if new_cumulative >= selection_point do
          {:halt, {genome, new_cumulative}}
        else
          {:cont, {nil, new_cumulative}}
        end
      end)
    
    selected || List.first(members)
  end

  defp elite_selection(members) do
    Enum.max_by(members, &(&1.fitness || 0.0))
  end

  defp apply_genome_mutations(genome, mutation_config, species) do
    # Adjust mutation rates based on species characteristics
    adjusted_config = adjust_mutation_rates(mutation_config, species)
    
    genome
    |> maybe_mutate_weights(adjusted_config)
    |> maybe_add_node(adjusted_config)
    |> maybe_add_connection(adjusted_config)
    |> maybe_disable_connection(adjusted_config)
  end

  defp adjust_mutation_rates(mutation_config, species) do
    # Increase mutation rates for stagnant species
    stagnation_factor = if species.stagnation_counter > 5, do: 1.5, else: 1.0
    
    # Decrease mutation rates for very young species  
    age_factor = if species.age < 5, do: 0.7, else: 1.0
    
    adjustment = stagnation_factor * age_factor
    
    Map.new(mutation_config, fn {key, rate} ->
      {key, rate * adjustment}
    end)
  end

  defp maybe_mutate_weights(genome, config) do
    rate = Map.get(config, :weight_mutation_rate, 0.8)
    if :rand.uniform() < rate do
      Genome.mutate_weights(genome)
    else
      genome
    end
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
end