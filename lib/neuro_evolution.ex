defmodule NeuroEvolution do
  @moduledoc """
  A comprehensive TWEANN (Topology and Weight Evolving Artificial Neural Network) 
  neuroevolution library for Elixir with GPU acceleration via Nx.

  This library provides:
  - TWEANN genome representation with topology evolution
  - Substrate encodings for spatial neural networks (HyperNEAT)
  - Neural plasticity mechanisms (Hebbian, STDP, BCM, Oja's rule)
  - GPU-optimized batch evaluation using Nx tensors
  - Population management with speciation and adaptive dynamics
  - Comprehensive mutation and crossover operators

  ## Examples

      # Basic TWEANN evolution
      population = NeuroEvolution.Population.new(100, 3, 2)
      
      fitness_fn = fn genome ->
        # Your fitness evaluation logic
        evaluate_genome(genome)
      end
      
      evolved_population = NeuroEvolution.evolve(population, fitness_fn, generations: 100)

      # HyperNEAT with substrate encoding
      substrate_config = %{
        geometry_type: :grid,
        dimensions: 2,
        resolution: {10, 10}
      }
      
      hyperneat = NeuroEvolution.Substrate.HyperNEAT.new([10, 10], [5, 5], [3, 3], 
        substrate_config: substrate_config)
      
      phenotype = NeuroEvolution.Substrate.HyperNEAT.decode_substrate(hyperneat)

      # Neural plasticity
      plasticity_config = %{
        plasticity_type: :hebbian,
        learning_rate: 0.01,
        homeostasis: true
      }
      
      plastic_population = NeuroEvolution.Population.new(100, 3, 2, 
        plasticity: plasticity_config)

  ## Configuration Options

  ### Population Configuration
  - `:population_size` - Number of individuals (default: 100)
  - `:speciation_threshold` - Distance threshold for speciation (default: 3.0)
  - `:fitness_sharing` - Enable fitness sharing within species (default: true)
  - `:elitism_rate` - Proportion of elite individuals preserved (default: 0.1)

  ### Mutation Configuration
  - `:weight_mutation_rate` - Probability of weight mutations (default: 0.8)
  - `:add_node_rate` - Probability of adding nodes (default: 0.03)
  - `:add_connection_rate` - Probability of adding connections (default: 0.05)

  ### Substrate Configuration
  - `:geometry_type` - Substrate geometry (:grid, :circular, :hexagonal)
  - `:dimensions` - Spatial dimensions (2 or 3)
  - `:connection_function` - How to query connections (:distance_based, :all_to_all)

  ### Plasticity Configuration  
  - `:plasticity_type` - Type of plasticity (:hebbian, :stdp, :bcm, :oja)
  - `:learning_rate` - Plasticity learning rate (default: 0.01)
  - `:homeostasis` - Enable homeostatic mechanisms (default: false)
  """

  alias NeuroEvolution.Population.Population
  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Evaluator.BatchEvaluator
  alias NeuroEvolution.Substrate.{Substrate, HyperNEAT}
  alias NeuroEvolution.Plasticity.NeuralPlasticity

  @doc """
  Creates a new TWEANN population.

  ## Parameters
  - `population_size` - Number of individuals in the population
  - `input_count` - Number of input nodes
  - `output_count` - Number of output nodes
  - `opts` - Configuration options

  ## Examples

      population = NeuroEvolution.new_population(100, 3, 2)
      
      # With substrate encoding
      population = NeuroEvolution.new_population(100, 9, 4, 
        substrate: %{geometry_type: :grid, dimensions: 2, resolution: {3, 3}})
      
      # With plasticity
      population = NeuroEvolution.new_population(100, 3, 2,
        plasticity: %{plasticity_type: :hebbian, learning_rate: 0.01})
  """
  def new_population(population_size, input_count, output_count, opts \\ []) do
    Population.new(population_size, input_count, output_count, opts)
  end

  @doc """
  Evolves a population for multiple generations.

  ## Parameters
  - `population` - The population to evolve
  - `fitness_fn` - Function that evaluates genome fitness
  - `opts` - Evolution options

  ## Options
  - `:generations` - Number of generations to evolve (default: 100)
  - `:target_fitness` - Stop when this fitness is reached
  - `:batch_evaluation` - Use GPU batch evaluation (default: true)
  - `:adaptive_population` - Enable adaptive population sizing (default: false)

  ## Examples

      evolved = NeuroEvolution.evolve(population, &fitness_function/1, 
        generations: 200, target_fitness: 0.95)
  """
  def evolve(%Population{} = population, fitness_fn, opts \\ []) do
    generations = Keyword.get(opts, :generations, 100)
    target_fitness = Keyword.get(opts, :target_fitness)
    batch_evaluation = Keyword.get(opts, :batch_evaluation, true)
    adaptive_population = Keyword.get(opts, :adaptive_population, false)

    evaluator = if batch_evaluation do
      BatchEvaluator.new(device: :cuda, plasticity: has_plasticity?(population))
    end

    evolution_loop(population, fitness_fn, evaluator, generations, target_fitness, adaptive_population, 0)
  end

  @doc """
  Creates a HyperNEAT system for substrate-based evolution.

  ## Parameters
  - `input_dims` - Input layer dimensions
  - `hidden_dims` - Hidden layer dimensions  
  - `output_dims` - Output layer dimensions
  - `opts` - Configuration options

  ## Examples

      hyperneat = NeuroEvolution.new_hyperneat([10, 10], [5, 5], [3, 3],
        connection_threshold: 0.3,
        leo_enabled: true)
  """
  def new_hyperneat(input_dims, hidden_dims, output_dims, opts \\ []) do
    HyperNEAT.new(input_dims, hidden_dims, output_dims, opts)
  end

  @doc """
  Creates a substrate for spatial neural networks.

  ## Parameters
  - `config` - Substrate configuration

  ## Examples

      substrate = NeuroEvolution.new_substrate(%{
        geometry_type: :grid,
        dimensions: 2,
        resolution: {10, 10}
      })
  """
  def new_substrate(config) do
    Substrate.new(config)
  end

  @doc """
  Creates a neural plasticity configuration.

  ## Parameters
  - `plasticity_type` - Type of plasticity rule
  - `opts` - Configuration options

  ## Examples

      plasticity = NeuroEvolution.new_plasticity(:hebbian, 
        learning_rate: 0.01, homeostasis: true)
        
      plasticity = NeuroEvolution.new_plasticity(:stdp,
        a_plus: 0.1, a_minus: 0.12, tau_plus: 20.0)
  """
  def new_plasticity(plasticity_type, opts \\ []) do
    NeuralPlasticity.new(plasticity_type, opts)
  end

  @doc """
  Evaluates a single genome.

  ## Parameters
  - `genome` - The genome to evaluate
  - `inputs` - Input data for evaluation
  - `opts` - Evaluation options

  ## Examples

      outputs = NeuroEvolution.evaluate_genome(genome, [0.5, 0.3, 0.8])
  """
  def evaluate_genome(%Genome{} = genome, inputs, opts \\ []) do
    # Convert genome to Nx tensor format and evaluate
    max_nodes = Keyword.get(opts, :max_nodes, 100)
    plasticity_enabled = Keyword.get(opts, :plasticity, false)
    
    tensor_representation = Genome.to_nx_tensor(genome, max_nodes)
    
    # Simplified evaluation - in practice would use full batch evaluator
    simulate_forward_pass(tensor_representation, inputs, plasticity_enabled)
  end

  @doc """
  Gets population statistics and diversity metrics.

  ## Parameters
  - `population` - The population to analyze

  ## Examples

      stats = NeuroEvolution.get_population_stats(population)
      IO.inspect(stats.diversity_metrics)
  """
  def get_population_stats(%Population{} = population) do
    diversity = Population.get_population_diversity(population)
    
    %{
      generation: population.generation,
      population_size: population.population_size,
      species_count: length(population.species),
      best_fitness: population.best_fitness,
      avg_fitness: population.avg_fitness,
      diversity_metrics: diversity,
      stagnation_counter: population.stagnation_counter
    }
  end

  @doc """
  Saves a population to a file.

  ## Parameters
  - `population` - The population to save
  - `filename` - File path to save to

  ## Examples

      NeuroEvolution.save_population(population, "evolved_population.json")
  """
  def save_population(%Population{} = population, filename) do
    serialized = Jason.encode!(population)
    File.write!(filename, serialized)
  end

  @doc """
  Loads a population from a file.

  ## Parameters
  - `filename` - File path to load from

  ## Examples

      population = NeuroEvolution.load_population("evolved_population.json")
  """
  def load_population(filename) do
    case File.read(filename) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} -> deserialize_population(data)
          {:error, reason} -> {:error, "JSON decode error: #{reason}"}
        end
      {:error, reason} ->
        {:error, "File read error: #{reason}"}
    end
  end

  @doc """
  Creates a simple fitness function for XOR problem.

  ## Examples

      fitness_fn = NeuroEvolution.xor_fitness()
      evolved = NeuroEvolution.evolve(population, fitness_fn)
  """
  def xor_fitness do
    fn genome ->
      test_cases = [
        {[0.0, 0.0], [0.0]},
        {[0.0, 1.0], [1.0]},
        {[1.0, 0.0], [1.0]},
        {[1.0, 1.0], [0.0]}
      ]
      
      total_error = 
        Enum.reduce(test_cases, 0.0, fn {inputs, expected}, acc ->
          outputs = evaluate_genome(genome, inputs)
          error = calculate_mse(outputs, expected)
          acc + error
        end)
      
      # Convert error to fitness (higher is better)
      4.0 - total_error
    end
  end

  # Private functions

  defp evolution_loop(population, _fitness_fn, _evaluator, 0, _target_fitness, _adaptive, generation) do
    IO.puts("Evolution completed after #{generation} generations")
    population
  end

  defp evolution_loop(population, fitness_fn, evaluator, remaining_generations, target_fitness, adaptive_population, generation) do
    # Evaluate population
    evaluated_population = if evaluator do
      # Use batch evaluation
      inputs = generate_test_inputs()  # Placeholder
      BatchEvaluator.evaluate_population(evaluator, population.genomes, inputs, fitness_fn)
      |> then(&%{population | genomes: &1})
    else
      Population.evaluate_fitness(population, fitness_fn)
    end
    
    # Check termination criteria
    if target_fitness && evaluated_population.best_fitness && evaluated_population.best_fitness >= target_fitness do
      IO.puts("Target fitness #{target_fitness} reached at generation #{generation}")
      evaluated_population
    else
      # Evolve for one generation
      next_population = 
        evaluated_population
        |> Population.evolve(fitness_fn)
        |> then(fn pop ->
          if adaptive_population do
            Population.adaptive_population_control(pop)
          else
            pop
          end
        end)
      
      # Print progress
      if rem(generation, 10) == 0 do
        stats = get_population_stats(next_population)
        IO.puts("Generation #{generation}: Best=#{Float.round(stats.best_fitness || 0.0, 4)}, Avg=#{Float.round(stats.avg_fitness, 4)}, Species=#{stats.species_count}")
      end
      
      evolution_loop(next_population, fitness_fn, evaluator, remaining_generations - 1, target_fitness, adaptive_population, generation + 1)
    end
  end

  defp has_plasticity?(%Population{} = population) do
    case List.first(population.genomes) do
      %Genome{plasticity_config: nil} -> false
      %Genome{plasticity_config: _config} -> true
      _ -> false
    end
  end

  defp simulate_forward_pass(tensor_representation, inputs, _plasticity_enabled) do
    # Simplified forward pass simulation
    # In practice, this would use the full Nx-based evaluation
    input_size = length(inputs)
    output_size = length(Map.get(tensor_representation, :output_mask, []) |> Nx.to_list() |> Enum.filter(&(&1 == 1)))
    
    # Placeholder: return random outputs based on input
    for _i <- 1..max(output_size, 1) do
      Enum.sum(inputs) / input_size + :rand.normal(0.0, 0.1)
    end
  end

  defp generate_test_inputs do
    # Placeholder for generating test inputs
    [[0.5, 0.3, 0.8]]
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
      1.0  # Maximum error if size mismatch
    end
  end

  defp deserialize_population(data) do
    try do
      population = %Population{
        genomes: deserialize_genomes(data["genomes"]),
        species: deserialize_species(data["species"]),
        generation: data["generation"] || 0,
        population_size: data["population_size"] || 100,
        best_fitness: data["best_fitness"],
        avg_fitness: data["avg_fitness"] || 0.0,
        stagnation_counter: data["stagnation_counter"] || 0,
        innovation_number: data["innovation_number"] || 1000,
        config: deserialize_config(data["config"])
      }
      {:ok, population}
    rescue
      error -> {:error, "Deserialization failed: #{inspect(error)}"}
    end
  end

  defp deserialize_genomes(genomes_data) when is_list(genomes_data) do
    Enum.map(genomes_data, &deserialize_genome/1)
  end

  defp deserialize_genome(genome_data) do
    %Genome{
      id: genome_data["id"],
      nodes: deserialize_nodes(genome_data["nodes"]),
      connections: deserialize_connections(genome_data["connections"]),
      inputs: genome_data["inputs"] || [],
      outputs: genome_data["outputs"] || [],
      fitness: genome_data["fitness"],
      species_id: genome_data["species_id"],
      generation: genome_data["generation"] || 0,
      substrate_config: genome_data["substrate_config"],
      plasticity_config: genome_data["plasticity_config"]
    }
  end

  defp deserialize_nodes(nodes_data) when is_map(nodes_data) do
    nodes_data
    |> Enum.map(fn {id_str, node_data} ->
      {String.to_integer(id_str), deserialize_node(node_data)}
    end)
    |> Map.new()
  end

  defp deserialize_node(node_data) do
    %NeuroEvolution.TWEANN.Node{
      id: node_data["id"],
      type: String.to_atom(node_data["type"]),
      activation: String.to_atom(node_data["activation"]),
      bias: node_data["bias"] || 0.0,
      position: deserialize_position(node_data["position"]),
      plasticity_params: node_data["plasticity_params"]
    }
  end

  defp deserialize_connections(connections_data) when is_map(connections_data) do
    connections_data
    |> Enum.map(fn {id_str, conn_data} ->
      {String.to_integer(id_str), deserialize_connection(conn_data)}
    end)
    |> Map.new()
  end

  defp deserialize_connection(conn_data) do
    %NeuroEvolution.TWEANN.Connection{
      innovation_number: conn_data["innovation_number"],
      input_node: conn_data["input_node"],
      output_node: conn_data["output_node"],
      weight: conn_data["weight"] || 0.0,
      enabled: conn_data["enabled"] || true,
      plasticity_params: conn_data["plasticity_params"]
    }
  end

  defp deserialize_species(species_data) when is_list(species_data) do
    Enum.map(species_data, &deserialize_single_species/1)
  end

  defp deserialize_single_species(species_data) do
    %NeuroEvolution.Population.Species{
      id: species_data["id"],
      representative: deserialize_genome(species_data["representative"]),
      members: deserialize_genomes(species_data["members"]),
      size: species_data["size"] || 0,
      best_fitness: species_data["best_fitness"],
      avg_fitness: species_data["avg_fitness"] || 0.0,
      age: species_data["age"] || 0,
      stagnation_counter: species_data["stagnation_counter"] || 0,
      last_improvement: species_data["last_improvement"] || 0,
      offspring_allocation: species_data["offspring_allocation"] || 0
    }
  end

  defp deserialize_config(config_data) when is_map(config_data) do
    # Convert string keys to atoms and reconstruct nested config
    config_data
    |> Enum.map(fn {key, value} ->
      {String.to_atom(key), deserialize_config_value(value)}
    end)
    |> Map.new()
  end

  defp deserialize_config(nil), do: %{}

  defp deserialize_config_value(value) when is_map(value) do
    deserialize_config(value)
  end

  defp deserialize_config_value(value), do: value

  defp deserialize_position(nil), do: nil
  defp deserialize_position([x, y]), do: {x, y}
  defp deserialize_position([x, y, z]), do: {x, y, z}
  defp deserialize_position(pos), do: pos
end