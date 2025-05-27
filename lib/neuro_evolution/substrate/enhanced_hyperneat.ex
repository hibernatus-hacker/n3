defmodule NeuroEvolution.Substrate.EnhancedHyperNEAT do
  @moduledoc """
  Enhanced HyperNEAT implementation with Nx/Axon integration for GPU acceleration.
  
  This module provides a high-performance implementation of HyperNEAT using:
  1. Axon-based CPPN for neural network operations
  2. Vectorized substrate operations using Nx
  3. GPU-accelerated spatial computations
  4. End-to-end differentiable substrate optimization
  
  This implementation significantly improves performance and capabilities compared
  to the standard HyperNEAT implementation, especially for large-scale networks.
  """

  alias NeuroEvolution.Substrate.{
    Substrate,
    AxonCPPN,
    VectorizedSubstrate,
    DifferentiableOptimization
  }
  # Uncomment if needed
  # alias NeuroEvolution.Substrate.GPUSpatialOps
  # alias NeuroEvolution.TWEANN.Genome

  defstruct [
    :cppn,
    :substrate,
    :connection_threshold,
    :max_weight,
    :distance_function,
    :leo_enabled,
    :use_gpu
  ]

  @type t :: %__MODULE__{
    cppn: {Axon.t(), map()},
    substrate: Substrate.t(),
    connection_threshold: float(),
    max_weight: float(),
    distance_function: atom(),
    leo_enabled: boolean(),
    use_gpu: boolean()
  }

  @doc """
  Creates a new EnhancedHyperNEAT instance.
  
  ## Parameters
  - input_dims: Dimensions of the input layer
  - hidden_dims: Dimensions of the hidden layer
  - output_dims: Dimensions of the output layer
  - opts: Additional options
  
  ## Options
  - connection_threshold: Threshold for connections (default: 0.2)
  - max_weight: Maximum weight value (default: 5.0)
  - distance_function: Distance function to use (default: :euclidean)
  - leo_enabled: Whether to use Link Expression Output (default: false)
  - use_gpu: Whether to use GPU acceleration (default: true)
  - hidden_layers: CPPN hidden layer sizes (default: [16, 16])
  - activation: CPPN activation function (default: :tanh)
  
  ## Returns
  A new EnhancedHyperNEAT struct
  """
  def new(input_dims, hidden_dims, output_dims, opts \\ []) do
    # Create substrate
    substrate_config = %{
      input_geometry: Keyword.get(opts, :input_geometry, :grid),
      hidden_geometry: Keyword.get(opts, :hidden_geometry, :grid),
      output_geometry: Keyword.get(opts, :output_geometry, :grid),
      input_dimensions: input_dims,
      hidden_dimensions: hidden_dims,
      output_dimensions: output_dims,
      resolution: case input_dims do
        [x, y] -> {x, y}
        [x, y, z] -> {x, y, z}
        [x] -> x
        _ -> 10
      end,
      connection_function: Keyword.get(opts, :connection_function, :distance_based)
    }

    substrate = Substrate.new(substrate_config)

    # Calculate CPPN input/output dimensions
    cppn_input_dims = calculate_cppn_inputs(input_dims)
    cppn_output_dims = if Keyword.get(opts, :leo_enabled, false), do: 2, else: 1

    # Create CPPN using Axon
    hidden_layers = Keyword.get(opts, :hidden_layers, [16, 16])
    activation = Keyword.get(opts, :activation, :tanh)
    
    cppn = AxonCPPN.new(
      cppn_input_dims,
      hidden_layers,
      cppn_output_dims,
      [activation: activation]
    )

    %__MODULE__{
      cppn: cppn,
      substrate: substrate,
      connection_threshold: Keyword.get(opts, :connection_threshold, 0.2),
      max_weight: Keyword.get(opts, :max_weight, 5.0),
      distance_function: Keyword.get(opts, :distance_function, :euclidean),
      leo_enabled: Keyword.get(opts, :leo_enabled, false),
      use_gpu: Keyword.get(opts, :use_gpu, true)
    }
  end

  @doc """
  Decodes the substrate into a genome using the CPPN.
  
  ## Parameters
  - hyperneat: The EnhancedHyperNEAT struct
  
  ## Returns
  A genome representing the substrate network
  """
  def decode_substrate(%__MODULE__{} = hyperneat) do
    # Vectorize substrate for efficient processing
    vectorized_substrate = VectorizedSubstrate.from_substrate(hyperneat.substrate)
    
    # Create genome from substrate
    VectorizedSubstrate.create_genome_from_substrate(
      vectorized_substrate,
      hyperneat.cppn,
      [
        threshold: hyperneat.connection_threshold,
        max_weight: hyperneat.max_weight
      ]
    )
  end

  @doc """
  Mutates the CPPN parameters.
  
  ## Parameters
  - hyperneat: The EnhancedHyperNEAT struct
  - mutation_rate: Rate of mutation (default: 0.1)
  - mutation_power: Power of mutation (default: 0.5)
  
  ## Returns
  Updated EnhancedHyperNEAT struct
  """
  def mutate_cppn(%__MODULE__{} = hyperneat, mutation_rate \\ 0.1, mutation_power \\ 0.5) do
    {model, params} = hyperneat.cppn
    
    # Apply random perturbations to parameters
    mutated_params = Enum.map(params, fn param ->
      # Generate random values with same shape as param
      # Replace Nx.random_normal with Nx.Random.normal
      random = Nx.Random.normal(param.shape, 0.0, mutation_power)
      
      # Generate mutation mask
      # Replace Nx.random_uniform with Nx.Random.uniform
      mask = Nx.Random.uniform(param.shape) < mutation_rate
      
      # Apply mutations only where mask is true
      Nx.add(param, Nx.multiply(random, mask))
    end)
    
    %{hyperneat | cppn: {model, mutated_params}}
  end

  @doc """
  Performs crossover between two CPPNs.
  
  ## Parameters
  - parent1: First parent EnhancedHyperNEAT
  - parent2: Second parent EnhancedHyperNEAT
  
  ## Returns
  Child EnhancedHyperNEAT struct
  """
  def crossover_cppn(%__MODULE__{} = parent1, %__MODULE__{} = parent2) do
    {model1, params1} = parent1.cppn
    {_model2, params2} = parent2.cppn
    
    # Perform crossover by randomly selecting parameters from each parent
    crossed_params = Enum.zip_with(params1, params2, fn p1, p2 ->
      # Generate random mask
      # Replace Nx.random_uniform with Nx.Random.uniform
      mask = Nx.Random.uniform(p1.shape) < 0.5
      
      # Select from parent1 where mask is true, otherwise from parent2
      # Use Nx operations for tensor arithmetic
      Nx.add(Nx.multiply(p1, mask), Nx.multiply(p2, Nx.subtract(1, mask)))
    end)
    
    %{parent1 | cppn: {model1, crossed_params}}
  end

  @doc """
  Optimizes the CPPN parameters using gradient descent.
  
  ## Parameters
  - hyperneat: The EnhancedHyperNEAT struct
  - dataset: Tuple of {inputs, targets} for training
  - opts: Optimization options
  
  ## Returns
  Updated EnhancedHyperNEAT struct with optimized CPPN
  """
  def optimize_cppn(%__MODULE__{} = hyperneat, dataset, opts \\ []) do
    {model, params} = hyperneat.cppn
    
    # Perform optimization
    {_loss, optimized_params} = DifferentiableOptimization.optimize_cppn(
      model,
      params,
      dataset,
      opts
    )
    
    %{hyperneat | cppn: {model, optimized_params}}
  end

  @doc """
  Optimizes the substrate for a specific task.
  
  ## Parameters
  - hyperneat: The EnhancedHyperNEAT struct
  - task_fn: Function that evaluates the substrate network on a task
  - opts: Optimization options
  
  ## Returns
  Updated EnhancedHyperNEAT struct with optimized substrate and CPPN
  """
  def optimize_for_task(%__MODULE__{} = hyperneat, task_fn, opts \\ []) do
    {optimized_substrate, optimized_cppn} = DifferentiableOptimization.optimize_substrate_for_task(
      hyperneat.substrate,
      hyperneat.cppn,
      task_fn,
      opts
    )
    
    %{hyperneat | substrate: optimized_substrate, cppn: optimized_cppn}
  end

  @doc """
  Evaluates a genome on the T-Maze task using the enhanced substrate architecture.
  
  This is an example of how to use the enhanced substrate architecture with
  the T-Maze task from the NeuroEvolution library.
  
  ## Parameters
  - hyperneat: The EnhancedHyperNEAT struct
  - opts: Evaluation options
  
  ## Returns
  Evaluation results
  """
  def evaluate_on_tmaze(%__MODULE__{} = hyperneat, opts \\ []) do
    # Decode substrate into genome
    genome = decode_substrate(hyperneat)
    
    # Evaluate on T-Maze task
    NeuroEvolution.Environments.TMaze.evaluate(genome, opts)
  end

  @doc """
  Evolves a population of EnhancedHyperNEAT instances for a specific task.
  
  ## Parameters
  - population: List of EnhancedHyperNEAT instances
  - fitness_fn: Function that evaluates fitness
  - generations: Number of generations to evolve
  - opts: Evolution options
  
  ## Returns
  Tuple of {evolved_population, stats}
  """
  def evolve(population, fitness_fn, generations, opts \\ []) do
    population_size = length(population)
    elitism = Keyword.get(opts, :elitism, 2)
    tournament_size = Keyword.get(opts, :tournament_size, 3)
    mutation_rate = Keyword.get(opts, :mutation_rate, 0.1)
    crossover_rate = Keyword.get(opts, :crossover_rate, 0.5)
    
    # Evolution loop
    Enum.reduce(1..generations, {population, []}, fn gen, {pop, stats} ->
      # Evaluate population
      evaluated_pop = Enum.map(pop, fn hyperneat ->
        fitness = fitness_fn.(hyperneat)
        {hyperneat, fitness}
      end)
      
      # Sort by fitness
      sorted_pop = Enum.sort_by(evaluated_pop, fn {_, fitness} -> fitness end, :desc)
      
      # Extract stats
      best_fitness = elem(hd(sorted_pop), 1)
      avg_fitness = Enum.reduce(sorted_pop, 0, fn {_, f}, acc -> acc + f end) / population_size
      
      # Print progress
      IO.puts("Generation #{gen}: Best fitness = #{best_fitness}, Avg fitness = #{avg_fitness}")
      
      # Select elites
      elites = Enum.take(sorted_pop, elitism) |> Enum.map(fn {h, _} -> h end)
      
      # Create new population
      new_pop = elites ++ create_offspring(sorted_pop, population_size - elitism, tournament_size, mutation_rate, crossover_rate)
      
      # Return updated population and stats
      {new_pop, stats ++ [%{generation: gen, best_fitness: best_fitness, avg_fitness: avg_fitness}]}
    end)
  end

  # Private functions

  defp calculate_cppn_inputs(dimensions) do
    # Calculate number of inputs based on dimensionality
    # For each dimension, we need coordinates for both source and target
    # plus a bias input
    length(dimensions) * 2 + 1
  end

  defp create_offspring(population, offspring_count, tournament_size, mutation_rate, crossover_rate) do
    Enum.map(1..offspring_count, fn _ ->
      if :rand.uniform() < crossover_rate do
        # Select parents using tournament selection
        parent1 = tournament_select(population, tournament_size)
        parent2 = tournament_select(population, tournament_size)
        
        # Perform crossover
        child = crossover_cppn(parent1, parent2)
        
        # Mutate child
        mutate_cppn(child, mutation_rate)
      else
        # Select parent using tournament selection
        parent = tournament_select(population, tournament_size)
        
        # Mutate parent
        mutate_cppn(parent, mutation_rate)
      end
    end)
  end

  defp tournament_select(population, tournament_size) do
    # Select random individuals for tournament
    tournament = Enum.take_random(population, tournament_size)
    
    # Return the best individual
    {winner, _} = Enum.max_by(tournament, fn {_, fitness} -> fitness end)
    winner
  end
end
