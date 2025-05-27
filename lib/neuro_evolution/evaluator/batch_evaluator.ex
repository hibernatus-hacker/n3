defmodule NeuroEvolution.Evaluator.TopologyCluster do
  @moduledoc """
  Represents a cluster of genomes with similar topology for batch processing.
  """

  defstruct [
    :genomes,
    :max_nodes,
    :batch_size,
    :signature
  ]

  @type t :: %__MODULE__{
    genomes: [NeuroEvolution.TWEANN.Genome.t()],
    max_nodes: integer(),
    batch_size: integer(),
    signature: tuple()
  }
end

defmodule NeuroEvolution.Evaluator.BatchEvaluator do
  @moduledoc """
  GPU-optimized batch evaluation system for TWEANN networks using Nx tensors.
  Handles topology clustering and efficient parallel evaluation.
  """

  require Logger

  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Evaluator.TopologyCluster

  import Nx.Defn

  defstruct [
    :clusters,
    :compiled_networks,
    :batch_size,
    :max_topology_size,
    :plasticity_enabled,
    :device,
    :memory_manager,
    :adaptive_batching
  ]

  @type t :: %__MODULE__{
    clusters: [TopologyCluster.t()],
    compiled_networks: %{atom() => Nx.Container.t()},
    batch_size: integer(),
    max_topology_size: integer(),
    plasticity_enabled: boolean(),
    device: atom(),
    memory_manager: map(),
    adaptive_batching: boolean()
  }

  def new(opts \\ []) do
    device = Keyword.get(opts, :device, :cuda)
    memory_manager = initialize_memory_manager(device, opts)
    
    %__MODULE__{
      clusters: [],
      compiled_networks: %{},
      batch_size: Keyword.get(opts, :batch_size, 32),
      max_topology_size: Keyword.get(opts, :max_topology_size, 100),
      plasticity_enabled: Keyword.get(opts, :plasticity, false),
      device: device,
      memory_manager: memory_manager,
      adaptive_batching: Keyword.get(opts, :adaptive_batching, true)
    }
  end

  def evaluate_population(%__MODULE__{} = evaluator, genomes, inputs, fitness_fn) do
    # Check available GPU memory and adjust batch sizes if needed
    updated_evaluator = if evaluator.adaptive_batching do
      adjust_batch_sizes_for_memory(evaluator, length(genomes))
    else
      evaluator
    end
    
    # Cluster genomes by topology similarity with memory-aware batching
    clusters = cluster_genomes_by_topology_with_memory(genomes, updated_evaluator)
    
    # Evaluate each cluster in parallel with memory monitoring and proper error handling
    results = 
      clusters
      |> Task.async_stream(fn cluster ->
        with_memory_monitoring(updated_evaluator, fn ->
          evaluate_cluster(updated_evaluator, cluster, inputs, fitness_fn)
        end)
      end, 
        max_concurrency: get_optimal_concurrency(updated_evaluator),
        timeout: 60_000,  # 60 second timeout per cluster
        on_timeout: :kill_task,
        ordered: false
      )
      |> Enum.reduce([], fn 
        {:ok, result}, acc -> [result | acc]
        {:error, reason}, acc -> 
          Logger.warning("Cluster evaluation failed: #{inspect(reason)}")
          acc
        {:exit, reason}, acc ->
          Logger.warning("Cluster evaluation task exited: #{inspect(reason)}")
          acc
      end)
      |> List.flatten()
    
    # Explicitly trigger garbage collection to clean up tensor memory
    :erlang.garbage_collect()
    
    # Return results in original genome order
    sort_results_by_genome_id(results, genomes)
  end

  def evaluate_cluster(%__MODULE__{} = evaluator, cluster, inputs, fitness_fn) do
    # Compile cluster to GPU-friendly tensor representation
    {tensor_batch, genome_mapping} = compile_cluster_to_tensors(cluster, evaluator.max_topology_size)
    
    try do
      # Run batch forward propagation on GPU
      outputs = forward_propagate_batch(tensor_batch, inputs, evaluator.plasticity_enabled)
      
      # Calculate fitness for each genome
      calculate_batch_fitness(outputs, genome_mapping, fitness_fn)
    after
      # Clean up tensor memory after evaluation
      :erlang.garbage_collect()
    end
  end

  def update_plasticity_batch(%__MODULE__{} = evaluator, cluster, inputs, outputs) when evaluator.plasticity_enabled do
    {tensor_batch, genome_mapping} = compile_cluster_to_tensors(cluster, evaluator.max_topology_size)
    
    # Update plastic weights based on activity
    updated_tensors = update_plastic_weights_batch(tensor_batch, inputs, outputs)
    
    # Convert back to genome representation
    convert_tensors_to_genomes(updated_tensors, genome_mapping)
  end

  def update_plasticity_batch(%__MODULE__{} = _evaluator, cluster, _inputs, _outputs) do
    # Return unchanged if plasticity disabled
    cluster
  end

  # GPU-optimized tensor operations

  defn forward_propagate_batch(tensor_batch, inputs, plasticity_enabled) do
    %{
      adjacency: adjacency_matrices,
      weights: weight_matrices,
      features: node_features,
      input_mask: input_masks,
      output_mask: output_masks,
      plastic_weights: plastic_weights
    } = tensor_batch

    batch_size = Nx.axis_size(adjacency_matrices, 0)
    max_nodes = Nx.axis_size(adjacency_matrices, 1)
    
    # Initialize activations
    activations = Nx.broadcast(0.0, {batch_size, max_nodes})
    
    # Set input activations
    activations = set_input_activations(activations, inputs, input_masks)
    
    # Compute effective weights (base + plastic)
    effective_weights = if plasticity_enabled do
      weight_matrices + plastic_weights
    else
      weight_matrices
    end
    
    # Forward propagation through multiple time steps
    final_activations = forward_steps(activations, adjacency_matrices, effective_weights, node_features, 5)
    
    # Extract outputs
    extract_outputs(final_activations, output_masks)
  end

  defn forward_steps(activations, adjacency, weights, features, num_steps) do
    initial_state = {activations, adjacency, weights, features, 0}
    
    {final_activations, _, _, _, _} = 
      while state = initial_state, Nx.less(elem(state, 4), num_steps) do
        {current_activations, adj, w, feat, step} = state
        
        # Compute weighted inputs
        weighted_inputs = Nx.dot(current_activations, [1], adj * w, [1])
        
        # Add bias from node features
        biases = feat[[.., 2]]  # Assuming bias is stored in features[:, 2]
        inputs_with_bias = weighted_inputs + biases
        
        # Apply activation functions
        new_activations = apply_activation_functions(inputs_with_bias, feat)
        
        {new_activations, adj, w, feat, step + 1}
      end
    
    final_activations
  end

  defn apply_activation_functions(inputs, features) do
    # Extract activation type from features (assuming it's in features[:, 1])
    activation_types = features[[.., 1]]
    
    # Apply different activation functions based on type
    # 0.0 = linear, 0.5 = tanh, 1.0 = relu
    linear_mask = Nx.equal(activation_types, 0.0)
    tanh_mask = Nx.equal(activation_types, 0.5)
    relu_mask = Nx.equal(activation_types, 1.0)
    
    linear_outputs = inputs
    tanh_outputs = Nx.tanh(inputs)
    relu_outputs = Nx.max(0.0, inputs)
    
    outputs = linear_outputs * linear_mask + tanh_outputs * tanh_mask + relu_outputs * relu_mask
    outputs
  end

  defn set_input_activations(activations, inputs, input_masks) do
    batch_size = Nx.axis_size(activations, 0)
    _input_size = Nx.axis_size(inputs, 1)
    
    # Broadcast inputs to match activation tensor shape
    expanded_inputs = Nx.broadcast(inputs, {batch_size, Nx.axis_size(activations, 1)})
    
    # Apply input mask to set only input nodes
    input_activations = expanded_inputs * input_masks
    
    # Combine with existing activations (for non-input nodes)
    activations * (1.0 - input_masks) + input_activations
  end

  defn extract_outputs(activations, output_masks) do
    # Extract only output node activations
    output_activations = activations * output_masks
    
    # Sum across nodes to get output vector for each genome
    Nx.sum(output_activations, axes: [1])
  end

  defn update_plastic_weights_batch(tensor_batch, inputs, outputs) do
    %{
      adjacency: adjacency_matrices,
      weights: _weight_matrices,
      plastic_weights: plastic_weights
    } = tensor_batch

    # Get pre and post synaptic activities
    pre_activities = get_presynaptic_activities(inputs, adjacency_matrices)
    post_activities = get_postsynaptic_activities(outputs, adjacency_matrices)
    
    # Apply plasticity rules
    updated_plastic_weights = apply_plasticity_rules_batch(
      plastic_weights, 
      pre_activities, 
      post_activities, 
      %{}
    )
    
    %{tensor_batch | plastic_weights: updated_plastic_weights}
  end

  defn apply_plasticity_rules_batch(plastic_weights, pre_activities, post_activities, _plasticity_params) do
    # Hebbian rule: Δw = η * pre * post
    learning_rate = 0.01
    
    delta_weights = learning_rate * pre_activities * post_activities
    
    # Update plastic weights with decay
    decay_rate = 0.99
    updated_weights = plastic_weights * decay_rate + delta_weights
    
    # Clamp weights
    max_plastic = 2.0
    Nx.clip(updated_weights, -max_plastic, max_plastic)
  end

  defn get_presynaptic_activities(inputs, adjacency_matrices) do
    # Extract pre-synaptic activities for each connection
    batch_size = Nx.axis_size(adjacency_matrices, 0)
    max_nodes = Nx.axis_size(adjacency_matrices, 1)
    input_size = Nx.axis_size(inputs, 1)
    
    # Create a padded input tensor that matches the max_nodes dimension
    # This ensures we can handle any size of input compared to the adjacency matrix
    padded_inputs = if input_size < max_nodes do
      # Pad the inputs to match max_nodes
      padding = max_nodes - input_size
      Nx.pad(inputs, 0.0, [{0, 0, 0}, {0, padding, 0}])
    else
      # If inputs are already large enough, just take the first max_nodes columns
      Nx.slice(inputs, [0, 0], [batch_size, max_nodes])
    end
    
    # Now the dimensions should match for broadcasting
    Nx.take_along_axis(padded_inputs, adjacency_matrices, axis: 1)
  end

  defn get_postsynaptic_activities(outputs, adjacency_matrices) do
    # Extract post-synaptic activities for each connection
    batch_size = Nx.axis_size(adjacency_matrices, 0)
    max_nodes = Nx.axis_size(adjacency_matrices, 1)
    output_size = Nx.axis_size(outputs, 1)
    
    # Create a padded output tensor that matches the max_nodes dimension
    # This ensures we can handle any size of output compared to the adjacency matrix
    padded_outputs = if output_size < max_nodes do
      # Pad the outputs to match max_nodes
      padding = max_nodes - output_size
      Nx.pad(outputs, 0.0, [{0, 0, 0}, {0, padding, 0}])
    else
      # If outputs are already large enough, just take the first max_nodes columns
      Nx.slice(outputs, [0, 0], [batch_size, max_nodes])
    end
    
    # Now the dimensions should match for broadcasting
    Nx.take_along_axis(padded_outputs, adjacency_matrices, axis: 1)
  end

  # Helper functions

  # Experimental function - kept for future reference but not currently used
  defp __cluster_genomes_by_topology(genomes, max_topology_size) do
    # Group genomes by similar topology characteristics
    genomes
    |> Enum.group_by(&topology_signature/1)
    |> Map.values()
    |> Enum.map(&create_topology_cluster(&1, max_topology_size))
  end

  defp topology_signature(genome) do
    node_count = map_size(genome.nodes)
    connection_count = map_size(genome.connections)
    
    # Create a signature based on topology characteristics
    {
      min(node_count, 20),  # Bucket by node count (up to 20)
      min(connection_count, 50),  # Bucket by connection count (up to 50)
      length(genome.inputs),
      length(genome.outputs)
    }
  end

  defp create_topology_cluster(genomes, max_topology_size) do
    max_nodes = genomes
                |> Enum.map(&map_size(&1.nodes))
                |> Enum.max()
                |> min(max_topology_size)
    
    %TopologyCluster{
      genomes: genomes,
      max_nodes: max_nodes,
      batch_size: length(genomes)
    }
  end

  defp compile_cluster_to_tensors(cluster, max_topology_size) do
    _batch_size = length(cluster.genomes)
    max_nodes = min(cluster.max_nodes, max_topology_size)
    
    # Convert each genome to tensor representation
    tensor_data = 
      cluster.genomes
      |> Enum.map(&Genome.to_nx_tensor(&1, max_nodes))
      |> combine_tensor_batch()
    
    genome_mapping = 
      cluster.genomes
      |> Enum.with_index()
      |> Enum.map(fn {genome, idx} -> {idx, genome.id} end)
      |> Map.new()
    
    {tensor_data, genome_mapping}
  end

  defp combine_tensor_batch(tensor_list) do
    # Stack individual tensors into batch tensors
    %{
      adjacency: stack_tensors(tensor_list, :adjacency),
      weights: stack_tensors(tensor_list, :weights),
      features: stack_tensors(tensor_list, :features),
      input_mask: stack_tensors(tensor_list, :input_mask),
      output_mask: stack_tensors(tensor_list, :output_mask),
      plastic_weights: initialize_plastic_weights(tensor_list)
    }
  end

  defp stack_tensors(tensor_list, key) do
    tensors = Enum.map(tensor_list, &Map.get(&1, key))
    Nx.stack(tensors)
  end

  defp initialize_plastic_weights(tensor_list) do
    # Initialize plastic weights to zero
    weight_tensors = Enum.map(tensor_list, &Map.get(&1, :weights))
    zeros = Enum.map(weight_tensors, &Nx.broadcast(0.0, Nx.shape(&1)))
    Nx.stack(zeros)
  end

  defp calculate_batch_fitness(outputs, genome_mapping, fitness_fn) do
    outputs
    |> Nx.to_list()
    |> Enum.with_index()
    |> Enum.map(fn {output, idx} ->
      genome_id = Map.get(genome_mapping, idx)
      fitness = fitness_fn.(output)
      {genome_id, fitness}
    end)
  end

  defp sort_results_by_genome_id(results, genomes) do
    result_map = Map.new(results)
    
    Enum.map(genomes, fn genome ->
      fitness = Map.get(result_map, genome.id, 0.0)
      %{genome | fitness: fitness}
    end)
  end

  defp convert_tensors_to_genomes(updated_tensors, genome_mapping) do
    # Extract individual genome tensors from batch
    %{
      adjacency: adjacency_batch,
      weights: weights_batch,
      features: features_batch,
      plastic_weights: plastic_weights_batch
    } = updated_tensors

    batch_size = Nx.axis_size(adjacency_batch, 0)
    
    # Convert each genome in the batch
    for batch_idx <- 0..(batch_size - 1) do
      genome_id = Map.get(genome_mapping, batch_idx)
      
      # Extract tensors for this specific genome
      genome_tensors = %{
        adjacency: Nx.slice(adjacency_batch, [batch_idx, 0, 0], [1, :all, :all]) |> Nx.squeeze([0]),
        weights: Nx.slice(weights_batch, [batch_idx, 0, 0], [1, :all, :all]) |> Nx.squeeze([0]),
        features: Nx.slice(features_batch, [batch_idx, 0, 0], [1, :all, :all]) |> Nx.squeeze([0])
      }
      
      plastic_weights = Nx.slice(plastic_weights_batch, [batch_idx, 0, 0], [1, :all, :all]) |> Nx.squeeze([0])
      
      # This would need the original genome to properly convert back
      # For now, we'll return the genome_id and tensor data for further processing
      {genome_id, genome_tensors, plastic_weights}
    end
  end

  def apply_plasticity_updates_to_genomes(cluster_genomes, tensor_updates) do
    # Apply tensor updates back to the original genomes
    tensor_updates
    |> Enum.map(fn {genome_id, _genome_tensors, plastic_weights} ->
      # Find the original genome
      original_genome = Enum.find(cluster_genomes, &(&1.id == genome_id))
      
      if original_genome do
        # Apply plasticity updates
        Genome.update_plasticity_from_tensor(original_genome, plastic_weights)
      else
        nil
      end
    end)
    |> Enum.filter(&(&1 != nil))
  end

  # Memory Management Functions

  defp initialize_memory_manager(device, opts) do
    %{
      device: device,
      max_memory_mb: Keyword.get(opts, :max_memory_mb, 8000),  # 8GB default
      memory_usage_mb: 0,
      memory_threshold: Keyword.get(opts, :memory_threshold, 0.85),  # 85% threshold
      batch_size_history: [],
      oom_count: 0
    }
  end

  defp adjust_batch_sizes_for_memory(%__MODULE__{} = evaluator, population_size) do
    memory_manager = evaluator.memory_manager
    
    # Estimate memory requirements
    estimated_memory = estimate_memory_usage(population_size, evaluator.max_topology_size, evaluator.plasticity_enabled)
    
    # Adjust batch size if estimated memory exceeds threshold
    new_batch_size = if estimated_memory > memory_manager.max_memory_mb * memory_manager.memory_threshold do
      # Reduce batch size proportionally
      reduction_factor = (memory_manager.max_memory_mb * memory_manager.memory_threshold) / estimated_memory
      max(round(evaluator.batch_size * reduction_factor), 1)
    else
      evaluator.batch_size
    end
    
    %{evaluator | batch_size: new_batch_size}
  end

  defp cluster_genomes_by_topology_with_memory(genomes, %__MODULE__{} = evaluator) do
    # Use memory-aware clustering that considers batch size limits
    max_cluster_size = evaluator.batch_size
    
    genomes
    |> Enum.group_by(&topology_signature/1)
    |> Map.values()
    |> Enum.flat_map(&split_large_clusters(&1, max_cluster_size, evaluator.max_topology_size))
  end

  defp split_large_clusters(genomes, max_cluster_size, max_topology_size) when length(genomes) <= max_cluster_size do
    [create_topology_cluster(genomes, max_topology_size)]
  end

  defp split_large_clusters(genomes, max_cluster_size, max_topology_size) do
    # Split large clusters into smaller memory-friendly chunks
    genomes
    |> Enum.chunk_every(max_cluster_size)
    |> Enum.map(&create_topology_cluster(&1, max_topology_size))
  end

  defp with_memory_monitoring(%__MODULE__{} = evaluator, evaluation_fn) do
    start_time = System.monotonic_time(:millisecond)
    
    try do
      result = evaluation_fn.()
      
      # Update memory statistics
      end_time = System.monotonic_time(:millisecond)
      execution_time = end_time - start_time
      
      # Log successful execution for future memory estimation
      update_memory_statistics(evaluator, execution_time, :success)
      
      result
    rescue
      error ->
        # Handle potential out-of-memory errors
        case error do
          %{message: message} ->
            if String.contains?(message, "out of memory") do
              update_memory_statistics(evaluator, 0, :oom)
              {:error, :out_of_memory}
            else
              reraise error, __STACKTRACE__
            end
          _ ->
            reraise error, __STACKTRACE__
        end
    end
  end

  defp get_optimal_concurrency(%__MODULE__{device: :cuda} = evaluator) do
    # For GPU, limit concurrency to avoid memory contention
    base_concurrency = case evaluator.memory_manager.oom_count do
      0 -> System.schedulers_online()
      1..2 -> max(System.schedulers_online() - 1, 1)
      _ -> 1  # Conservative approach after multiple OOMs
    end
    
    min(base_concurrency, 4)  # Cap at 4 for GPU workloads
  end

  defp get_optimal_concurrency(%__MODULE__{device: :cpu}) do
    System.schedulers_online()
  end

  defp estimate_memory_usage(population_size, max_topology_size, plasticity_enabled) do
    # Rough estimation based on tensor sizes
    # Each genome tensor: max_topology_size^2 for adjacency and weights
    # Plus features and masks
    
    base_memory_per_genome = max_topology_size * max_topology_size * 4 * 2  # float32 for adj + weights
    feature_memory = max_topology_size * 4 * 4  # 4 features per node
    mask_memory = max_topology_size * 4 * 2  # input and output masks
    
    total_per_genome = base_memory_per_genome + feature_memory + mask_memory
    
    # Add plasticity overhead if enabled
    total_per_genome = if plasticity_enabled do
      total_per_genome + base_memory_per_genome  # Plastic weights
    else
      total_per_genome
    end
    
    # Convert to MB and add overhead for batch processing
    total_bytes = population_size * total_per_genome
    overhead_factor = 1.5  # 50% overhead for processing
    
    (total_bytes * overhead_factor) / (1024 * 1024)
  end

  defp update_memory_statistics(%__MODULE__{} = _evaluator, _execution_time, status) do
    # In a real implementation, this would update global state
    # For now, we'll just log the information
    case status do
      :success ->
        # Could update estimations based on successful runs
        :ok
      :oom ->
        # Could trigger batch size reduction for future evaluations
        :ok
    end
  end
end