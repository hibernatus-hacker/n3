defmodule NeuroEvolution.Substrate.VectorizedSubstrate do
  @moduledoc """
  Vectorized implementation of substrate operations using Nx tensors.
  
  This module provides high-performance, GPU-accelerated implementations of
  substrate operations for HyperNEAT, enabling efficient processing of large-scale
  neural networks through vectorized operations.
  """

  alias NeuroEvolution.Substrate.{Substrate, AxonCPPN}
  alias NeuroEvolution.TWEANN.{Genome, Node, Connection}
  
  import Nx.Defn
  
  # Configure to use GPU when available
  # Commented out unused module attributes
  # @defn_compiler EXLA
  # @defn_options [compiler: EXLA, client: :cuda]
  
  @doc """
  Creates a new vectorized substrate from a standard substrate.
  
  ## Parameters
  - substrate: The standard Substrate struct
  
  ## Returns
  A map containing tensor representations of the substrate
  """
  def from_substrate(%Substrate{} = substrate) do
    # Convert node positions to tensors
    input_positions = nodes_to_tensor(substrate.input_positions)
    hidden_positions = nodes_to_tensor(substrate.hidden_positions)
    output_positions = nodes_to_tensor(substrate.output_positions)
    
    # Store additional metadata
    %{
      input_positions: input_positions,
      hidden_positions: hidden_positions,
      output_positions: output_positions,
      dimensions: substrate.dimensions,
      geometry_type: substrate.geometry_type,
      connection_threshold: substrate.distance_threshold,
      boundary_conditions: substrate.boundary_conditions
    }
  end
  
  @doc """
  Generates all possible connection queries between layers in a vectorized manner.
  
  ## Parameters
  - source_positions: Tensor of source node positions
  - target_positions: Tensor of target node positions
  
  ## Returns
  Tensor of all possible connection queries
  """
  defn generate_connection_queries(source_positions, target_positions) do
    # Get dimensions
    source_count = Nx.axis_size(source_positions, 0)
    target_count = Nx.axis_size(target_positions, 0)
    pos_dims = Nx.axis_size(source_positions, 1)
    
    # Reshape for broadcasting
    sources = Nx.reshape(source_positions, {source_count, 1, pos_dims})
    targets = Nx.reshape(target_positions, {1, target_count, pos_dims})
    
    # Broadcast to create all pairs
    sources_broadcast = Nx.broadcast(sources, {source_count, target_count, pos_dims})
    targets_broadcast = Nx.broadcast(targets, {source_count, target_count, pos_dims})
    
    # Combine source and target positions
    queries = Nx.concatenate([
      Nx.reshape(sources_broadcast, {source_count * target_count, pos_dims}),
      Nx.reshape(targets_broadcast, {source_count * target_count, pos_dims})
    ], axis: 1)
    
    queries
  end
  
  @doc """
  Computes distances between all pairs of nodes in a vectorized manner.
  
  ## Parameters
  - source_positions: Tensor of source node positions
  - target_positions: Tensor of target node positions
  - distance_function: Type of distance function to use (:euclidean, :manhattan, etc.)
  
  ## Returns
  Tensor of distances between all pairs of nodes
  """
  defn compute_distances(source_positions, target_positions, distance_function \\ :euclidean) do
    # Get dimensions
    source_count = Nx.axis_size(source_positions, 0)
    target_count = Nx.axis_size(target_positions, 0)
    pos_dims = Nx.axis_size(source_positions, 1)
    
    # Reshape for broadcasting
    sources = Nx.reshape(source_positions, {source_count, 1, pos_dims})
    targets = Nx.reshape(target_positions, {1, target_count, pos_dims})
    
    # Compute differences
    differences = sources - targets
    
    # Compute distances based on the selected distance function
    case distance_function do
      :euclidean ->
        squared_diffs = Nx.pow(differences, 2)
        Nx.sqrt(Nx.sum(squared_diffs, axes: [2]))
        
      :manhattan ->
        Nx.sum(Nx.abs(differences), axes: [2])
        
      :chebyshev ->
        Nx.reduce_max(Nx.abs(differences), axes: [2])
        
      _ ->
        # Default to euclidean
        squared_diffs = Nx.pow(differences, 2)
        Nx.sqrt(Nx.sum(squared_diffs, axes: [2]))
    end
  end
  
  @doc """
  Applies a connection threshold to a weight matrix in a vectorized manner.
  
  ## Parameters
  - weight_matrix: Tensor of connection weights
  - threshold: Threshold value for connections
  - absolute: Whether to use absolute value for thresholding
  
  ## Returns
  Binary connectivity matrix
  """
  defn apply_threshold(weight_matrix, threshold, absolute \\ true) do
    if absolute do
      Nx.greater(Nx.abs(weight_matrix), threshold)
    else
      Nx.greater(weight_matrix, threshold)
    end
  end
  
  @doc """
  Finds the k-nearest neighbors for each node in a set of positions.
  
  ## Parameters
  - positions: Tensor of node positions
  - k: Number of neighbors to find
  
  ## Returns
  Indices of k-nearest neighbors for each node
  """
  defn k_nearest_neighbors(positions, k) do
    # Compute pairwise distances
    distances = compute_distances(positions, positions)
    
    # Set diagonal to infinity to exclude self-connections
    node_count = Nx.axis_size(positions, 0)
    diagonal_indices = Nx.stack([Nx.iota({node_count}), Nx.iota({node_count})], axis: 1)
    distances = Nx.indexed_put(distances, diagonal_indices, Nx.Constants.infinity())
    
    # Get indices of k smallest distances for each node
    Nx.argsort(distances, axis: 1)
    |> Nx.slice([0, 0], [node_count, k])
  end
  
  @doc """
  Generates a small-world network connectivity pattern in a vectorized manner.
  
  ## Parameters
  - positions: Tensor of node positions
  - k: Initial number of neighbors (must be even)
  - rewiring_prob: Probability of rewiring each edge
  
  ## Returns
  Binary connectivity matrix
  """
  def generate_small_world_connectivity(positions, k \\ 4, rewiring_prob \\ 0.1) do
    node_count = Nx.axis_size(positions, 0)
    
    # Create initial regular lattice connectivity
    # Each node is connected to its k nearest neighbors
    nearest_neighbors = k_nearest_neighbors(positions, k)
    
    # Initialize connectivity matrix
    connectivity = Nx.broadcast(0, {node_count, node_count})
    
    # Set connections to nearest neighbors
    # This would be a loop in imperative code, but we need a different approach in defn
    # For simplicity, we'll return the indices and handle the matrix creation outside defn
    Enum.reduce(0..(node_count - 1), connectivity, fn source_idx, conn ->
      # Get indices of k nearest neighbors for this source
      neighbor_indices = Nx.slice(nearest_neighbors, [source_idx, 0], [1, k])
      
      # Create a mask for the k nearest neighbors
      mask = Nx.broadcast(0, {node_count})
      mask = Nx.indexed_put(mask, Nx.new_axis(neighbor_indices, 1), Nx.broadcast(1, {k}))
      
      # Update connectivity matrix for this source
      Nx.indexed_put(conn, Nx.tensor([[source_idx]]), Nx.reshape(mask, {1, node_count}))
    end)
    |> apply_rewiring(rewiring_prob)
  end
  
  defp apply_rewiring(connectivity, rewiring_prob) do
    # Apply small-world rewiring to the connectivity matrix
    # With probability p, rewire each connection to a random target
    n_sources = Nx.axis_size(connectivity, 0)
    n_targets = Nx.axis_size(connectivity, 1)
    
    # Generate random values for rewiring decisions
    rewire_mask = Nx.less(Nx.Random.uniform({n_sources, n_targets}), rewiring_prob)
    
    # Only consider rewiring existing connections
    rewire_mask = Nx.logical_and(rewire_mask, Nx.equal(connectivity, 1))
    
    # Count connections to rewire
    num_rewires = Nx.sum(rewire_mask) |> Nx.to_number()
    
    if num_rewires > 0 do
      # Find indices of connections to rewire
      # Custom implementation to find non-zero indices
      rewire_indices = rewire_mask
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.filter(fn {val, _idx} -> val > 0 end)
        |> Enum.map(fn {_val, idx} -> 
          # Convert flat index to 2D coordinates
          row = div(idx, n_targets)
          col = rem(idx, n_targets)
          [row, col]
        end)
        |> Nx.tensor()
      
      # Remove these connections
      connectivity_without_rewired = Nx.select(
        rewire_mask,
        Nx.broadcast(0, {n_sources, n_targets}),
        connectivity
      )
      
      # Create new random connections
      Enum.reduce(0..(num_rewires - 1), connectivity_without_rewired, fn i, conn ->
        # Get source index for this rewire
        source_idx = Nx.to_number(Nx.slice(rewire_indices, [i, 0], [1, 1]))
        
        # Choose a random target that isn't already connected
        current_connections = Nx.slice(conn, [source_idx, 0], [1, n_targets])
        available_targets = Nx.logical_not(Nx.equal(current_connections, 1))
        
        # If there are available targets, choose one randomly
        if Nx.to_number(Nx.sum(available_targets)) > 0 do
          # Create probability distribution over available targets
          probs = Nx.select(
            available_targets,
            Nx.broadcast(1.0, {1, n_targets}),
            Nx.broadcast(0.0, {1, n_targets})
          )
          probs = probs / Nx.sum(probs)
          
          # Sample a target based on probabilities
          target_idx = sample_categorical(probs)
          
          # Add new connection
          Nx.indexed_put(conn, Nx.tensor([[source_idx, target_idx]]), Nx.tensor(1))
        else
          # No available targets, keep as is
          conn
        end
      end)
    else
      # No connections to rewire
      connectivity
    end
  end
  
  defp sample_categorical(probs) do
    # Sample from categorical distribution defined by probs
    # This is a simplified implementation that works for 1D probability vectors
    n = Nx.axis_size(probs, 1)
    
    # Generate a random value
    r = Nx.Random.uniform({1}) |> Nx.to_number()
    
    # Convert probs to cumulative distribution
    cumulative = Nx.cumulative_sum(probs, axis: 1)
    
    # Find the first index where cumulative exceeds r
    Enum.find_index(Nx.to_flat_list(cumulative), fn p -> p >= r end) || (n - 1)
  end
  
  @doc """
  Creates a genome from a substrate and CPPN in a vectorized manner.
  
  ## Parameters
  - substrate: The substrate struct or tensor representation
  - cppn_model: The Axon CPPN model
  - cppn_params: The CPPN model parameters
  - opts: Additional options
  
  ## Returns
  A genome representing the substrate network
  """
  def create_genome_from_substrate(substrate, {cppn_model, cppn_params}, opts \\ []) do
    # Extract substrate data
    {input_positions, hidden_positions, output_positions} = extract_positions(substrate)
    
    # Generate all possible connections
    input_hidden_queries = generate_connection_queries(input_positions, hidden_positions)
    hidden_hidden_queries = generate_connection_queries(hidden_positions, hidden_positions)
    hidden_output_queries = generate_connection_queries(hidden_positions, output_positions)
    
    # Process queries through CPPN
    input_hidden_weights = AxonCPPN.batch_process_queries(cppn_model, cppn_params, input_hidden_queries)
    hidden_hidden_weights = AxonCPPN.batch_process_queries(cppn_model, cppn_params, hidden_hidden_queries)
    hidden_output_weights = AxonCPPN.batch_process_queries(cppn_model, cppn_params, hidden_output_queries)
    
    # Apply threshold
    threshold = Keyword.get(opts, :threshold, 0.2)
    input_hidden_conn = apply_threshold(input_hidden_weights, threshold)
    hidden_hidden_conn = apply_threshold(hidden_hidden_weights, threshold)
    hidden_output_conn = apply_threshold(hidden_output_weights, threshold)
    
    # Create genome
    create_genome_from_matrices(
      input_positions, 
      hidden_positions, 
      output_positions,
      input_hidden_conn,
      input_hidden_weights,
      hidden_hidden_conn,
      hidden_hidden_weights,
      hidden_output_conn,
      hidden_output_weights,
      opts
    )
  end
  
  # Private functions
  
  defp nodes_to_tensor(positions) do
    # Convert list of position tuples to tensor
    positions
    |> Enum.map(&Tuple.to_list/1)
    |> Nx.tensor()
  end
  
  defp extract_positions(substrate) do
    case substrate do
      %Substrate{} = s ->
        {
          nodes_to_tensor(s.input_positions),
          nodes_to_tensor(s.hidden_positions),
          nodes_to_tensor(s.output_positions)
        }
      
      %{input_positions: i, hidden_positions: h, output_positions: o} ->
        {i, h, o}
      
      _ ->
        raise "Invalid substrate format"
    end
  end
  
  defp create_genome_from_matrices(
    input_positions, 
    hidden_positions, 
    output_positions,
    input_hidden_conn,
    input_hidden_weights,
    hidden_hidden_conn,
    hidden_hidden_weights,
    hidden_output_conn,
    hidden_output_weights,
    opts
  ) do
    # Create nodes
    input_nodes = create_nodes(input_positions, :input, 1)
    hidden_nodes = create_nodes(hidden_positions, :hidden, length(input_nodes) + 1)
    output_nodes = create_nodes(output_positions, :output, length(input_nodes) + length(hidden_nodes) + 1)
    
    all_nodes = input_nodes ++ hidden_nodes ++ output_nodes
    
    # Create node map for easy lookup
    nodes_map = Map.new(all_nodes, fn node -> {node.id, node} end)
    
    # Create connections
    input_hidden_connections = create_connections(
      input_nodes, 
      hidden_nodes, 
      input_hidden_conn, 
      input_hidden_weights
    )
    
    hidden_hidden_connections = create_connections(
      hidden_nodes, 
      hidden_nodes, 
      hidden_hidden_conn, 
      hidden_hidden_weights
    )
    
    hidden_output_connections = create_connections(
      hidden_nodes, 
      output_nodes, 
      hidden_output_conn, 
      hidden_output_weights
    )
    
    all_connections = input_hidden_connections ++ hidden_hidden_connections ++ hidden_output_connections
    
    # Create genome
    activation = Keyword.get(opts, :activation, :tanh)
    
    Genome.new(
      length(input_nodes),
      length(output_nodes),
      [
        nodes: nodes_map,
        connections: Map.new(all_connections, fn conn -> {{conn.from, conn.to}, conn} end),
        activation_function: activation
      ]
    )
  end
  
  defp create_nodes(positions, type, start_id) do
    positions
    |> Nx.to_list()
    |> Enum.with_index(start_id)
    |> Enum.map(fn {pos, id} -> 
      Node.new(id, type, :tanh, List.to_tuple(pos))
    end)
  end
  
  defp create_connections(source_nodes, target_nodes, connectivity_matrix, weight_matrix) do
    connectivity_list = Nx.to_list(connectivity_matrix)
    weight_list = Nx.to_list(weight_matrix)
    
    for {source, s_idx} <- Enum.with_index(source_nodes),
        {target, t_idx} <- Enum.with_index(target_nodes),
        connectivity_list |> Enum.at(s_idx) |> Enum.at(t_idx) == 1 do
      
      weight = weight_list |> Enum.at(s_idx) |> Enum.at(t_idx)
      
      %Connection{
        from: source.id,
        to: target.id,
        weight: weight,
        enabled: true,
        innovation: generate_innovation_number(source.id, target.id)
      }
    end
  end
  
  defp generate_innovation_number(source_id, target_id) do
    # Simple hash function for innovation numbers
    Integer.to_string(source_id * 1000 + target_id)
  end
end
