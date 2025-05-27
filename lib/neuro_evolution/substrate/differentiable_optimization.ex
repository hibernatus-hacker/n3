defmodule NeuroEvolution.Substrate.DifferentiableOptimization do
  @moduledoc """
  End-to-end differentiable optimization for substrate-based neuroevolution.
  
  This module provides gradient-based optimization capabilities for CPPNs and substrate
  networks, enabling more efficient exploration of the search space compared to
  traditional evolutionary approaches.
  """

  alias NeuroEvolution.Substrate.{AxonCPPN, VectorizedSubstrate, GPUSpatialOps}
  import Nx.Defn
  
  @doc """
  Defines a loss function for substrate optimization.
  
  ## Parameters
  - predictions: Predicted outputs from the substrate network
  - targets: Target outputs
  - loss_type: Type of loss function (:mse, :mae, :huber, etc.)
  
  ## Returns
  Loss value
  """
  defn substrate_loss(predictions, targets, loss_type \\ :mse) do
    case loss_type do
      :mse ->
        Nx.mean(Nx.pow(predictions - targets, 2))
        
      :mae ->
        Nx.mean(Nx.abs(predictions - targets))
        
      :huber ->
        # Huber loss with delta = 1.0
        delta = 1.0
        abs_diff = Nx.abs(predictions - targets)
        squared_loss = 0.5 * Nx.pow(abs_diff, 2)
        linear_loss = delta * (abs_diff - 0.5 * delta)
        
        mask = Nx.less_equal(abs_diff, delta)
        loss = mask * squared_loss + (1 - mask) * linear_loss
        Nx.mean(loss)
        
      _ ->
        # Default to MSE
        Nx.mean(Nx.pow(predictions - targets, 2))
    end
  end
  
  @doc """
  Performs a single optimization step for CPPN parameters.
  
  ## Parameters
  - model: The Axon CPPN model
  - params: Current model parameters
  - inputs: Training inputs
  - targets: Training targets
  - learning_rate: Learning rate for optimization
  - loss_type: Type of loss function
  
  ## Returns
  Tuple of {loss, updated_params}
  """
  defn optimize_step(model, params, inputs, targets, learning_rate \\ 0.01, loss_type \\ :mse) do
    {loss, gradients} = value_and_grad(params, fn p -> 
      predictions = Axon.predict(model, p, inputs)
      substrate_loss(predictions, targets, loss_type)
    end)
    
    # Update parameters using gradients
    # Replace Nx.Defn.Kernel.map with Enum.zip_with
    updated_params = Enum.zip_with(params, gradients, fn param, grad ->
      Nx.subtract(param, Nx.multiply(learning_rate, grad))
    end)
    
    {loss, updated_params}
  end
  
  @doc """
  Runs a full optimization loop for CPPN parameters.
  
  ## Parameters
  - model: The Axon CPPN model
  - initial_params: Initial model parameters
  - dataset: Tuple of {inputs, targets}
  - opts: Optimization options
  
  ## Options
  - iterations: Number of iterations (default: 1000)
  - learning_rate: Learning rate (default: 0.01)
  - loss_type: Type of loss function (default: :mse)
  - batch_size: Batch size for training (default: 32)
  - verbose: Whether to print progress (default: true)
  
  ## Returns
  Tuple of {final_loss, optimized_params}
  """
  def optimize_cppn(model, initial_params, dataset, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 1000)
    learning_rate = Keyword.get(opts, :learning_rate, 0.01)
    loss_type = Keyword.get(opts, :loss_type, :mse)
    batch_size = Keyword.get(opts, :batch_size, 32)
    verbose = Keyword.get(opts, :verbose, true)
    
    # Extract inputs and targets from dataset
    {inputs, targets} = dataset
    
    # Convert to tensors if not already
    inputs_tensor = to_tensor(inputs)
    targets_tensor = to_tensor(targets)
    
    # Create batches
    batches = create_batches(inputs_tensor, targets_tensor, batch_size)
    
    # Optimization loop
    Enum.reduce(1..iterations, {0, initial_params}, fn i, {_prev_loss, params} ->
      # Get a batch
      {batch_inputs, batch_targets} = get_batch(batches, i, batch_size)
      
      # Perform optimization step
      {loss, new_params} = optimize_step(model, params, batch_inputs, batch_targets, learning_rate, loss_type)
      
      # Print progress if verbose
      if verbose and rem(i, max(1, div(iterations, 10))) == 0 do
        IO.puts("Iteration #{i}/#{iterations}, Loss: #{Nx.to_number(loss)}")
      end
      
      {loss, new_params}
    end)
  end
  
  @doc """
  Optimizes a substrate network end-to-end for a specific task.
  
  ## Parameters
  - substrate: The substrate configuration
  - cppn: Tuple of {model, params} for the CPPN
  - task_fn: Function that evaluates the substrate network on a task
  - opts: Optimization options
  
  ## Options
  - iterations: Number of iterations (default: 100)
  - learning_rate: Learning rate (default: 0.01)
  - connection_threshold: Threshold for connections (default: 0.2)
  - batch_size: Batch size for training (default: 16)
  - verbose: Whether to print progress (default: true)
  
  ## Returns
  Tuple of {optimized_substrate, optimized_cppn}
  """
  def optimize_substrate_for_task(substrate, {cppn_model, cppn_params}, task_fn, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 100)
    learning_rate = Keyword.get(opts, :learning_rate, 0.01)
    connection_threshold = Keyword.get(opts, :connection_threshold, 0.2)
    verbose = Keyword.get(opts, :verbose, true)
    
    # Vectorize substrate
    vectorized_substrate = VectorizedSubstrate.from_substrate(substrate)
    
    # Optimization loop
    Enum.reduce(1..iterations, {vectorized_substrate, {cppn_model, cppn_params}}, fn i, {current_substrate, {model, params}} ->
      # Generate substrate network from current CPPN
      genome = VectorizedSubstrate.create_genome_from_substrate(
        current_substrate,
        {model, params},
        [threshold: connection_threshold]
      )
      
      # Evaluate on task to get training data
      {inputs, expected_outputs} = task_fn.(genome)
      
      # Convert to tensors
      inputs_tensor = to_tensor(inputs)
      targets_tensor = to_tensor(expected_outputs)
      
      # Perform optimization step
      {loss, new_params} = optimize_step(model, params, inputs_tensor, targets_tensor, learning_rate)
      
      # Print progress if verbose
      if verbose and rem(i, max(1, div(iterations, 10))) == 0 do
        IO.puts("Iteration #{i}/#{iterations}, Loss: #{Nx.to_number(loss)}")
      end
      
      # Return updated substrate and CPPN
      {current_substrate, {model, new_params}}
    end)
  end
  
  @doc """
  Performs gradient-based optimization of substrate topology.
  
  This is a more advanced technique that optimizes both the CPPN parameters
  and the substrate topology simultaneously.
  
  ## Parameters
  - substrate: Initial substrate configuration
  - cppn: Tuple of {model, params} for the CPPN
  - task_fn: Function that evaluates the substrate network on a task
  - opts: Optimization options
  
  ## Returns
  Tuple of {optimized_substrate, optimized_cppn}
  """
  def optimize_substrate_topology(substrate, {cppn_model, cppn_params}, task_fn, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 50)
    learning_rate = Keyword.get(opts, :learning_rate, 0.01)
    topology_iterations = Keyword.get(opts, :topology_iterations, 5)
    verbose = Keyword.get(opts, :verbose, true)
    
    # Vectorize substrate
    vectorized_substrate = VectorizedSubstrate.from_substrate(substrate)
    
    # Outer loop for topology optimization
    Enum.reduce(1..topology_iterations, {vectorized_substrate, {cppn_model, cppn_params}}, fn t, {current_substrate, current_cppn} ->
      # Inner loop for CPPN parameter optimization
      {optimized_substrate, optimized_cppn} = optimize_substrate_for_task(
        current_substrate,
        current_cppn,
        task_fn,
        [iterations: iterations, learning_rate: learning_rate, verbose: verbose]
      )
      
      # Adapt substrate topology based on CPPN outputs
      adapted_substrate = adapt_substrate_topology(optimized_substrate, optimized_cppn)
      
      # Print progress if verbose
      if verbose do
        IO.puts("Topology Iteration #{t}/#{topology_iterations} completed")
      end
      
      {adapted_substrate, optimized_cppn}
    end)
  end
  
  @doc """
  Performs meta-learning for CPPN architectures.
  
  This approach optimizes the CPPN architecture itself, not just its parameters,
  to find more effective encoding strategies for substrate networks.
  
  ## Parameters
  - initial_cppn: Initial CPPN configuration
  - task_fn: Function that evaluates the substrate network on a task
  
  ## Returns
  Optimized CPPN architecture
  """
  def meta_learn_cppn_architecture(initial_cppn, _task_fn, _opts \\ []) do
    # This is a placeholder implementation
    # In a real implementation, we would use techniques like neural architecture search
    # or evolutionary strategies to optimize the CPPN architecture
    initial_cppn
  end
  
  # Private functions
  
  defp to_tensor(data) do
    if is_struct(data, Nx.Tensor) do
      data
    else
      Nx.tensor(data)
    end
  end
  
  defp create_batches(inputs, targets, batch_size) do
    # Get total size
    n = Nx.axis_size(inputs, 0)
    
    # Calculate number of batches
    num_batches = div(n + batch_size - 1, batch_size)
    
    # Create indices for each batch
    Enum.map(0..(num_batches - 1), fn i ->
      start_idx = i * batch_size
      end_idx = min((i + 1) * batch_size, n)
      
      batch_inputs = Nx.slice(inputs, [start_idx, 0], [end_idx - start_idx, Nx.axis_size(inputs, 1)])
      batch_targets = Nx.slice(targets, [start_idx, 0], [end_idx - start_idx, Nx.axis_size(targets, 1)])
      
      {batch_inputs, batch_targets}
    end)
  end
  
  defp get_batch(batches, iteration, _batch_size) do
    # Get batch index, cycling through available batches
    batch_idx = rem(iteration - 1, length(batches))
    Enum.at(batches, batch_idx)
  end
  
  defp adapt_substrate_topology(substrate, {cppn_model, cppn_params}) do
    # Analyze CPPN outputs to identify regions of high complexity and adapt substrate topology
    # This implementation uses an adaptive subdivision approach based on CPPN output variance
    
    # Extract substrate dimensions and node positions
    _input_nodes = substrate.input_nodes
    hidden_nodes = substrate.hidden_nodes
    _output_nodes = substrate.output_nodes
    
    # Create a grid of query points for sampling the CPPN
    resolution = 20  # Number of points per dimension
    grid = create_sampling_grid(substrate, resolution)
    
    # Sample CPPN at grid points
    cppn_outputs = sample_cppn_at_points(grid, {cppn_model, cppn_params})
    
    # Identify regions of high complexity using GPUSpatialOps
    high_complexity_regions = identify_high_complexity_regions(grid, cppn_outputs)
    
    # Generate new nodes in high complexity regions
    new_hidden_nodes = generate_nodes_in_complex_regions(high_complexity_regions, hidden_nodes)
    
    # Update substrate with new nodes
    %{substrate | hidden_nodes: hidden_nodes ++ new_hidden_nodes}
  end
  
  defp create_sampling_grid(substrate, resolution) do
    # Determine dimensionality from substrate
    dims = case List.first(substrate.input_nodes) do
      {_x} -> 1
      {_x, _y} -> 2
      {_x, _y, _z} -> 3
      _ -> 2  # Default to 2D if unknown
    end
    
    # Create grid based on dimensionality
    case dims do
      1 ->
        # 1D grid
        Nx.linspace(-1.0, 1.0, n: resolution)
        |> Nx.reshape({resolution, 1})
        
      2 ->
        # 2D grid
        x_coords = Nx.linspace(-1.0, 1.0, n: resolution)
        y_coords = Nx.linspace(-1.0, 1.0, n: resolution)
        
        # Create meshgrid
        {x_grid, y_grid} = Nx.broadcast_vectors([x_coords, y_coords])
        
        # Reshape to 2D points
        x_flat = Nx.reshape(x_grid, {:auto, 1})
        y_flat = Nx.reshape(y_grid, {:auto, 1})
        
        Nx.concatenate([x_flat, y_flat], axis: 1)
        
      3 ->
        # 3D grid
        x_coords = Nx.linspace(-1.0, 1.0, n: resolution)
        y_coords = Nx.linspace(-1.0, 1.0, n: resolution)
        z_coords = Nx.linspace(-1.0, 1.0, n: resolution)
        
        # Create meshgrid (simplified for 3D)
        # This is a simplified approach - in practice we might use a more efficient method
        points = for x <- Nx.to_flat_list(x_coords),
                     y <- Nx.to_flat_list(y_coords),
                     z <- Nx.to_flat_list(z_coords) do
          [x, y, z]
        end
        
        Nx.tensor(points)
    end
  end
  
  defp sample_cppn_at_points(grid, {cppn_model, cppn_params}) do
    # Sample CPPN at all grid points
    AxonCPPN.batch_process_queries(cppn_model, cppn_params, grid)
  end
  
  defp identify_high_complexity_regions(grid, cppn_outputs) do
    # Use GPUSpatialOps to identify regions of high complexity
    # This is based on variance in CPPN outputs
    variance_threshold = 0.1  # Threshold for considering a region as high complexity
    max_depth = 3  # Maximum subdivision depth
    
    # Call GPUSpatialOps to identify high complexity regions
    GPUSpatialOps.identify_high_variance_regions(grid, cppn_outputs, variance_threshold, max_depth)
  end
  
  defp generate_nodes_in_complex_regions(high_complexity_regions, existing_nodes) do
    # Generate new nodes in high complexity regions
    # Avoid placing nodes too close to existing ones
    min_distance = 0.1  # Minimum distance between nodes
    
    # Convert existing nodes to tensor for efficient distance computation
    existing_nodes_tensor = nodes_to_tensor(existing_nodes)
    
    # Filter points that are too close to existing nodes
    filtered_points = filter_by_distance(high_complexity_regions, existing_nodes_tensor, min_distance)
    
    # Convert filtered points back to node format
    tensor_to_nodes(filtered_points)
  end
  
  defp nodes_to_tensor(nodes) do
    # Convert list of node coordinates to tensor
    # Determine dimensionality from first node
    _dims = case List.first(nodes) do
      {_x} -> 1
      {_x, _y} -> 2
      {_x, _y, _z} -> 3
      _ -> 2  # Default to 2D if unknown
    end
    
    # Convert nodes to list of lists
    points = Enum.map(nodes, fn
      {x} -> [x]
      {x, y} -> [x, y]
      {x, y, z} -> [x, y, z]
      _ -> [0.0, 0.0]  # Default for unknown format
    end)
    
    # Convert to tensor
    Nx.tensor(points)
  end
  
  defp tensor_to_nodes(tensor) do
    # Convert tensor of coordinates back to node format
    # Determine dimensionality from tensor shape
    _dims = Nx.axis_size(tensor, 1)
    
    # Convert tensor to list of tuples
    tensor
    |> Nx.to_list()
    |> Enum.map(fn
      [x] -> {x}
      [x, y] -> {x, y}
      [x, y, z] -> {x, y, z}
      _ -> {0.0, 0.0}  # Default for unknown format
    end)
  end
  
  defp filter_by_distance(points, existing_points, min_distance) do
    # Filter points that are too close to existing points
    # Compute pairwise distances
    distances = GPUSpatialOps.compute_pairwise_distances(points, existing_points)
    
    # Find minimum distance for each point to any existing point
    min_distances = Nx.reduce_min(distances, axes: [1])
    
    # Create mask for points that are far enough from existing points
    mask = Nx.greater_equal(min_distances, min_distance)
    
    # Apply mask to filter points
    # Custom implementation to find non-zero indices
    indices = mask
      |> Nx.flatten()
      |> Nx.to_flat_list()
      |> Enum.with_index()
      |> Enum.filter(fn {val, _idx} -> val > 0 end)
      |> Enum.map(fn {_val, idx} -> idx end)
    
    Nx.take(points, Nx.tensor(indices))
  end
end
