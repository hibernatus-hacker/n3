defmodule NeuroEvolution.Substrate.GPUSpatialOps do
  @moduledoc """
  GPU-accelerated spatial operations for substrate-based neuroevolution.
  
  This module provides efficient implementations of spatial operations
  using Nx for GPU acceleration, including:
  - Distance calculations
  - Density distributions
  - Spatial interpolation
  - Dimensionality reduction
  """
  
  alias NeuroEvolution.Substrate.AxonCPPN
  import Nx.Defn
  
  # Configure to use GPU when available
  # Commented out unused module attribute
  # @default_defn_options [compiler: EXLA]
  
  @doc """
  Computes pairwise distances between two sets of points in a vectorized manner.
  
  ## Parameters
  - points1: First set of points as a tensor
  - points2: Second set of points as a tensor
  - distance_type: Type of distance metric to use (:euclidean, :manhattan, :cosine)
  
  ## Returns
  A tensor containing pairwise distances
  """
  defn compute_pairwise_distances(points1, points2, distance_fn \\ :euclidean) do
    # Compute pairwise distances between two sets of points
    # This is a vectorized implementation that avoids explicit loops
    
    # Get dimensions
    n = Nx.axis_size(points1, 0)
    m = Nx.axis_size(points2, 0)
    d = Nx.axis_size(points1, 1)
    
    # Reshape for broadcasting
    p1 = Nx.reshape(points1, {n, 1, d})
    p2 = Nx.reshape(points2, {1, m, d})
    
    # Compute distances based on the specified distance function
    case distance_fn do
      :euclidean ->
        # Euclidean distance: sqrt(sum((x_i - y_i)^2))
        diffs = p1 - p2
        squared_diffs = Nx.pow(diffs, 2)
        sum_squared_diffs = Nx.sum(squared_diffs, axes: [2])
        Nx.sqrt(sum_squared_diffs)
        
      :manhattan ->
        # Manhattan distance: sum(|x_i - y_i|)
        diffs = p1 - p2
        abs_diffs = Nx.abs(diffs)
        Nx.sum(abs_diffs, axes: [2])
        
      :cosine ->
        # Cosine distance: 1 - (x·y)/(||x||·||y||)
        # Compute dot products
        dot_products = Nx.sum(p1 * p2, axes: [2])
        
        # Compute norms
        norm1 = Nx.sqrt(Nx.sum(Nx.pow(p1, 2), axes: [2]))
        norm2 = Nx.sqrt(Nx.sum(Nx.pow(p2, 2), axes: [2]))
        
        # Compute cosine similarity
        similarity = dot_products / (norm1 * norm2)
        
        # Convert to distance
        1 - similarity
        
      _ ->
        # Default to Euclidean
        diffs = p1 - p2
        squared_diffs = Nx.pow(diffs, 2)
        sum_squared_diffs = Nx.sum(squared_diffs, axes: [2])
        Nx.sqrt(sum_squared_diffs)
    end
  end
  
  @doc """
  Creates a grid of points in the specified dimensions.
  
  ## Parameters
  - resolution: Number of points in each dimension
  - dimensions: Dimensionality of the grid (1D, 2D, or 3D)
  
  ## Returns
  A tuple of coordinate tensors
  """
  def create_grid_coordinates(resolution, dimensions \\ 2) do
    case {dimensions, resolution} do
      {1, res} when is_integer(res) ->
        # 1D grid
        x = Nx.linspace(-1.0, 1.0, res)
        {x}
        
      {2, {width, height}} ->
        # 2D grid
        x = Nx.linspace(-1.0, 1.0, width)
        y = Nx.linspace(-1.0, 1.0, height)
        {x, y}
        
      {2, res} when is_integer(res) ->
        # 2D grid with same resolution in both dimensions
        x = Nx.linspace(-1.0, 1.0, res)
        y = Nx.linspace(-1.0, 1.0, res)
        {x, y}
        
      {3, {width, height, depth}} ->
        # 3D grid
        x = Nx.linspace(-1.0, 1.0, width)
        y = Nx.linspace(-1.0, 1.0, height)
        z = Nx.linspace(-1.0, 1.0, depth)
        {x, y, z}
        
      {3, res} when is_integer(res) ->
        # 3D grid with same resolution in all dimensions
        x = Nx.linspace(-1.0, 1.0, res)
        y = Nx.linspace(-1.0, 1.0, res)
        z = Nx.linspace(-1.0, 1.0, res)
        {x, y, z}
        
      _ ->
        # Default to 2D grid
        x = Nx.linspace(-1.0, 1.0, 10)
        y = Nx.linspace(-1.0, 1.0, 10)
        {x, y}
    end
  end
  
  @doc """
  Generates a density distribution for points in a substrate space.
  
  ## Parameters
  - points: Set of points as a tensor
  - resolution: Resolution of the density grid as {width, height} or {width, height, depth}
  - sigma: Standard deviation for the Gaussian kernel
  
  ## Returns
  A tensor representing the density distribution
  """
  def generate_density_distribution(points, resolution, sigma \\ 1.0) do
    # Create grid coordinates
    {x_grid, y_grid} = create_grid_coordinates(resolution, 2)
    
    # Reshape grid coordinates for computation
    n_grid_points = Nx.size(x_grid)
    x_flat = Nx.reshape(x_grid, {n_grid_points, 1})
    y_flat = Nx.reshape(y_grid, {n_grid_points, 1})
    grid_points = Nx.concatenate([x_flat, y_flat], axis: 1)
    
    # Compute distances from each grid point to each input point
    distances = compute_pairwise_distances(grid_points, points)
    
    # Apply Gaussian kernel
    gaussian = Nx.exp(-Nx.pow(distances, 2) / (2 * sigma * sigma))
    
    # Sum contributions from all points
    density = Nx.sum(gaussian, axes: [1])
    
    # Normalize
    density / Nx.sum(density)
  end
  
  @doc """
  Performs spatial interpolation between points in the substrate.
  
  ## Parameters
  - points: Set of points with known values
  - values: Values at the known points
  - query_points: Points at which to interpolate
  - method: Interpolation method (:linear, :cubic, :nearest)
  
  ## Returns
  Interpolated values at the query points
  """
  defn interpolate(points, values, query_points, method \\ :linear) do
    # Compute distances from query points to known points
    distances = compute_pairwise_distances(query_points, points)
    
    case method do
      :nearest ->
        # Find index of nearest point for each query point
        nearest_indices = Nx.argmin(distances, axis: 1)
        
        # Get values of nearest points
        Nx.take(values, nearest_indices)
        
      :linear ->
        # Compute weights based on inverse distance
        weights = 1.0 / Nx.max(distances, 1.0e-10)
        
        # Normalize weights
        weights = weights / Nx.sum(weights, axes: [1], keep_axes: true)
        
        # Compute weighted average
        Nx.sum(weights * Nx.reshape(values, {1, Nx.size(values)}), axes: [1])
        
      :cubic ->
        # For cubic interpolation, we'd need a more complex implementation
        # This is a simplified version that falls back to linear
        weights = 1.0 / Nx.pow(Nx.max(distances, 1.0e-10), 2)
        weights = weights / Nx.sum(weights, axes: [1], keep_axes: true)
        Nx.sum(weights * Nx.reshape(values, {1, Nx.size(values)}), axes: [1])
        
      _ ->
        # Default to linear
        weights = 1.0 / Nx.max(distances, 1.0e-10)
        weights = weights / Nx.sum(weights, axes: [1], keep_axes: true)
        Nx.sum(weights * Nx.reshape(values, {1, Nx.size(values)}), axes: [1])
    end
  end
  
  @doc """
  Performs adaptive sampling of a substrate space based on variance.
  
  This is a key component of ES-HyperNEAT, which adaptively samples the substrate
  space based on the complexity of the underlying pattern.
  
  ## Parameters
  - cppn: Tuple of {model, params} for the CPPN
  - resolution: Initial sampling resolution as {width, height} or {width, height, depth}
  - variance_threshold: Threshold for further subdivision
  - max_depth: Maximum recursion depth for subdivision
  
  ## Returns
  A tensor of sampled points
  """
  def adaptive_sample(cppn, resolution, variance_threshold \\ 0.1, max_depth \\ 4) do
    # Create initial grid
    {x_grid, y_grid} = create_grid_coordinates(resolution, 2)
    
    # Reshape grid coordinates for computation
    n_grid_points = Nx.size(x_grid)
    x_flat = Nx.reshape(x_grid, {n_grid_points, 1})
    y_flat = Nx.reshape(y_grid, {n_grid_points, 1})
    grid_points = Nx.concatenate([x_flat, y_flat], axis: 1)
    
    # Sample the CPPN at these points
    {model, params} = cppn
    initial_values = AxonCPPN.batch_process_queries(model, params, grid_points)
    
    # Recursively subdivide regions with high variance
    sampled_points = adaptive_subdivide(
      cppn,
      grid_points,
      initial_values,
      variance_threshold,
      max_depth,
      0
    )
    
    sampled_points
  end
  
  @doc """
  Recursively subdivides regions based on variance in CPPN outputs.
  
  ## Parameters
  - cppn: Tuple of {model, params} for the CPPN
  - grid: Current grid of points
  - values: CPPN outputs at grid points
  - variance_threshold: Threshold for further subdivision
  - max_depth: Maximum recursion depth
  - current_depth: Current recursion depth
  
  ## Returns
  A tensor of adaptively sampled points
  """
  def adaptive_subdivide(cppn, grid, values, variance_threshold, max_depth, current_depth) do
    # Base case: maximum depth reached
    if current_depth >= max_depth do
      grid
    else
      # Compute variance in each region
      variances = compute_regional_variance(grid, values)
      
      # Identify regions with variance above threshold
      high_variance_mask = Nx.greater(variances, variance_threshold)
      
      # If no high variance regions, return current grid
      if Nx.sum(high_variance_mask) == 0 do
        grid
      else
        # Subdivide high variance regions
        new_points = subdivide_regions(grid, high_variance_mask)
        
        # Combine with existing grid
        combined_grid = Nx.concatenate([grid, new_points], axis: 0)
        
        # Sample CPPN at new points
        {model, params} = cppn
        new_values = AxonCPPN.batch_process_queries(model, params, new_points)
        combined_values = Nx.concatenate([values, new_values], axis: 0)
        
        # Recursive call with increased depth
        adaptive_subdivide(
          cppn,
          combined_grid,
          combined_values,
          variance_threshold,
          max_depth,
          current_depth + 1
        )
      end
    end
  end
  
  @doc """
  Identifies regions of high variance in CPPN outputs.
  
  ## Parameters
  - cppn: Tuple of {model, params} for the CPPN
  - grid: Grid of points to evaluate
  - variance_threshold: Threshold for high variance
  - max_regions: Maximum number of regions to return
  
  ## Returns
  A tensor of points in high variance regions
  """
  def identify_high_variance_regions(cppn, grid, variance_threshold \\ 0.1, max_regions \\ 10) do
    # Sample the CPPN at the grid points
    {model, params} = cppn
    values = AxonCPPN.batch_process_queries(model, params, grid)
    
    # Compute variance in each region
    variances = compute_regional_variance(grid, values)
    
    # Identify regions with variance above threshold
    high_variance_mask = Nx.greater(variances, variance_threshold)
    
    # Find indices where mask is true (equivalent to argwhere)
    indices = Nx.flatten(Nx.argsort(high_variance_mask, direction: :desc))
    high_variance_count = Nx.sum(high_variance_mask) |> Nx.to_number()
    high_variance_indices = Nx.slice(indices, [0], [high_variance_count])
    
    # Limit to max_regions if necessary
    high_variance_indices = if high_variance_count > max_regions do
      Nx.slice(high_variance_indices, [0], [max_regions])
    else
      high_variance_indices
    end
    
    # Extract points in high variance regions
    high_variance_points = Nx.take(grid, high_variance_indices)
    
    high_variance_points
  end
  
  @doc """
  Implements t-SNE dimensionality reduction.
  
  ## Parameters
  - data: High-dimensional data tensor
  - output_dims: Dimensionality of the output (default: 2)
  - perplexity: Perplexity parameter for t-SNE (default: 30.0)
  - iterations: Number of iterations (default: 1000)
  
  ## Returns
  Low-dimensional embedding of the data
  """
  def tsne(data, output_dims \\ 2, perplexity \\ 30.0, iterations \\ 1000) do
    # Get dimensions
    n = Nx.axis_size(data, 0)
    
    # Step 1: Compute pairwise distances
    distances = compute_pairwise_distances(data, data)
    
    # Step 2: Convert distances to probabilities (P matrix)
    p_matrix = compute_p_matrix(distances, perplexity)
    
    # Step 3: Initialize low-dimensional embedding with small random values
    y = Nx.tensor(
      Enum.map(1..n, fn _ -> 
        Enum.map(1..output_dims, fn _ -> :rand.normal(0.0, 0.0001) end)
      end)
    )
    
    # Step 4: Gradient descent to optimize embedding
    learning_rate = 200.0
    momentum = 0.5
    prev_gradient = Nx.broadcast(0.0, {n, output_dims})
    
    # Perform gradient descent iterations
    y = Enum.reduce(1..iterations, y, fn iter, y_current ->
      # Compute pairwise distances in low-dimensional space
      y_distances = compute_pairwise_distances(y_current, y_current)
      
      # Compute Q matrix (t-distribution in low-dimensional space)
      q_matrix = compute_q_matrix(y_distances)
      
      # Compute gradient
      gradient = compute_tsne_gradient(p_matrix, q_matrix, y_current)
      
      # Update with momentum
      momentum_term = if iter > 20, do: 0.8, else: momentum
      update = momentum_term * prev_gradient - learning_rate * gradient
      
      # Apply update
      y_current + update
    end)
    
    y
  end
  
  # Private helper functions
  
  defp compute_regional_variance(grid, values) do
    # Compute variance in spatial regions
    # This is a simplified implementation that computes global variance
    # In a real implementation, we would use quadtree/octree to compute regional variance
    
    # Compute mean value
    mean = Nx.mean(values)
    
    # Compute squared differences from mean
    squared_diffs = Nx.pow(values - mean, 2)
    
    # Compute variance
    variance = Nx.mean(squared_diffs)
    
    # For now, return the same variance for all points
    Nx.broadcast(variance, {Nx.axis_size(grid, 0)})
  end
  
  defp subdivide_regions(grid, high_variance_mask) do
    # Subdivide regions with high variance by adding new points
    # Extract grid points in high variance regions
    dims = Nx.axis_size(grid, 1)
    
    # Find indices where mask is true (equivalent to argwhere)
    indices = Nx.flatten(Nx.argsort(high_variance_mask, direction: :desc))
    high_variance_count = Nx.sum(high_variance_mask) |> Nx.to_number()
    high_variance_indices = Nx.slice(indices, [0], [high_variance_count])
    
    # If no high variance regions, return empty tensor
    if high_variance_count == 0 do
      # Return empty tensor with same shape as grid but 0 rows
      Nx.broadcast(0, {0, dims})
    else
      # Extract points in high variance regions
      high_variance_points = Nx.take(grid, high_variance_indices)
      
      # For each high variance point, generate new points around it
      # This is a simplified approach; a more sophisticated approach would use
      # quadtree/octree subdivision
      
      # Define offsets for new points (in normalized coordinates)
      offsets = case dims do
        1 ->
          Nx.tensor([[-0.1], [0.1]])
        2 ->
          Nx.tensor([[-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1], [0.1, 0.1]])
        3 ->
          Nx.tensor([
            [-0.1, -0.1, -0.1], [-0.1, -0.1, 0.1], [-0.1, 0.1, -0.1], [-0.1, 0.1, 0.1],
            [0.1, -0.1, -0.1], [0.1, -0.1, 0.1], [0.1, 0.1, -0.1], [0.1, 0.1, 0.1]
          ])
        _ ->
          Nx.tensor([[-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1], [0.1, 0.1]])
      end
      
      # Generate new points by adding offsets to high variance points
      num_high_variance = Nx.axis_size(high_variance_points, 0)
      num_offsets = Nx.axis_size(offsets, 0)
      
      # Reshape for broadcasting
      points_expanded = Nx.reshape(high_variance_points, {num_high_variance, 1, dims})
      offsets_expanded = Nx.reshape(offsets, {1, num_offsets, dims})
      
      # Add offsets to points
      new_points = points_expanded + offsets_expanded
      
      # Reshape to 2D tensor
      new_points = Nx.reshape(new_points, {num_high_variance * num_offsets, dims})
      
      # Ensure new points are within bounds [-1, 1]
      new_points = Nx.min(new_points, 1.0)
      new_points = Nx.max(new_points, -1.0)
      
      new_points
    end
  end
  
  defp compute_p_matrix(distances, perplexity) do
    n = Nx.axis_size(distances, 0)
    
    # Set diagonal to infinity to ensure zero probability for self-pairs
    indices = Nx.tensor(Enum.map(0..(n-1), fn i -> [i, i] end))
    distances = Nx.indexed_put(distances, indices, Nx.broadcast(:infinity, {n}))
    
    # Convert distances to probabilities using Gaussian kernel
    # This is a simplified version; real t-SNE uses binary search for sigma
    sigma = perplexity / 3.0
    p_conditional = Nx.exp(-Nx.pow(distances, 2) / (2 * sigma * sigma))
    
    # Normalize rows to sum to 1
    row_sums = Nx.sum(p_conditional, axes: [1])
    row_sums = Nx.reshape(row_sums, {n, 1})
    p_conditional = p_conditional / row_sums
    
    # Symmetrize and normalize
    p_matrix = (p_conditional + Nx.transpose(p_conditional)) / (2 * n)
    
    # Add small epsilon to avoid numerical issues
    p_matrix = Nx.max(p_matrix, Nx.tensor(1.0e-12))
    
    p_matrix
  end
  
  defp compute_q_matrix(distances) do
    n = Nx.axis_size(distances, 0)
    
    # Set diagonal to infinity to ensure zero probability for self-pairs
    indices = Nx.tensor(Enum.map(0..(n-1), fn i -> [i, i] end))
    distances = Nx.indexed_put(distances, indices, Nx.broadcast(:infinity, {n}))
    
    # Convert distances to probabilities using t-distribution
    q_unnormalized = 1.0 / (1.0 + distances)
    
    # Normalize to sum to 1
    q_sum = Nx.sum(q_unnormalized)
    q_matrix = q_unnormalized / q_sum
    
    # Add small epsilon to avoid numerical issues
    q_matrix = Nx.max(q_matrix, Nx.tensor(1.0e-12))
    
    q_matrix
  end
  
  defp compute_tsne_gradient(p_matrix, q_matrix, y) do
    n = Nx.axis_size(y, 0)
    
    # Compute difference between P and Q
    pq_diff = p_matrix - q_matrix
    
    # Compute gradient
    gradient = Nx.broadcast(0.0, Nx.shape(y))
    
    # For each point, compute its contribution to the gradient
    Enum.reduce(0..(n-1), gradient, fn i, grad ->
      # Get all differences between point i and all other points
      y_i = Nx.slice(y, [i, 0], [1, Nx.axis_size(y, 1)])
      y_i = Nx.broadcast(y_i, Nx.shape(y))
      y_diffs = y - y_i
      
      # Get the PQ differences for point i
      pq_factors = Nx.slice(pq_diff, [i, 0], [1, n])
      pq_factors = Nx.reshape(pq_factors, {n, 1})
      
      # Compute gradient contribution for point i
      point_grad = Nx.sum(pq_factors * y_diffs, axes: [0])
      
      # Update gradient for point i
      Nx.indexed_put(grad, Nx.tensor([[i, 0]]), point_grad)
    end)
  end
end
