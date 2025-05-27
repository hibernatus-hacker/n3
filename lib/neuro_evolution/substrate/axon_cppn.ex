defmodule NeuroEvolution.Substrate.AxonCPPN do
  @moduledoc """
  Compositional Pattern Producing Network (CPPN) implementation using Axon for GPU acceleration.
  
  This module provides a high-performance implementation of CPPNs using Axon neural networks,
  enabling efficient substrate querying and weight generation for HyperNEAT.
  """

  # Remove unused aliases and imports
  require Nx

  @doc """
  Creates a new CPPN model using Axon.
  
  ## Parameters
  - input_dims: Number of input dimensions (depends on substrate dimensionality)
  - hidden_layers: List of hidden layer sizes
  - output_dims: Number of output dimensions (typically 1 for weight, 2 if using LEO)
  - opts: Additional options for the model
  
  ## Options
  - activation: Activation function for hidden layers (default: :tanh)
  - output_activation: Activation function for output layer (default: :tanh)
  - batch_size: Batch size for training (default: 32)
  - learning_rate: Learning rate for training (default: 0.01)
  
  ## Returns
  A tuple containing the Axon model and initialized parameters
  """
  def new(input_dims, hidden_layers \\ [16, 16], output_dims \\ 1, opts \\ []) do
    activation = Keyword.get(opts, :activation, :tanh)
    output_activation = Keyword.get(opts, :output_activation, :tanh)
    
    # Build the model
    model =
      Axon.input("input", shape: {nil, input_dims})
      |> build_hidden_layers(hidden_layers, activation)
      |> Axon.dense(output_dims, activation: output_activation)
    
    # Initialize parameters using the updated Axon.build function
    # This replaces the deprecated Axon.init function
    params = Axon.build(model, compiler: EXLA)
    
    {model, params}
  end
  
  @doc """
  Performs forward pass through the CPPN to generate weights or other outputs.
  
  ## Parameters
  - model: The Axon model
  - params: The model parameters
  - inputs: Input tensor or list of inputs
  
  ## Returns
  Output tensor from the CPPN
  """
  def forward(model, params, inputs) when is_list(inputs) do
    try do
      inputs_tensor = Nx.tensor(inputs)
      forward(model, params, inputs_tensor)
    rescue
      e in ArgumentError ->
        raise ArgumentError, message: "Failed to convert inputs to tensor: #{inspect(e.message)}"
      e ->
        raise "Unexpected error in CPPN forward pass: #{inspect(e)}"
    end
  end
  
  def forward(model, params, %Nx.Tensor{} = inputs) do
    try do
      # Ensure inputs are properly shaped for batch processing
      inputs = reshape_for_batch(inputs)
      Axon.predict(model, params, inputs)
    rescue
      # Handle shape errors without relying on Nx.ShapeError
      e in ArgumentError ->
        if String.contains?(inspect(e.message), "shape") do
          # This is likely a shape error
          raise ArgumentError, message: "Shape mismatch in CPPN forward pass: #{inspect(e.message)}"
        else
          raise ArgumentError, message: "Invalid arguments for CPPN forward pass: #{inspect(e.message)}"
        end
      e ->
        raise "Unexpected error in CPPN forward pass: #{inspect(e)}"
    end
  end
  
  @doc """
  Processes a batch of substrate queries through the CPPN.
  
  ## Parameters
  - model: The Axon model
  - params: The model parameters
  - queries: List of substrate coordinate pairs to query
  - opts: Additional options for batch processing
  
  ## Options
  - batch_size: Size of batches to process (default: 1000)
  - device: Device to use for computation (:cuda or :host, default: :cuda)
  
  ## Returns
  Tensor of weights/outputs for each query
  """
  def batch_process_queries(model, params, queries, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 1000)
    device = Keyword.get(opts, :device, :cuda)
    
    try do
      # Convert queries to tensor format
      queries_tensor = Nx.tensor(queries)
      
      # Ensure proper shape for batch processing
      queries_tensor = reshape_for_batch(queries_tensor)
      
      # Process in batches to manage memory efficiently
      if length(queries) > batch_size do
        process_in_batches(model, params, queries_tensor, batch_size, device)
      else
        # Run through the model
        Axon.predict(model, params, queries_tensor)
      end
    rescue
      e in RuntimeError ->
        if String.contains?(inspect(e.message), "memory") do
          # If memory error, try with smaller batch size or on CPU
          IO.puts("Warning: GPU memory issue detected. Falling back to smaller batches or CPU.")
          new_batch_size = div(batch_size, 2)
          new_device = if device == :cuda, do: :host, else: :host
          batch_process_queries(model, params, queries, [batch_size: new_batch_size, device: new_device])
        else
          reraise e, __STACKTRACE__
        end
      e ->
        reraise e, __STACKTRACE__
    end
  end
  
  @doc """
  Processes queries in batches to manage GPU memory efficiently.
  
  ## Parameters
  - model: The Axon model
  - params: The model parameters
  - queries_tensor: Tensor of queries to process
  - batch_size: Size of batches to process
  - device: Device to use for computation
  
  ## Returns
  Tensor of weights/outputs for all queries
  """
  def process_in_batches(model, params, queries_tensor, batch_size, _device) do
    # Get total number of queries
    num_queries = Nx.axis_size(queries_tensor, 0)
    
    # Calculate number of batches
    num_batches = div(num_queries + batch_size - 1, batch_size)
    
    # Process each batch and collect results
    results = Enum.map(0..(num_batches - 1), fn batch_idx ->
      # Calculate start and end indices for this batch
      start_idx = batch_idx * batch_size
      end_idx = min((batch_idx + 1) * batch_size, num_queries)
      batch_size = end_idx - start_idx
      
      # Extract batch
      batch = Nx.slice(queries_tensor, [start_idx, 0], [batch_size, Nx.axis_size(queries_tensor, 1)])
      
      # Process batch
      Axon.predict(model, params, batch)
    end)
    
    # Concatenate results
    Nx.concatenate(results)
  end
  
  @doc """
  Generates a complete connectivity matrix for a substrate using vectorized operations.
  
  ## Parameters
  - model: The Axon model
  - params: The model parameters
  - source_positions: List of source node positions
  - target_positions: List of target node positions
  - opts: Additional options
  
  ## Options
  - threshold: Connection threshold (default: 0.2)
  - batch_size: Size of batches for processing (default: 1000)
  - leo_enabled: Whether to use Link Expression Output (default: false)
  
  ## Returns
  A tuple containing {connectivity_matrix, weight_matrix}
  """
  def generate_connectivity_matrix(model, params, source_positions, target_positions, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.2)
    batch_size = Keyword.get(opts, :batch_size, 1000)
    leo_enabled = Keyword.get(opts, :leo_enabled, false)
    
    # Generate all possible connection queries
    connection_queries = generate_all_connection_queries(source_positions, target_positions)
    
    # Process in batches to avoid memory issues
    {connectivity_matrix, weight_matrix} = 
      process_queries_in_batches(model, params, connection_queries, batch_size, threshold, leo_enabled)
    
    {connectivity_matrix, weight_matrix}
  end
  
  @doc """
  Trains the CPPN model using a set of input-output pairs.
  
  ## Parameters
  - model: The Axon model
  - params: Initial model parameters
  - inputs: Training inputs
  - targets: Training targets
  - opts: Training options
  
  ## Options
  - epochs: Number of training epochs (default: 100)
  - learning_rate: Learning rate (default: 0.01)
  - batch_size: Batch size (default: 32)
  
  ## Returns
  Updated model parameters
  """
  def train(model, params, inputs, targets, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 100)
    learning_rate = Keyword.get(opts, :learning_rate, 0.01)
    batch_size = Keyword.get(opts, :batch_size, 32)
    
    # Prepare data
    inputs_tensor = Nx.tensor(inputs)
    targets_tensor = Nx.tensor(targets)
    
    # Define optimizer and loss
    # Use Polaris.Optimizers instead of deprecated Axon.Optimizers
    optimizer = Polaris.Optimizers.adam(learning_rate: learning_rate)
    loss = :mean_squared_error
    
    # Train the model
    trained_params = Axon.Loop.trainer(model, loss, optimizer)
    |> Axon.Loop.run(
      Stream.repeatedly(fn -> {inputs_tensor, targets_tensor} end) |> Stream.take(epochs),
      params,
      epochs: epochs,
      compiler: EXLA,
      batch_size: batch_size
    )
    
    trained_params
  end
  
  # Private functions
  
  defp build_hidden_layers(input, [], _activation), do: input
  
  defp build_hidden_layers(input, [size | rest], activation) do
    input
    |> Axon.dense(size, activation: activation)
    |> build_hidden_layers(rest, activation)
  end
  
  defp reshape_for_batch(%Nx.Tensor{} = tensor) do
    shape = Nx.shape(tensor)
    
    case shape do
      {_n, _m} -> tensor  # Already in batch shape
      {n} -> Nx.reshape(tensor, {1, n})  # Single input, reshape to batch of 1
      _ -> 
        # For higher dimensional tensors, flatten to 2D
        flattened_size = Nx.size(tensor) |> div(Nx.axis_size(tensor, 0))
        Nx.reshape(tensor, {Nx.axis_size(tensor, 0), flattened_size})
    end
  end
  
  defp generate_all_connection_queries(source_positions, target_positions) do
    # Generate all possible pairs of source and target positions
    for source_pos <- source_positions, target_pos <- target_positions do
      # Flatten the positions into a single vector for the CPPN input
      List.flatten(source_pos) ++ List.flatten(target_pos)
    end
  end
  
  defp process_queries_in_batches(model, params, queries, batch_size, threshold, leo_enabled) do
    # Calculate dimensions
    num_queries = length(queries)
    num_sources = length(Enum.uniq_by(queries, &Enum.take(&1, div(length(List.first(queries)), 2))))
    num_targets = div(num_queries, num_sources)
    
    # Process in batches
    {connectivity_data, weight_data} =
      Enum.chunk_every(queries, batch_size)
      |> Enum.map(fn batch ->
        outputs = batch_process_queries(model, params, batch)
        
        if leo_enabled do
          # Extract weights and expression outputs
          weights = Nx.slice(outputs, [0, 0], [Nx.axis_size(outputs, 0), 1])
          expressions = Nx.slice(outputs, [0, 1], [Nx.axis_size(outputs, 0), 1])
          
          # Apply threshold to expressions
          connectivity = Nx.greater(expressions, threshold)
          
          {connectivity, weights}
        else
          # Apply threshold directly to weights for connectivity
          connectivity = Nx.greater(Nx.abs(outputs), threshold)
          
          {connectivity, outputs}
        end
      end)
      |> Enum.reduce({[], []}, fn {batch_conn, batch_weights}, {conn_acc, weights_acc} ->
        {conn_acc ++ Nx.to_list(batch_conn), weights_acc ++ Nx.to_list(batch_weights)}
      end)
    
    # Reshape into matrices
    connectivity_matrix = Nx.tensor(connectivity_data) |> Nx.reshape({num_sources, num_targets})
    weight_matrix = Nx.tensor(weight_data) |> Nx.reshape({num_sources, num_targets})
    
    {connectivity_matrix, weight_matrix}
  end
end
