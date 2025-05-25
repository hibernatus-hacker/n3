defmodule NeuroEvolution.TWEANN.Genome do
  @moduledoc """
  TWEANN genome representation with support for topology and weight evolution.
  Includes innovation numbers for crossover alignment and substrate encodings.
  """

  alias NeuroEvolution.TWEANN.{Node, Connection}
  
  defstruct [
    :id,
    :nodes,
    :connections,
    :inputs,
    :outputs,
    :fitness,
    :species_id,
    :generation,
    :substrate_config,
    :plasticity_config
  ]

  @type t :: %__MODULE__{
    id: binary(),
    nodes: %{integer() => Node.t()},
    connections: %{integer() => Connection.t()},
    inputs: [integer()],
    outputs: [integer()],
    fitness: float() | nil,
    species_id: integer() | nil,
    generation: integer(),
    substrate_config: map() | nil,
    plasticity_config: map() | nil
  }

  def new(input_count, output_count, opts \\ []) do
    substrate_config = Keyword.get(opts, :substrate)
    plasticity_config = Keyword.get(opts, :plasticity)
    
    {nodes, inputs, outputs} = create_initial_topology(input_count, output_count, substrate_config)
    
    %__MODULE__{
      id: generate_id(),
      nodes: nodes,
      connections: %{},
      inputs: inputs,
      outputs: outputs,
      fitness: nil,
      species_id: nil,
      generation: 0,
      substrate_config: substrate_config,
      plasticity_config: plasticity_config
    }
  end

  def add_node(%__MODULE__{} = genome, connection_innovation) do
    connection = genome.connections[connection_innovation]
    return_if_nil(connection, genome)

    new_node_id = get_next_node_id(genome)
    innovation1 = get_innovation_number()
    innovation2 = get_innovation_number()

    new_node = Node.new(new_node_id, :hidden, get_activation_function(genome))
    
    conn1 = Connection.new(connection.from, new_node_id, 1.0, innovation1, true)
    conn2 = Connection.new(new_node_id, connection.to, connection.weight, innovation2, true)

    updated_connections = 
      genome.connections
      |> Map.put(connection_innovation, %{connection | enabled: false})
      |> Map.put(innovation1, conn1)
      |> Map.put(innovation2, conn2)

    %{genome | 
      nodes: Map.put(genome.nodes, new_node_id, new_node),
      connections: updated_connections
    }
  end

  def add_connection(%__MODULE__{} = genome, from_id, to_id) do
    if valid_connection?(genome, from_id, to_id) do
      innovation = get_innovation_number()
      weight = :rand.normal(0.0, 1.0)
      
      connection = Connection.new(from_id, to_id, weight, innovation, true)
      
      %{genome | connections: Map.put(genome.connections, innovation, connection)}
    else
      genome
    end
  end

  def mutate_weights(%__MODULE__{} = genome, mutation_rate \\ 0.1, perturbation_strength \\ 0.1) do
    updated_connections = 
      Enum.reduce(genome.connections, %{}, fn {id, conn}, acc ->
        # Apply mutation based on the mutation rate
        # This ensures that higher mutation rates cause more weight changes
        new_weight = 
          if :rand.uniform() < mutation_rate do
            # Sometimes completely replace the weight (10% chance)
            if :rand.uniform() < 0.1 do
              :rand.normal(0.0, 1.0) 
            else
              # Apply a significant perturbation to ensure the weight changes noticeably
              # Use a larger perturbation to ensure weights change significantly
              conn.weight + :rand.normal(0.0, perturbation_strength * 2.0)
            end
          else
            conn.weight
          end
        
        Map.put(acc, id, %{conn | weight: new_weight})
      end)
    
    %{genome | connections: updated_connections}
  end

  def crossover(%__MODULE__{} = parent1, %__MODULE__{} = parent2) do
    # Determine which parent is more fit
    {dominant, recessive} = 
      if (parent1.fitness || 0.0) > (parent2.fitness || 0.0) do
        {parent1, parent2}
      else
        {parent2, parent1}
      end

    # Strongly preserve the dominant parent's structure
    # First, copy all nodes from the dominant parent
    child_nodes = dominant.nodes
    
    # For connections, heavily bias towards the dominant parent
    child_connections = 
      Enum.reduce(dominant.connections, %{}, fn {innovation, conn}, acc ->
        case Map.get(recessive.connections, innovation) do
          nil -> 
            # Always include connections unique to the dominant parent
            Map.put(acc, innovation, conn)
          recessive_conn ->
            # For matching connections, strongly bias towards the dominant parent (90% chance)
            chosen_conn = if :rand.uniform() < 0.9, do: conn, else: recessive_conn
            enabled = conn.enabled and recessive_conn.enabled
            Map.put(acc, innovation, %{chosen_conn | enabled: enabled})
        end
      end)

    # Include very few connections unique to the recessive parent (10% chance)
    child_connections =
      Enum.reduce(recessive.connections, child_connections, fn {innovation, conn}, acc ->
        if not Map.has_key?(dominant.connections, innovation) and :rand.uniform() < 0.1 do
          Map.put(acc, innovation, conn)
        else
          acc
        end
      end)

    # Merge nodes, ensuring we have all nodes needed for the connections
    child_nodes = merge_nodes(dominant.nodes, recessive.nodes, child_connections)

    %__MODULE__{
      id: generate_id(),
      nodes: child_nodes,
      connections: child_connections,
      inputs: dominant.inputs,
      outputs: dominant.outputs,
      fitness: nil,
      species_id: nil,
      generation: max(dominant.generation, recessive.generation) + 1,
      substrate_config: dominant.substrate_config,
      plasticity_config: dominant.plasticity_config
    }
  end

  def distance(%__MODULE__{} = genome1, %__MODULE__{} = genome2) do
    # Check if genomes are identical (same ID)
    if genome1.id == genome2.id do
      0.0  # Identical genomes have zero distance
    else
      # Coefficients for the distance calculation
      c1 = 1.0  # excess coefficient
      c2 = 1.0  # disjoint coefficient  
      c3 = 3.0  # weight difference coefficient - increased to make weight differences more significant
  
      # Get the innovation numbers from both genomes
      innovations1 = MapSet.new(Map.keys(genome1.connections))
      innovations2 = MapSet.new(Map.keys(genome2.connections))
      
      # Find matching, excess, and disjoint genes
      matching = MapSet.intersection(innovations1, innovations2)
      excess = calculate_excess(innovations1, innovations2)
      disjoint = calculate_disjoint(innovations1, innovations2, excess)
      
      # If the genomes have completely different structures, add a minimum distance
      # This ensures that different genomes always have a non-zero distance
      min_distance = 0.1
      
      # Calculate weight differences for matching genes
      weight_diff = calculate_weight_difference(genome1.connections, genome2.connections, matching)
      
      # Normalize by the size of the larger genome
      n = max(map_size(genome1.connections), map_size(genome2.connections))
      n = if n < 20, do: 1, else: n
  
      # Calculate structural distance
      structural_distance = (c1 * excess + c2 * disjoint) / n
      
      # Calculate total distance
      # For similar genomes with small mutations, weight_diff will be small
      # For dissimilar genomes with structural differences, excess and disjoint will be large
      total_distance = structural_distance + c3 * weight_diff
      
      # Ensure we always return a non-zero distance for different genomes
      # This is crucial for speciation and diversity calculations
      max(total_distance, min_distance)
    end
  end

  def to_nx_tensor(%__MODULE__{} = genome, max_nodes) do
    adjacency_matrix = build_adjacency_matrix(genome, max_nodes)
    weight_matrix = build_weight_matrix(genome, max_nodes)
    node_features = build_node_features(genome, max_nodes)
    
    %{
      adjacency: adjacency_matrix,
      weights: weight_matrix,
      features: node_features,
      input_mask: build_input_mask(genome, max_nodes),
      output_mask: build_output_mask(genome, max_nodes)
    }
  end

  @doc """
  Converts tensor representation back to genome, updating weights and connections.
  Used primarily for applying plasticity updates from GPU computations.
  """
  def from_nx_tensor(%__MODULE__{} = original_genome, tensor_data) do
    %{
      adjacency: adjacency_matrix,
      weights: weight_matrix,
      features: _node_features
    } = tensor_data

    # Extract updated connections from weight matrix
    updated_connections = extract_connections_from_tensors(
      original_genome.connections,
      adjacency_matrix,
      weight_matrix
    )

    %{original_genome | connections: updated_connections}
  end

  @doc """
  Updates plasticity weights in genome from tensor representation.
  """
  def update_plasticity_from_tensor(%__MODULE__{} = genome, plastic_weights_tensor) do
    updated_connections = 
      genome.connections
      |> Enum.map(fn {id, conn} ->
        # Extract plastic weight for this connection
        plastic_weight = extract_plastic_weight(conn, plastic_weights_tensor)
        
        updated_plasticity_params = 
          (conn.plasticity_params || %{})
          |> Map.put(:plastic_weight, plastic_weight)
        
        {id, %{conn | plasticity_params: updated_plasticity_params}}
      end)
      |> Map.new()

    %{genome | connections: updated_connections}
  end

  # Private functions

  defp create_initial_topology(input_count, output_count, substrate_config) do
    case substrate_config do
      nil -> 
        create_basic_topology(input_count, output_count)
      config -> 
        create_substrate_topology(input_count, output_count, config)
    end
  end

  defp create_basic_topology(input_count, output_count) do
    input_nodes = 
      for i <- 1..input_count, into: %{} do
        {i, Node.new(i, :input, :linear)}
      end
    
    output_nodes =
      for i <- (input_count + 1)..(input_count + output_count), into: %{} do
        {i, Node.new(i, :output, :tanh)}
      end
    
    nodes = Map.merge(input_nodes, output_nodes)
    inputs = Enum.to_list(1..input_count)
    outputs = Enum.to_list((input_count + 1)..(input_count + output_count))
    
    {nodes, inputs, outputs}
  end

  defp create_substrate_topology(input_count, output_count, config) do
    substrate_nodes = generate_substrate_nodes(config)
    
    input_nodes = 
      for i <- 1..input_count, into: %{} do
        pos = get_substrate_position(i, :input, config)
        {i, Node.new(i, :input, :linear, pos)}
      end
    
    output_nodes =
      for i <- (input_count + 1)..(input_count + output_count), into: %{} do
        pos = get_substrate_position(i - input_count, :output, config)
        {i, Node.new(i, :output, :tanh, pos)}
      end
    
    hidden_nodes =
      for {id, pos} <- substrate_nodes, into: %{} do
        node_id = input_count + output_count + id
        {node_id, Node.new(node_id, :hidden, :tanh, pos)}
      end
    
    nodes = Map.merge(input_nodes, output_nodes) |> Map.merge(hidden_nodes)
    inputs = Enum.to_list(1..input_count)
    outputs = Enum.to_list((input_count + 1)..(input_count + output_count))
    
    {nodes, inputs, outputs}
  end

  defp generate_substrate_nodes(config) do
    case Map.get(config, :dimensions, 2) do
      2 -> generate_2d_substrate(config)
      3 -> generate_3d_substrate(config)
      _ -> []
    end
  end

  defp generate_2d_substrate(config) do
    resolution = Map.get(config, :resolution, 10)
    {width, height} = normalize_substrate_resolution(resolution)
    
    for x <- 0..(width - 1),
        y <- 0..(height - 1),
        do: {x * height + y, {x / max(1, width - 1), y / max(1, height - 1)}}
  end

  defp generate_3d_substrate(config) do
    resolution = Map.get(config, :resolution, 10)
    {width, height, depth} = normalize_substrate_resolution_3d(resolution)
    
    for x <- 0..(width - 1),
        y <- 0..(height - 1),
        z <- 0..(depth - 1),
        do: {x * height * depth + y * depth + z, 
             {x / max(1, width - 1), y / max(1, height - 1), z / max(1, depth - 1)}}
  end

  defp normalize_substrate_resolution(res) when is_integer(res), do: {res, res}
  defp normalize_substrate_resolution({w, h}), do: {w, h}
  defp normalize_substrate_resolution(_res), do: {10, 10}

  defp normalize_substrate_resolution_3d(res) when is_integer(res), do: {res, res, res}
  defp normalize_substrate_resolution_3d({w, h}), do: {w, h, w}
  defp normalize_substrate_resolution_3d({w, h, d}), do: {w, h, d}
  defp normalize_substrate_resolution_3d(_), do: {10, 10, 10}

  defp get_substrate_position(index, type, _config) do
    case type do
      :input -> [] |> Enum.at(index - 1, {0.0, 0.0})
      :output -> [] |> Enum.at(index - 1, {1.0, 1.0})
    end
  end

  defp get_activation_function(%{plasticity_config: nil}), do: :tanh
  defp get_activation_function(%{plasticity_config: %{adaptive_activation: true}}), do: :adaptive
  defp get_activation_function(_), do: :tanh

  defp build_adjacency_matrix(genome, max_nodes) do
    matrix = Nx.broadcast(0, {max_nodes, max_nodes})
    
    Enum.reduce(genome.connections, matrix, fn {_id, conn}, acc ->
      if conn.enabled and conn.from <= max_nodes and conn.to <= max_nodes do
        Nx.put_slice(acc, [conn.from - 1, conn.to - 1], Nx.tensor([[1]]))
      else
        acc
      end
    end)
  end

  defp build_weight_matrix(genome, max_nodes) do
    matrix = Nx.broadcast(0.0, {max_nodes, max_nodes})
    
    Enum.reduce(genome.connections, matrix, fn {_id, conn}, acc ->
      if conn.enabled and conn.from <= max_nodes and conn.to <= max_nodes do
        Nx.put_slice(acc, [conn.from - 1, conn.to - 1], Nx.tensor([[conn.weight]]))
      else
        acc
      end
    end)
  end

  defp build_node_features(genome, max_nodes) do
    features = Nx.broadcast(0.0, {max_nodes, 4})  # [type, activation, x_pos, y_pos]
    
    Enum.reduce(genome.nodes, features, fn {id, node}, acc ->
      if id <= max_nodes do
        type_val = case node.type do
          :input -> 1.0
          :hidden -> 0.5
          :output -> 0.0
        end
        
        activation_val = case node.activation do
          :linear -> 0.0
          :tanh -> 0.5
          :relu -> 1.0
          :adaptive -> 0.25
        end
        
        {x, y} = node.position || {0.0, 0.0}
        
        Nx.put_slice(acc, [id - 1, 0], Nx.tensor([[type_val, activation_val, x, y]]))
      else
        acc
      end
    end)
  end

  defp build_input_mask(genome, max_nodes) do
    mask = Nx.broadcast(0, {max_nodes})
    
    Enum.reduce(genome.inputs, mask, fn input_id, acc ->
      if input_id <= max_nodes do
        Nx.put_slice(acc, [input_id - 1], Nx.tensor([1]))
      else
        acc
      end
    end)
  end

  defp build_output_mask(genome, max_nodes) do
    mask = Nx.broadcast(0, {max_nodes})
    
    Enum.reduce(genome.outputs, mask, fn output_id, acc ->
      if output_id <= max_nodes do
        Nx.put_slice(acc, [output_id - 1], Nx.tensor([1]))
      else
        acc
      end
    end)
  end

  defp get_next_node_id(genome) do
    case Map.keys(genome.nodes) do
      [] -> 1
      keys -> Enum.max(keys) + 1
    end
  end

  defp get_innovation_number do
    System.unique_integer([:positive])
  end

  defp generate_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16()
  end

  defp return_if_nil(nil, default), do: default
  defp return_if_nil(value, _default), do: value

  defp extract_connections_from_tensors(original_connections, adjacency_matrix, weight_matrix) do
    # Convert tensors to Elixir data
    adjacency_data = Nx.to_list(adjacency_matrix)
    weights_data = Nx.to_list(weight_matrix)
    
    # Update existing connections with new weights
    original_connections
    |> Enum.map(fn {id, conn} ->
      if conn.from <= length(adjacency_data) and conn.to <= length(List.first(adjacency_data) || []) do
        # Extract weight from tensor (adjust for 0-based indexing)
        row_idx = conn.from - 1
        col_idx = conn.to - 1
        
        if row_idx >= 0 and col_idx >= 0 do
          new_weight = weights_data
                      |> Enum.at(row_idx, [])
                      |> Enum.at(col_idx, conn.weight)
          
          {id, %{conn | weight: new_weight}}
        else
          {id, conn}
        end
      else
        {id, conn}
      end
    end)
    |> Map.new()
  end

  defp extract_plastic_weight(conn, plastic_weights_tensor) do
    # Extract plastic weight for this specific connection
    plastic_data = Nx.to_list(plastic_weights_tensor)
    
    if conn.from <= length(plastic_data) and conn.to <= length(List.first(plastic_data) || []) do
      row_idx = conn.from - 1
      col_idx = conn.to - 1
      
      if row_idx >= 0 and col_idx >= 0 do
        plastic_data
        |> Enum.at(row_idx, [])
        |> Enum.at(col_idx, 0.0)
      else
        0.0
      end
    else
      0.0
    end
  end

  defp valid_connection?(genome, from_id, to_id) do
    with %Node{} = from_node <- Map.get(genome.nodes, from_id),
         %Node{} = to_node <- Map.get(genome.nodes, to_id) do
      
      from_node.type != :output and
      to_node.type != :input and
      from_id != to_id and
      not connection_exists?(genome, from_id, to_id) and
      not creates_cycle?(genome, from_id, to_id)
    else
      _ -> false
    end
  end

  defp connection_exists?(genome, from_id, to_id) do
    Enum.any?(genome.connections, fn {_id, conn} ->
      conn.from == from_id and conn.to == to_id and conn.enabled
    end)
  end

  defp creates_cycle?(genome, from_id, to_id) do
    reachable_from_to = get_reachable_nodes(genome, to_id, MapSet.new())
    MapSet.member?(reachable_from_to, from_id)
  end

  defp get_reachable_nodes(genome, node_id, visited) do
    if MapSet.member?(visited, node_id) do
      visited
    else
      new_visited = MapSet.put(visited, node_id)
      
      outgoing_connections = 
        Enum.filter(genome.connections, fn {_id, conn} ->
          conn.from == node_id and conn.enabled
        end)
      
      Enum.reduce(outgoing_connections, new_visited, fn {_id, conn}, acc ->
        get_reachable_nodes(genome, conn.to, acc)
      end)
    end
  end

  defp merge_nodes(nodes1, nodes2, connections) do
    required_nodes = 
      connections
      |> Enum.flat_map(fn {_id, conn} -> [conn.from, conn.to] end)
      |> MapSet.new()
    
    Map.merge(nodes1, nodes2)
    |> Enum.filter(fn {id, _node} -> MapSet.member?(required_nodes, id) end)
    |> Map.new()
  end

  defp calculate_excess(innovations1, innovations2) do
    max1 = if MapSet.size(innovations1) > 0, do: Enum.max(innovations1), else: 0
    max2 = if MapSet.size(innovations2) > 0, do: Enum.max(innovations2), else: 0
    
    cond do
      max1 > max2 -> 
        MapSet.filter(innovations1, &(&1 > max2)) |> MapSet.size()
      max2 > max1 -> 
        MapSet.filter(innovations2, &(&1 > max1)) |> MapSet.size()
      true -> 
        0
    end
  end

  defp calculate_disjoint(innovations1, innovations2, excess) do
    total_disjoint = MapSet.union(innovations1, innovations2) 
                    |> MapSet.difference(MapSet.intersection(innovations1, innovations2))
                    |> MapSet.size()
    
    total_disjoint - excess
  end

  defp calculate_weight_difference(connections1, connections2, matching_innovations) do
    if MapSet.size(matching_innovations) == 0 do
      0.0
    else
      total_diff = 
        Enum.reduce(matching_innovations, 0.0, fn innovation, acc ->
          weight1 = connections1[innovation].weight
          weight2 = connections2[innovation].weight
          acc + abs(weight1 - weight2)
        end)
      
      total_diff / MapSet.size(matching_innovations)
    end
  end
end

defimpl Jason.Encoder, for: NeuroEvolution.TWEANN.Genome do
  def encode(genome, opts) do
    Jason.Encode.map(%{
      id: genome.id,
      nodes: Map.values(genome.nodes),
      connections: Map.values(genome.connections),
      inputs: genome.inputs,
      outputs: genome.outputs,
      fitness: genome.fitness,
      species_id: genome.species_id,
      generation: genome.generation,
      substrate_config: genome.substrate_config,
      plasticity_config: genome.plasticity_config
    }, opts)
  end
end