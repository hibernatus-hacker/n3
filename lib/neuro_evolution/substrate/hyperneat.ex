defmodule NeuroEvolution.Substrate.HyperNEAT do
  @moduledoc """
  HyperNEAT implementation for evolving large-scale neural networks
  through compositional pattern producing networks (CPPNs).
  """

  alias NeuroEvolution.TWEANN.{Genome, Node, Connection}
  alias NeuroEvolution.Substrate.Substrate

  defstruct [
    :cppn,
    :substrate,
    :connection_threshold,
    :max_weight,
    :distance_function,
    :leo_enabled  # Link Expression Output
  ]

  @type t :: %__MODULE__{
    cppn: Genome.t(),
    substrate: Substrate.t(),
    connection_threshold: float(),
    max_weight: float(),
    distance_function: atom(),
    leo_enabled: boolean()
  }

  def new(input_dims, hidden_dims, output_dims, opts \\ []) do
    # Create layer configs with exact sizes based on dimensions
    input_size = case input_dims do
      [x, y] -> x * y
      [x, y, z] -> x * y * z
      [x] -> x
      _ -> 10
    end
    
    hidden_size = case hidden_dims do
      [x, y] -> x * y
      [x, y, z] -> x * y * z
      [x] -> x
      _ -> 5
    end
    
    output_size = case output_dims do
      [x, y] -> x * y
      [x, y, z] -> x * y * z
      [x] -> x
      _ -> 2
    end
    
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
      input_layer_config: %{size: input_size},
      hidden_layer_config: %{size: hidden_size},
      output_layer_config: %{size: output_size},
      connection_function: Keyword.get(opts, :connection_function, :distance_based)
    }

    substrate = Substrate.new(substrate_config)

    # CPPN inputs: [x1, y1, z1, x2, y2, z2, bias] for 3D, fewer for 2D
    cppn_inputs = calculate_cppn_inputs(input_dims, opts)
    cppn_outputs = if Keyword.get(opts, :leo_enabled, false), do: 2, else: 1

    cppn = Genome.new(cppn_inputs, cppn_outputs, substrate: substrate_config)

    %__MODULE__{
      cppn: cppn,
      substrate: substrate,
      connection_threshold: Keyword.get(opts, :connection_threshold, 0.2),
      max_weight: Keyword.get(opts, :max_weight, 5.0),
      distance_function: Keyword.get(opts, :distance_function, :euclidean),
      leo_enabled: Keyword.get(opts, :leo_enabled, false)
    }
  end

  def decode_substrate(%__MODULE__{} = hyperneat) do
    substrate_nodes = create_substrate_nodes(hyperneat.substrate)
    connections = query_connections(hyperneat, substrate_nodes)

    build_phenotype_network(substrate_nodes, connections, hyperneat.substrate)
  end

  def query_connections(%__MODULE__{} = hyperneat, substrate_nodes) do
    substrate_nodes
    |> generate_connection_queries(hyperneat.substrate)
    |> Enum.map(&query_cppn_for_connection(hyperneat, &1))
    |> Enum.filter(&connection_expressed?(&1, hyperneat.connection_threshold))
  end

  def query_cppn_for_connection(%__MODULE__{} = hyperneat, {source_pos, target_pos, source_id, target_id}) do
    cppn_input = build_cppn_input(source_pos, target_pos, hyperneat.substrate)
    cppn_output = evaluate_cppn(hyperneat.cppn, cppn_input)

    weight = extract_weight(cppn_output, hyperneat.max_weight)
    expression = if hyperneat.leo_enabled, do: extract_expression(cppn_output), else: 1.0

    %{
      from: source_id,
      to: target_id,
      weight: weight,
      expression: expression,
      distance: calculate_distance(source_pos, target_pos, hyperneat.distance_function)
    }
  end

  def mutate_cppn(%__MODULE__{} = hyperneat, mutation_config \\ %{}) do
    mutated_cppn = apply_cppn_mutations(hyperneat.cppn, mutation_config)
    %{hyperneat | cppn: mutated_cppn}
  end

  def crossover_hyperneat(%__MODULE__{} = parent1, %__MODULE__{} = parent2) do
    child_cppn = Genome.crossover(parent1.cppn, parent2.cppn)

    %__MODULE__{
      cppn: child_cppn,
      substrate: parent1.substrate,  # Assume same substrate
      connection_threshold: parent1.connection_threshold,
      max_weight: parent1.max_weight,
      distance_function: parent1.distance_function,
      leo_enabled: parent1.leo_enabled
    }
  end

  def evolve_substrate_topology(%__MODULE__{} = hyperneat, evolution_params \\ %{}) do
    new_substrate = evolve_substrate_structure(hyperneat.substrate, evolution_params)

    # Update CPPN if substrate dimensionality changed
    updated_cppn =
      if substrate_dimensionality_changed?(hyperneat.substrate, new_substrate) do
        adapt_cppn_to_substrate(hyperneat.cppn, new_substrate)
      else
        hyperneat.cppn
      end

    %{hyperneat | substrate: new_substrate, cppn: updated_cppn}
  end

  def calculate_substrate_complexity(substrate_nodes, connections) do
    node_count = length(substrate_nodes)
    connection_count = length(connections)
    
    # Handle both HyperNEAT connection maps and TWEANN Connection structs
    active_connections = Enum.count(connections, fn conn ->
      cond do
        is_map_key(conn, :expression) -> conn.expression > 0.5
        is_map_key(conn, :enabled) -> conn.enabled
        true -> true  # Default to true if neither key exists
      end
    end)
    
    connectivity_density = if node_count > 0, do: active_connections / (node_count * node_count), else: 0.0
    
    %{
      nodes: node_count,
      connections: connection_count,
      active_connections: active_connections,
      density: connectivity_density
    }
  end

  # Private functions

  defp calculate_cppn_inputs(dimensions, opts) do
    base_inputs = case length(dimensions) do
      2 -> 5  # [x1, y1, x2, y2, bias]
      3 -> 7  # [x1, y1, z1, x2, y2, z2, bias]
      _ -> 3  # [x1, x2, bias] for 1D
    end

    additional_inputs = Keyword.get(opts, :additional_cppn_inputs, 0)
    base_inputs + additional_inputs
  end

  defp create_substrate_nodes(%Substrate{} = substrate) do
    input_nodes = create_layer_nodes(substrate.input_positions, :input)
    hidden_nodes = create_layer_nodes(substrate.hidden_positions, :hidden)
    output_nodes = create_layer_nodes(substrate.output_positions, :output)

    input_nodes ++ hidden_nodes ++ output_nodes
  end

  defp create_layer_nodes(positions, layer_type) do
    positions
    |> Enum.with_index(1)
    |> Enum.map(fn {pos, idx} ->
      node_id = generate_node_id(layer_type, idx)
      Node.new(node_id, layer_type, :tanh, pos)
    end)
  end

  defp generate_node_id(:input, idx), do: idx
  defp generate_node_id(:hidden, idx), do: 1000 + idx
  defp generate_node_id(:output, idx), do: 2000 + idx

  defp generate_connection_queries(substrate_nodes, %Substrate{} = substrate) do
    case substrate.connection_function do
      :all_to_all -> generate_all_to_all_queries(substrate_nodes)
      :layer_to_layer -> generate_layer_to_layer_queries(substrate_nodes)
      :distance_based -> generate_distance_based_queries(substrate_nodes, substrate)
      :topological -> generate_topological_queries(substrate_nodes, substrate)
    end
  end

  defp generate_all_to_all_queries(substrate_nodes) do
    for source <- substrate_nodes,
        target <- substrate_nodes,
        source.id != target.id,
        valid_connection_type?(source.type, target.type),
        do: {source.position, target.position, source.id, target.id}
  end

  defp generate_layer_to_layer_queries(substrate_nodes) do
    inputs = Enum.filter(substrate_nodes, &(&1.type == :input))
    hiddens = Enum.filter(substrate_nodes, &(&1.type == :hidden))
    outputs = Enum.filter(substrate_nodes, &(&1.type == :output))

    input_to_hidden = layer_connections(inputs, hiddens)
    hidden_to_hidden = layer_connections(hiddens, hiddens)
    hidden_to_output = layer_connections(hiddens, outputs)

    input_to_hidden ++ hidden_to_hidden ++ hidden_to_output
  end

  defp generate_distance_based_queries(substrate_nodes, %Substrate{distance_threshold: threshold}) do
    for source <- substrate_nodes,
        target <- substrate_nodes,
        source.id != target.id,
        valid_connection_type?(source.type, target.type),
        Node.euclidean_distance(source, target) <= threshold,
        do: {source.position, target.position, source.id, target.id}
  end

  defp generate_topological_queries(substrate_nodes, %Substrate{topology: topology}) do
    case topology do
      :grid -> generate_grid_topology_queries(substrate_nodes)
      :random -> generate_random_topology_queries(substrate_nodes)
      :small_world -> generate_small_world_queries(substrate_nodes)
      _ -> generate_all_to_all_queries(substrate_nodes)
    end
  end

  defp generate_grid_topology_queries(substrate_nodes) do
    # Connect nodes to their spatial neighbors in a grid
    for source <- substrate_nodes,
        target <- substrate_nodes,
        source.id != target.id,
        valid_connection_type?(source.type, target.type),
        grid_neighbors?(source.position, target.position),
        do: {source.position, target.position, source.id, target.id}
  end

  # Small world queries implementation has been moved - see line 538
  # Legacy function - kept for reference but replaced with improved implementation
  defp __generate_small_world_queries_old(substrate_nodes) do
    # Watts-Strogatz small-world network model
    # Start with regular lattice, then rewire with probability
    rewiring_probability = 0.1
    k_neighbors = 4  # Each node connects to k nearest neighbors
    
    # Generate regular lattice connections
    regular_connections = __generate_k_nearest_neighbors(substrate_nodes, k_neighbors)
    
    # Rewire some connections randomly
    rewired_connections = 
      regular_connections
      |> Enum.map(fn connection ->
        if :rand.uniform() < rewiring_probability do
          __rewire_connection(connection, substrate_nodes)
        else
          connection
        end
      end)
    
    rewired_connections
  end

  defp layer_connections(source_layer, target_layer) do
    for source <- source_layer,
        target <- target_layer,
        source.id != target.id,
        do: {source.position, target.position, source.id, target.id}
  end

  defp valid_connection_type?(:output, _), do: false
  defp valid_connection_type?(_, :input), do: false
  defp valid_connection_type?(_, _), do: true

  defp grid_neighbors?({x1, y1}, {x2, y2}) do
    # Check if two positions are grid neighbors (adjacent or diagonal)
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    
    # Adjacent (including diagonal) neighbors
    (dx <= 1 and dy <= 1) and not (dx == 0 and dy == 0)
  end

  defp grid_neighbors?({x1, y1, z1}, {x2, y2, z2}) do
    # 3D grid neighbors
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dz = abs(z1 - z2)
    
    (dx <= 1 and dy <= 1 and dz <= 1) and not (dx == 0 and dy == 0 and dz == 0)
  end

  defp grid_neighbors?(_, _), do: false

  # Experimental function - kept for future reference but not currently used
  defp __generate_k_nearest_neighbors(substrate_nodes, k) do
    # For each node, connect to its k nearest neighbors
    substrate_nodes
    |> Enum.flat_map(fn source ->
      substrate_nodes
      |> Enum.filter(&(&1.id != source.id))
      |> Enum.filter(&valid_connection_type?(source.type, &1.type))
      |> Enum.sort_by(&Node.euclidean_distance(source, &1))
      |> Enum.take(k)
      |> Enum.map(&{source.position, &1.position, source.id, &1.id})
    end)
  end

  # Experimental function - kept for future reference but not currently used
  defp __rewire_connection({source_pos, _target_pos, source_id, _target_id}, substrate_nodes) do
    # Find the source node
    source_node = Enum.find(substrate_nodes, &(&1.id == source_id))
    
    # Pick a random target (excluding self and invalid connection types)
    valid_targets = 
      substrate_nodes
      |> Enum.filter(&(&1.id != source_id))
      |> Enum.filter(&valid_connection_type?(source_node.type, &1.type))
    
    if length(valid_targets) > 0 do
      target_node = Enum.random(valid_targets)
      {source_pos, target_node.position, source_id, target_node.id}
    else
      # If no valid targets, keep original connection
      {source_pos, source_pos, source_id, source_id}  # This will be filtered out later
    end
  end

  defp build_cppn_input(source_pos, target_pos, %Substrate{} = _substrate) do
    case {source_pos, target_pos} do
      {{x1, y1}, {x2, y2}} ->
        [x1, y1, x2, y2, 1.0]  # bias

      {{x1, y1, z1}, {x2, y2, z2}} ->
        [x1, y1, z1, x2, y2, z2, 1.0]  # bias

      _ ->
        [0.0, 0.0, 0.0, 0.0, 1.0]  # fallback
    end
  end

  defp evaluate_cppn(_cppn_genome, input) do
    # This would integrate with the Nx-based network evaluator
    # For now, simplified evaluation
    case length(input) do
      5 -> [simulate_cppn_output(input), 0.8]  # [weight, expression]
      7 -> [simulate_cppn_output(input), 0.8]
      _ -> [0.0, 0.0]
    end
  end

  defp simulate_cppn_output(input) do
    # Simplified CPPN simulation - in practice this would use the full network
    weighted_sum = Enum.with_index(input) |> Enum.reduce(0.0, fn {val, idx}, acc ->
      acc + val * :math.sin(idx + 1)
    end)

    :math.tanh(weighted_sum)
  end

  defp extract_weight([weight_output | _], max_weight) do
    clamped_weight = max(-max_weight, min(max_weight, weight_output))
    clamped_weight
  end

  defp extract_expression([_, expression_output | _]), do: 1.0 / (1.0 + :math.exp(-expression_output))
  defp extract_expression(_), do: 1.0

  defp connection_expressed?(%{expression: expression}, threshold) do
    expression > threshold
  end

  defp calculate_distance(pos1, pos2, :euclidean) do
    case {pos1, pos2} do
      {{x1, y1}, {x2, y2}} ->
        :math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

      {{x1, y1, z1}, {x2, y2, z2}} ->
        :math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))
    end
  end

  defp calculate_distance(pos1, pos2, :manhattan) do
    case {pos1, pos2} do
      {{x1, y1}, {x2, y2}} ->
        abs(x2 - x1) + abs(y2 - y1)

      {{x1, y1, z1}, {x2, y2, z2}} ->
        abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1)
    end
  end

  defp build_phenotype_network(substrate_nodes, connections, %Substrate{} = substrate) do
    # Convert substrate representation to standard TWEANN genome
    nodes_map =
      substrate_nodes
      |> Enum.map(&{&1.id, &1})
      |> Map.new()

    connections_map =
      connections
      |> Enum.with_index(1)
      |> Enum.map(fn {conn, idx} ->
        tweann_conn = Connection.new(conn.from, conn.to, conn.weight, idx, true)
        {idx, tweann_conn}
      end)
      |> Map.new()

    input_ids = Enum.filter(substrate_nodes, &(&1.type == :input)) |> Enum.map(&(&1.id))
    output_ids = Enum.filter(substrate_nodes, &(&1.type == :output)) |> Enum.map(&(&1.id))

    %Genome{
      id: :crypto.strong_rand_bytes(16) |> Base.encode16(),
      nodes: nodes_map,
      connections: connections_map,
      inputs: input_ids,
      outputs: output_ids,
      fitness: nil,
      species_id: nil,
      generation: 0,
      substrate_config: substrate
    }
  end

  defp apply_cppn_mutations(cppn, mutation_config) do
    mutation_rate = Map.get(mutation_config, :weight_mutation_rate, 0.1)
    add_node_rate = Map.get(mutation_config, :add_node_rate, 0.05)
    add_connection_rate = Map.get(mutation_config, :add_connection_rate, 0.1)

    cppn
    |> maybe_mutate_weights(mutation_rate)
    |> maybe_add_node(add_node_rate)
    |> maybe_add_connection(add_connection_rate)
  end

  defp maybe_mutate_weights(cppn, rate) do
    if :rand.uniform() < rate do
      Genome.mutate_weights(cppn, rate)
    else
      cppn
    end
  end

  defp maybe_add_node(cppn, rate) do
    if :rand.uniform() < rate and map_size(cppn.connections) > 0 do
      connection_keys = Map.keys(cppn.connections)
      random_connection = Enum.random(connection_keys)
      Genome.add_node(cppn, random_connection)
    else
      cppn
    end
  end

  defp maybe_add_connection(cppn, rate) do
    if :rand.uniform() < rate do
      node_ids = Map.keys(cppn.nodes)
      if length(node_ids) >= 2 do
        from_id = Enum.random(node_ids)
        to_id = Enum.random(node_ids)
        Genome.add_connection(cppn, from_id, to_id)
      else
        cppn
      end
    else
      cppn
    end
  end

  defp evolve_substrate_structure(substrate, _evolution_params) do
    # Placeholder for substrate evolution - could add/remove layers, change geometry
    substrate
  end

  defp substrate_dimensionality_changed?(%Substrate{} = old, %Substrate{} = new) do
    old_dims = calculate_substrate_dimensions(old)
    new_dims = calculate_substrate_dimensions(new)
    old_dims != new_dims
  end

  defp calculate_substrate_dimensions(%Substrate{} = substrate) do
    sample_position = List.first(substrate.input_positions) || {0.0, 0.0}
    case sample_position do
      {_, _} -> 2
      {_, _, _} -> 3
      _ -> 1
    end
  end

  defp adapt_cppn_to_substrate(cppn, _new_substrate) do
    # Adjust CPPN input/output structure for new substrate dimensionality
    # This is a placeholder - would need more sophisticated adaptation
    cppn
  end

  # This is the actual implementation used
  defp generate_random_topology_queries(substrate_nodes) do
    connection_probability = 0.1

    for source <- substrate_nodes,
        target <- substrate_nodes,
        source.id != target.id,
        valid_connection_type?(source.type, target.type),
        :rand.uniform() < connection_probability,
        do: {source.position, target.position, source.id, target.id}
  end

  # This is the actual implementation used
  defp generate_small_world_queries(substrate_nodes) do
    # Combine local grid connections with some random long-range connections
    grid_queries = generate_grid_topology_queries(substrate_nodes)
    random_queries = generate_random_topology_queries(substrate_nodes)

    # Take subset of random queries to maintain small-world property
    random_subset_size = div(length(random_queries), 10)
    random_subset = Enum.take_random(random_queries, random_subset_size)

    grid_queries ++ random_subset
  end
end
