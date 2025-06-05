defmodule NeuroEvolution.Substrate.Substrate do
  @moduledoc """
  Substrate representation for spatial neural networks.
  Defines the geometry and connectivity patterns for HyperNEAT and ES-HyperNEAT.
  """

  defstruct [
    :input_positions,
    :hidden_positions,
    :output_positions,
    :geometry_type,
    :dimensions,
    :resolution,
    :connection_function,
    :distance_threshold,
    :topology,
    :boundary_conditions,
    :symmetry
  ]

  @type geometry_type :: :grid | :circular | :hexagonal | :random | :custom
  @type boundary_condition :: :periodic | :reflective | :absorbing
  @type symmetry_type :: :none | :bilateral | :radial | :translational

  @type t :: %__MODULE__{
    input_positions: [tuple()],
    hidden_positions: [tuple()],
    output_positions: [tuple()],
    geometry_type: geometry_type(),
    dimensions: integer(),
    resolution: integer() | tuple(),
    connection_function: atom(),
    distance_threshold: float(),
    topology: atom(),
    boundary_conditions: boundary_condition(),
    symmetry: symmetry_type()
  }

  def new(config) do
    geometry_type = Map.get(config, :geometry_type, :grid)
    dimensions = Map.get(config, :dimensions, 2)
    resolution = Map.get(config, :resolution, 10)
    
    input_positions = generate_layer_positions(:input, geometry_type, config)
    hidden_positions = generate_layer_positions(:hidden, geometry_type, config)
    output_positions = generate_layer_positions(:output, geometry_type, config)
    
    %__MODULE__{
      input_positions: input_positions,
      hidden_positions: hidden_positions,
      output_positions: output_positions,
      geometry_type: geometry_type,
      dimensions: dimensions,
      resolution: resolution,
      connection_function: Map.get(config, :connection_function, :distance_based),
      distance_threshold: Map.get(config, :distance_threshold, 1.0),
      topology: Map.get(config, :topology, :grid),
      boundary_conditions: Map.get(config, :boundary_conditions, :absorbing),
      symmetry: Map.get(config, :symmetry, :none)
    }
  end

  def grid_2d(width, height, _layers \\ [:input, :hidden, :output]) do
    config = %{
      geometry_type: :grid,
      dimensions: 2,
      resolution: {width, height},
      input_layer_config: %{position: :bottom, size: width},
      hidden_layer_config: %{position: :middle, size: width * height},
      output_layer_config: %{position: :top, size: width}
    }
    
    new(config)
  end

  # Convenience function with 2 parameters
  def grid_2d(size) when is_integer(size) do
    grid_2d(size, size)
  end

  def grid_3d(width, height, depth, _layers \\ [:input, :hidden, :output]) do
    config = %{
      geometry_type: :grid,
      dimensions: 3,
      resolution: {width, height, depth},
      input_layer_config: %{position: :front, size: width * height},
      hidden_layer_config: %{position: :middle, size: width * height * depth},
      output_layer_config: %{position: :back, size: width * height}
    }
    
    new(config)
  end

  def circular(radius, num_layers \\ 3, nodes_per_layer \\ 8) do
    config = %{
      geometry_type: :circular,
      dimensions: 2,
      radius: radius,
      num_layers: num_layers,
      nodes_per_layer: nodes_per_layer
    }
    
    new(config)
  end

  def hexagonal(radius, layers \\ [:input, :hidden, :output]) do
    config = %{
      geometry_type: :hexagonal,
      dimensions: 2,
      radius: radius,
      layers: layers
    }
    
    new(config)
  end

  def custom(input_positions, hidden_positions, output_positions, opts \\ []) do
    config = Map.merge(%{
      geometry_type: :custom,
      input_positions: input_positions,
      hidden_positions: hidden_positions,
      output_positions: output_positions
    }, Map.new(opts))
    
    new(config)
  end

  def add_symmetry(%__MODULE__{} = substrate, symmetry_type) do
    case symmetry_type do
      :bilateral -> add_bilateral_symmetry(substrate)
      :radial -> add_radial_symmetry(substrate)
      :translational -> add_translational_symmetry(substrate)
      _ -> substrate
    end
  end

  def scale_positions(%__MODULE__{} = substrate, scale_factor) do
    %{substrate |
      input_positions: Enum.map(substrate.input_positions, &scale_position(&1, scale_factor)),
      hidden_positions: Enum.map(substrate.hidden_positions, &scale_position(&1, scale_factor)),
      output_positions: Enum.map(substrate.output_positions, &scale_position(&1, scale_factor))
    }
  end

  def translate_positions(%__MODULE__{} = substrate, offset) do
    %{substrate |
      input_positions: Enum.map(substrate.input_positions, &translate_position(&1, offset)),
      hidden_positions: Enum.map(substrate.hidden_positions, &translate_position(&1, offset)),
      output_positions: Enum.map(substrate.output_positions, &translate_position(&1, offset))
    }
  end

  def rotate_positions(%__MODULE__{} = substrate, angle, axis \\ :z) do
    rotation_fn = case substrate.dimensions do
      2 -> &rotate_2d(&1, angle)
      3 -> &rotate_3d(&1, angle, axis)
      _ -> &(&1)  # No rotation for 1D
    end
    
    %{substrate |
      input_positions: Enum.map(substrate.input_positions, rotation_fn),
      hidden_positions: Enum.map(substrate.hidden_positions, rotation_fn),
      output_positions: Enum.map(substrate.output_positions, rotation_fn)
    }
  end

  def get_neighbors(%__MODULE__{} = substrate, position, radius \\ 1.0) do
    all_positions = substrate.input_positions ++ substrate.hidden_positions ++ substrate.output_positions
    
    all_positions
    |> Enum.filter(&(calculate_distance(position, &1) <= radius))
    |> Enum.reject(&(&1 == position))
  end

  def apply_boundary_conditions(%__MODULE__{} = substrate, position) do
    case substrate.boundary_conditions do
      :periodic -> apply_periodic_boundary(position, substrate)
      :reflective -> apply_reflective_boundary(position, substrate)
      :absorbing -> position  # No change for absorbing boundaries
    end
  end

  def get_topology_connections(%__MODULE__{} = substrate) do
    case substrate.topology do
      :grid -> get_grid_connections(substrate)
      :random -> get_random_connections(substrate)
      :small_world -> get_small_world_connections(substrate)
      :scale_free -> get_scale_free_connections(substrate)
      _ -> []
    end
  end

  def calculate_substrate_volume(%__MODULE__{} = substrate) do
    all_positions = substrate.input_positions ++ substrate.hidden_positions ++ substrate.output_positions
    
    case substrate.dimensions do
      1 -> calculate_1d_volume(all_positions)
      2 -> calculate_2d_area(all_positions)
      3 -> calculate_3d_volume(all_positions)
      _ -> 0.0
    end
  end

  def visualize_substrate(%__MODULE__{} = substrate, format \\ :text) do
    case format do
      :text -> visualize_text(substrate)
      :coordinates -> visualize_coordinates(substrate)
      :adjacency -> visualize_adjacency_matrix(substrate)
    end
  end

  # Private functions

  defp generate_layer_positions(layer_type, geometry_type, config) do
    case geometry_type do
      :grid -> generate_grid_positions(layer_type, config)
      :circular -> generate_circular_positions(layer_type, config)
      :hexagonal -> generate_hexagonal_positions(layer_type, config)
      :random -> generate_random_positions(layer_type, config)
      :custom -> Map.get(config, :"#{layer_type}_positions", [])
    end
  end

  defp generate_grid_positions(layer_type, config) do
    dimensions = Map.get(config, :dimensions, 2)
    resolution = Map.get(config, :resolution, 10)
    layer_config = Map.get(config, :"#{layer_type}_layer_config", %{})
    
    case dimensions do
      1 -> generate_1d_grid(layer_type, resolution, layer_config)
      2 -> generate_2d_grid(layer_type, resolution, layer_config)
      3 -> generate_3d_grid(layer_type, resolution, layer_config)
    end
  end

  defp generate_1d_grid(layer_type, resolution, layer_config) do
    size = Map.get(layer_config, :size, resolution)
    position_offset = get_layer_offset(layer_type, 1)
    
    for i <- 0..(size - 1) do
      {i / max(1, size - 1) + position_offset}
    end
  end

  defp generate_2d_grid(layer_type, resolution, layer_config) do
    {width, height} = normalize_resolution(resolution)
    size = Map.get(layer_config, :size, width * height)
    y_offset = get_layer_offset(layer_type, 2)
    
    positions_per_row = min(width, size)
    num_rows = div(size + positions_per_row - 1, positions_per_row)
    
    for row <- 0..(num_rows - 1),
        col <- 0..(positions_per_row - 1),
        row * positions_per_row + col < size do
      x = col / max(1, positions_per_row - 1)
      y = y_offset + row / max(1, num_rows)
      {x, y}
    end
  end

  defp generate_3d_grid(layer_type, resolution, layer_config) do
    {width, height, depth} = normalize_resolution_3d(resolution)
    size = Map.get(layer_config, :size, width * height * depth)
    z_offset = get_layer_offset(layer_type, 3)
    
    positions_per_slice = width * height
    num_slices = div(size + positions_per_slice - 1, positions_per_slice)
    
    for slice <- 0..(num_slices - 1),
        row <- 0..(height - 1),
        col <- 0..(width - 1),
        slice * positions_per_slice + row * width + col < size do
      x = col / max(1, width - 1)
      y = row / max(1, height - 1) 
      z = z_offset + slice / max(1, num_slices)
      {x, y, z}
    end
  end

  defp generate_circular_positions(layer_type, config) do
    radius = Map.get(config, :radius, 1.0)
    nodes_per_layer = Map.get(config, :nodes_per_layer, 8)
    layer_radius = get_circular_layer_radius(layer_type, radius)
    
    for i <- 0..(nodes_per_layer - 1) do
      angle = 2 * :math.pi() * i / nodes_per_layer
      x = layer_radius * :math.cos(angle)
      y = layer_radius * :math.sin(angle)
      {x, y}
    end
  end

  defp generate_hexagonal_positions(layer_type, config) do
    radius = Map.get(config, :radius, 1.0)
    layer_radius = get_circular_layer_radius(layer_type, radius)
    
    # Generate hexagonal grid positions
    _hex_positions = []
    
    for q <- -layer_radius..layer_radius,
        r <- max(-layer_radius, -q - layer_radius)..min(layer_radius, -q + layer_radius) do
      {x, y} = hex_to_cartesian(q, r)
      {x, y}
    end
  end

  defp generate_random_positions(layer_type, config) do
    count = Map.get(config, :"#{layer_type}_count", 10)
    dimensions = Map.get(config, :dimensions, 2)
    
    for _ <- 1..count do
      case dimensions do
        1 -> {:rand.uniform()}
        2 -> {:rand.uniform(), :rand.uniform()}
        3 -> {:rand.uniform(), :rand.uniform(), :rand.uniform()}
      end
    end
  end

  defp normalize_resolution(resolution) when is_integer(resolution), do: {resolution, resolution}
  defp normalize_resolution({width, height}), do: {width, height}
  defp normalize_resolution({width, height, _depth}), do: {width, height}

  defp normalize_resolution_3d(resolution) when is_integer(resolution), do: {resolution, resolution, resolution}
  defp normalize_resolution_3d({width, height}), do: {width, height, width}
  defp normalize_resolution_3d({width, height, depth}), do: {width, height, depth}

  defp get_layer_offset(:input, 1), do: 0.0
  defp get_layer_offset(:hidden, 1), do: 0.5
  defp get_layer_offset(:output, 1), do: 1.0

  defp get_layer_offset(:input, 2), do: 0.0
  defp get_layer_offset(:hidden, 2), do: 0.5  
  defp get_layer_offset(:output, 2), do: 1.0

  defp get_layer_offset(:input, 3), do: 0.0
  defp get_layer_offset(:hidden, 3), do: 0.5
  defp get_layer_offset(:output, 3), do: 1.0

  defp get_circular_layer_radius(:input, radius), do: radius * 0.3
  defp get_circular_layer_radius(:hidden, radius), do: radius * 0.6
  defp get_circular_layer_radius(:output, radius), do: radius

  defp hex_to_cartesian(q, r) do
    x = 3.0/2.0 * q
    y = :math.sqrt(3.0) * (r + q/2.0)
    {x, y}
  end

  defp scale_position({x}, scale), do: {x * scale}
  defp scale_position({x, y}, scale), do: {x * scale, y * scale}
  defp scale_position({x, y, z}, scale), do: {x * scale, y * scale, z * scale}

  defp translate_position({x}, {dx}), do: {x + dx}
  defp translate_position({x, y}, {dx, dy}), do: {x + dx, y + dy}
  defp translate_position({x, y, z}, {dx, dy, dz}), do: {x + dx, y + dy, z + dz}

  defp rotate_2d({x, y}, angle) do
    cos_a = :math.cos(angle)
    sin_a = :math.sin(angle)
    {x * cos_a - y * sin_a, x * sin_a + y * cos_a}
  end

  defp rotate_3d({x, y, z}, angle, :x) do
    cos_a = :math.cos(angle)
    sin_a = :math.sin(angle)
    {x, y * cos_a - z * sin_a, y * sin_a + z * cos_a}
  end

  defp rotate_3d({x, y, z}, angle, :y) do
    cos_a = :math.cos(angle)
    sin_a = :math.sin(angle)
    {x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a}
  end

  defp rotate_3d({x, y, z}, angle, :z) do
    cos_a = :math.cos(angle)
    sin_a = :math.sin(angle)
    {x * cos_a - y * sin_a, x * sin_a + y * cos_a, z}
  end

  defp add_bilateral_symmetry(%__MODULE__{} = substrate) do
    # Mirror positions across vertical axis
    mirrored_positions = fn positions ->
      mirrored = Enum.map(positions, fn
        {x, y} -> {-x, y}
        {x, y, z} -> {-x, y, z}
        pos -> pos
      end)
      positions ++ mirrored
    end
    
    %{substrate |
      input_positions: mirrored_positions.(substrate.input_positions),
      hidden_positions: mirrored_positions.(substrate.hidden_positions),
      output_positions: mirrored_positions.(substrate.output_positions),
      symmetry: :bilateral
    }
  end

  defp add_radial_symmetry(%__MODULE__{} = substrate) do
    # Create radial copies (4-fold symmetry)
    radial_positions = fn positions ->
      angles = [0, :math.pi/2, :math.pi, 3*:math.pi/2]
      
      Enum.flat_map(angles, fn angle ->
        Enum.map(positions, &rotate_2d(&1, angle))
      end)
    end
    
    %{substrate |
      input_positions: radial_positions.(substrate.input_positions),
      hidden_positions: radial_positions.(substrate.hidden_positions),
      output_positions: radial_positions.(substrate.output_positions),
      symmetry: :radial
    }
  end

  defp add_translational_symmetry(%__MODULE__{} = substrate) do
    # Create translated copies
    offset = {1.0, 0.0}
    translated_positions = fn positions ->
      translated = Enum.map(positions, &translate_position(&1, offset))
      positions ++ translated
    end
    
    %{substrate |
      input_positions: translated_positions.(substrate.input_positions),
      hidden_positions: translated_positions.(substrate.hidden_positions),
      output_positions: translated_positions.(substrate.output_positions),
      symmetry: :translational
    }
  end

  defp calculate_distance({x1}, {x2}), do: abs(x2 - x1)
  defp calculate_distance({x1, y1}, {x2, y2}) do
    :math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
  end
  defp calculate_distance({x1, y1, z1}, {x2, y2, z2}) do
    :math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))
  end

  defp apply_periodic_boundary(position, %__MODULE__{} = _substrate) do
    # Wrap coordinates to stay within [0, 1] bounds
    case position do
      {x} -> {:math.fmod(x + 1.0, 1.0)}
      {x, y} -> {:math.fmod(x + 1.0, 1.0), :math.fmod(y + 1.0, 1.0)}
      {x, y, z} -> {:math.fmod(x + 1.0, 1.0), :math.fmod(y + 1.0, 1.0), :math.fmod(z + 1.0, 1.0)}
    end
  end

  defp apply_reflective_boundary(position, %__MODULE__{} = _substrate) do
    # Reflect coordinates at boundaries
    reflect = fn coord ->
      cond do
        coord < 0.0 -> -coord
        coord > 1.0 -> 2.0 - coord
        true -> coord
      end
    end
    
    case position do
      {x} -> {reflect.(x)}
      {x, y} -> {reflect.(x), reflect.(y)}
      {x, y, z} -> {reflect.(x), reflect.(y), reflect.(z)}
    end
  end

  defp get_grid_connections(%__MODULE__{} = _substrate) do
    # Grid topology connections (neighbors)
    []  # Implementation would generate neighbor connections
  end

  defp get_random_connections(%__MODULE__{} = _substrate) do
    # Random topology connections
    []  # Implementation would generate random connections
  end

  defp get_small_world_connections(%__MODULE__{} = _substrate) do
    # Small world topology (local + some random long-range)
    []  # Implementation would generate small-world network
  end

  defp get_scale_free_connections(%__MODULE__{} = _substrate) do
    # Scale-free topology (preferential attachment)
    []  # Implementation would generate scale-free network
  end

  defp calculate_1d_volume(positions) do
    xs = Enum.map(positions, fn {x} -> x end)
    Enum.max(xs) - Enum.min(xs)
  end

  defp calculate_2d_area(positions) do
    xs = Enum.map(positions, fn {x, _} -> x end)
    ys = Enum.map(positions, fn {_, y} -> y end)
    
    width = Enum.max(xs) - Enum.min(xs)
    height = Enum.max(ys) - Enum.min(ys)
    width * height
  end

  defp calculate_3d_volume(positions) do
    xs = Enum.map(positions, fn {x, _, _} -> x end)
    ys = Enum.map(positions, fn {_, y, _} -> y end)
    zs = Enum.map(positions, fn {_, _, z} -> z end)
    
    width = Enum.max(xs) - Enum.min(xs)
    height = Enum.max(ys) - Enum.min(ys)
    depth = Enum.max(zs) - Enum.min(zs)
    width * height * depth
  end

  defp visualize_text(%__MODULE__{} = substrate) do
    """
    Substrate Configuration:
    - Geometry: #{substrate.geometry_type}
    - Dimensions: #{substrate.dimensions}
    - Input nodes: #{length(substrate.input_positions)}
    - Hidden nodes: #{length(substrate.hidden_positions)}  
    - Output nodes: #{length(substrate.output_positions)}
    - Total nodes: #{length(substrate.input_positions) + length(substrate.hidden_positions) + length(substrate.output_positions)}
    """
  end

  defp visualize_coordinates(%__MODULE__{} = substrate) do
    %{
      input: substrate.input_positions,
      hidden: substrate.hidden_positions,
      output: substrate.output_positions
    }
  end

  defp visualize_adjacency_matrix(%__MODULE__{} = substrate) do
    # Generate adjacency matrix representation
    all_positions = substrate.input_positions ++ substrate.hidden_positions ++ substrate.output_positions
    size = length(all_positions)
    
    # This would generate actual adjacency matrix based on connectivity rules
    %{
      size: size,
      positions: all_positions,
      matrix: "#{size}x#{size} adjacency matrix"
    }
  end
end