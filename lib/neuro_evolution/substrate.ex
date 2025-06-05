defmodule NeuroEvolution.Substrate do
  @moduledoc """
  Convenience module that delegates to NeuroEvolution.Substrate.Substrate.
  """
  
  defdelegate new(config), to: NeuroEvolution.Substrate.Substrate
  defdelegate grid_2d(width, height), to: NeuroEvolution.Substrate.Substrate
  defdelegate grid_3d(width, height, depth), to: NeuroEvolution.Substrate.Substrate
  defdelegate circular(radius, num_layers \\ 3, nodes_per_layer \\ 8), to: NeuroEvolution.Substrate.Substrate
  defdelegate hexagonal(radius, layers \\ [:input, :hidden, :output]), to: NeuroEvolution.Substrate.Substrate
  defdelegate custom(input_positions, hidden_positions, output_positions, opts \\ []), to: NeuroEvolution.Substrate.Substrate
  defdelegate add_symmetry(substrate, symmetry_type), to: NeuroEvolution.Substrate.Substrate
  defdelegate scale_positions(substrate, scale_factor), to: NeuroEvolution.Substrate.Substrate
  defdelegate translate_positions(substrate, offset), to: NeuroEvolution.Substrate.Substrate
  defdelegate rotate_positions(substrate, angle, axis \\ :z), to: NeuroEvolution.Substrate.Substrate
  defdelegate get_neighbors(substrate, position, radius \\ 1.0), to: NeuroEvolution.Substrate.Substrate
  defdelegate apply_boundary_conditions(substrate, position), to: NeuroEvolution.Substrate.Substrate
  defdelegate get_topology_connections(substrate), to: NeuroEvolution.Substrate.Substrate
  defdelegate calculate_substrate_volume(substrate), to: NeuroEvolution.Substrate.Substrate
  defdelegate visualize_substrate(substrate, format \\ :text), to: NeuroEvolution.Substrate.Substrate
end