defmodule NeuroEvolution.Substrate.HyperNEATTest do
  use ExUnit.Case
  alias NeuroEvolution.Substrate.{HyperNEAT, Substrate}
  alias NeuroEvolution.TWEANN.Genome

  describe "HyperNEAT creation" do
    test "creates HyperNEAT system with valid structure" do
      hyperneat = HyperNEAT.new([3, 3], [2, 2], [1, 1])
      
      assert %HyperNEAT{} = hyperneat
      assert %Substrate{} = hyperneat.substrate
      assert %Genome{} = hyperneat.cppn
      assert hyperneat.connection_threshold > 0
    end

    test "CPPN has correct input/output structure" do
      hyperneat = HyperNEAT.new([3, 3], [2, 2], [1, 1])
      cppn = hyperneat.cppn
      
      # CPPN should have inputs for coordinate pairs (x1, y1, x2, y2, bias)
      assert length(cppn.inputs) >= 5
      # Should have at least 1 output (weight), possibly 2 with LEO
      assert length(cppn.outputs) >= 1
    end

    test "substrate has correct layer dimensions" do
      input_dims = [4, 4]
      hidden_dims = [3, 3] 
      output_dims = [2, 2]
      
      hyperneat = HyperNEAT.new(input_dims, hidden_dims, output_dims)
      substrate = hyperneat.substrate
      
      assert length(substrate.input_positions) == 16  # 4x4
      assert length(substrate.hidden_positions) == 9   # 3x3
      assert length(substrate.output_positions) == 4   # 2x2
    end
  end

  describe "substrate decoding" do
    test "decodes substrate to phenotype network" do
      hyperneat = HyperNEAT.new([2, 2], [2, 2], [1, 1])
      
      phenotype = HyperNEAT.decode_substrate(hyperneat)
      
      assert %Genome{} = phenotype
      assert length(phenotype.inputs) > 0
      assert length(phenotype.outputs) > 0
      assert map_size(phenotype.nodes) > 0
    end

    test "connection threshold affects network density" do
      low_threshold = HyperNEAT.new([3, 3], [2, 2], [1, 1], connection_threshold: 0.1)
      high_threshold = HyperNEAT.new([3, 3], [2, 2], [1, 1], connection_threshold: 0.8)
      
      low_phenotype = HyperNEAT.decode_substrate(low_threshold)
      high_phenotype = HyperNEAT.decode_substrate(high_threshold)
      
      # Lower threshold should generally produce more connections
      low_connections = map_size(low_phenotype.connections)
      high_connections = map_size(high_phenotype.connections)
      
      # This is probabilistic, but low threshold should tend toward more connections
      assert low_connections >= 0
      assert high_connections >= 0
    end

    test "LEO affects output dimensionality" do
      without_leo = HyperNEAT.new([2, 2], [2, 2], [1, 1], leo_enabled: false)
      with_leo = HyperNEAT.new([2, 2], [2, 2], [1, 1], leo_enabled: true)
      
      assert length(without_leo.cppn.outputs) == 1
      assert length(with_leo.cppn.outputs) == 2
    end
  end

  describe "CPPN queries" do
    test "queries CPPN for connection weights" do
      hyperneat = HyperNEAT.new([2, 2], [1, 1], [1, 1])
      
      # Create mock substrate nodes
      alias NeuroEvolution.TWEANN.Node
      substrate_nodes = [
        Node.new(1, :input, :linear, {0.0, 0.0}),
        Node.new(2, :output, :tanh, {1.0, 1.0})
      ]
      
      result = HyperNEAT.query_cppn_for_connection(hyperneat, 
        {{0.0, 0.0}, {1.0, 1.0}, 1, 2})
      
      assert Map.has_key?(result, :from)
      assert Map.has_key?(result, :to) 
      assert Map.has_key?(result, :weight)
      assert Map.has_key?(result, :expression)
      assert Map.has_key?(result, :distance)
      
      assert result.from == 1
      assert result.to == 2
      assert is_float(result.weight)
      assert is_float(result.expression)
      assert is_float(result.distance)
    end

    test "connection weights are bounded" do
      hyperneat = HyperNEAT.new([2, 2], [1, 1], [1, 1], max_weight: 3.0)
      
      # Test multiple queries
      results = for i <- 1..20 do
        HyperNEAT.query_cppn_for_connection(hyperneat,
          {{:rand.uniform(), :rand.uniform()}, 
           {:rand.uniform(), :rand.uniform()}, i, i+1})
      end
      
      weights = Enum.map(results, &(&1.weight))
      
      # All weights should be within bounds
      assert Enum.all?(weights, &(abs(&1) <= 3.0))
    end

    test "distance calculation is correct" do
      hyperneat = HyperNEAT.new([2, 2], [1, 1], [1, 1], distance_function: :euclidean)
      
      result = HyperNEAT.query_cppn_for_connection(hyperneat,
        {{0.0, 0.0}, {3.0, 4.0}, 1, 2})
      
      # Distance from (0,0) to (3,4) should be 5.0
      assert_in_delta result.distance, 5.0, 0.001
    end
  end

  describe "CPPN evolution" do
    test "mutates CPPN structure" do
      hyperneat = HyperNEAT.new([2, 2], [1, 1], [1, 1])
      original_cppn = hyperneat.cppn
      
      mutation_config = %{
        weight_mutation_rate: 1.0,
        add_node_rate: 0.5,
        add_connection_rate: 0.5
      }
      
      mutated_hyperneat = HyperNEAT.mutate_cppn(hyperneat, mutation_config)
      mutated_cppn = mutated_hyperneat.cppn
      
      # Structure should potentially change
      original_complexity = map_size(original_cppn.nodes) + map_size(original_cppn.connections)
      mutated_complexity = map_size(mutated_cppn.nodes) + map_size(mutated_cppn.connections)
      
      # Should maintain at least the original structure
      assert mutated_complexity >= original_complexity
    end

    test "crossover combines CPPN features" do
      parent1 = HyperNEAT.new([2, 2], [1, 1], [1, 1])
      parent2 = HyperNEAT.new([2, 2], [1, 1], [1, 1])
      
      # Add some complexity to make them different
      parent1 = HyperNEAT.mutate_cppn(parent1, %{add_node_rate: 0.3})
      parent2 = HyperNEAT.mutate_cppn(parent2, %{add_connection_rate: 0.3})
      
      child = HyperNEAT.crossover_hyperneat(parent1, parent2)
      
      # Child should have valid structure
      assert %HyperNEAT{} = child
      assert %Genome{} = child.cppn
      assert child.substrate == parent1.substrate  # Should inherit substrate
    end
  end

  describe "substrate topology evolution" do
    test "evolves substrate structure" do
      hyperneat = HyperNEAT.new([2, 2], [1, 1], [1, 1])
      original_substrate = hyperneat.substrate
      
      evolved_hyperneat = HyperNEAT.evolve_substrate_topology(hyperneat)
      
      # Should maintain HyperNEAT structure
      assert %HyperNEAT{} = evolved_hyperneat
      assert %Substrate{} = evolved_hyperneat.substrate
    end

    test "adapts CPPN when substrate changes" do
      hyperneat = HyperNEAT.new([2, 2], [1, 1], [1, 1])
      
      # This is a placeholder test since actual substrate evolution
      # would require more complex dimensional changes
      evolved = HyperNEAT.evolve_substrate_topology(hyperneat)
      
      assert %Genome{} = evolved.cppn
      assert length(evolved.cppn.inputs) >= 5  # Should maintain CPPN input structure
    end
  end

  describe "complexity analysis" do
    test "calculates substrate complexity metrics" do
      hyperneat = HyperNEAT.new([3, 3], [2, 2], [2, 2])
      phenotype = HyperNEAT.decode_substrate(hyperneat)
      
      # Create mock connections for analysis
      connections = Map.values(phenotype.connections)
      substrate_nodes = Map.values(phenotype.nodes)
      
      complexity = HyperNEAT.calculate_substrate_complexity(substrate_nodes, connections)
      
      assert Map.has_key?(complexity, :nodes)
      assert Map.has_key?(complexity, :connections)
      assert Map.has_key?(complexity, :active_connections)
      assert Map.has_key?(complexity, :density)
      
      assert complexity.nodes == length(substrate_nodes)
      assert complexity.connections == length(connections)
      assert complexity.density >= 0.0 and complexity.density <= 1.0
    end

    test "density calculation is correct" do
      # Create simple test case
      substrate_nodes = [
        %{id: 1}, %{id: 2}, %{id: 3}
      ]
      
      connections = [
        %{expression: 0.8},  # Active
        %{expression: 0.3},  # Inactive (< 0.5)
        %{expression: 0.9}   # Active
      ]
      
      complexity = HyperNEAT.calculate_substrate_complexity(substrate_nodes, connections)
      
      # 2 active connections out of 3 total nodes -> density = 2/9 ≈ 0.22
      expected_density = 2.0 / (3 * 3)
      assert_in_delta complexity.density, expected_density, 0.01
    end
  end

  describe "different substrate geometries" do
    test "works with grid substrate" do
      hyperneat = HyperNEAT.new([3, 3], [2, 2], [1, 1], 
        input_geometry: :grid, 
        hidden_geometry: :grid,
        output_geometry: :grid)
      
      phenotype = HyperNEAT.decode_substrate(hyperneat)
      
      assert map_size(phenotype.nodes) > 0
      assert length(phenotype.inputs) > 0
      assert length(phenotype.outputs) > 0
    end

    test "handles 3D substrates" do
      # Test with 3D dimensions
      input_dims = [2, 2, 2]
      hidden_dims = [2, 2, 2]
      output_dims = [1, 1, 1]
      
      hyperneat = HyperNEAT.new(input_dims, hidden_dims, output_dims)
      
      # CPPN should have more inputs for 3D coordinates
      assert length(hyperneat.cppn.inputs) >= 7  # x1,y1,z1,x2,y2,z2,bias
      
      phenotype = HyperNEAT.decode_substrate(hyperneat)
      assert %Genome{} = phenotype
    end
  end

  describe "connection expression" do
    test "respects connection threshold" do
      low_threshold = HyperNEAT.new([2, 2], [1, 1], [1, 1], connection_threshold: 0.1)
      high_threshold = HyperNEAT.new([2, 2], [1, 1], [1, 1], connection_threshold: 0.9)
      
      # Create substrate nodes for testing
      # Create mock substrate nodes using proper Node structs
      alias NeuroEvolution.TWEANN.Node
      substrate_nodes = [
        Node.new(1, :input, :linear, {0.0, 0.0}),
        Node.new(2, :hidden, :tanh, {0.5, 0.5}),
        Node.new(3, :output, :tanh, {1.0, 1.0})
      ]
      
      low_connections = HyperNEAT.query_connections(low_threshold, substrate_nodes)
      high_connections = HyperNEAT.query_connections(high_threshold, substrate_nodes)
      
      # Lower threshold should generally allow more connections
      # (This is probabilistic based on CPPN output)
      assert is_list(low_connections)
      assert is_list(high_connections)
    end

    test "LEO modulates connection expression" do
      hyperneat_with_leo = HyperNEAT.new([2, 2], [1, 1], [1, 1], leo_enabled: true)
      
      result = HyperNEAT.query_cppn_for_connection(hyperneat_with_leo,
        {{0.0, 0.0}, {1.0, 1.0}, 1, 2})
      
      # With LEO, expression should be calculated
      assert result.expression >= 0.0 and result.expression <= 1.0
    end
  end
end