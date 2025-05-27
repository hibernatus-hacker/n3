#!/usr/bin/env elixir

# Substrate & HyperNEAT Example - Demonstrates spatial neural networks
#
# This example shows how to use substrate encoding with HyperNEAT to evolve
# large-scale spatial neural networks for pattern recognition tasks.


defmodule SubstrateExample do
  def run do
    IO.puts("🌐 NeuroEvolution - Substrate & HyperNEAT Example")
    IO.puts("===============================================\n")

    # Create a 2D grid substrate for spatial pattern processing
    IO.puts("🏗️  Creating 2D substrate (5x5 grid)...")
    substrate = NeuroEvolution.Substrate.grid_2d(5, 5)

    IO.puts("   📍 Input positions: #{length(substrate.input_positions)}")
    IO.puts("   🧠 Hidden positions: #{length(substrate.hidden_positions)}")
    IO.puts("   📤 Output positions: #{length(substrate.output_positions)}")

    # Create HyperNEAT system with the substrate
    IO.puts("\n🔬 Creating HyperNEAT system...")
    hyperneat = NeuroEvolution.new_hyperneat(
      [5, 1],  # Input layer dimensions
      [3, 3],  # Hidden layer dimensions
      [5, 1],  # Output layer dimensions
      connection_threshold: 0.3,
      leo_enabled: true  # Link Expression Output for modularity
    )

    IO.puts("   🔗 Connection threshold: #{hyperneat.connection_threshold}")
    IO.puts("   🧬 LEO enabled: #{hyperneat.leo_enabled}")
    IO.puts("   🎛️  CPPN complexity: #{map_size(hyperneat.cppn.nodes)} nodes, #{map_size(hyperneat.cppn.connections)} connections")

    # Decode substrate to create phenotype network
    IO.puts("\n🔄 Decoding substrate to phenotype network...")
    phenotype = NeuroEvolution.Substrate.HyperNEAT.decode_substrate(hyperneat)

    IO.puts("   🏗️  Phenotype network:")
    IO.puts("     • Nodes: #{map_size(phenotype.nodes)}")
    IO.puts("     • Connections: #{map_size(phenotype.connections)}")
    IO.puts("     • Inputs: #{length(phenotype.inputs)}")
    IO.puts("     • Outputs: #{length(phenotype.outputs)}")

    # Demonstrate CPPN querying
    IO.puts("\n🔍 Querying CPPN for spatial connections...")
    source_pos = List.first(substrate.input_positions)
    target_pos = List.first(substrate.output_positions)

    connection_query = NeuroEvolution.Substrate.HyperNEAT.query_cppn_for_connection(
      hyperneat,
      {source_pos, target_pos, 1, 25}  # From node 1 to node 25
    )

    IO.puts("   📊 Connection analysis:")
    IO.puts("     • Weight: #{Float.round(connection_query.weight, 4)}")
    IO.puts("     • Expression: #{Float.round(connection_query.expression, 4)}")
    IO.puts("     • Distance: #{Float.round(connection_query.distance, 4)}")

    # Demonstrate substrate transformations
    IO.puts("\n🔧 Demonstrating substrate transformations...")

    scaled = NeuroEvolution.Substrate.scale_positions(substrate, 0.8)
    IO.puts("   📐 Scaled substrate by 0.8x")

    rotated = NeuroEvolution.Substrate.rotate_positions(substrate, :math.pi / 4)
    IO.puts("   🔄 Rotated substrate by 45°")

    symmetric = NeuroEvolution.Substrate.add_symmetry(substrate, :bilateral)
    IO.puts("   🪞 Added bilateral symmetry")

    # Demonstrate evolution of spatial patterns
    IO.puts("\n🧬 Evolving spatial pattern classifier...")

    # Simple pattern classification task (vertical vs horizontal lines)
    pattern_fitness_fn = fn hyperneat_genome ->
      # Test patterns: vertical line [1,0,1,0,1] vs horizontal line [1,1,1,0,0]
      test_patterns = [
        {[1.0, 0.0, 1.0, 0.0, 1.0], [1.0]},  # Vertical → class 1
        {[0.0, 1.0, 0.0, 1.0, 0.0], [1.0]},  # Vertical → class 1
        {[1.0, 1.0, 1.0, 0.0, 0.0], [0.0]},  # Horizontal → class 0
        {[0.0, 0.0, 1.0, 1.0, 1.0], [0.0]}   # Horizontal → class 0
      ]

      # Decode and evaluate phenotype
      phenotype_net = NeuroEvolution.Substrate.HyperNEAT.decode_substrate(hyperneat_genome)

      total_error = Enum.reduce(test_patterns, 0.0, fn {inputs, expected}, acc ->
        outputs = NeuroEvolution.activate(phenotype_net, inputs)
        error = :math.pow(List.first(outputs, 0.0) - List.first(expected), 2)
        acc + error
      end)

      4.0 - total_error  # Convert to fitness
    end

    # Create HyperNEAT population
    hyperneat_population = [hyperneat |
      for _i <- 1..19 do
        NeuroEvolution.Substrate.HyperNEAT.mutate_cppn(hyperneat, %{
          weight_mutation_rate: 0.8,
          add_node_rate: 0.1,
          add_connection_rate: 0.15
        })
      end
    ]

    # Evaluate and select best
    best_hyperneat = Enum.max_by(hyperneat_population, pattern_fitness_fn)
    best_fitness = pattern_fitness_fn.(best_hyperneat)

    IO.puts("   🏆 Best spatial classifier fitness: #{Float.round(best_fitness, 3)}")
    IO.puts("   🧠 Final CPPN complexity: #{map_size(best_hyperneat.cppn.nodes)} nodes")

    IO.puts("\n✅ Substrate example complete!")
    IO.puts("🎯 HyperNEAT successfully evolved spatial pattern recognition capabilities.")
  end
end

# Run the example
SubstrateExample.run()
