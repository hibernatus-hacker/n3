#!/usr/bin/env elixir

# Simple CartPole Example - Demonstrates basic CartPole environment functionality
#
# This example shows how to use the CartPole environment with a simple neural network
# to solve the classic control problem.

defmodule CartPoleSimple do
  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Environments.CartPole
  
  def run do
    IO.puts("🧠 NeuroEvolution - Simple CartPole Example")
    IO.puts("======================================================\n")
    
    # Start Python bridge for gym environment
    IO.puts("🔗 Starting Python bridge...")
    {:ok, _bridge} = NeuroEvolution.Environments.PythonBridge.start_link()
    
    # Create a simple genome for CartPole (4 inputs, 2 outputs)
    IO.puts("\n🧬 Creating a simple genome for CartPole...")
    
    genome = create_simple_genome()
    
    IO.puts("   📊 Genome structure:")
    IO.puts("     • Inputs: 4 (cart position, cart velocity, pole angle, pole velocity)")
    IO.puts("     • Outputs: 2 (left, right)")
    IO.puts("     • Connections: #{map_size(genome.connections)}")
    
    # Evaluate the genome on CartPole
    IO.puts("\n🎮 Evaluating genome on CartPole...")
    
    scores = for trial <- 1..5 do
      score = CartPole.evaluate(genome, 1) # Run 1 trial per iteration
      IO.puts("   Trial #{trial}: #{score} steps")
      score
    end
    
    avg_score = Enum.sum(scores) / length(scores)
    IO.puts("\n📈 Average score: #{Float.round(avg_score, 1)} steps")
    
    # Clean up
    NeuroEvolution.Environments.PythonBridge.stop()
    
    IO.puts("\n✅ CartPole example complete!")
  end
  
  # Create a simple feed-forward neural network for CartPole
  defp create_simple_genome do
    # Create a basic genome with 4 inputs and 2 outputs
    genome = Genome.new(4, 2)
    
    # Add a hidden layer with 4 nodes
    genome = add_hidden_layer(genome, 4)
    
    # Connect all inputs to all hidden nodes
    genome = connect_layers(genome, 1..4, 5..8)
    
    # Connect all hidden nodes to all outputs
    genome = connect_layers(genome, 5..8, 9..10)
    
    # Initialize weights with small random values
    genome = randomize_weights(genome)
    
    genome
  end
  
  # Add a hidden layer with n nodes to the genome
  defp add_hidden_layer(genome, n) do
    # Get the next available node ID
    next_id = Enum.max(Map.keys(genome.nodes)) + 1
    
    # Create n hidden nodes
    hidden_nodes = for i <- 0..(n-1), into: %{} do
      node_id = next_id + i
      {node_id, %{type: :hidden, activation: :tanh, bias: 0.0}}
    end
    
    # Add the hidden nodes to the genome
    %{genome | nodes: Map.merge(genome.nodes, hidden_nodes)}
  end
  
  # Connect all nodes in source_range to all nodes in target_range
  defp connect_layers(genome, source_range, target_range) do
    # Create connections between all source and target nodes
    connections = for source_id <- source_range, target_id <- target_range, into: %{} do
      conn_id = "#{source_id}_#{target_id}"
      {conn_id, %{from: source_id, to: target_id, weight: 0.0, enabled: true}}
    end
    
    # Add the connections to the genome
    %{genome | connections: Map.merge(genome.connections, connections)}
  end
  
  # Initialize all connection weights with small random values
  defp randomize_weights(genome) do
    # Update each connection with a random weight
    connections = Map.new(genome.connections, fn {id, conn} ->
      # Generate a random weight between -0.5 and 0.5
      weight = :rand.uniform() - 0.5
      {id, %{conn | weight: weight}}
    end)
    
    # Return the genome with randomized weights
    %{genome | connections: connections}
  end
end

# Run the example
CartPoleSimple.run()
