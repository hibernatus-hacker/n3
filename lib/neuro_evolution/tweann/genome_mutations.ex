defmodule NeuroEvolution.TWEANN.GenomeMutations do
  @moduledoc """
  Provides comprehensive mutation operations for TWEANN genomes.
  
  This module implements various mutation strategies for topology and weight evolution,
  supporting the hybrid reinforcement learning approach that combines traditional RL
  with neuroevolution.
  """
  
  # No aliases needed
  
  @doc """
  Applies multiple mutation operations to a genome based on specified rates.
  
  ## Parameters
  - genome: The genome to mutate
  - opts: Mutation options
    - weight_mutation_rate: Rate of weight mutations (default: 0.8)
    - weight_mutation_power: Strength of weight perturbations (default: 0.5)
    - add_node_rate: Rate of adding new nodes (default: 0.03)
    - add_connection_rate: Rate of adding new connections (default: 0.05)
    - enable_connection_rate: Rate of enabling disabled connections (default: 0.01)
    - disable_connection_rate: Rate of disabling connections (default: 0.01)
  
  ## Returns
  - The mutated genome
  """
  def mutate(genome, opts \\ []) do
    # Default mutation rates
    opts = Keyword.merge([
      weight_mutation_rate: 0.8,
      weight_mutation_power: 0.5,
      add_node_rate: 0.03,
      add_connection_rate: 0.05,
      enable_connection_rate: 0.01,
      disable_connection_rate: 0.01
    ], opts)
    
    # Extract mutation rates from options
    weight_mutation_rate = opts[:weight_mutation_rate]
    weight_mutation_power = opts[:weight_mutation_power]
    add_node_rate = opts[:add_node_rate]
    add_connection_rate = opts[:add_connection_rate]
    enable_connection_rate = opts[:enable_connection_rate]
    disable_connection_rate = opts[:disable_connection_rate]
    
    # Apply mutations in sequence
    genome
    |> mutate_weights(weight_mutation_rate, weight_mutation_power)
    |> maybe_add_node(add_node_rate)
    |> maybe_add_connection(add_connection_rate)
    |> maybe_toggle_connections(enable_connection_rate, disable_connection_rate)
  end
  
  @doc """
  Mutates the weights of a genome's connections.
  
  ## Parameters
  - genome: The genome to mutate
  - mutation_rate: Probability of mutating each weight (default: 0.8)
  - perturbation_strength: Strength of weight perturbations (default: 0.5)
  
  ## Returns
  - The genome with mutated weights
  """
  def mutate_weights(genome, mutation_rate \\ 0.8, perturbation_strength \\ 0.5) do
    # Validate parameters
    mutation_rate = max(0.0, min(1.0, mutation_rate))
    perturbation_strength = max(0.1, perturbation_strength)
    
    # Mutate connection weights
    updated_connections = 
      Enum.reduce(genome.connections, %{}, fn {id, conn}, acc ->
        # Apply mutation based on the mutation rate
        new_weight = 
          if :rand.uniform() < mutation_rate do
            # Sometimes completely replace the weight (10% chance)
            if :rand.uniform() < 0.1 do
              :rand.normal(0.0, 1.0) 
            else
              # Apply a perturbation to the weight
              conn.weight + :rand.normal(0.0, perturbation_strength)
            end
          else
            conn.weight
          end
        
        # Ensure weight is within reasonable bounds
        bounded_weight = max(-4.0, min(4.0, new_weight))
        
        Map.put(acc, id, %{conn | weight: bounded_weight})
      end)
    
    %{genome | connections: updated_connections}
  end
  
  @doc """
  Potentially adds a new node to the genome based on the specified rate.
  
  ## Parameters
  - genome: The genome to mutate
  - add_rate: Probability of adding a node (default: 0.03)
  
  ## Returns
  - The genome, potentially with a new node
  """
  def maybe_add_node(genome, add_rate \\ 0.03) do
    if :rand.uniform() < add_rate do
      add_node(genome)
    else
      genome
    end
  end
  
  @doc """
  Adds a new node to the genome by splitting an existing connection.
  
  ## Parameters
  - genome: The genome to mutate
  
  ## Returns
  - The genome with a new node
  """
  def add_node(genome) do
    # Can't add a node if there are no connections
    if map_size(genome.connections) == 0 do
      genome
    else
      # Select a random enabled connection to split
      enabled_connections = 
        genome.connections
        |> Enum.filter(fn {_id, conn} -> conn.enabled end)
        
      if length(enabled_connections) == 0 do
        genome
      else
        {conn_id, connection} = Enum.random(enabled_connections)
        
        # Parse the connection ID to get source and target
        [from, to] = String.split(conn_id, "_")
        
        # Disable the original connection
        updated_connections = Map.put(genome.connections, conn_id, %{connection | enabled: false})
        
        # Create a new node
        next_node_id = get_next_node_id(genome)
        new_node = %{
          type: :hidden,
          activation: :tanh,
          position: nil,
          bias: 0.0,
          plasticity_params: nil
        }
        
        # Create two new connections
        # 1. From the source to the new node (weight = 1.0)
        # 2. From the new node to the target (weight = original connection weight)
        conn1_id = "#{from}_#{next_node_id}"
        conn2_id = "#{next_node_id}_#{to}"
        
        conn1 = %{
          from: from,
          to: next_node_id,
          weight: 1.0,
          enabled: true,
          recurrent: false,
          plasticity_type: nil
        }
        
        conn2 = %{
          from: next_node_id,
          to: to,
          weight: connection.weight,
          enabled: true,
          recurrent: false,
          plasticity_type: nil
        }
        
        # Update the genome
        updated_connections = 
          updated_connections
          |> Map.put(conn1_id, conn1)
          |> Map.put(conn2_id, conn2)
        
        updated_nodes = Map.put(genome.nodes, next_node_id, new_node)
        
        %{genome | 
          nodes: updated_nodes, 
          connections: updated_connections
        }
      end
    end
  end
  
  @doc """
  Potentially adds a new connection to the genome based on the specified rate.
  
  ## Parameters
  - genome: The genome to mutate
  - add_rate: Probability of adding a connection (default: 0.05)
  
  ## Returns
  - The genome, potentially with a new connection
  """
  def maybe_add_connection(genome, add_rate \\ 0.05) do
    if :rand.uniform() < add_rate do
      add_connection(genome)
    else
      genome
    end
  end
  
  @doc """
  Adds a new connection between two unconnected nodes in the genome.
  
  ## Parameters
  - genome: The genome to mutate
  
  ## Returns
  - The genome with a new connection
  """
  def add_connection(genome) do
    # Get all node IDs
    node_ids = Map.keys(genome.nodes)
    
    # Can't add a connection if there are fewer than 2 nodes
    if length(node_ids) < 2 do
      genome
    else
      # Find potential source and target nodes
      potential_sources = node_ids -- genome.outputs
      potential_targets = node_ids -- genome.inputs
      
      # Can't add a connection if there are no valid source or target nodes
      if length(potential_sources) == 0 or length(potential_targets) == 0 do
        genome
      else
        # Try to find an unconnected pair of nodes
        max_attempts = 20
        
        find_unconnected_pair = fn ->
          source_id = Enum.random(potential_sources)
          target_id = Enum.random(potential_targets)
          
          # Check if this connection already exists
          conn_id = "#{source_id}_#{target_id}"
          
          if Map.has_key?(genome.connections, conn_id) do
            # Connection already exists
            :already_exists
          else
            # Check for cycles (except for recurrent connections)
            if creates_cycle?(genome, source_id, target_id) do
              # Would create a cycle, mark as recurrent
              {:ok, source_id, target_id, true}
            else
              # Valid new connection
              {:ok, source_id, target_id, false}
            end
          end
        end
        
        # Try to find an unconnected pair
        result = 
          Enum.reduce_while(1..max_attempts, :not_found, fn _i, _acc ->
            case find_unconnected_pair.() do
              :already_exists -> {:cont, :not_found}
              result -> {:halt, result}
            end
          end)
        
        case result do
          {:ok, source_id, target_id, _recurrent} ->
            # Create the new connection
            conn_id = "#{source_id}_#{target_id}"
            
            new_connection = %{
              from: source_id,
              to: target_id,
              weight: :rand.normal(0.0, 1.0),
              enabled: true,
              plasticity_type: nil
            }
            
            # Update the genome
            updated_connections = Map.put(genome.connections, conn_id, new_connection)
            %{genome | connections: updated_connections}
            
          _ ->
            # Couldn't find an unconnected pair
            genome
        end
      end
    end
  end
  
  @doc """
  Potentially toggles connection states (enabled/disabled) based on the specified rates.
  
  ## Parameters
  - genome: The genome to mutate
  - enable_rate: Probability of enabling a disabled connection (default: 0.01)
  - disable_rate: Probability of disabling an enabled connection (default: 0.01)
  
  ## Returns
  - The genome with potentially toggled connections
  """
  def maybe_toggle_connections(genome, enable_rate \\ 0.01, disable_rate \\ 0.01) do
    # Toggle connection states
    updated_connections = 
      Enum.reduce(genome.connections, %{}, fn {id, conn}, acc ->
        new_enabled = 
          cond do
            conn.enabled and :rand.uniform() < disable_rate ->
              false
            not conn.enabled and :rand.uniform() < enable_rate ->
              true
            true ->
              conn.enabled
          end
        
        Map.put(acc, id, %{conn | enabled: new_enabled})
      end)
    
    %{genome | connections: updated_connections}
  end
  
  # Helper functions
  
  # Check if adding a connection from source_id to target_id would create a cycle
  defp creates_cycle?(genome, source_id, target_id) do
    # If the target is an input or the source is an output, it's a cycle
    if target_id in genome.inputs or source_id in genome.outputs do
      true
    else
      # Check if there's a path from target to source
      reachable_from_target = reachable_nodes(genome, target_id)
      source_id in reachable_from_target
    end
  end
  
  # Find all nodes reachable from a given node
  defp reachable_nodes(genome, start_node) do
    # Initialize with just the start node
    reachable = MapSet.new([start_node])
    
    # Find all outgoing connections from the start node
    outgoing = 
      genome.connections
      |> Enum.filter(fn {id, conn} -> 
        conn.enabled and String.starts_with?(id, "#{start_node}_")
      end)
      |> Enum.map(fn {id, _conn} -> 
        [_from, to] = String.split(id, "_")
        to
      end)
    
    # Recursively find reachable nodes
    Enum.reduce(outgoing, reachable, fn target, acc ->
      if MapSet.member?(acc, target) do
        # Already visited this node
        acc
      else
        # Add this node and recursively find its reachable nodes
        new_acc = MapSet.put(acc, target)
        reachable_from_target = reachable_nodes(genome, target)
        MapSet.union(new_acc, reachable_from_target)
      end
    end)
  end
  
  # Get the next available node ID
  defp get_next_node_id(genome) do
    case Map.keys(genome.nodes) do
      [] -> 1
      keys -> 
        # Filter out string keys (like "memory" or "balance")
        numeric_keys = Enum.filter(keys, &is_integer/1)
        if Enum.empty?(numeric_keys), do: 1, else: Enum.max(numeric_keys) + 1
    end
  end
end
