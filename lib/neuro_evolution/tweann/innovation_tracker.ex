defmodule NeuroEvolution.TWEANN.InnovationTracker do
  @moduledoc """
  Global innovation number tracking for NEAT algorithm.
  
  This module ensures that identical mutations (same source and target nodes)
  receive the same innovation number across the entire population, which is
  critical for proper crossover alignment in NEAT.
  """
  
  use GenServer
  require Logger
  
  # Type definitions
  @type mutation_key :: {from :: integer(), to :: integer(), type :: :connection | :node}
  @type innovation_number :: integer()
  
  # Server state
  defstruct [
    :innovation_map,      # %{mutation_key => innovation_number}
    :next_innovation,     # next available innovation number
    :generation,          # current generation number
    :stats                # tracking statistics
  ]
  
  @doc """
  Starts the innovation tracker as a supervised process.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Gets an innovation number for a connection mutation.
  If this exact mutation has been seen before, returns the existing number.
  Otherwise, assigns a new innovation number.
  """
  def get_connection_innovation(from_node, to_node) do
    mutation_key = {from_node, to_node, :connection}
    GenServer.call(__MODULE__, {:get_innovation, mutation_key})
  end
  
  @doc """
  Gets an innovation number for a node mutation.
  Node mutations split an existing connection, so we track based on the 
  connection being split.
  """
  def get_node_innovation(from_node, to_node) do
    mutation_key = {from_node, to_node, :node}
    GenServer.call(__MODULE__, {:get_innovation, mutation_key})
  end
  
  @doc """
  Resets the innovation tracker for a new generation.
  This clears the mutation history but preserves the innovation counter
  to maintain global uniqueness.
  """
  def new_generation(generation_number) do
    GenServer.cast(__MODULE__, {:new_generation, generation_number})
  end
  
  @doc """
  Gets current innovation tracking statistics.
  """
  def get_stats do
    GenServer.call(__MODULE__, :get_stats)
  end
  
  @doc """
  Resets the entire innovation tracker (use with caution).
  """
  def reset do
    GenServer.cast(__MODULE__, :reset)
  end
  
  # GenServer callbacks
  
  def init(_opts) do
    Logger.info("Starting innovation tracker")
    
    state = %__MODULE__{
      innovation_map: %{},
      next_innovation: 1,
      generation: 0,
      stats: %{
        total_innovations: 0,
        connection_innovations: 0,
        node_innovations: 0,
        reused_innovations: 0
      }
    }
    
    {:ok, state}
  end
  
  def handle_call({:get_innovation, mutation_key}, _from, state) do
    case Map.get(state.innovation_map, mutation_key) do
      nil ->
        # New mutation - assign next innovation number
        innovation_number = state.next_innovation
        new_innovation_map = Map.put(state.innovation_map, mutation_key, innovation_number)
        
        # Update statistics
        {_, _, mutation_type} = mutation_key
        new_stats = state.stats
                   |> Map.update!(:total_innovations, &(&1 + 1))
                   |> Map.update!(mutation_type_to_stat_key(mutation_type), &(&1 + 1))
        
        new_state = %{state |
          innovation_map: new_innovation_map,
          next_innovation: state.next_innovation + 1,
          stats: new_stats
        }
        
        Logger.debug("Assigned new innovation #{innovation_number} for #{inspect(mutation_key)}")
        {:reply, innovation_number, new_state}
      
      existing_innovation ->
        # Existing mutation - reuse innovation number
        new_stats = Map.update!(state.stats, :reused_innovations, &(&1 + 1))
        new_state = %{state | stats: new_stats}
        
        Logger.debug("Reused innovation #{existing_innovation} for #{inspect(mutation_key)}")
        {:reply, existing_innovation, new_state}
    end
  end
  
  def handle_call(:get_stats, _from, state) do
    enhanced_stats = Map.merge(state.stats, %{
      current_generation: state.generation,
      next_innovation: state.next_innovation,
      unique_mutations: map_size(state.innovation_map)
    })
    
    {:reply, enhanced_stats, state}
  end
  
  def handle_cast({:new_generation, generation_number}, state) do
    Logger.info("Innovation tracker advancing to generation #{generation_number}")
    
    # Clear the innovation map but keep the innovation counter
    # This allows for reuse within a generation but maintains global uniqueness
    new_state = %{state |
      innovation_map: %{},
      generation: generation_number,
      stats: Map.merge(state.stats, %{
        reused_innovations: 0  # Reset per-generation counter
      })
    }
    
    {:noreply, new_state}
  end
  
  def handle_cast(:reset, _state) do
    Logger.warning("Innovation tracker reset - all innovation history lost")
    
    new_state = %__MODULE__{
      innovation_map: %{},
      next_innovation: 1,
      generation: 0,
      stats: %{
        total_innovations: 0,
        connection_innovations: 0,
        node_innovations: 0,
        reused_innovations: 0
      }
    }
    
    {:noreply, new_state}
  end
  
  # Private functions
  
  defp mutation_type_to_stat_key(:connection), do: :connection_innovations
  defp mutation_type_to_stat_key(:node), do: :node_innovations
end