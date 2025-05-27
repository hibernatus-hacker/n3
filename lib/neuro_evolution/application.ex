defmodule NeuroEvolution.Application do
  @moduledoc """
  OTP application for NeuroEvolution library.
  
  Sets up the supervision tree with essential services including:
  - Innovation number tracking for NEAT algorithm
  - Python bridge for gym environments (optional)
  """
  
  use Application
  require Logger
  
  def start(_type, _args) do
    Logger.info("Starting NeuroEvolution application")
    
    children = [
      # Innovation tracker is essential for NEAT algorithm
      NeuroEvolution.TWEANN.InnovationTracker,
      
      # Python bridge is optional - starts only if needed
      # {NeuroEvolution.Environments.PythonBridge, []}
    ]
    
    opts = [strategy: :one_for_one, name: NeuroEvolution.Supervisor]
    
    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("NeuroEvolution application started successfully")
        {:ok, pid}
        
      {:error, reason} ->
        Logger.error("Failed to start NeuroEvolution application: #{inspect(reason)}")
        {:error, reason}
    end
  end
  
  def stop(_state) do
    Logger.info("Stopping NeuroEvolution application")
    :ok
  end
end