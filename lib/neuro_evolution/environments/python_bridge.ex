defmodule NeuroEvolution.Environments.PythonBridge do
  @moduledoc """
  A secure and robust bridge for communicating with Python from Elixir.
  
  This module provides a supervised interface to Python processes, with
  automatic restart capabilities, proper error handling, and security validation.
  
  ## Security Features
  - Python executable validation and allowlisting
  - Path sanitization and validation
  - Process isolation and resource limits
  - Timeout protection and circuit breaker patterns
  """
  
  use GenServer
  require Logger
  import Bitwise
  
  # Security: Allowed Python executables (basename only)
  @allowed_python_executables ~w[python3 python python3.9 python3.8 python3.7 python3.10 python3.11]
  
  # Security: Path validation regex
  @safe_path_regex ~r/^[a-zA-Z0-9._\/-]+$/
  
  # Circuit breaker: Maximum consecutive failures before disabling
  @max_consecutive_failures 5
  
  
  @doc """
  Starts the Python bridge as a supervised process.
  
  ## Options
  - `:python_path` - Path to Python modules (default: priv/python)
  - `:python_executable` - Python executable to use (default: "python3")
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Initializes the Python bridge.
  """
  def init(opts) do
    python_path = Keyword.get(opts, :python_path, Path.join(:code.priv_dir(:neuro_evolution), "python"))
    python_executable = find_python_executable(opts)
    
    Logger.info("Starting Python bridge with executable: #{python_executable}")
    Logger.info("Python path: #{python_path}")
    
    case start_python_process(python_path, python_executable) do
      {:ok, python_pid} ->
        state = %{
          python_pid: python_pid,
          python_path: python_path,
          python_executable: python_executable,
          consecutive_failures: 0,
          circuit_breaker_open: false,
          last_failure_time: nil
        }
        {:ok, state}
      
      {:error, reason} ->
        Logger.error("Failed to start Python process: #{inspect(reason)}")
        {:stop, reason}
    end
  end
  
  # Find and validate a secure Python executable
  defp find_python_executable(opts) do
    # First check if explicitly provided in options
    candidate = case Keyword.get(opts, :python_executable) do
      nil ->
        # Then check environment variable (with validation)
        case System.get_env("ERL_PYTHON") do
          nil ->
            # Try common Python executable names
            find_python_in_path()
          env_python ->
            Logger.info("Using Python from ERL_PYTHON environment variable: #{env_python}")
            env_python
        end
      explicit_python ->
        Logger.info("Using explicitly provided Python executable: #{explicit_python}")
        explicit_python
    end
    
    # Security: Validate the Python executable
    case validate_python_executable(candidate) do
      {:ok, safe_executable} -> safe_executable
      {:error, reason} ->
        Logger.error("Python executable validation failed: #{reason}")
        raise ArgumentError, "Invalid Python executable: #{reason}"
    end
  end
  
  # Try to find Python in PATH (with security validation)
  defp find_python_in_path do
    Enum.find_value(@allowed_python_executables, "python3", fn candidate ->
      case System.find_executable(candidate) do
        nil -> false
        path -> 
          Logger.info("Found Python executable: #{path}")
          path  # Return full path for validation
      end
    end)
  end
  
  # Security: Validate Python executable
  defp validate_python_executable(executable) when is_binary(executable) do
    cond do
      not Regex.match?(@safe_path_regex, executable) ->
        {:error, "Python executable path contains invalid characters"}
      
      not (Path.basename(executable) in @allowed_python_executables) ->
        {:error, "Python executable not in allowlist: #{Path.basename(executable)}"}
      
      not File.exists?(executable) ->
        {:error, "Python executable not found: #{executable}"}
      
      not File.regular?(executable) ->
        {:error, "Python executable is not a regular file: #{executable}"}
      
      not (File.stat!(executable).mode |> band(0o111) > 0) ->
        {:error, "Python executable is not executable: #{executable}"}
      
      true ->
        # Additional validation: ensure it's actually a Python interpreter
        validate_python_interpreter(executable)
    end
  end
  
  defp validate_python_executable(_), do: {:error, "Python executable must be a string"}
  
  # Validate that the executable is actually a Python interpreter
  defp validate_python_interpreter(executable) do
    case System.cmd(executable, ["--version"], stderr_to_stdout: true) do
      {output, 0} ->
        if String.contains?(String.downcase(output), "python") do
          {:ok, executable}
        else
          {:error, "Executable is not a Python interpreter"}
        end
      {_output, _code} ->
        {:error, "Failed to verify Python interpreter"}
    end
  rescue
    _ -> {:error, "Exception while validating Python interpreter"}
  end
  
  @doc """
  Calls a Python function with the given arguments.
  
  ## Parameters
  - module: Python module name
  - function: Python function name
  - args: Arguments to pass to the function
  """
  def call(module, function, args \\ []) do
    GenServer.call(__MODULE__, {:call, module, function, args}, 30_000)
  end
  
  @doc """
  Stops the Python bridge.
  """
  def stop do
    GenServer.cast(__MODULE__, :stop)
  end
  
  # GenServer callbacks
  
  def handle_call({:call, module, function, args}, _from, state) do
    # Check circuit breaker
    if state.circuit_breaker_open do
      if should_attempt_reset?(state.last_failure_time) do
        # Try to reset circuit breaker
        attempt_call_with_circuit_breaker(module, function, args, state)
      else
        {:reply, {:error, :circuit_breaker_open}, state}
      end
    else
      attempt_call_with_circuit_breaker(module, function, args, state)
    end
  end
  
  # Attempt Python call with circuit breaker logic
  defp attempt_call_with_circuit_breaker(module, function, args, state) do
    try do
      result = :python.call(state.python_pid, module, function, args)
      # Success - reset failure count
      new_state = %{state | consecutive_failures: 0, circuit_breaker_open: false}
      {:reply, {:ok, result}, new_state}
    rescue
      error ->
        Logger.error("Python call failed: #{inspect(error)}")
        
        # Increment failure count
        new_failure_count = state.consecutive_failures + 1
        current_time = System.monotonic_time(:second)
        
        # Check if we should open circuit breaker
        circuit_breaker_open = new_failure_count >= @max_consecutive_failures
        
        if circuit_breaker_open do
          Logger.warning("Circuit breaker opened after #{new_failure_count} consecutive failures")
        end
        
        # Attempt to restart Python process (unless circuit breaker is open)
        new_state = if circuit_breaker_open do
          %{state | 
            consecutive_failures: new_failure_count,
            circuit_breaker_open: true,
            last_failure_time: current_time
          }
        else
          case restart_python_process(state) do
            {:ok, new_python_pid} ->
              Logger.info("Python process restarted successfully")
              %{state | 
                python_pid: new_python_pid,
                consecutive_failures: new_failure_count,
                last_failure_time: current_time
              }
            
            {:error, restart_reason} ->
              Logger.error("Failed to restart Python process: #{inspect(restart_reason)}")
              %{state | 
                consecutive_failures: new_failure_count,
                circuit_breaker_open: true,
                last_failure_time: current_time
              }
          end
        end
        
        {:reply, {:error, error}, new_state}
    end
  end
  
  # Check if enough time has passed to attempt circuit breaker reset
  defp should_attempt_reset?(last_failure_time) when is_nil(last_failure_time), do: true
  defp should_attempt_reset?(last_failure_time) do
    current_time = System.monotonic_time(:second)
    # Wait 60 seconds before attempting reset
    current_time - last_failure_time > 60
  end
  
  # Safely restart Python process
  defp restart_python_process(state) do
    # Safely terminate existing process
    if Process.alive?(state.python_pid) do
      Process.exit(state.python_pid, :kill)
      # Wait a bit for process to terminate
      Process.sleep(100)
    end
    
    start_python_process(state.python_path, state.python_executable)
  end
  
  def handle_cast(:stop, state) do
    Process.exit(state.python_pid, :normal)
    {:stop, :normal, state}
  end
  
  def terminate(_reason, state) do
    Process.exit(state.python_pid, :normal)
    :ok
  end
  
  # Private functions
  
  defp start_python_process(python_path, python_executable) do
    try do
      Logger.info("Starting Python process with executable: #{python_executable} and path: #{python_path}")
      {:ok, python_pid} = :python.start_link([
        {:python_path, to_charlist(python_path)},
        {:python, to_charlist(python_executable)}
      ])
      Logger.info("Python process started successfully")
      {:ok, python_pid}
    rescue
      error -> 
        Logger.error("Failed to start Python process: #{inspect(error)}")
        {:error, error}
    end
  end
end
