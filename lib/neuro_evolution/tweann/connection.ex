defmodule NeuroEvolution.TWEANN.Connection do
  @moduledoc """
  Neural network connection representation with support for plastic weights
  and various plasticity rules (Hebbian, STDP, Oja's rule).
  """

  defstruct [
    :from,
    :to,
    :weight,
    :innovation,
    :enabled,
    :plasticity_params,
    :plasticity_state
  ]

  @type t :: %__MODULE__{
    from: integer(),
    to: integer(),
    weight: float(),
    innovation: integer(),
    enabled: boolean(),
    plasticity_params: map() | nil,
    plasticity_state: map() | nil
  }

  def new(from, to, weight, innovation, enabled \\ true) do
    %__MODULE__{
      from: from,
      to: to,
      weight: weight,
      innovation: innovation,
      enabled: enabled,
      plasticity_params: nil,
      plasticity_state: nil
    }
  end

  def with_plasticity(%__MODULE__{} = connection, plasticity_type, params \\ %{}) do
    plasticity_params = Map.merge(default_plasticity_params(plasticity_type), params)
    
    %{connection | 
      plasticity_params: Map.put(plasticity_params, :type, plasticity_type),
      plasticity_state: initial_plasticity_state(plasticity_type)
    }
  end

  def update_weight(%__MODULE__{plasticity_params: nil} = connection, _pre_activity, _post_activity) do
    connection
  end

  def update_weight(%__MODULE__{plasticity_params: params} = connection, pre_activity, post_activity) do
    case Map.get(params, :type) do
      :hebbian -> update_hebbian_weight(connection, pre_activity, post_activity)
      :stdp -> update_stdp_weight(connection, pre_activity, post_activity)
      :oja -> update_oja_weight(connection, pre_activity, post_activity)
      :bcm -> update_bcm_weight(connection, pre_activity, post_activity)
      _ -> connection
    end
  end

  def get_effective_weight(%__MODULE__{} = connection) do
    base_weight = connection.weight
    
    case connection.plasticity_params do
      nil -> 
        base_weight
      params ->
        plastic_component = Map.get(connection.plasticity_state || %{}, :plastic_weight, 0.0)
        modulation = Map.get(params, :modulation_strength, 1.0)
        
        base_weight + modulation * plastic_component
    end
  end

  def reset_plasticity_state(%__MODULE__{plasticity_params: nil} = connection), do: connection
  
  def reset_plasticity_state(%__MODULE__{plasticity_params: params} = connection) do
    plasticity_type = Map.get(params, :type)
    %{connection | plasticity_state: initial_plasticity_state(plasticity_type)}
  end

  def decay_plasticity(%__MODULE__{plasticity_params: nil} = connection), do: connection
  
  def decay_plasticity(%__MODULE__{plasticity_params: params} = connection) do
    decay_rate = Map.get(params, :decay_rate, 0.99)
    
    new_state = 
      case connection.plasticity_state do
        nil -> nil
        state ->
          Enum.reduce(state, %{}, fn {key, value}, acc ->
            case key do
              :plastic_weight -> Map.put(acc, key, value * decay_rate)
              :trace -> Map.put(acc, key, value * decay_rate)
              _ -> Map.put(acc, key, value)
            end
          end)
      end
    
    %{connection | plasticity_state: new_state}
  end

  def clip_weight(%__MODULE__{} = connection, min_weight \\ -5.0, max_weight \\ 5.0) do
    clipped_weight = connection.weight |> max(min_weight) |> min(max_weight)
    
    clipped_plastic = 
      case connection.plasticity_state do
        nil -> nil
        state ->
          plastic_weight = Map.get(state, :plastic_weight, 0.0)
          clipped_plastic_weight = plastic_weight |> max(min_weight) |> min(max_weight)
          Map.put(state, :plastic_weight, clipped_plastic_weight)
      end
    
    %{connection | weight: clipped_weight, plasticity_state: clipped_plastic}
  end

  # Private functions

  defp default_plasticity_params(:hebbian) do
    %{
      learning_rate: 0.01,
      decay_rate: 0.99,
      modulation_strength: 1.0,
      threshold: 0.0
    }
  end

  defp default_plasticity_params(:stdp) do
    %{
      a_plus: 0.1,
      a_minus: 0.12,
      tau_plus: 20.0,
      tau_minus: 20.0,
      max_delay: 100.0,
      modulation_strength: 1.0
    }
  end

  defp default_plasticity_params(:oja) do
    %{
      learning_rate: 0.01,
      modulation_strength: 1.0,
      decay_rate: 0.99
    }
  end

  defp default_plasticity_params(:bcm) do
    %{
      learning_rate: 0.01,
      threshold_rate: 0.001,
      modulation_strength: 1.0,
      decay_rate: 0.99
    }
  end

  defp initial_plasticity_state(:hebbian) do
    %{
      plastic_weight: 0.0,
      trace: 0.0
    }
  end

  defp initial_plasticity_state(:stdp) do
    %{
      plastic_weight: 0.0,
      pre_spike_times: [],
      post_spike_times: []
    }
  end

  defp initial_plasticity_state(:oja) do
    %{
      plastic_weight: 0.0,
      post_trace: 0.0
    }
  end

  defp initial_plasticity_state(:bcm) do
    %{
      plastic_weight: 0.0,
      threshold: 1.0,
      post_trace: 0.0
    }
  end

  defp update_hebbian_weight(%__MODULE__{} = connection, pre_activity, post_activity) do
    params = connection.plasticity_params
    state = connection.plasticity_state || %{}
    
    learning_rate = Map.get(params, :learning_rate, 0.01)
    threshold = Map.get(params, :threshold, 0.0)
    
    # Classic Hebbian rule: Δw = η * pre * post
    delta_w = learning_rate * pre_activity * (post_activity - threshold)
    
    current_plastic = Map.get(state, :plastic_weight, 0.0)
    new_plastic_weight = current_plastic + delta_w
    
    new_state = Map.put(state, :plastic_weight, new_plastic_weight)
    %{connection | plasticity_state: new_state}
  end

  defp update_stdp_weight(%__MODULE__{} = connection, pre_activity, post_activity) do
    params = connection.plasticity_params
    state = connection.plasticity_state || %{}
    
    a_plus = Map.get(params, :a_plus, 0.1)
    a_minus = Map.get(params, :a_minus, 0.12)
    tau_plus = Map.get(params, :tau_plus, 20.0)
    _tau_minus = Map.get(params, :tau_minus, 20.0)
    
    # Simplified STDP - in practice would need spike timing information
    # This is a rate-based approximation
    current_plastic = Map.get(state, :plastic_weight, 0.0)
    
    # Simplified rule based on correlation with temporal asymmetry bias
    correlation = pre_activity * post_activity
    temporal_bias = if pre_activity > post_activity, do: a_plus, else: -a_minus
    
    delta_w = temporal_bias * correlation * :math.exp(-abs(pre_activity - post_activity) / tau_plus)
    new_plastic_weight = current_plastic + delta_w
    
    new_state = Map.put(state, :plastic_weight, new_plastic_weight)
    %{connection | plasticity_state: new_state}
  end

  defp update_oja_weight(%__MODULE__{} = connection, pre_activity, post_activity) do
    params = connection.plasticity_params
    state = connection.plasticity_state || %{}
    
    learning_rate = Map.get(params, :learning_rate, 0.01)
    current_plastic = Map.get(state, :plastic_weight, 0.0)
    
    # Oja's rule: Δw = η * post * (pre - post * w)
    current_weight = connection.weight + current_plastic
    delta_w = learning_rate * post_activity * (pre_activity - post_activity * current_weight)
    
    new_plastic_weight = current_plastic + delta_w
    new_state = Map.put(state, :plastic_weight, new_plastic_weight)
    
    %{connection | plasticity_state: new_state}
  end

  defp update_bcm_weight(%__MODULE__{} = connection, pre_activity, post_activity) do
    params = connection.plasticity_params
    state = connection.plasticity_state || %{}
    
    learning_rate = Map.get(params, :learning_rate, 0.01)
    threshold_rate = Map.get(params, :threshold_rate, 0.001)
    
    current_plastic = Map.get(state, :plastic_weight, 0.0)
    current_threshold = Map.get(state, :threshold, 1.0)
    
    # BCM rule: Δw = η * pre * post * (post - θ)
    delta_w = learning_rate * pre_activity * post_activity * (post_activity - current_threshold)
    
    # Update sliding threshold: τ * dθ/dt = post² - θ
    new_threshold = current_threshold + threshold_rate * (post_activity * post_activity - current_threshold)
    
    new_plastic_weight = current_plastic + delta_w
    new_state = %{
      plastic_weight: new_plastic_weight,
      threshold: new_threshold
    }
    
    %{connection | plasticity_state: new_state}
  end
end

defimpl Jason.Encoder, for: NeuroEvolution.TWEANN.Connection do
  def encode(connection, opts) do
    Jason.Encode.map(%{
      from: connection.from,
      to: connection.to,
      weight: connection.weight,
      innovation: connection.innovation,
      enabled: connection.enabled,
      plasticity_params: connection.plasticity_params,
      plasticity_state: connection.plasticity_state
    }, opts)
  end
end