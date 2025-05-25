defmodule NeuroEvolution.TWEANN.Node do
  @moduledoc """
  Neural network node representation with support for various activation functions
  and spatial positioning for substrate-based networks.
  """

  defstruct [
    :id,
    :type,
    :activation,
    :position,
    :bias,
    :plasticity_params
  ]

  @type node_type :: :input | :hidden | :output
  @type activation_type :: :linear | :tanh | :relu | :sigmoid | :adaptive | :plastic
  @type position :: {float(), float()} | {float(), float(), float()} | nil

  @type t :: %__MODULE__{
    id: integer(),
    type: node_type(),
    activation: activation_type(),
    position: position(),
    bias: float(),
    plasticity_params: map() | nil
  }

  def new(id, type, activation \\ :tanh, position \\ nil) do
    %__MODULE__{
      id: id,
      type: type,
      activation: activation,
      position: position,
      bias: if(type == :input, do: 0.0, else: :rand.normal(0.0, 0.1)),
      plasticity_params: nil
    }
  end

  def with_plasticity(%__MODULE__{} = node, plasticity_type, params \\ %{}) do
    plasticity_params = Map.merge(default_plasticity_params(plasticity_type), params)
    
    %{node | 
      activation: :plastic,
      plasticity_params: Map.put(plasticity_params, :type, plasticity_type)
    }
  end

  def activate(%__MODULE__{activation: :linear}, input, _state), do: {input, nil}
  def activate(%__MODULE__{activation: :tanh}, input, _state), do: {:math.tanh(input), nil}
  def activate(%__MODULE__{activation: :relu}, input, _state), do: {max(0.0, input), nil}
  def activate(%__MODULE__{activation: :sigmoid}, input, _state) do
    {1.0 / (1.0 + :math.exp(-input)), nil}
  end

  def activate(%__MODULE__{activation: :adaptive, plasticity_params: params}, input, state) do
    alpha = Map.get(params, :alpha, 0.1)
    current_activation = Map.get(state || %{}, :activation_threshold, 0.0)
    
    new_threshold = current_activation + alpha * (input - current_activation)
    output = :math.tanh(input - new_threshold)
    
    new_state = Map.put(state || %{}, :activation_threshold, new_threshold)
    {output, new_state}
  end

  def activate(%__MODULE__{activation: :plastic, plasticity_params: params}, input, state) do
    case Map.get(params, :type) do
      :hebbian -> activate_hebbian(input, params, state)
      :stdp -> activate_stdp(input, params, state)
      :oja -> activate_oja(input, params, state)
      :intrinsic_plasticity -> activate_intrinsic_plasticity(input, params, state)
      :adaptive_threshold -> activate_adaptive_threshold(input, params, state)
      _ -> {:math.tanh(input), state}
    end
  end

  def distance(%__MODULE__{} = node1, %__MODULE__{} = node2) do
    position_distance(node1.position, node2.position)
  end

  def euclidean_distance(%__MODULE__{position: pos1}, %__MODULE__{position: pos2}) do
    case {pos1, pos2} do
      {{x1, y1}, {x2, y2}} ->
        :math.sqrt(:math.pow(x2 - x1, 2) + :math.pow(y2 - y1, 2))
      
      {{x1, y1, z1}, {x2, y2, z2}} ->
        :math.sqrt(:math.pow(x2 - x1, 2) + :math.pow(y2 - y1, 2) + :math.pow(z2 - z1, 2))
      
      _ -> 
        0.0
    end
  end

  def manhattan_distance(%__MODULE__{position: pos1}, %__MODULE__{position: pos2}) do
    case {pos1, pos2} do
      {{x1, y1}, {x2, y2}} ->
        abs(x2 - x1) + abs(y2 - y1)
      
      {{x1, y1, z1}, {x2, y2, z2}} ->
        abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1)
      
      _ -> 
        0.0
    end
  end

  def update_plasticity_state(%__MODULE__{plasticity_params: nil} = node, _input, _output, state) do
    {node, state}
  end

  def update_plasticity_state(%__MODULE__{plasticity_params: params} = node, input, output, state) do
    case Map.get(params, :type) do
      :hebbian -> update_hebbian_state(node, input, output, state)
      :stdp -> update_stdp_state(node, input, output, state)
      :oja -> update_oja_state(node, input, output, state)
      :intrinsic_plasticity -> update_intrinsic_plasticity_state(node, input, output, state)
      :adaptive_threshold -> update_adaptive_threshold_state(node, input, output, state)
      _ -> {node, state}
    end
  end

  # Private functions

  defp default_plasticity_params(:hebbian) do
    %{
      learning_rate: 0.01,
      decay_rate: 0.99,
      max_weight: 5.0,
      min_weight: -5.0
    }
  end

  defp default_plasticity_params(:stdp) do
    %{
      a_plus: 0.1,
      a_minus: 0.12,
      tau_plus: 20.0,
      tau_minus: 20.0,
      spike_threshold: 0.5
    }
  end

  defp default_plasticity_params(:oja) do
    %{
      learning_rate: 0.01,
      normalization_factor: 1.0
    }
  end
  
  defp default_plasticity_params(:intrinsic_plasticity) do
    %{
      learning_rate: 0.01,
      target_mean: 0.2,
      target_variance: 1.0,
      gain: 1.0,
      bias: 0.0
    }
  end
  
  defp default_plasticity_params(:adaptive_threshold) do
    %{
      threshold_baseline: 0.5,
      adaptation_rate: 0.01,
      recovery_rate: 0.001,
      max_threshold: 1.0
    }
  end

  defp activate_hebbian(input, params, state) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    current_activity = Map.get(state || %{}, :activity, 0.0)
    
    output = :math.tanh(input)
    new_activity = current_activity + learning_rate * input * output
    
    new_state = Map.put(state || %{}, :activity, new_activity)
    {output, new_state}
  end

  defp activate_stdp(input, params, state) do
    spike_threshold = Map.get(params, :spike_threshold, 0.5)
    last_spike_time = Map.get(state || %{}, :last_spike_time, -1000.0)
    current_time = Map.get(state || %{}, :current_time, 0.0)
    
    output = :math.tanh(input)
    
    new_state = 
      if output > spike_threshold do
        state 
        |> Map.put(:last_spike_time, current_time)
        |> Map.put(:current_time, current_time + 1.0)
      else
        Map.put(state || %{}, :current_time, current_time + 1.0)
      end
    
    {output, new_state}
  end

  defp activate_oja(input, params, state) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    weights = Map.get(state || %{}, :weights, [])
    
    output = :math.tanh(input)
    
    # Oja's rule weight update would typically happen at connection level
    new_state = Map.put(state || %{}, :last_output, output)
    {output, new_state}
  end

  defp update_hebbian_state(node, input, output, state) do
    params = node.plasticity_params
    learning_rate = Map.get(params, :learning_rate, 0.01)
    decay_rate = Map.get(params, :decay_rate, 0.99)
    
    current_trace = Map.get(state || %{}, :hebbian_trace, 0.0)
    new_trace = decay_rate * current_trace + learning_rate * input * output
    
    new_state = Map.put(state || %{}, :hebbian_trace, new_trace)
    {node, new_state}
  end

  defp update_stdp_state(node, _input, output, state) do
    params = node.plasticity_params
    spike_threshold = Map.get(params, :spike_threshold, 0.5)
    current_time = Map.get(state || %{}, :current_time, 0.0)
    
    new_state = 
      if output > spike_threshold do
        Map.put(state || %{}, :last_spike_time, current_time)
      else
        state || %{}
      end
    
    {node, new_state}
  end

  defp update_oja_state(node, input, output, state) do
    params = node.plasticity_params
    learning_rate = Map.get(params, :learning_rate, 0.01)
    
    # Store input/output for connection weight updates
    new_state = 
      (state || %{})
      |> Map.put(:last_input, input)
      |> Map.put(:last_output, output)
      |> Map.put(:oja_activity, learning_rate * output * output)
    
    {node, new_state}
  end

  defp activate_intrinsic_plasticity(input, params, state) do
    # Get current gain and bias from state or use defaults
    gain = Map.get(state || %{}, :gain, 1.0)
    bias = Map.get(state || %{}, :bias, 0.0)
    
    # Apply intrinsic plasticity transformation
    transformed_input = gain * input + bias
    output = :math.tanh(transformed_input)
    
    # Store last values in state
    new_state = 
      (state || %{})
      |> Map.put(:last_input, input)
      |> Map.put(:last_output, output)
      |> Map.put(:last_transformed_input, transformed_input)
    
    {output, new_state}
  end
  
  defp activate_adaptive_threshold(input, params, state) do
    # Get threshold from state or use baseline
    threshold_baseline = Map.get(params, :threshold_baseline, 0.5)
    threshold = Map.get(state || %{}, :threshold, threshold_baseline)
    
    # Apply threshold to input
    effective_input = input - threshold
    output = :math.tanh(effective_input)
    
    # Store values in state
    new_state = 
      (state || %{})
      |> Map.put(:last_input, input)
      |> Map.put(:last_output, output)
      |> Map.put(:last_effective_input, effective_input)
    
    {output, new_state}
  end

  defp position_distance(nil, nil), do: 0.0
  defp position_distance(nil, _), do: Float.max_finite()
  defp position_distance(_, nil), do: Float.max_finite()
  
  defp position_distance({x1, y1}, {x2, y2}) do
    :math.sqrt(:math.pow(x2 - x1, 2) + :math.pow(y2 - y1, 2))
  end
  
  defp position_distance({x1, y1, z1}, {x2, y2, z2}) do
    :math.sqrt(:math.pow(x2 - x1, 2) + :math.pow(y2 - y1, 2) + :math.pow(z2 - z1, 2))
  end
  
  defp update_intrinsic_plasticity_state(node, input, output, state) do
    # Update state based on intrinsic plasticity
    params = node.plasticity_params
    learning_rate = Map.get(params, :learning_rate, 0.01)
    target_mean = Map.get(params, :target_mean, 0.2)
    
    # Get current gain and bias
    gain = Map.get(state || %{}, :gain, 1.0)
    bias = Map.get(state || %{}, :bias, 0.0)
    
    # Update gain and bias based on output statistics
    delta_bias = learning_rate * (target_mean - output)
    delta_gain = learning_rate * (1.0/gain + input * (target_mean - output))
    
    new_gain = gain + delta_gain
    new_bias = bias + delta_bias
    
    # Update node parameters
    new_state = Map.merge(state || %{}, %{gain: new_gain, bias: new_bias})
    {node, new_state}
  end
  
  defp update_adaptive_threshold_state(node, input, output, state) do
    # Update state based on adaptive threshold
    params = node.plasticity_params
    adaptation_rate = Map.get(params, :adaptation_rate, 0.01)
    recovery_rate = Map.get(params, :recovery_rate, 0.001)
    threshold_baseline = Map.get(params, :threshold_baseline, 0.5)
    max_threshold = Map.get(params, :max_threshold, 1.0)
    
    # Get current threshold
    threshold = Map.get(state || %{}, :threshold, threshold_baseline)
    
    # Update threshold based on activity
    new_threshold = cond do
      output > threshold -> min(threshold + adaptation_rate, max_threshold)
      true -> max(threshold - recovery_rate, threshold_baseline)
    end
    
    # Update state
    new_state = Map.put(state || %{}, :threshold, new_threshold)
    {node, new_state}
  end
end

defimpl Jason.Encoder, for: NeuroEvolution.TWEANN.Node do
  def encode(node, opts) do
    Jason.Encode.map(%{
      id: node.id,
      type: node.type,
      activation: node.activation,
      position: node.position,
      bias: node.bias,
      plasticity_params: node.plasticity_params
    }, opts)
  end
end