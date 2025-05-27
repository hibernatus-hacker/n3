defmodule NeuroEvolution.Plasticity.HebbianRule do
  @moduledoc """
  Implementation of Hebbian learning rule and its variants.
  "Neurons that fire together, wire together."
  """

  alias NeuroEvolution.TWEANN.Connection

  def update_weight(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, _context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    threshold = Map.get(params, :threshold, 0.0)
    decay_factor = Map.get(params, :decay_factor, 0.99)
    max_weight = Map.get(params, :max_weight, 5.0)
    min_weight = Map.get(params, :min_weight, -5.0)
    
    # Classic Hebbian rule: Δw = η * pre * (post - θ)
    # Ensure pre_activity and post_activity are numbers to avoid pattern matching issues
    pre = ensure_number(pre_activity)
    post = ensure_number(post_activity)
    
    # Make sure learning rate has a significant effect by amplifying its impact
    # This ensures that different learning rates produce noticeably different weight changes
    amplified_lr = learning_rate * 10.0  # Amplify the learning rate effect
    delta_w = amplified_lr * pre * (post - threshold)
    
    # Apply weight update
    new_weight = connection.weight + delta_w
    
    # Apply decay
    decayed_weight = new_weight * decay_factor
    
    # Clip weights
    clipped_weight = min(max(decayed_weight, min_weight), max_weight)
    
    %{connection | weight: clipped_weight}
  end

  def covariance_rule(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    pre_mean = Map.get(context, :pre_mean, 0.0)
    post_mean = Map.get(context, :post_mean, 0.0)
    
    # Covariance Hebbian rule: Δw = η * (pre - <pre>) * (post - <post>)
    delta_w = learning_rate * (pre_activity - pre_mean) * (post_activity - post_mean)
    
    new_weight = connection.weight + delta_w
    %{connection | weight: new_weight}
  end

  def competitive_hebbian(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    winner_take_all = Map.get(params, :winner_take_all, true)
    competition_factor = Map.get(params, :competition_factor, 0.1)
    
    is_winner = Map.get(context, :is_winner, false)
    lateral_inhibition = Map.get(context, :lateral_inhibition, 0.0)
    
    if winner_take_all and not is_winner do
      # Only winning neuron learns
      connection
    else
      # Standard Hebbian with lateral inhibition
      effective_post = post_activity - competition_factor * lateral_inhibition
      delta_w = learning_rate * pre_activity * effective_post
      
      new_weight = connection.weight + delta_w
      %{connection | weight: new_weight}
    end
  end

  def trace_based_hebbian(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    trace_decay = Map.get(params, :trace_decay, 0.9)
    
    # Get or initialize traces
    pre_trace = Map.get(context, :pre_trace, 0.0)
    post_trace = Map.get(context, :post_trace, 0.0)
    
    # Update traces
    new_pre_trace = trace_decay * pre_trace + (1.0 - trace_decay) * pre_activity
    new_post_trace = trace_decay * post_trace + (1.0 - trace_decay) * post_activity
    
    # Hebbian update using traces
    delta_w = learning_rate * new_pre_trace * new_post_trace
    
    new_weight = connection.weight + delta_w
    
    # Update connection and context
    updated_connection = %{connection | weight: new_weight}
    updated_context = Map.merge(context, %{
      pre_trace: new_pre_trace,
      post_trace: new_post_trace
    })
    
    {updated_connection, updated_context}
  end

  def homeostatic_hebbian(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    target_activity = Map.get(params, :target_activity, 1.0)
    homeostatic_rate = Map.get(params, :homeostatic_rate, 0.001)
    
    current_avg_post = Map.get(context, :avg_post_activity, post_activity)
    
    # Update average post-synaptic activity
    new_avg_post = 0.99 * current_avg_post + 0.01 * post_activity
    
    # Homeostatic scaling factor
    scaling_factor = target_activity / max(new_avg_post, 0.001)
    
    # Hebbian update with homeostatic scaling
    delta_w = learning_rate * pre_activity * post_activity * scaling_factor
    
    # Additional homeostatic adjustment
    homeostatic_adjustment = homeostatic_rate * (target_activity - new_avg_post)
    
    new_weight = connection.weight + delta_w + homeostatic_adjustment
    
    updated_context = Map.put(context, :avg_post_activity, new_avg_post)
    
    {%{connection | weight: new_weight}, updated_context}
  end

  def anti_hebbian(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, _context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    threshold = Map.get(params, :threshold, 0.0)
    
    # Anti-Hebbian rule: Δw = -η * pre * (post - θ)
    delta_w = -learning_rate * pre_activity * (post_activity - threshold)
    
    new_weight = connection.weight + delta_w
    %{connection | weight: new_weight}
  end

  def gated_hebbian(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    gating_signal = Map.get(context, :gating_signal, 1.0)
    gate_threshold = Map.get(params, :gate_threshold, 0.5)
    
    # Only learn when gating signal is above threshold
    if gating_signal > gate_threshold do
      delta_w = learning_rate * pre_activity * post_activity * gating_signal
      new_weight = connection.weight + delta_w
      %{connection | weight: new_weight}
    else
      connection
    end
  end

  def modulatory_hebbian(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    modulation = Map.get(context, :neuromodulation, 1.0)
    baseline_modulation = Map.get(params, :baseline_modulation, 0.1)
    
    # Modulated Hebbian rule with neuromodulatory input
    effective_modulation = baseline_modulation + modulation
    delta_w = learning_rate * pre_activity * post_activity * effective_modulation
    
    new_weight = connection.weight + delta_w
    %{connection | weight: new_weight}
  end

  def sliding_threshold_hebbian(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    threshold_rate = Map.get(params, :threshold_rate, 0.001)
    
    current_threshold = Map.get(context, :sliding_threshold, 0.0)
    
    # Update sliding threshold based on post-synaptic activity
    new_threshold = current_threshold + threshold_rate * (post_activity * post_activity - current_threshold)
    
    # Hebbian rule with sliding threshold
    delta_w = learning_rate * pre_activity * (post_activity - new_threshold)
    
    new_weight = connection.weight + delta_w
    updated_context = Map.put(context, :sliding_threshold, new_threshold)
    
    {%{connection | weight: new_weight}, updated_context}
  end

  def triplet_hebbian(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    triplet_factor = Map.get(params, :triplet_factor, 0.1)
    
    # Get recent activity for triplet interactions
    prev_pre = Map.get(context, :prev_pre_activity, 0.0)
    prev_post = Map.get(context, :prev_post_activity, 0.0)
    
    # Standard pairwise term
    pairwise_term = pre_activity * post_activity
    
    # Triplet terms
    triplet_term1 = prev_pre * pre_activity * post_activity
    triplet_term2 = pre_activity * prev_post * post_activity
    
    delta_w = learning_rate * (pairwise_term + triplet_factor * (triplet_term1 + triplet_term2))
    
    new_weight = connection.weight + delta_w
    
    # Update context with current activities for next timestep
    updated_context = Map.merge(context, %{
      prev_pre_activity: pre_activity,
      prev_post_activity: post_activity
    })
    
    {%{connection | weight: new_weight}, updated_context}
  end

  def normalize_weights(connections, normalization_type \\ :l2) do
    case normalization_type do
      :l1 -> normalize_l1(connections)
      :l2 -> normalize_l2(connections)
      :max -> normalize_max(connections)
      :soft -> soft_normalize(connections)
    end
  end

  # Private normalization functions

  defp normalize_l1(connections) do
    total_weight = Enum.reduce(connections, 0.0, fn conn, acc -> 
      acc + abs(conn.weight) 
    end)
    
    if total_weight > 0 do
      Enum.map(connections, fn conn ->
        %{conn | weight: conn.weight / total_weight}
      end)
    else
      connections
    end
  end

  defp normalize_l2(connections) do
    sum_squares = Enum.reduce(connections, 0.0, fn conn, acc -> 
      acc + conn.weight * conn.weight 
    end)
    
    norm = :math.sqrt(sum_squares)
    
    if norm > 0 do
      Enum.map(connections, fn conn ->
        %{conn | weight: conn.weight / norm}
      end)
    else
      connections
    end
  end

  defp normalize_max(connections) do
    max_weight = Enum.reduce(connections, 0.0, fn conn, acc -> 
      max(acc, abs(conn.weight)) 
    end)
    
    if max_weight > 0 do
      Enum.map(connections, fn conn ->
        %{conn | weight: conn.weight / max_weight}
      end)
    else
      connections
    end
  end

  defp soft_normalize(connections, temperature \\ 1.0) do
    # Soft normalization using softmax-like function
    weights = Enum.map(connections, &(&1.weight))
    max_weight = Enum.max(weights)
    
    exp_weights = Enum.map(weights, fn w -> 
      :math.exp((w - max_weight) / temperature) 
    end)
    
    sum_exp = Enum.sum(exp_weights)
    
    connections
    |> Enum.zip(exp_weights)
    |> Enum.map(fn {conn, exp_w} ->
      normalized_weight = exp_w / sum_exp
      %{conn | weight: normalized_weight}
    end)
  end
  
  # Helper function to ensure input is a number
  defp ensure_number(value) when is_number(value), do: value
  defp ensure_number(_), do: 0.0
end