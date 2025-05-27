defmodule NeuroEvolution.Plasticity.STDPRule do
  @moduledoc """
  Spike-Timing Dependent Plasticity (STDP) implementation.
  Synaptic strength depends on the relative timing of pre and post-synaptic spikes.
  """

  alias NeuroEvolution.TWEANN.Connection

  defstruct [
    :a_plus,      # LTP amplitude
    :a_minus,     # LTD amplitude  
    :tau_plus,    # LTP time constant
    :tau_minus,   # LTD time constant
    :w_max,       # Maximum weight
    :w_min,       # Minimum weight
    :spike_threshold
  ]

  @type t :: %__MODULE__{
    a_plus: float(),
    a_minus: float(),
    tau_plus: float(),
    tau_minus: float(),
    w_max: float(),
    w_min: float(),
    spike_threshold: float()
  }

  def new(opts \\ []) do
    %__MODULE__{
      a_plus: Keyword.get(opts, :a_plus, 0.1),
      a_minus: Keyword.get(opts, :a_minus, 0.12),
      tau_plus: Keyword.get(opts, :tau_plus, 20.0),
      tau_minus: Keyword.get(opts, :tau_minus, 20.0),
      w_max: Keyword.get(opts, :w_max, 5.0),
      w_min: Keyword.get(opts, :w_min, -5.0),
      spike_threshold: Keyword.get(opts, :spike_threshold, 0.5)
    }
  end

  def update_weight(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    stdp_params = merge_default_params(params)
    
    # Convert activities to spike times if needed
    {pre_spike_times, post_spike_times} = extract_spike_times(pre_activity, post_activity, stdp_params, context)
    
    # Calculate STDP weight change
    delta_w = calculate_stdp_update(pre_spike_times, post_spike_times, stdp_params, connection.weight)
    
    # Apply weight change
    new_weight = connection.weight + delta_w
    clamped_weight = clamp_weight(new_weight, stdp_params.w_min, stdp_params.w_max)
    
    %{connection | weight: clamped_weight}
  end

  def additive_stdp(%Connection{} = connection, pre_spikes, post_spikes, params \\ %{}) do
    stdp_params = merge_default_params(params)
    
    delta_w = Enum.reduce(pre_spikes, 0.0, fn t_pre, acc ->
      ltp_contribution = Enum.reduce(post_spikes, 0.0, fn t_post, ltp_acc ->
        if t_post > t_pre do
          dt = t_post - t_pre
          ltp_acc + stdp_params.a_plus * :math.exp(-dt / stdp_params.tau_plus)
        else
          ltp_acc
        end
      end)
      
      ltd_contribution = Enum.reduce(post_spikes, 0.0, fn t_post, ltd_acc ->
        if t_pre > t_post do
          dt = t_pre - t_post
          ltd_acc - stdp_params.a_minus * :math.exp(-dt / stdp_params.tau_minus)
        else
          ltd_acc
        end
      end)
      
      acc + ltp_contribution + ltd_contribution
    end)
    
    new_weight = connection.weight + delta_w
    clamped_weight = clamp_weight(new_weight, stdp_params.w_min, stdp_params.w_max)
    
    %{connection | weight: clamped_weight}
  end

  def multiplicative_stdp(%Connection{} = connection, pre_spikes, post_spikes, params \\ %{}) do
    stdp_params = merge_default_params(params)
    current_weight = connection.weight
    
    delta_w = Enum.reduce(pre_spikes, 0.0, fn t_pre, acc ->
      ltp_contribution = Enum.reduce(post_spikes, 0.0, fn t_post, ltp_acc ->
        if t_post > t_pre do
          dt = t_post - t_pre
          # Multiplicative LTP: depends on current weight
          weight_dependence = stdp_params.w_max - current_weight
          ltp_acc + stdp_params.a_plus * weight_dependence * :math.exp(-dt / stdp_params.tau_plus)
        else
          ltp_acc
        end
      end)
      
      ltd_contribution = Enum.reduce(post_spikes, 0.0, fn t_post, ltd_acc ->
        if t_pre > t_post do
          dt = t_pre - t_post
          # Multiplicative LTD: depends on current weight
          weight_dependence = current_weight - stdp_params.w_min
          ltd_acc - stdp_params.a_minus * weight_dependence * :math.exp(-dt / stdp_params.tau_minus)
        else
          ltd_acc
        end
      end)
      
      acc + ltp_contribution + ltd_contribution
    end)
    
    new_weight = current_weight + delta_w
    clamped_weight = clamp_weight(new_weight, stdp_params.w_min, stdp_params.w_max)
    
    %{connection | weight: clamped_weight}
  end

  def triplet_stdp(%Connection{} = connection, pre_spikes, post_spikes, params \\ %{}, context \\ %{}) do
    stdp_params = merge_default_params(params)
    
    # Triplet STDP parameters
    a2_plus = Map.get(params, :a2_plus, 0.01)
    a2_minus = Map.get(params, :a2_minus, 0.01)
    a3_plus = Map.get(params, :a3_plus, 0.01)
    a3_minus = Map.get(params, :a3_minus, 0.01)
    tau_x = Map.get(params, :tau_x, 15.0)
    tau_y = Map.get(params, :tau_y, 30.0)
    
    # Get trace values from context
    pre_trace_1 = Map.get(context, :pre_trace_1, 0.0)
    pre_trace_2 = Map.get(context, :pre_trace_2, 0.0)
    post_trace_1 = Map.get(context, :post_trace_1, 0.0)
    post_trace_2 = Map.get(context, :post_trace_2, 0.0)
    
    delta_w = calculate_triplet_update(pre_spikes, post_spikes, 
                                     {pre_trace_1, pre_trace_2, post_trace_1, post_trace_2},
                                     {a2_plus, a2_minus, a3_plus, a3_minus, tau_x, tau_y})
    
    new_weight = connection.weight + delta_w
    clamped_weight = clamp_weight(new_weight, stdp_params.w_min, stdp_params.w_max)
    
    %{connection | weight: clamped_weight}
  end

  def voltage_based_stdp(%Connection{} = connection, pre_voltage, post_voltage, params \\ %{}, context \\ %{}) do
    # Voltage-based STDP for rate-coded neurons
    learning_rate = Map.get(params, :learning_rate, 0.01)
    voltage_threshold = Map.get(params, :voltage_threshold, 0.0)
    tau_voltage = Map.get(params, :tau_voltage, 10.0)
    
    # Get voltage traces from context
    pre_trace = Map.get(context, :pre_voltage_trace, 0.0)
    post_trace = Map.get(context, :post_voltage_trace, 0.0)
    
    # Update traces
    decay_factor = :math.exp(-1.0 / tau_voltage)
    new_pre_trace = decay_factor * pre_trace + (1.0 - decay_factor) * (pre_voltage - voltage_threshold)
    new_post_trace = decay_factor * post_trace + (1.0 - decay_factor) * (post_voltage - voltage_threshold)
    
    # STDP-like rule for voltages
    delta_w = learning_rate * new_pre_trace * new_post_trace
    
    new_weight = connection.weight + delta_w
    
    # Update context
    updated_context = Map.merge(context, %{
      pre_voltage_trace: new_pre_trace,
      post_voltage_trace: new_post_trace
    })
    
    {%{connection | weight: new_weight}, updated_context}
  end

  def calcium_based_stdp(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    # Calcium-based plasticity model
    ca_threshold_low = Map.get(params, :ca_threshold_low, 0.3)
    ca_threshold_high = Map.get(params, :ca_threshold_high, 0.7)
    learning_rate = Map.get(params, :learning_rate, 0.01)
    ca_decay = Map.get(params, :ca_decay, 0.9)
    
    # Get calcium concentration from context
    ca_concentration = Map.get(context, :calcium_concentration, 0.0)
    
    # Update calcium based on pre and post activity
    ca_influx = 0.1 * pre_activity * post_activity
    new_ca = ca_decay * ca_concentration + ca_influx
    
    # Determine plasticity direction based on calcium levels
    delta_w = cond do
      new_ca > ca_threshold_high ->
        # High calcium -> LTP
        learning_rate * (new_ca - ca_threshold_high)
      
      new_ca < ca_threshold_low ->
        # Low calcium -> LTD
        -learning_rate * (ca_threshold_low - new_ca)
      
      true ->
        # Intermediate calcium -> no change
        0.0
    end
    
    new_weight = connection.weight + delta_w
    
    updated_context = Map.put(context, :calcium_concentration, new_ca)
    
    {%{connection | weight: new_weight}, updated_context}
  end

  def homeostatic_stdp(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    stdp_params = merge_default_params(params)
    target_rate = Map.get(params, :target_firing_rate, 1.0)
    homeostatic_strength = Map.get(params, :homeostatic_strength, 0.1)
    
    # Get firing rate from context
    current_rate = Map.get(context, :post_firing_rate, post_activity)
    
    # Update firing rate estimate
    rate_decay = 0.99
    new_rate = rate_decay * current_rate + (1.0 - rate_decay) * post_activity
    
    # Homeostatic scaling factor
    scaling_factor = 1.0 + homeostatic_strength * (target_rate - new_rate) / target_rate
    
    # Apply STDP with homeostatic scaling
    {pre_spikes, post_spikes} = extract_spike_times(pre_activity, post_activity, stdp_params, context)
    base_delta_w = calculate_stdp_update(pre_spikes, post_spikes, stdp_params, connection.weight)
    
    delta_w = base_delta_w * scaling_factor
    
    new_weight = connection.weight + delta_w
    clamped_weight = clamp_weight(new_weight, stdp_params.w_min, stdp_params.w_max)
    
    updated_context = Map.put(context, :post_firing_rate, new_rate)
    
    {%{connection | weight: clamped_weight}, updated_context}
  end

  def metaplastic_stdp(%Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    stdp_params = merge_default_params(params)
    
    # Metaplasticity parameters
    meta_learning_rate = Map.get(params, :meta_learning_rate, 0.001)
    activity_threshold = Map.get(params, :activity_threshold, 1.0)
    
    # Get metaplastic state from context
    meta_variable = Map.get(context, :metaplastic_variable, 1.0)
    
    # Update metaplastic variable based on recent activity
    recent_activity = Map.get(context, :recent_post_activity, post_activity)
    new_recent_activity = 0.95 * recent_activity + 0.05 * post_activity
    
    # BCM-like metaplasticity
    new_meta_variable = meta_variable + meta_learning_rate * (new_recent_activity - activity_threshold)
    clamped_meta = max(0.1, min(10.0, new_meta_variable))
    
    # Apply STDP with metaplastic modulation
    {pre_spikes, post_spikes} = extract_spike_times(pre_activity, post_activity, stdp_params, context)
    base_delta_w = calculate_stdp_update(pre_spikes, post_spikes, stdp_params, connection.weight)
    
    # Modulate STDP by metaplastic variable
    delta_w = base_delta_w * clamped_meta
    
    new_weight = connection.weight + delta_w
    clamped_weight = clamp_weight(new_weight, stdp_params.w_min, stdp_params.w_max)
    
    updated_context = Map.merge(context, %{
      metaplastic_variable: clamped_meta,
      recent_post_activity: new_recent_activity
    })
    
    {%{connection | weight: clamped_weight}, updated_context}
  end

  # Private helper functions

  defp merge_default_params(params) do
    defaults = %{
      a_plus: 0.1,
      a_minus: 0.12,
      tau_plus: 20.0,
      tau_minus: 20.0,
      w_max: 5.0,
      w_min: -5.0,
      spike_threshold: 0.5
    }
    
    Map.merge(defaults, params)
  end

  defp extract_spike_times(pre_activity, post_activity, params, context) do
    current_time = Map.get(context, :current_time, 0.0)
    spike_threshold = params.spike_threshold
    
    # Check if explicit spike times are provided in context
    pre_spike_time = Map.get(context, :pre_spike_time)
    post_spike_time = Map.get(context, :post_spike_time)
    
    # Use explicit spike times if available, otherwise infer from activity
    pre_spikes = cond do
      pre_spike_time != nil -> [pre_spike_time]
      pre_activity > spike_threshold -> [current_time]
      true -> []
    end
    
    post_spikes = cond do
      post_spike_time != nil -> [post_spike_time]
      post_activity > spike_threshold -> [current_time]
      true -> []
    end
    
    {pre_spikes, post_spikes}
  end

  defp calculate_stdp_update(pre_spikes, post_spikes, params, _current_weight) do
    Enum.reduce(pre_spikes, 0.0, fn t_pre, acc ->
      ltp_contribution = Enum.reduce(post_spikes, 0.0, fn t_post, ltp_acc ->
        if t_post > t_pre do
          dt = t_post - t_pre
          ltp_acc + params.a_plus * :math.exp(-dt / params.tau_plus)
        else
          ltp_acc
        end
      end)
      
      ltd_contribution = Enum.reduce(post_spikes, 0.0, fn t_post, ltd_acc ->
        if t_pre > t_post do
          dt = t_pre - t_post
          ltd_acc - params.a_minus * :math.exp(-dt / params.tau_minus)
        else
          ltd_acc
        end
      end)
      
      acc + ltp_contribution + ltd_contribution
    end)
  end

  defp calculate_triplet_update(pre_spikes, post_spikes, traces, triplet_params) do
    {_pre_trace_1, _pre_trace_2, _post_trace_1, _post_trace_2} = traces
    {a2_plus, a2_minus, a3_plus, a3_minus, _tau_x, _tau_y} = triplet_params
    
    # Simplified triplet STDP calculation
    # In practice, this would involve more complex trace dynamics
    pairwise_term = calculate_pairwise_contribution(pre_spikes, post_spikes, a2_plus, a2_minus)
    triplet_term = calculate_triplet_contribution(pre_spikes, post_spikes, traces, a3_plus, a3_minus)
    
    pairwise_term + triplet_term
  end

  defp calculate_pairwise_contribution(pre_spikes, post_spikes, a2_plus, a2_minus) do
    # Standard pairwise STDP
    Enum.reduce(pre_spikes, 0.0, fn t_pre, acc ->
      contribution = Enum.reduce(post_spikes, 0.0, fn t_post, pair_acc ->
        cond do
          t_post > t_pre -> pair_acc + a2_plus
          t_pre > t_post -> pair_acc - a2_minus
          true -> pair_acc
        end
      end)
      acc + contribution
    end)
  end

  defp calculate_triplet_contribution(pre_spikes, post_spikes, _traces, a3_plus, a3_minus) do
    # Simplified triplet contribution
    # Real implementation would use exponential traces
    if length(pre_spikes) > 0 and length(post_spikes) > 1 do
      a3_plus * 0.1  # Placeholder for triplet LTP
    else
      0.0
    end + 
    if length(post_spikes) > 0 and length(pre_spikes) > 1 do
      -a3_minus * 0.1  # Placeholder for triplet LTD
    else
      0.0
    end
  end

  defp clamp_weight(weight, min_weight, max_weight) do
    min(max(weight, min_weight), max_weight)
  end
end