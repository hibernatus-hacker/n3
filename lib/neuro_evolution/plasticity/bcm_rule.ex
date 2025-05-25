defmodule NeuroEvolution.Plasticity.BCMRule do
  @moduledoc """
  Implementation of the Bienenstock-Cooper-Munro (BCM) plasticity rule.
  BCM learning provides homeostatic regulation of synaptic weights.
  """

  @behaviour NeuroEvolution.Plasticity.PlasticityRule

  @impl true
  def update_weight(weight, pre_activity, post_activity, learning_rate, params \\ %{}) do
    threshold = Map.get(params, :threshold, 1.0)
    
    # BCM rule: Δw = η * pre * post * (post - θ)
    # where θ is the modification threshold
    weight_change = learning_rate * pre_activity * post_activity * (post_activity - threshold)
    
    weight + weight_change
  end

  @impl true
  def update_threshold(current_threshold, post_activity, time_constant \\ 1000.0) do
    # Sliding threshold based on recent post-synaptic activity
    decay = 1.0 / time_constant
    current_threshold * (1.0 - decay) + post_activity * post_activity * decay
  end
end