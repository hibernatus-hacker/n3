defmodule NeuroEvolution.Plasticity.BCMRule do
  @moduledoc """
  Implementation of the Bienenstock-Cooper-Munro (BCM) plasticity rule.
  BCM learning provides homeostatic regulation of synaptic weights.
  """

  @behaviour NeuroEvolution.Plasticity.PlasticityRule

  @impl true
  def update_weight(%NeuroEvolution.TWEANN.Connection{} = connection, pre_activity, post_activity, params \\ %{}, context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    threshold = Map.get(params, :threshold, 1.0)
    
    # BCM rule: Δw = η * pre * post * (post - θ)
    # where θ is the modification threshold
    weight_change = learning_rate * pre_activity * post_activity * (post_activity - threshold)
    
    new_weight = connection.weight + weight_change
    %{connection | weight: new_weight}
  end

  @impl true
  def update_threshold(current_threshold, post_activity, time_constant \\ 1000.0) do
    # Sliding threshold based on recent post-synaptic activity
    decay = 1.0 / time_constant
    current_threshold * (1.0 - decay) + post_activity * post_activity * decay
  end
end