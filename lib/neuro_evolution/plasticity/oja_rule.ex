defmodule NeuroEvolution.Plasticity.OjaRule do
  @moduledoc """
  Implementation of Oja's plasticity rule for normalized Hebbian learning.
  Provides weight normalization to prevent runaway growth.
  """

  @behaviour NeuroEvolution.Plasticity.PlasticityRule

  @impl true
  def update_weight(%NeuroEvolution.TWEANN.Connection{} = connection, pre_activity, post_activity, params \\ %{}, _context \\ %{}) do
    learning_rate = Map.get(params, :learning_rate, 0.01)
    
    # Oja's rule: Δw = η * post * (pre - post * weight)
    # This normalizes weights and prevents unlimited growth
    weight_change = learning_rate * post_activity * (pre_activity - post_activity * connection.weight)
    
    new_weight = connection.weight + weight_change
    %{connection | weight: new_weight}
  end

  @impl true
  def normalize_weights(weights) do
    # Ensure weight vector has unit norm
    norm = :math.sqrt(Enum.reduce(weights, 0.0, fn w, acc -> acc + w * w end))
    
    if norm > 0.0 do
      Enum.map(weights, fn w -> w / norm end)
    else
      weights
    end
  end
end