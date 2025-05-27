defmodule NeuroEvolution.Plasticity.PlasticityRule do
  @moduledoc """
  Behaviour defining the interface for neural plasticity rules.
  
  Plasticity rules modify neural network weights based on activity patterns.
  Implementations should provide the required callbacks.
  """
  
  @doc """
  Updates a weight based on pre and post-synaptic activity.
  
  ## Parameters
  - weight: The current weight value
  - pre_activity: Activity of the pre-synaptic neuron
  - post_activity: Activity of the post-synaptic neuron
  - learning_rate: Rate of weight change
  - params: Additional parameters specific to the rule
  
  ## Returns
  The updated weight value
  """
  @callback update_weight(
    weight :: float(),
    pre_activity :: float(),
    post_activity :: float(),
    learning_rate :: float(),
    params :: map()
  ) :: float()
  
  @doc """
  Updates a threshold value based on neural activity.
  
  ## Parameters
  - current_threshold: The current threshold value
  - post_activity: Activity of the post-synaptic neuron
  - time_constant: Time constant for threshold adaptation
  
  ## Returns
  The updated threshold value
  """
  @callback update_threshold(
    current_threshold :: float(),
    post_activity :: float(),
    time_constant :: float()
  ) :: float()
  
  @doc """
  Normalizes weights according to the plasticity rule.
  
  ## Parameters
  - weights: List of weights to normalize
  
  ## Returns
  The normalized weights
  """
  @callback normalize_weights(weights :: list(float())) :: list(float())
  
  @optional_callbacks [update_threshold: 3, normalize_weights: 1]
end
