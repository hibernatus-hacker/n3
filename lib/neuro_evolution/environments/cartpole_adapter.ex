defmodule NeuroEvolution.Environments.CartPoleAdapter do
  @moduledoc """
  An adapter for the CartPole environment that provides a consistent interface
  for both neuroevolution and Q-learning approaches.
  
  This module wraps the CartPole environment and provides methods for:
  1. Resetting the environment
  2. Taking steps in the environment
  3. Rendering the environment
  4. Converting between different action formats
  """
  
  @doc """
  Initializes the CartPole environment.
  
  ## Returns
  - :ok if successful
  """
  def init do
    # Ensure the Python bridge is started
    case Process.whereis(NeuroEvolution.Environments.PythonBridge) do
      nil ->
        {:ok, _pid} = NeuroEvolution.Environments.PythonBridge.start_link()
      _pid ->
        :ok
    end
    
    # Initialize the CartPole environment
    {:ok, _} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :create_env, [])
    :ok
  end
  
  @doc """
  Resets the CartPole environment.
  
  ## Returns
  - {:ok, state} tuple with the initial state
  """
  def reset do
    # Reset the environment
    {:ok, observation} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :reset_env, [])
    
    # Normalize the observation
    normalized_observation = normalize_observation(observation)
    
    {:ok, normalized_observation}
  end
  
  @doc """
  Takes a step in the CartPole environment.
  
  ## Parameters
  - action: The action to take (0 for left, 1 for right)
  
  ## Returns
  - {:ok, {next_state, reward, done}} tuple with the next state, reward, and done flag
  """
  def step(action) do
    # Ensure action is an integer (0 or 1)
    action_int = if is_float(action), do: round(action), else: action
    
    # Take a step in the environment
    {:ok, {new_observation, reward, done}} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :step_env, [action_int])
    
    # Normalize the observation
    normalized_observation = normalize_observation(new_observation)
    
    {:ok, {normalized_observation, reward, done}}
  end
  
  @doc """
  Renders the CartPole environment.
  
  ## Returns
  - :ok if successful
  """
  def render do
    {:ok, _} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :render_env, [])
    :ok
  end
  
  @doc """
  Closes the CartPole environment.
  
  ## Returns
  - :ok if successful
  """
  def close do
    {:ok, _} = NeuroEvolution.Environments.PythonBridge.call(:simple_cartpole, :close_env, [])
    :ok
  end
  
  @doc """
  Normalizes the observation for better neural network performance.
  
  ## Parameters
  - observation: The raw observation from the environment
  
  ## Returns
  - Normalized observation
  """
  def normalize_observation([cart_pos, cart_vel, pole_angle, pole_vel]) do
    # Typical ranges for CartPole values
    # Position: [-2.4, 2.4], Velocity: [-inf, inf] but typically [-10, 10]
    # Angle: [-0.21, 0.21] radians, Angular velocity: [-inf, inf] but typically [-10, 10]
    
    [
      cart_pos / 2.4,  # Normalize position
      :math.tanh(cart_vel / 5.0),  # Normalize velocity using tanh
      pole_angle / 0.21,  # Normalize angle
      :math.tanh(pole_vel / 5.0)  # Normalize angular velocity using tanh
    ]
  end
  
  @doc """
  Evaluates a genome on the CartPole task using the Q-Network approach.
  
  ## Parameters
  - genome: The genome to evaluate
  - num_episodes: Number of episodes to evaluate for
  - max_steps: Maximum number of steps per episode
  - opts: Options for evaluation
    - render: Whether to render the environment
    - use_q_learning: Whether to use Q-learning updates
    - exploration_rate: Initial exploration rate
  
  ## Returns
  - Average reward across all episodes
  """
  def evaluate_with_q_learning(genome, num_episodes \\ 1, max_steps \\ 500, opts \\ []) do
    # Default options
    opts = Keyword.merge([
      render: false,
      use_q_learning: true,
      exploration_rate: 0.1
    ], opts)
    
    # Initialize the environment
    init()
    
    # Create a Q-network from the genome
    q_network = NeuroEvolution.RL.QNetwork.new(genome, [
      exploration_rate: opts[:exploration_rate],
      experience_replay: opts[:use_q_learning]
    ])
    
    # Evaluate the Q-network
    {stats, _} = NeuroEvolution.RL.QNetwork.evaluate(
      q_network,
      num_episodes,
      max_steps,
      __MODULE__,
      opts[:render]
    )
    
    # Calculate average reward
    avg_reward = Enum.sum(stats.episode_rewards) / length(stats.episode_rewards)
    
    avg_reward
  end
  
  @doc """
  Trains a genome on the CartPole task using the Q-Network approach.
  
  ## Parameters
  - genome: The genome to train
  - num_episodes: Number of episodes to train for
  - max_steps: Maximum number of steps per episode
  - opts: Options for training
    - learning_rate: Learning rate for Q-learning updates
    - discount_factor: Discount factor for future rewards
    - exploration_rate: Initial exploration rate
    - exploration_decay: Decay rate for exploration
    - experience_replay: Whether to use experience replay
  
  ## Returns
  - {trained_genome, training_stats} tuple with the trained genome and training statistics
  """
  def train_with_q_learning(genome, num_episodes \\ 500, max_steps \\ 500, opts \\ []) do
    # Default options
    opts = Keyword.merge([
      learning_rate: 0.1,
      discount_factor: 0.99,
      exploration_rate: 1.0,
      exploration_decay: 0.995,
      experience_replay: true
    ], opts)
    
    # Initialize the environment
    init()
    
    # Create a Q-network from the genome
    q_network = NeuroEvolution.RL.QNetwork.new(genome, [
      learning_rate: opts[:learning_rate],
      discount_factor: opts[:discount_factor],
      exploration_rate: opts[:exploration_rate],
      exploration_decay: opts[:exploration_decay],
      experience_replay: opts[:experience_replay]
    ])
    
    # Train the Q-network
    {trained_q_network, training_stats} = NeuroEvolution.RL.QNetwork.train(
      q_network,
      num_episodes,
      max_steps,
      __MODULE__
    )
    
    # Return the trained genome and training statistics
    {trained_q_network.genome, training_stats}
  end
  
  @doc """
  Creates a hybrid training approach that combines neuroevolution with Q-learning.
  
  ## Parameters
  - population: The initial population
  - num_generations: Number of generations for neuroevolution
  - num_episodes: Number of episodes for Q-learning per genome
  - fitness_fn: The fitness function for neuroevolution
  - opts: Options for hybrid training
    - q_learning_frequency: How often to apply Q-learning (default: every generation)
    - q_learning_episodes: Number of episodes for Q-learning per genome (default: 50)
    - elitism: Number of elite genomes to preserve (default: 2)
    - reporter: Function to report progress
  
  ## Returns
  - The evolved population
  """
  def hybrid_train(population, num_generations, fitness_fn, opts \\ []) do
    # Default options
    opts = Keyword.merge([
      q_learning_frequency: 1,
      q_learning_episodes: 50,
      elitism: 2,
      reporter: fn pop, gen -> 
        IO.puts("Generation #{gen}: Best=#{pop.best_fitness}, Avg=#{pop.avg_fitness}")
      end
    ], opts)
    
    # Initialize the environment
    init()
    
    # Define a hybrid fitness function that incorporates Q-learning
    hybrid_fitness_fn = fn genome ->
      # Apply Q-learning if it's time
      genome = if rem(population.generation, opts[:q_learning_frequency]) == 0 do
        # Train with Q-learning
        {trained_genome, _} = train_with_q_learning(
          genome,
          opts[:q_learning_episodes],
          500,
          [exploration_rate: 0.3]
        )
        
        trained_genome
      else
        genome
      end
      
      # Evaluate with the original fitness function
      fitness_fn.(genome)
    end
    
    # Evolve the population with the hybrid fitness function
    NeuroEvolution.evolve(population, hybrid_fitness_fn, 
      generations: num_generations,
      reporter: opts[:reporter]
    )
  end
end
