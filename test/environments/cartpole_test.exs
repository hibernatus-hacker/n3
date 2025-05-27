defmodule NeuroEvolution.Environments.CartPoleTest do
  use ExUnit.Case
  
  alias NeuroEvolution.RL.QNetwork
  alias NeuroEvolution.Environments.CartPole
  
  # These tests may be skipped if Python bridge is not available
  @moduletag :cartpole
  
  setup do
    # Start Python bridge for tests
    case NeuroEvolution.Environments.PythonBridge.start_link() do
      {:ok, _pid} ->
        :ok
      {:error, _reason} ->
        IO.puts("⚠️ Skipping CartPole tests - Python bridge not available")
        {:skip, "Python bridge not available"}
    end
  end
  
  describe "CartPole environment" do
    test "initializes environment" do
      assert {:ok, _env} = CartPole.init()
    end
    
    test "resets environment" do
      {:ok, env} = CartPole.init()
      assert {:ok, state, _info} = CartPole.reset(env)
      assert length(state) == 4  # CartPole has 4 state variables
    end
    
    test "takes step in environment" do
      {:ok, env} = CartPole.init()
      {:ok, state, _info} = CartPole.reset(env)
      
      # Take action 0 (left)
      assert {:ok, next_state, reward, done, _info} = CartPole.step(env, 0)
      
      # State should be different after taking action
      assert next_state != state
      assert is_number(reward)
      assert is_boolean(done)
    end
    
    test "evaluates simple genome" do
      # Create a basic genome for testing
      genome = NeuroEvolution.new_genome(4, 2)
      
      # Evaluate for a few steps
      score = CartPole.evaluate(genome, max_steps: 10)
      
      # Score should be a number representing how many steps the agent survived
      assert is_number(score)
      assert score <= 10
    end
    
    @tag :slow
    test "evaluates plastic genome" do
      # Create a genome with plasticity
      plasticity_config = %{
        plasticity_type: :hebbian,
        learning_rate: 0.05
      }
      
      genome = NeuroEvolution.new_genome(4, 2, plasticity: plasticity_config)
      
      # Evaluate for a few steps
      score = CartPole.evaluate(genome, max_steps: 10)
      
      # Score should be a number
      assert is_number(score)
    end
  end
  
  describe "Q-learning integration" do
    @tag :slow
    test "creates specialized Q-network genome" do
      # Create specialized genome for Q-learning
      q_genome = QNetwork.create_cartpole_genome()
      
      # Should have correct structure
      assert length(q_genome.inputs) == 4  # CartPole has 4 state variables
      assert length(q_genome.outputs) == 2  # CartPole has 2 actions
      
      # Should have recurrent connections
      has_recurrent = Enum.any?(q_genome.connections, fn {_id, conn} ->
        conn.recurrent
      end)
      
      assert has_recurrent
    end
    
    @tag :slow
    test "Q-network can predict values" do
      # Create Q-network
      q_genome = QNetwork.create_cartpole_genome()
      q_network = QNetwork.new(q_genome)
      
      # Generate random state
      state = [0.0, 0.0, 0.0, 0.0]
      
      # Select action
      {action, _updated_q_network} = QNetwork.select_action(q_network, state)
      
      # Action should be valid for CartPole (0 or 1)
      assert action in [0, 1]
    end
    
    @tag :slow
    test "Q-network updates from experience" do
      # Create Q-network
      q_genome = QNetwork.create_cartpole_genome()
      q_network = QNetwork.new(q_genome, exploration_rate: 0.5)
      
      # Create a sample transition
      state = [0.0, 0.0, 0.0, 0.0]
      action = 0
      reward = 1.0
      next_state = [0.01, 0.01, 0.01, 0.01]
      done = false
      
      # Update Q-network
      updated_q_network = QNetwork.update(q_network, {state, action, reward, next_state, done})
      
      # Q-network should be updated
      assert updated_q_network != q_network
    end
    
    @tag :slow
    test "hybrid learning approach" do
      # Skip this test in CI environments or if it would take too long
      if System.get_env("CI") do
        IO.puts("Skipping hybrid learning test in CI environment")
      else
        # Create specialized genome for Q-learning
        q_genome = QNetwork.create_cartpole_genome()
        q_network = QNetwork.new(q_genome, 
          learning_rate: 0.1,
          exploration_rate: 1.0,
          exploration_decay: 0.9
        )
        
        # Train for a very small number of episodes
        {trained_q_network, stats} = QNetwork.train(q_network, 2, 100, CartPole)
        
        # Should have some statistics
        assert length(stats.episode_rewards) == 2
        
        # Should have reduced exploration rate
        assert trained_q_network.exploration_rate < q_network.exploration_rate
        
        # Convert to genome for evolution
        trained_genome = trained_q_network.genome
        
        # Create a small population with the trained genome
        population = %{
          genomes: %{
            0 => trained_genome,
            1 => NeuroEvolution.TWEANN.Genome.mutate(trained_genome),
            2 => NeuroEvolution.TWEANN.Genome.mutate(trained_genome)
          }
        }
        
        # Define a simple fitness function
        fitness_fn = fn genome ->
          CartPole.evaluate(genome, max_steps: 50)
        end
        
        # Evolve for just 1 generation
        evolved_population = NeuroEvolution.evolve(population, fitness_fn, generations: 1)
        
        # Should have a best genome
        best_genome = NeuroEvolution.get_best_genome(evolved_population)
        assert best_genome != nil
      end
    end
  end
end
