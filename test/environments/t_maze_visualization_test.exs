defmodule NeuroEvolution.Environments.TMazeVisualizationTest do
  use ExUnit.Case
  
  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Environments.TMaze
  alias NeuroEvolution.Environments.TMazeVisualization

  describe "T-Maze visualization" do
    test "renders maze with agent and reward positions" do
      # Test rendering the maze with different agent positions
      start_maze = TMazeVisualization.render_maze(:start, :left)
      corridor_maze = TMazeVisualization.render_maze(:corridor, :left)
      junction_maze = TMazeVisualization.render_maze(:junction, :left)
      left_maze = TMazeVisualization.render_maze(:left, :left)
      right_maze = TMazeVisualization.render_maze(:right, :right)
      
      # Verify the rendered mazes are strings
      assert is_binary(start_maze)
      assert is_binary(corridor_maze)
      assert is_binary(junction_maze)
      assert is_binary(left_maze)
      assert is_binary(right_maze)
      
      # Verify the agent position is correctly rendered
      assert String.contains?(start_maze, "A")
      assert String.contains?(corridor_maze, "A")
      assert String.contains?(junction_maze, "A")
      assert String.contains?(left_maze, "A")
      assert String.contains?(right_maze, "A")
      
      # Verify the reward position is correctly rendered
      assert String.contains?(left_maze, "R")
      assert String.contains?(right_maze, "R")
    end
    
    test "generates trial report" do
      # Create a test genome
      genome = create_test_genome()
      
      # Generate a trial report
      report = TMazeVisualization.trial_report(genome, :left, :left, true)
      
      # Verify the report contains expected information
      assert String.contains?(report, "Genome ID")
      assert String.contains?(report, "Reward Location: left")
      assert String.contains?(report, "Agent's Choice: left")
      assert String.contains?(report, "Success: true")
    end
    
    test "generates summary report" do
      # Create test metrics
      metrics = %{
        overall_success_rate: 0.75,
        left_success_rate: 0.8,
        right_success_rate: 0.7,
        total_score: 15,
        num_trials: 20
      }
      
      # Generate a summary report
      report = TMazeVisualization.summary_report(metrics)
      
      # Verify the report contains expected information
      assert String.contains?(report, "Total Trials: 20")
      assert String.contains?(report, "Overall Success Rate: 75.0%")
      assert String.contains?(report, "Left Reward Success Rate: 80.0%")
      assert String.contains?(report, "Right Reward Success Rate: 70.0%")
      assert String.contains?(report, "Total Score: 15 / 20")
    end
    
    test "generates learning progress report" do
      # Create test generation statistics
      generations = [
        %{best_fitness: 2.0, avg_fitness: 1.0},
        %{best_fitness: 3.0, avg_fitness: 1.5},
        %{best_fitness: 4.0, avg_fitness: 2.0},
        %{best_fitness: 5.0, avg_fitness: 2.5},
        %{best_fitness: 6.0, avg_fitness: 3.0}
      ]
      
      # Generate a learning progress report
      report = TMazeVisualization.learning_progress_report(generations)
      
      # Verify the report contains expected information
      assert String.contains?(report, "Generations: 5")
      assert String.contains?(report, "Initial Best Fitness: 2.0")
      assert String.contains?(report, "Final Best Fitness: 6.0")
      assert String.contains?(report, "Improvement: 4.0")
    end
  end
  
  # Helper function to create a test genome
  defp create_test_genome do
    Genome.new(3, 2, plasticity: %{plasticity_type: :hebbian, learning_rate: 0.1})
  end
end
