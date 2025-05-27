defmodule NeuroEvolution.Environments.TMazeVisualization do
  @moduledoc """
  Visualization utilities for the T-Maze environment.
  
  This module provides functions to visualize the T-Maze environment, including
  rendering the maze with agent and reward positions, and generating various
  reports for analysis.
  """
  
  # Uncomment if needed
  # alias NeuroEvolution.TWEANN.Genome
  
  @doc """
  Renders the T-Maze with the agent at the specified position and reward at the specified location.
  
  ## Parameters
  - agent_position: The position of the agent (:start, :corridor, :junction, :left, :right)
  - reward_location: The location of the reward (:left or :right)
  
  ## Returns
  - A string representation of the maze
  """
  def render_maze(agent_position, reward_location) do
    # Create the base maze structure
    maze = [
      "┌───────────────┐",
      "│               │",
      "│               │",
      "│               │",
      "├───────┬───────┤",
      "│       │       │",
      "│       │       │",
      "│       │       │",
      "└───────┴───────┘"
    ]
    
    # Place the agent based on position
    maze = place_agent(maze, agent_position)
    
    # Place the reward based on location
    maze = place_reward(maze, reward_location)
    
    # Convert the maze to a string
    Enum.join(maze, "\n")
  end
  
  @doc """
  Generates a report for a single trial.
  
  ## Parameters
  - genome: The genome being evaluated
  - reward_location: The location of the reward (:left or :right)
  - agent_choice: The agent's choice (:left or :right)
  - success: Whether the agent successfully found the reward
  
  ## Returns
  - A formatted string report
  """
  def trial_report(genome, reward_location, agent_choice, success) do
    """
    T-Maze Trial Report
    ==================
    
    Genome ID: #{genome.id}
    Reward Location: #{reward_location}
    Agent's Choice: #{agent_choice}
    Success: #{success}
    
    #{render_maze(agent_choice, reward_location)}
    """
  end
  
  @doc """
  Generates a summary report for multiple trials.
  
  ## Parameters
  - metrics: A map containing metrics about the trials
  
  ## Returns
  - A formatted string report
  """
  def summary_report(metrics) do
    # Calculate percentages
    overall_percentage = metrics.overall_success_rate * 100
    left_percentage = metrics.left_success_rate * 100
    right_percentage = metrics.right_success_rate * 100
    
    """
    T-Maze Summary Report
    ====================
    
    Total Trials: #{metrics.num_trials}
    Overall Success Rate: #{:io_lib.format("~.1f", [overall_percentage])}%
    Left Reward Success Rate: #{:io_lib.format("~.1f", [left_percentage])}%
    Right Reward Success Rate: #{:io_lib.format("~.1f", [right_percentage])}%
    Total Score: #{metrics.total_score} / #{metrics.num_trials}
    """
  end
  
  @doc """
  Generates a learning progress report across generations.
  
  ## Parameters
  - generations: A list of generation statistics
  
  ## Returns
  - A formatted string report
  """
  def learning_progress_report(generations) do
    # Extract best and average fitness for each generation
    generation_data = Enum.with_index(generations, 1)
    |> Enum.map(fn {gen_stats, gen_num} ->
      "Generation #{gen_num}: Best Fitness = #{gen_stats.best_fitness}, Average Fitness = #{gen_stats.avg_fitness}"
    end)
    |> Enum.join("\n")
    
    # Calculate additional statistics
    num_generations = length(generations)
    initial_best_fitness = List.first(generations).best_fitness
    final_best_fitness = List.last(generations).best_fitness
    improvement = final_best_fitness - initial_best_fitness
    
    """
    T-Maze Learning Progress Report
    =============================
    
    Generations: #{num_generations}
    Initial Best Fitness: #{initial_best_fitness}
    Final Best Fitness: #{final_best_fitness}
    Improvement: #{improvement}
    
    Generation Details:
    #{generation_data}
    """
  end
  
  # Private helper functions
  
  defp place_agent(maze, agent_position) do
    case agent_position do
      :start ->
        replace_at_position(maze, 1, 9, "A")
      :corridor ->
        replace_at_position(maze, 2, 9, "A")
      :junction ->
        replace_at_position(maze, 3, 9, "A")
      :left ->
        replace_at_position(maze, 6, 3, "A")
      :right ->
        replace_at_position(maze, 6, 15, "A")
    end
  end
  
  defp place_reward(maze, reward_location) do
    case reward_location do
      :left ->
        replace_at_position(maze, 6, 1, "R")
      :right ->
        replace_at_position(maze, 6, 17, "R")
    end
  end
  
  defp replace_at_position(maze, row, col, char) do
    row_string = Enum.at(maze, row)
    
    # Replace the character at the specified column
    {prefix, suffix} = String.split_at(row_string, col)
    {_, suffix} = String.split_at(suffix, 1)
    new_row = prefix <> char <> suffix
    
    # Replace the row in the maze
    List.replace_at(maze, row, new_row)
  end
end
