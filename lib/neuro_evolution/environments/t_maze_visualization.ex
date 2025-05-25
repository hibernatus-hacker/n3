defmodule NeuroEvolution.Environments.TMazeVisualization do
  @moduledoc """
  Visualization utilities for the T-Maze environment.
  
  This module provides functions to visualize the T-Maze environment, agent behavior,
  and learning progress.
  """
  
  @doc """
  Generates an ASCII representation of the T-Maze with the agent's position and reward location.
  
  ## Parameters
  - position: The agent's current position (:start, :corridor, :junction, :left, :right)
  - reward_location: The location of the reward (:left or :right)
  
  ## Returns
  - A string containing the ASCII representation of the maze
  """
  def render_maze(position, reward_location) do
    # Define the maze layout
    maze = [
      "  +---+---+  ",
      "  |       |  ",
      "  L       R  ",
      "  +   +---+  ",
      "  |   |      ",
      "  |   |      ",
      "  |   |      ",
      "  +---+      ",
      "              "
    ]
    
    # Replace characters based on position and reward
    maze = 
      case position do
        :start -> replace_at(maze, 6, 2, "A")
        :corridor -> replace_at(maze, 4, 2, "A")
        :junction -> replace_at(maze, 2, 2, "A")
        :left -> replace_at(maze, 2, 1, "A") # Fixed position for left arm
        :right -> replace_at(maze, 2, 9, "A") # Fixed position for right arm
        _ -> maze
      end
    
    # Mark the reward location
    maze = 
      case reward_location do
        :left -> replace_at(maze, 2, 1, "R") # Fixed position for left reward
        :right -> replace_at(maze, 2, 9, "R") # Fixed position for right reward
        _ -> maze
      end
    
    # Join the maze rows into a single string
    Enum.join(maze, "\n")
  end
  
  @doc """
  Generates a text report of a T-Maze trial.
  
  ## Parameters
  - genome: The genome being evaluated
  - reward_location: The location of the reward (:left or :right)
  - choice: The agent's final choice (:left or :right)
  - success: Whether the agent found the reward
  
  ## Returns
  - A string containing the trial report
  """
  def trial_report(genome, reward_location, choice, success) do
    """
    T-Maze Trial Report
    ==================
    Genome ID: #{genome.id}
    Reward Location: #{reward_location}
    Agent's Choice: #{choice}
    Success: #{success}
    
    #{render_maze(choice, reward_location)}
    """
  end
  
  @doc """
  Generates a summary report of multiple T-Maze trials.
  
  ## Parameters
  - metrics: A map containing evaluation metrics
  
  ## Returns
  - A string containing the summary report
  """
  def summary_report(metrics) do
    """
    T-Maze Evaluation Summary
    ========================
    Total Trials: #{metrics.num_trials}
    Overall Success Rate: #{Float.round(metrics.overall_success_rate * 100, 2)}%
    Left Reward Success Rate: #{Float.round(metrics.left_success_rate * 100, 2)}%
    Right Reward Success Rate: #{Float.round(metrics.right_success_rate * 100, 2)}%
    Total Score: #{metrics.total_score} / #{metrics.num_trials}
    """
  end
  
  @doc """
  Generates a learning progress report for T-Maze evolution.
  
  ## Parameters
  - generations: A list of generation statistics
  
  ## Returns
  - A string containing the learning progress report
  """
  def learning_progress_report(generations) do
    # Extract generation numbers and best fitness values
    {gen_nums, best_fitnesses} = Enum.map_reduce(generations, 1, fn gen, acc ->
      {{acc, gen.best_fitness}, acc + 1}
    end) |> elem(0) |> Enum.unzip()
    
    # Generate the report
    """
    T-Maze Learning Progress
    =======================
    Generations: #{length(generations)}
    Initial Best Fitness: #{Float.round(List.first(best_fitnesses), 2)}
    Final Best Fitness: #{Float.round(List.last(best_fitnesses), 2)}
    Improvement: #{Float.round(List.last(best_fitnesses) - List.first(best_fitnesses), 2)}
    
    Generation Progress:
    #{generation_progress_chart(gen_nums, best_fitnesses)}
    """
  end
  
  # Helper function to replace a character in a specific position in the maze
  defp replace_at(maze, row, col, char) do
    List.update_at(maze, row, fn line ->
      String.graphemes(line)
      |> List.update_at(col, fn _ -> char end)
      |> Enum.join("")
    end)
  end
  
  # Helper function to generate a simple ASCII chart of learning progress
  defp generation_progress_chart(generations, fitnesses) do
    max_fitness = Enum.max(fitnesses)
    chart_height = 10
    
    # Generate the chart rows
    rows = for row <- 1..chart_height do
      threshold = max_fitness * (chart_height - row + 1) / chart_height
      
      # Create a row with markers for fitness values that exceed the threshold
      row_chars = Enum.map(fitnesses, fn fitness ->
        if fitness >= threshold, do: "#", else: " "
      end)
      
      # Add y-axis label
      y_label = Float.round(threshold, 2) |> Float.to_string() |> String.pad_leading(5)
      "#{y_label} |#{Enum.join(row_chars, "")}"
    end
    
    # Add x-axis
    x_axis = "      " <> String.duplicate("-", length(generations))
    
    # Add x-axis labels
    x_labels = "      " <> Enum.map_join(generations, fn _ -> " " end)
    
    # Combine all parts
    Enum.join(rows ++ [x_axis, x_labels], "\n")
  end
end
