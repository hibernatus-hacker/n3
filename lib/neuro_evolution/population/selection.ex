defmodule NeuroEvolution.Population.Selection do
  @moduledoc """
  Selection strategies for evolutionary algorithms.
  """

  def tournament_select(population, tournament_size \\ 3) do
    tournament_size = min(tournament_size, length(population))
    
    if tournament_size > 0 do
      tournament = Enum.take_random(population, tournament_size)
      Enum.max_by(tournament, &(&1.fitness || 0.0))
    else
      Enum.random(population)
    end
  end

  def roulette_select(population) do
    total_fitness = 
      population
      |> Enum.map(&(&1.fitness || 0.0))
      |> Enum.sum()
    
    if total_fitness > 0 do
      selection_point = :rand.uniform() * total_fitness
      
      {selected, _} = 
        Enum.reduce_while(population, {nil, 0.0}, fn genome, {_, cumulative} ->
          new_cumulative = cumulative + (genome.fitness || 0.0)
          if new_cumulative >= selection_point do
            {:halt, {genome, new_cumulative}}
          else
            {:cont, {nil, new_cumulative}}
          end
        end)
      
      selected || List.first(population)
    else
      Enum.random(population)
    end
  end

  def rank_select(population) do
    ranked_population = 
      population
      |> Enum.with_index()
      |> Enum.sort_by(fn {genome, _} -> genome.fitness || 0.0 end, :desc)
    
    # Linear ranking: best gets rank n, worst gets rank 1
    total_rank = length(population) * (length(population) + 1) / 2
    selection_point = :rand.uniform() * total_rank
    
    {selected, _} = 
      Enum.reduce_while(ranked_population, {nil, 0}, fn {{genome, _}, rank}, {_, cumulative} ->
        new_cumulative = cumulative + (length(population) - rank)
        if new_cumulative >= selection_point do
          {:halt, {genome, new_cumulative}}
        else
          {:cont, {nil, new_cumulative}}
        end
      end)
    
    selected || List.first(population)
  end

  def elite_select(population) do
    Enum.max_by(population, &(&1.fitness || 0.0))
  end
end