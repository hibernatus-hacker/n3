defmodule VizWeb.VisualizationLive.TMaze do
  use VizWeb, :live_view
  
  alias NeuroEvolution.TWEANN.Genome
  alias NeuroEvolution.Environments.TMaze
  alias NeuroEvolution.Environments.TMazeVisualization

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      # Start a timer to simulate the agent's movement
      :timer.send_interval(500, self(), :update_simulation)
    end
    
    # Create a test genome for demonstration
    genome = create_test_genome()
    
    # Initialize the T-Maze state
    maze_state = %{
      genome: genome,
      position: :start,
      reward_location: :left,
      trial_step: 0,
      max_steps: 5,
      cue_visible: true,
      success: false,
      history: [],
      trials_completed: 0,
      successful_trials: 0
    }
    
    {:ok, assign(socket, 
      maze_state: maze_state,
      maze_view: TMazeVisualization.render_maze(:start, :left),
      running: false,
      auto_run: false,
      evolution_stats: []
    )}
  end

  @impl true
  def handle_event("toggle_simulation", _, socket) do
    running = not socket.assigns.running
    
    if running do
      Process.send_after(self(), :update_simulation, 500)
    end
    
    {:noreply, assign(socket, running: running)}
  end
  
  @impl true
  def handle_event("toggle_auto_run", _, socket) do
    {:noreply, assign(socket, auto_run: not socket.assigns.auto_run)}
  end
  
  @impl true
  def handle_event("reset_simulation", _, socket) do
    # Reset the maze state
    maze_state = %{
      genome: socket.assigns.maze_state.genome,
      position: :start,
      reward_location: Enum.random([:left, :right]),
      trial_step: 0,
      max_steps: 5,
      cue_visible: true,
      success: false,
      history: [],
      trials_completed: socket.assigns.maze_state.trials_completed,
      successful_trials: socket.assigns.maze_state.successful_trials
    }
    
    maze_view = TMazeVisualization.render_maze(
      maze_state.position, 
      maze_state.reward_location
    )
    
    {:noreply, assign(socket, maze_state: maze_state, maze_view: maze_view)}
  end
  
  @impl true
  def handle_event("evolve_network", _, socket) do
    # Create a small population for evolution
    population = NeuroEvolution.new_population(10, 3, 2, 
      %{
        speciation: %{enabled: true, compatibility_threshold: 1.0},
        mutation: %{
          weight_mutation_rate: 0.8,
          weight_perturbation: 0.5,
          add_node_rate: 0.03,
          add_connection_rate: 0.05
        },
        plasticity: %{
          enabled: true,
          plasticity_type: :hebbian,
          learning_rate: 0.1,
          modulation_enabled: true
        }
      }
    )
    
    # T-Maze fitness function
    fitness_fn = fn genome ->
      TMaze.evaluate(genome, 10)  # 10 trials
    end
    
    # Evolve for a few generations
    {evolved_pop, stats} = evolve_with_stats(population, fitness_fn, 5)
    
    # Extract the best genome
    best_genome = NeuroEvolution.get_best_genome(evolved_pop)
    
    # Reset the maze state with the new genome
    maze_state = %{
      genome: best_genome,
      position: :start,
      reward_location: Enum.random([:left, :right]),
      trial_step: 0,
      max_steps: 5,
      cue_visible: true,
      success: false,
      history: [],
      trials_completed: 0,
      successful_trials: 0
    }
    
    maze_view = TMazeVisualization.render_maze(
      maze_state.position, 
      maze_state.reward_location
    )
    
    {:noreply, assign(socket, 
      maze_state: maze_state, 
      maze_view: maze_view,
      evolution_stats: stats
    )}
  end
  
  @impl true
  def handle_info(:update_simulation, socket) do
    if socket.assigns.running do
      # Update the maze state
      {maze_state, maze_view} = update_maze_state(socket.assigns.maze_state)
      
      # Schedule the next update if auto-run is enabled or if we're still in the same trial
      if socket.assigns.auto_run or maze_state.trial_step < maze_state.max_steps do
        Process.send_after(self(), :update_simulation, 500)
      end
      
      {:noreply, assign(socket, maze_state: maze_state, maze_view: maze_view)}
    else
      {:noreply, socket}
    end
  end
  
  # Helper function to update the maze state based on the current state
  defp update_maze_state(state) do
    # If we've reached the end of a trial, reset for a new trial
    if state.trial_step >= state.max_steps do
      # Determine if the trial was successful
      success = state.position == state.reward_location
      
      # Update the trial statistics
      new_state = %{
        state |
        position: :start,
        reward_location: Enum.random([:left, :right]),
        trial_step: 0,
        cue_visible: true,
        success: success,
        history: state.history ++ [{state.reward_location, state.position, success}],
        trials_completed: state.trials_completed + 1,
        successful_trials: state.successful_trials + (if success, do: 1, else: 0)
      }
      
      # Render the new maze state
      maze_view = TMazeVisualization.render_maze(
        new_state.position, 
        new_state.reward_location
      )
      
      {new_state, maze_view}
    else
      # Get the current inputs based on the state
      inputs = get_inputs(state)
      
      # Create a simple evaluator for the genome
      evaluator = NeuroEvolution.Evaluator.BatchEvaluator.new(device: :cpu)
      
      # Get the outputs from the genome
      {outputs, updated_genome} = TMaze.evaluate_step(state.genome, evaluator, inputs)
      
      # Determine the next position based on the outputs
      next_position = determine_next_position(state.position, outputs)
      
      # Update the state
      new_state = %{
        state |
        genome: updated_genome,
        position: next_position,
        trial_step: state.trial_step + 1,
        cue_visible: state.trial_step == 0  # Only show the cue on the first step
      }
      
      # Render the new maze state
      maze_view = TMazeVisualization.render_maze(
        new_state.position, 
        new_state.reward_location
      )
      
      {new_state, maze_view}
    end
  end
  
  # Helper function to get the inputs for the neural network based on the current state
  defp get_inputs(state) do
    cue_left = if state.cue_visible and state.reward_location == :left, do: 1.0, else: 0.0
    cue_right = if state.cue_visible and state.reward_location == :right, do: 1.0, else: 0.0
    
    # Add a bias input
    [1.0, cue_left, cue_right]
  end
  
  # Helper function to determine the next position based on the current position and outputs
  defp determine_next_position(current_position, outputs) do
    [left_output, right_output] = outputs
    
    case current_position do
      :start -> :corridor
      :corridor -> :junction
      :junction -> 
        if left_output > right_output, do: :left, else: :right
      _ -> current_position  # Stay in place if we're already at the end
    end
  end
  
  # Helper function to create a test genome
  defp create_test_genome do
    Genome.new(3, 2, plasticity: %{plasticity_type: :hebbian, learning_rate: 0.1})
  end
  
  # Helper function to evolve a population and collect statistics
  defp evolve_with_stats(population, fitness_fn, generations) do
    Enum.reduce(1..generations, {population, []}, fn gen, {pop, stats} ->
      # Evolve the population
      evolved = NeuroEvolution.evolve(pop, fitness_fn)
      
      # Collect statistics
      gen_stats = %{
        generation: gen,
        best_fitness: evolved.best_fitness,
        avg_fitness: evolved.avg_fitness,
        species_count: length(evolved.species)
      }
      
      {evolved, stats ++ [gen_stats]}
    end)
  end
  
  @impl true
  def render(assigns) do
    ~H"""
    <div class="container mx-auto px-4 py-8">
      <h1 class="text-3xl font-bold mb-6">T-Maze Visualization</h1>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-xl font-semibold mb-4">Maze Environment</h2>
          
          <div class="mb-6 font-mono whitespace-pre bg-gray-100 p-4 rounded-md text-center">
            <%= @maze_view %>
          </div>
          
          <div class="flex space-x-4 mb-4">
            <button phx-click="toggle_simulation" class="px-4 py-2 bg-blue-500 text-white rounded-md">
              <%= if @running, do: "Pause", else: "Start" %>
            </button>
            <button phx-click="reset_simulation" class="px-4 py-2 bg-gray-500 text-white rounded-md">
              Reset
            </button>
            <button phx-click="evolve_network" class="px-4 py-2 bg-green-500 text-white rounded-md">
              Evolve Network
            </button>
          </div>
          
          <div class="flex items-center mb-4">
            <input 
              type="checkbox" 
              id="auto_run" 
              phx-click="toggle_auto_run" 
              class="mr-2"
              <%= if @auto_run, do: "checked" %>
            />
            <label for="auto_run">Auto-run trials</label>
          </div>
        </div>
        
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-xl font-semibold mb-4">Trial Statistics</h2>
          
          <div class="mb-4">
            <p><strong>Current Position:</strong> <%= @maze_state.position %></p>
            <p><strong>Reward Location:</strong> <%= @maze_state.reward_location %></p>
            <p><strong>Trial Step:</strong> <%= @maze_state.trial_step %> / <%= @maze_state.max_steps %></p>
            <p><strong>Cue Visible:</strong> <%= @maze_state.cue_visible %></p>
          </div>
          
          <div class="mb-4">
            <p><strong>Trials Completed:</strong> <%= @maze_state.trials_completed %></p>
            <p><strong>Successful Trials:</strong> <%= @maze_state.successful_trials %></p>
            <p><strong>Success Rate:</strong> 
              <%= if @maze_state.trials_completed > 0 do %>
                <%= Float.round(@maze_state.successful_trials / @maze_state.trials_completed * 100, 1) %>%
              <% else %>
                0.0%
              <% end %>
            </p>
          </div>
          
          <h3 class="text-lg font-semibold mb-2">Trial History</h3>
          <div class="bg-gray-100 p-4 rounded-md max-h-40 overflow-y-auto">
            <table class="w-full">
              <thead>
                <tr>
                  <th class="text-left">Trial</th>
                  <th class="text-left">Reward</th>
                  <th class="text-left">Choice</th>
                  <th class="text-left">Success</th>
                </tr>
              </thead>
              <tbody>
                <%= for {{reward, choice, success}, idx} <- Enum.with_index(@maze_state.history) do %>
                  <tr>
                    <td><%= idx + 1 %></td>
                    <td><%= reward %></td>
                    <td><%= choice %></td>
                    <td><%= success %></td>
                  </tr>
                <% end %>
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      <%= if length(@evolution_stats) > 0 do %>
        <div class="mt-8 bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-xl font-semibold mb-4">Evolution Statistics</h2>
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h3 class="text-lg font-semibold mb-2">Best Fitness</h3>
              <p class="text-2xl font-bold"><%= Float.round(List.last(@evolution_stats).best_fitness, 2) %></p>
            </div>
            <div>
              <h3 class="text-lg font-semibold mb-2">Generations</h3>
              <p class="text-2xl font-bold"><%= length(@evolution_stats) %></p>
            </div>
            <div>
              <h3 class="text-lg font-semibold mb-2">Species Count</h3>
              <p class="text-2xl font-bold"><%= List.last(@evolution_stats).species_count %></p>
            </div>
          </div>
          
          <div class="mt-4">
            <h3 class="text-lg font-semibold mb-2">Fitness Progress</h3>
            <div class="bg-gray-100 p-4 rounded-md">
              <div class="h-40 flex items-end">
                <%= for stat <- @evolution_stats do %>
                  <div class="flex-1 flex flex-col items-center">
                    <div 
                      class="bg-blue-500 w-8" 
                      style={"height: #{stat.best_fitness / Enum.max_by(@evolution_stats, & &1.best_fitness).best_fitness * 100}%"}
                    ></div>
                    <div class="mt-2 text-xs"><%= stat.generation %></div>
                  </div>
                <% end %>
              </div>
            </div>
          </div>
        </div>
      <% end %>
    </div>
    """
  end
end
