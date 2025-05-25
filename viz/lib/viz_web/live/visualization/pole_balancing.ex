defmodule VizWeb.VisualizationLive.PoleBalancing do
  use VizWeb, :live_view
  
  alias NeuroEvolution.TWEANN.Genome

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      # Start a timer to update the simulation
      :timer.send_interval(50, self(), :update_simulation)
    end
    
    # Create a test genome for demonstration
    genome = create_test_genome()
    
    # Initialize the pole balancing state
    state = %{
      genome: genome,
      position: 0.0,  # Cart position (-2.4 to 2.4)
      velocity: 0.0,  # Cart velocity
      angle: 0.1,     # Pole angle in radians
      angle_velocity: 0.0, # Angular velocity
      time: 0,
      max_time: 500,
      history: [],
      force: 0.0
    }
    
    {:ok, assign(socket, 
      state: state,
      running: false,
      auto_reset: true,
      evolution_stats: [],
      show_network: false
    )}
  end

  @impl true
  def handle_event("toggle_simulation", _, socket) do
    running = not socket.assigns.running
    
    if running do
      Process.send_after(self(), :update_simulation, 50)
    end
    
    {:noreply, assign(socket, running: running)}
  end
  
  @impl true
  def handle_event("toggle_auto_reset", _, socket) do
    {:noreply, assign(socket, auto_reset: not socket.assigns.auto_reset)}
  end
  
  @impl true
  def handle_event("toggle_network_view", _, socket) do
    {:noreply, assign(socket, show_network: not socket.assigns.show_network)}
  end
  
  @impl true
  def handle_event("reset_simulation", _, socket) do
    # Reset the simulation state
    state = %{
      genome: socket.assigns.state.genome,
      position: 0.0,
      velocity: 0.0,
      angle: 0.1,  # Start with a slight angle
      angle_velocity: 0.0,
      time: 0,
      max_time: 500,
      history: [],
      force: 0.0
    }
    
    {:noreply, assign(socket, state: state)}
  end
  
  @impl true
  def handle_event("evolve_network", _, socket) do
    # Create a small population for evolution
    population = NeuroEvolution.new_population(20, 4, 1, 
      %{
        speciation: %{enabled: true, compatibility_threshold: 1.0},
        mutation: %{
          weight_mutation_rate: 0.8,
          weight_perturbation: 0.5,
          add_node_rate: 0.03,
          add_connection_rate: 0.05
        }
      }
    )
    
    # Pole balancing fitness function
    fitness_fn = fn genome ->
      evaluate_pole_balancing(genome)
    end
    
    # Evolve for a few generations
    {evolved_pop, stats} = evolve_with_stats(population, fitness_fn, 5)
    
    # Extract the best genome
    best_genome = NeuroEvolution.get_best_genome(evolved_pop)
    
    # Reset the simulation with the new genome
    state = %{
      genome: best_genome,
      position: 0.0,
      velocity: 0.0,
      angle: 0.1,
      angle_velocity: 0.0,
      time: 0,
      max_time: 500,
      history: [],
      force: 0.0
    }
    
    {:noreply, assign(socket, 
      state: state, 
      evolution_stats: stats
    )}
  end
  
  @impl true
  def handle_info(:update_simulation, socket) do
    if socket.assigns.running do
      # Update the simulation state
      state = update_pole_balancing_state(socket.assigns.state)
      
      # Check if the simulation has failed
      failed = has_failed?(state)
      
      # If failed and auto-reset is enabled, reset the simulation
      {state, should_continue} = 
        if failed and socket.assigns.auto_reset do
          {%{state | 
            position: 0.0,
            velocity: 0.0,
            angle: 0.1,
            angle_velocity: 0.0,
            time: 0,
            history: []
          }, true}
        else
          # Otherwise, continue the simulation if it hasn't failed
          {state, not failed}
        end
      
      # Schedule the next update if we should continue
      if should_continue do
        Process.send_after(self(), :update_simulation, 50)
      end
      
      {:noreply, assign(socket, state: state, running: should_continue)}
    else
      {:noreply, socket}
    end
  end
  
  # Helper function to update the pole balancing state
  defp update_pole_balancing_state(state) do
    # Constants for the pole balancing simulation
    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    total_mass = mass_cart + mass_pole
    length = 0.5  # Half the pole length
    pole_mass_length = mass_pole * length
    force_mag = 10.0
    dt = 0.02  # Time step
    
    # Get the inputs for the neural network
    inputs = [
      state.position / 2.4,  # Normalize position
      state.velocity / 10.0,  # Normalize velocity
      state.angle / 0.2094,  # Normalize angle (12 degrees)
      state.angle_velocity / 10.0  # Normalize angular velocity
    ]
    
    # Create a simple evaluator
    evaluator = NeuroEvolution.Evaluator.BatchEvaluator.new(device: :cpu)
    
    # Get the output from the genome (simplified for visualization)
    outputs = manual_forward_propagate(state.genome, inputs)
    
    # Determine the force based on the network output
    force = if List.first(outputs) > 0.5, do: force_mag, else: -force_mag
    
    # Calculate the dynamics of the system
    temp = (force + pole_mass_length * state.angle_velocity * state.angle_velocity * :math.sin(state.angle)) / total_mass
    angle_acc = (gravity * :math.sin(state.angle) - :math.cos(state.angle) * temp) / 
                (length * (4.0/3.0 - mass_pole * :math.cos(state.angle) * :math.cos(state.angle) / total_mass))
    cart_acc = temp - pole_mass_length * angle_acc * :math.cos(state.angle) / total_mass
    
    # Update the state variables
    new_position = state.position + dt * state.velocity
    new_velocity = state.velocity + dt * cart_acc
    new_angle = state.angle + dt * state.angle_velocity
    new_angle_velocity = state.angle_velocity + dt * angle_acc
    
    # Update the state
    %{
      state |
      position: new_position,
      velocity: new_velocity,
      angle: new_angle,
      angle_velocity: new_angle_velocity,
      time: state.time + 1,
      history: state.history ++ [{new_position, new_angle}],
      force: force
    }
  end
  
  # Helper function to check if the simulation has failed
  defp has_failed?(state) do
    # Check if the cart has moved beyond the boundaries
    # or if the pole has fallen too far
    state.position < -2.4 or 
    state.position > 2.4 or 
    state.angle < -0.2094 or  # -12 degrees
    state.angle > 0.2094 or   # 12 degrees
    state.time >= state.max_time
  end
  
  # Helper function to evaluate a genome on the pole balancing task
  defp evaluate_pole_balancing(genome) do
    # Initialize the state
    state = %{
      genome: genome,
      position: 0.0,
      velocity: 0.0,
      angle: 0.1,
      angle_velocity: 0.0,
      time: 0,
      max_time: 500
    }
    
    # Run the simulation until failure
    run_simulation(state, 0)
  end
  
  # Helper function to run the simulation until failure
  defp run_simulation(state, steps) do
    if has_failed?(state) do
      # Return the number of steps as the fitness
      steps
    else
      # Update the state and continue
      new_state = update_pole_balancing_state(state)
      run_simulation(new_state, steps + 1)
    end
  end
  
  # Helper function to create a test genome
  defp create_test_genome do
    Genome.new(4, 1)
  end
  
  # A simplified forward propagation implementation that doesn't use Nx tensors
  defp manual_forward_propagate(genome, inputs) do
    # Initialize activations for all nodes
    activations = %{}
    
    # Set input activations
    activations = Enum.with_index(inputs, 1)
      |> Enum.reduce(activations, fn {value, idx}, acc ->
        Map.put(acc, idx, value)
      end)
    
    # Process hidden and output nodes in topological order
    sorted_nodes = genome.inputs ++ (Map.keys(genome.nodes) -- genome.inputs -- genome.outputs) ++ genome.outputs
    
    # Propagate signals through the network
    final_activations = Enum.reduce(sorted_nodes, activations, fn node_id, acc ->
      # Skip input nodes as they already have activations
      if node_id in genome.inputs do
        acc
      else
        # Get all incoming connections to this node
        incoming = Enum.filter(genome.connections, fn {_id, conn} -> 
          conn.to == node_id && conn.enabled
        end)
        
        # Sum weighted inputs
        weighted_sum = Enum.reduce(incoming, 0.0, fn {_id, conn}, sum ->
          from_activation = Map.get(acc, conn.from, 0.0)
          sum + from_activation * conn.weight
        end)
        
        # Apply activation function (sigmoid)
        activation = 1.0 / (1.0 + :math.exp(-weighted_sum))
        
        # Store the activation
        Map.put(acc, node_id, activation)
      end
    end)
    
    # Extract output activations
    Enum.map(genome.outputs, fn output_id ->
      Map.get(final_activations, output_id, 0.0)
    end)
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
  
  # Helper function to render the cart and pole
  defp render_cart_pole(position, angle) do
    # Canvas dimensions
    width = 600
    height = 200
    
    # Cart dimensions
    cart_width = 50
    cart_height = 30
    
    # Scale the position to fit the canvas
    scaled_position = position * 100 + width / 2
    
    # Calculate the pole endpoints
    pole_length = 100
    pole_end_x = scaled_position + pole_length * :math.sin(angle)
    pole_end_y = 150 - pole_length * :math.cos(angle)
    
    # Generate the SVG
    """
    <svg width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">
      <!-- Track -->
      <line x1="50" y1="150" x2="#{width - 50}" y2="150" stroke="black" stroke-width="2" />
      
      <!-- Cart -->
      <rect x="#{scaled_position - cart_width / 2}" y="#{150 - cart_height / 2}" width="#{cart_width}" height="#{cart_height}" fill="blue" />
      
      <!-- Pole -->
      <line x1="#{scaled_position}" y1="150" x2="#{pole_end_x}" y2="#{pole_end_y}" stroke="red" stroke-width="4" />
      
      <!-- Pole tip -->
      <circle cx="#{pole_end_x}" cy="#{pole_end_y}" r="5" fill="red" />
    </svg>
    """
  end
  
  # Helper function to render the neural network
  defp render_network(genome) do
    # Canvas dimensions
    width = 400
    height = 300
    
    # Node positions
    input_y = 50
    hidden_y = 150
    output_y = 250
    
    # Get all nodes
    input_nodes = genome.inputs
    output_nodes = genome.outputs
    hidden_nodes = Map.keys(genome.nodes) -- input_nodes -- output_nodes
    
    # Calculate node positions
    input_positions = calculate_node_positions(input_nodes, width, input_y)
    hidden_positions = calculate_node_positions(hidden_nodes, width, hidden_y)
    output_positions = calculate_node_positions(output_nodes, width, output_y)
    
    # Combine all node positions
    node_positions = Map.merge(input_positions, hidden_positions) |> Map.merge(output_positions)
    
    # Generate SVG for nodes
    nodes_svg = node_positions
      |> Enum.map(fn {node_id, {x, y}} ->
        node_type = cond do
          node_id in input_nodes -> "input"
          node_id in output_nodes -> "output"
          true -> "hidden"
        end
        
        color = case node_type do
          "input" -> "green"
          "output" -> "red"
          "hidden" -> "blue"
        end
        
        """
        <circle cx="#{x}" cy="#{y}" r="10" fill="#{color}" />
        <text x="#{x}" y="#{y + 4}" text-anchor="middle" fill="white" font-size="10">#{node_id}</text>
        """
      end)
      |> Enum.join("\n")
    
    # Generate SVG for connections
    connections_svg = genome.connections
      |> Enum.filter(fn {_id, conn} -> conn.enabled end)
      |> Enum.map(fn {_id, conn} ->
        from_pos = Map.get(node_positions, conn.from)
        to_pos = Map.get(node_positions, conn.to)
        
        if from_pos && to_pos do
          {from_x, from_y} = from_pos
          {to_x, to_y} = to_pos
          
          # Determine color based on weight
          color = cond do
            conn.weight > 0 -> "rgba(0, 0, 255, #{min(abs(conn.weight) / 2, 1)})"
            true -> "rgba(255, 0, 0, #{min(abs(conn.weight) / 2, 1)})"
          end
          
          # Determine line width based on weight
          width = min(abs(conn.weight) * 2, 4)
          
          """
          <line x1="#{from_x}" y1="#{from_y}" x2="#{to_x}" y2="#{to_y}" stroke="#{color}" stroke-width="#{width}" />
          """
        else
          ""
        end
      end)
      |> Enum.join("\n")
    
    # Generate the complete SVG
    """
    <svg width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">
      #{connections_svg}
      #{nodes_svg}
    </svg>
    """
  end
  
  # Helper function to calculate node positions
  defp calculate_node_positions(nodes, width, y) do
    node_count = length(nodes)
    
    if node_count > 0 do
      spacing = width / (node_count + 1)
      
      nodes
      |> Enum.with_index(1)
      |> Enum.map(fn {node_id, idx} ->
        {node_id, {idx * spacing, y}}
      end)
      |> Map.new()
    else
      %{}
    end
  end
  
  @impl true
  def render(assigns) do
    ~H"""
    <div class="container mx-auto px-4 py-8">
      <h1 class="text-3xl font-bold mb-6">Pole Balancing Visualization</h1>
      
      <div class="grid grid-cols-1 gap-8">
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-xl font-semibold mb-4">Simulation</h2>
          
          <div class="mb-6 bg-gray-100 p-4 rounded-md overflow-hidden">
            <%= raw render_cart_pole(@state.position, @state.angle) %>
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
              id="auto_reset" 
              phx-click="toggle_auto_reset" 
              class="mr-2"
              <%= if @auto_reset, do: "checked" %>
            />
            <label for="auto_reset">Auto-reset on failure</label>
          </div>
          
          <div class="flex items-center mb-4">
            <input 
              type="checkbox" 
              id="show_network" 
              phx-click="toggle_network_view" 
              class="mr-2"
              <%= if @show_network, do: "checked" %>
            />
            <label for="show_network">Show neural network</label>
          </div>
        </div>
        
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-xl font-semibold mb-4">Simulation Statistics</h2>
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <h3 class="text-lg font-semibold">Position</h3>
              <p class="text-2xl font-bold"><%= Float.round(@state.position, 3) %></p>
            </div>
            <div>
              <h3 class="text-lg font-semibold">Angle (degrees)</h3>
              <p class="text-2xl font-bold"><%= Float.round(@state.angle * 180 / :math.pi, 2) %>°</p>
            </div>
            <div>
              <h3 class="text-lg font-semibold">Time Steps</h3>
              <p class="text-2xl font-bold"><%= @state.time %></p>
            </div>
          </div>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 class="text-lg font-semibold">Velocity</h3>
              <p class="text-xl"><%= Float.round(@state.velocity, 3) %></p>
            </div>
            <div>
              <h3 class="text-lg font-semibold">Angular Velocity</h3>
              <p class="text-xl"><%= Float.round(@state.angle_velocity, 3) %></p>
            </div>
            <div>
              <h3 class="text-lg font-semibold">Force</h3>
              <p class="text-xl"><%= Float.round(@state.force, 3) %></p>
            </div>
          </div>
        </div>
        
        <%= if @show_network do %>
          <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-xl font-semibold mb-4">Neural Network</h2>
            
            <div class="mb-6 bg-gray-100 p-4 rounded-md overflow-hidden">
              <%= raw render_network(@state.genome) %>
            </div>
            
            <div class="text-sm text-gray-600">
              <p>Green: Input nodes (position, velocity, angle, angular velocity)</p>
              <p>Blue: Hidden nodes</p>
              <p>Red: Output node (force direction)</p>
              <p>Line color: Blue for positive weights, Red for negative weights</p>
              <p>Line thickness: Proportional to weight magnitude</p>
            </div>
          </div>
        <% end %>
        
        <%= if length(@evolution_stats) > 0 do %>
          <div class="bg-white p-6 rounded-lg shadow-md">
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
    </div>
    """
  end
end
