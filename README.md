# NeuroEvolution

A comprehensive TWEANN (Topology and Weight Evolving Artificial Neural Network) neuroevolution library for Elixir with advanced features for evolving adaptive neural networks.

## 🚀 Features

- **TWEANN Evolution**: Full topology and weight evolution with NEAT-compliant innovation tracking
- **Neural Plasticity**: Hebbian, STDP, BCM, and Oja's learning rules for adaptive networks
- **HyperNEAT Substrates**: Spatial pattern processing with compositional pattern producing networks (CPPNs)
- **GPU Acceleration**: High-performance batch evaluation via Nx and EXLA
- **Secure Gym Integration**: Safe Python bridge for OpenAI Gym environments
- **Robust Architecture**: Supervised processes with proper error handling and memory management

## 📚 Examples

The library includes 5 focused, high-quality examples demonstrating core capabilities:

### 🧠 Basic XOR Evolution (`basic_xor_example.exs`)
Learn the fundamentals of TWEANN evolution with the classic XOR problem.

### 🎯 CartPole Control (`cartpole_example.exs`) 
Evolve neural controllers for the CartPole balancing task using neural plasticity.

### 🌐 Spatial Networks (`substrate_example.exs`)
Explore HyperNEAT and substrate encoding for spatial pattern recognition.

### 🔄 Neural Plasticity (`plasticity_example.exs`)
Compare different plasticity rules (Hebbian, STDP, BCM, Oja) for adaptive learning.

### 🏋️ Gym Integration (`gym_integration_example.exs`)
Multi-environment evolution with secure OpenAI Gym integration.

## 🚀 Quick Start

```elixir
# Basic XOR evolution
population = NeuroEvolution.new_population(50, 2, 1)
fitness_fn = NeuroEvolution.xor_fitness()
evolved = NeuroEvolution.evolve(population, fitness_fn, generations: 50)
best = NeuroEvolution.get_best_genome(evolved)

# Test the evolved network
outputs = NeuroEvolution.activate(best, [1.0, 0.0])
IO.puts("XOR(1,0) = #{List.first(outputs)}")
```

### 🎮 Gym Environments

```elixir
# Start secure Python bridge
{:ok, _bridge} = NeuroEvolution.Environments.PythonBridge.start_link()

# Create controller for CartPole
genome = NeuroEvolution.new_genome(4, 2, plasticity: %{plasticity_type: :hebbian})
score = NeuroEvolution.Environments.CartPole.evaluate(genome)
genome = NeuroEvolution.new_genome(2, 3)

# Evaluate the genome on MountainCar
score = NeuroEvolution.Environments.GymAdapter.evaluate(genome, "MountainCar-v0")

# Create a population for MountainCar
population = NeuroEvolution.new_population(20, 2, 3)

# Evolve the population directly using the adapter
{evolved_pop, stats} = NeuroEvolution.Environments.GymAdapter.evolve(population, "MountainCar-v0", 10)

# Visualize the best genome's performance
best_genome = Enum.max_by(evolved_pop.genomes, fn genome -> genome.fitness || 0 end)
NeuroEvolution.Environments.GymAdapter.visualize(best_genome, "MountainCar-v0")
```

See the `examples/gym_environments_example.exs` script for examples with multiple environments.

### Enhanced Visualization

For detailed visualization and metrics of the CartPole environment, you can use the provided scripts:

```bash
# Run the enhanced CartPole visualization with 5 generations
./run_cartpole.sh 5

# Use simple visualization without metrics
./run_cartpole.sh --simple 5

# Visualize a specific network
./run_cartpole.sh --visualize best_network_gen_3.json

# Show help message
./run_cartpole.sh --help
```

The enhanced visualization includes:

- Performance metrics (rewards, episode lengths)
- Evolution progress charts
- Detailed animations of the CartPole behavior
- Saved network weights for later analysis

## Installation

Ensure you have Python installed with the required dependencies:

```bash
pip install -r priv/python/requirements.txt
```

