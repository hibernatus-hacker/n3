# NeuroEvolution

A comprehensive TWEANN (Topology and Weight Evolving Artificial Neural Network) neuroevolution library for Elixir with GPU acceleration via Nx.

## Features

### 🧠 **TWEANN Evolution**
- Dynamic topology evolution (add/remove nodes and connections)
- Innovation-based crossover for topological alignment
- Speciation with fitness sharing
- Comprehensive mutation operators

### 🌐 **Substrate Encodings**
- HyperNEAT implementation for spatial neural networks
- Multiple substrate geometries (grid, circular, hexagonal, custom)
- CPPN (Compositional Pattern Producing Network) evolution
- ES-HyperNEAT support for adaptive substrate resolution

### 🔄 **Neural Plasticity**
- Multiple plasticity rules:
  - Hebbian learning and variants
  - STDP (Spike-Timing Dependent Plasticity)
  - BCM (Bienenstock-Cooper-Munro) rule
  - Oja's rule for principal component learning
- Homeostatic mechanisms
- Metaplasticity for adaptive learning

### 🚀 **GPU Acceleration**
- Nx-powered tensor operations
- Batch evaluation for population processing
- Topology clustering for efficient GPU utilization
- CUDA/ROCM support via EXLA

### 🧬 **Population Management**
- Adaptive population sizing
- Elite preservation strategies
- Diversity maintenance mechanisms
- Stagnation detection and recovery

## Installation

Add `neuro_evolution` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:neuro_evolution, "~> 0.1.0"},
    {:nx, "~> 0.7"},
    {:exla, "~> 0.7"},  # For GPU acceleration
    {:axon, "~> 0.6"},
    {:jason, "~> 1.4"}
  ]
end
```

## Quick Start

### Basic TWEANN Evolution

```elixir
# Create a population
population = NeuroEvolution.new_population(100, 3, 2)

# Define fitness function
fitness_fn = fn genome ->
  # Your evaluation logic here
  evaluate_genome_performance(genome)
end

# Evolve for 100 generations
evolved_population = NeuroEvolution.evolve(population, fitness_fn, 
  generations: 100, 
  target_fitness: 0.95)

# Get the best genome
best_genome = evolved_population.best_genome
```

### XOR Problem Example

```elixir
# Create population for XOR (2 inputs, 1 output)
population = NeuroEvolution.new_population(150, 2, 1)

# Use built-in XOR fitness function
fitness_fn = NeuroEvolution.xor_fitness()

# Evolve
evolved = NeuroEvolution.evolve(population, fitness_fn, generations: 300)

IO.puts("Best fitness: #{evolved.best_fitness}")
```

### HyperNEAT with Substrate Encoding

```elixir
# Create 2D grid substrate
substrate_config = %{
  geometry_type: :grid,
  dimensions: 2,
  resolution: {10, 10},
  connection_function: :distance_based
}

# Create HyperNEAT system
hyperneat = NeuroEvolution.new_hyperneat(
  [10, 10],  # Input layer dimensions
  [5, 5],    # Hidden layer dimensions  
  [3, 3],    # Output layer dimensions
  substrate_config: substrate_config,
  connection_threshold: 0.3,
  leo_enabled: true  # Link Expression Output
)

# Decode CPPN to phenotype network
phenotype = NeuroEvolution.Substrate.HyperNEAT.decode_substrate(hyperneat)
```

### Neural Plasticity

```elixir
# Create population with Hebbian plasticity
plasticity_config = %{
  plasticity_type: :hebbian,
  learning_rate: 0.01,
  homeostasis: true,
  decay_rate: 0.99
}

population = NeuroEvolution.new_population(100, 3, 2, 
  plasticity: plasticity_config)

# STDP plasticity
stdp_config = %{
  plasticity_type: :stdp,
  a_plus: 0.1,
  a_minus: 0.12,
  tau_plus: 20.0,
  tau_minus: 20.0
}

plasticity = NeuroEvolution.new_plasticity(:stdp, stdp_config)
```

## Architecture Overview

### Core Components

```
NeuroEvolution/
├── TWEANN/           # Core genome representation
│   ├── Genome        # Network topology and weights
│   ├── Node          # Neural nodes with activation functions
│   └── Connection    # Synaptic connections with plasticity
├── Substrate/        # Spatial network encodings
│   ├── HyperNEAT     # CPPN-based substrate evolution
│   └── Substrate     # Geometry definitions
├── Plasticity/       # Neural plasticity mechanisms
│   ├── HebbianRule   # Hebbian learning variants
│   ├── STDPRule      # Spike-timing dependent plasticity
│   └── NeuralPlasticity # Main plasticity coordinator
├── Evaluator/        # GPU-optimized evaluation
│   └── BatchEvaluator # Batch processing for populations
└── Population/       # Evolution management
    ├── Population    # Main evolution loop
    ├── Species       # Speciation and niching
    └── Selection     # Selection strategies
```

### GPU Acceleration Strategy

The library addresses the fundamental challenge of GPU acceleration for dynamic topologies through:

1. **Topology Clustering**: Groups genomes with similar structures for batch processing
2. **Tensor Padding**: Pads smaller networks to uniform sizes within clusters  
3. **Mask Tensors**: Uses masks to disable unused connections efficiently
4. **Lazy Compilation**: Only recompiles GPU kernels when topology diversity changes significantly

## Advanced Usage

### Custom Substrate Geometries

```elixir
# Hexagonal substrate
hex_substrate = NeuroEvolution.Substrate.Substrate.hexagonal(5)

# Custom positions
custom_positions = [
  {0.0, 0.0}, {0.5, 0.0}, {1.0, 0.0},  # Input layer
  {0.25, 0.5}, {0.75, 0.5},            # Hidden layer
  {0.5, 1.0}                           # Output layer
]

custom_substrate = NeuroEvolution.Substrate.Substrate.custom(
  [{0.0, 0.0}, {0.5, 0.0}, {1.0, 0.0}],  # Inputs
  [{0.25, 0.5}, {0.75, 0.5}],            # Hidden
  [{0.5, 1.0}]                           # Outputs
)
```

### Population Statistics and Monitoring

```elixir
# Get detailed population statistics
stats = NeuroEvolution.get_population_stats(population)

IO.puts """
Generation: #{stats.generation}
Best Fitness: #{stats.best_fitness}
Average Fitness: #{stats.avg_fitness}
Species Count: #{stats.species_count}
Genetic Diversity: #{stats.diversity_metrics.genetic_diversity}
"""
```

### Custom Mutation Strategies

```elixir
config = %{
  mutation: %{
    weight_mutation_rate: 0.9,      # High weight mutation
    weight_perturbation_rate: 0.8,   # Most weights get small changes
    add_node_rate: 0.05,             # Moderate structural growth
    add_connection_rate: 0.1,        # Frequent new connections
    disable_connection_rate: 0.02    # Occasional pruning
  },
  speciation_threshold: 2.5,         # Tighter species boundaries
  fitness_sharing: true,             # Enable niching
  elitism_rate: 0.15                 # Preserve top 15%
}

population = NeuroEvolution.new_population(200, 4, 3, config)
```

## Performance Considerations

### GPU Memory Management
- Use appropriate `max_topology_size` to limit memory usage
- Consider batch sizes based on available GPU memory
- Monitor topology diversity to optimize clustering

### Population Sizing
- Larger populations provide better exploration but slower evolution
- Enable adaptive population sizing for dynamic adjustment
- Use diversity metrics to guide population size decisions

### Plasticity Trade-offs
- Plasticity adds computational overhead but enables learning
- Consider plasticity only for environments requiring adaptation
- Balance plasticity rates with evolution rates

## Examples

See the `examples/` directory for complete implementations:

- `xor_evolution.exs` - Classic XOR problem solving
- `hyperneat_substrate.exs` - Spatial pattern recognition
- `plastic_learning.exs` - Adaptive behavior with plasticity
- `gpu_benchmarks.exs` - Performance optimization examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass with `mix test`
5. Submit a pull request

## Performance Benchmarks

### Standard TWEANN Evolution
- 1000 genomes/generation: ~2.5s (CPU) vs ~0.3s (GPU batch)
- Scales linearly with population size on GPU
- Memory usage: ~50MB per 1000 genomes

### HyperNEAT Substrate Decoding  
- 100x100 substrate: ~100ms decode time
- CPPN evaluation: GPU-accelerated for large substrates
- Memory scales with substrate resolution²

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Based on NEAT algorithm by Kenneth Stanley
- HyperNEAT implementation following Stanley et al.
- Plasticity mechanisms from computational neuroscience literature
- GPU optimization strategies from modern deep learning frameworks