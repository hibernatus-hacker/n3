"""
Visualize N3's performance on the CartPole environment.

This script loads a neural network from a JSON file and visualizes its
performance on the CartPole environment.
"""

import gymnasium as gym
import json
import numpy as np
import time
import sys

# Ensure compatibility with different NumPy versions
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class SimpleNetwork:
    """A simple neural network implementation for CartPole."""
    
    def __init__(self, weights_file=None):
        """Initialize the network with weights from a file or randomly."""
        if weights_file:
            with open(weights_file, 'r') as f:
                data = json.load(f)
                self.input_weights = np.array(data['input_weights'])
                self.hidden_weights = np.array(data['hidden_weights'])
        else:
            # Random initialization for a simple 4-4-2 network
            self.input_weights = np.random.randn(4, 4) * 0.1
            self.hidden_weights = np.random.randn(4, 2) * 0.1
    
    def activate(self, inputs):
        """Forward pass through the network."""
        # Hidden layer
        hidden = np.tanh(np.dot(inputs, self.input_weights))
        # Output layer
        outputs = 1.0 / (1.0 + np.exp(-np.dot(hidden, self.hidden_weights)))
        return outputs
    
    def save_weights(self, filename):
        """Save the network weights to a file."""
        data = {
            'input_weights': self.input_weights.tolist(),
            'hidden_weights': self.hidden_weights.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

def run_episode(env, network, render=True):
    """Run a single episode with the given network."""
    observation, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        if render:
            env.render()
            time.sleep(0.01)  # Slow down for visualization
        
        # Get action from network
        outputs = network.activate(observation)
        action = 0 if outputs[0] > outputs[1] else 1
        
        # Take step in environment
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    return total_reward

def evolve_network(generations=5, population_size=20):
    """Simple evolution algorithm to evolve a network for CartPole."""
    env = gym.make('CartPole-v1', render_mode='human')
    
    # Initialize population
    population = [SimpleNetwork() for _ in range(population_size)]
    
    for generation in range(generations):
        # Evaluate population
        fitness_scores = []
        for i, network in enumerate(population):
            fitness = run_episode(env, network, render=(i == 0 and generation > 0))
            fitness_scores.append(fitness)
            print(f"Generation {generation+1}, Individual {i+1}: Fitness = {fitness}")
        
        # Get best network
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_network = population[best_idx]
        
        print(f"\nGeneration {generation+1} complete")
        print(f"Best fitness: {best_fitness}")
        
        # Save best network
        best_network.save_weights(f"best_network_gen_{generation+1}.json")
        
        if generation < generations - 1:
            # Create new population through selection and mutation
            new_population = [best_network]  # Elitism
            
            for _ in range(population_size - 1):
                # Tournament selection
                idx1, idx2 = np.random.randint(0, population_size, 2)
                parent = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
                
                # Create child through mutation
                child = SimpleNetwork()
                child.input_weights = parent.input_weights + np.random.randn(4, 4) * 0.1
                child.hidden_weights = parent.hidden_weights + np.random.randn(4, 2) * 0.1
                
                new_population.append(child)
            
            population = new_population
    
    # Visualize best network
    print("\nVisualizing best network...")
    best_network = population[np.argmax(fitness_scores)]
    final_fitness = run_episode(env, best_network, render=True)
    print(f"Final fitness: {final_fitness}")
    
    env.close()
    return best_network

if __name__ == "__main__":
    # Check if we should visualize or evolve
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        if len(sys.argv) > 2:
            # Visualize a specific network
            env = gym.make('CartPole-v1', render_mode='human')
            network = SimpleNetwork(sys.argv[2])
            fitness = run_episode(env, network, render=True)
            print(f"Fitness: {fitness}")
            env.close()
        else:
            print("Please specify a network file to visualize")
    else:
        # Evolve a new network
        generations = 5
        if len(sys.argv) > 1:
            try:
                generations = int(sys.argv[1])
            except ValueError:
                pass
        
        evolve_network(generations=generations)
