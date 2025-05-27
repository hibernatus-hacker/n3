"""
Enhanced visualization for N3's performance on the CartPole environment.

This script provides detailed metrics and visualizations for neural networks
trained on the CartPole environment.
"""

import gymnasium as gym
import json
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

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

class CartPoleVisualizer:
    """Class for visualizing CartPole performance with detailed metrics."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.metrics = {
            'rewards': [],
            'episode_lengths': [],
            'cart_positions': [],
            'pole_angles': [],
            'actions': []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs('cartpole_metrics', exist_ok=True)
    
    def run_episode(self, network, render=True, collect_metrics=True):
        """Run a single episode with the given network."""
        observation, _ = self.env.reset()
        total_reward = 0
        done = False
        step = 0
        
        episode_metrics = {
            'cart_positions': [],
            'pole_angles': [],
            'actions': []
        }
        
        while not done:
            if render:
                self.env.render()
                time.sleep(0.01)  # Slow down for visualization
            
            # Get action from network
            outputs = network.activate(observation)
            action = 0 if outputs[0] > outputs[1] else 1
            
            # Take step in environment
            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # Collect metrics
            if collect_metrics:
                episode_metrics['cart_positions'].append(observation[0])
                episode_metrics['pole_angles'].append(observation[2])
                episode_metrics['actions'].append(action)
        
        # Update overall metrics
        if collect_metrics:
            self.metrics['rewards'].append(total_reward)
            self.metrics['episode_lengths'].append(step)
            self.metrics['cart_positions'].append(episode_metrics['cart_positions'])
            self.metrics['pole_angles'].append(episode_metrics['pole_angles'])
            self.metrics['actions'].append(episode_metrics['actions'])
        
        return total_reward
    
    def plot_metrics(self, filename_prefix='cartpole'):
        """Plot the collected metrics."""
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot rewards
        axs[0].plot(self.metrics['rewards'])
        axs[0].set_title('Rewards per Episode')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        axs[0].grid(True)
        
        # Plot episode lengths
        axs[1].plot(self.metrics['episode_lengths'])
        axs[1].set_title('Episode Lengths')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Steps')
        axs[1].grid(True)
        
        # Plot pole angles for the last episode
        if self.metrics['pole_angles']:
            last_episode = self.metrics['pole_angles'][-1]
            axs[2].plot(last_episode)
            axs[2].set_title('Pole Angle (Last Episode)')
            axs[2].set_xlabel('Step')
            axs[2].set_ylabel('Angle (radians)')
            axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'cartpole_metrics/{filename_prefix}_metrics.png')
        plt.close()
        
        # Create animation of cart positions and pole angles
        if self.metrics['cart_positions'] and self.metrics['pole_angles']:
            self.create_animation(filename_prefix)
    
    def create_animation(self, filename_prefix):
        """Create an animation of the CartPole performance."""
        # Use the last episode for animation
        cart_positions = self.metrics['cart_positions'][-1]
        pole_angles = self.metrics['pole_angles'][-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-1, 1)
        
        # Cart and pole objects
        cart = plt.Rectangle((0, -0.1), 0.4, 0.2, fill=True, color='blue')
        pole = plt.Line2D([0, 0], [0, 0], lw=3, color='red')
        
        ax.add_patch(cart)
        ax.add_line(pole)
        
        def init():
            cart.set_xy((-0.2, -0.1))
            pole.set_data([0, 0], [0, 0])
            return cart, pole
        
        def animate(i):
            cart_pos = cart_positions[i]
            pole_angle = pole_angles[i]
            
            # Update cart position
            cart.set_xy((cart_pos - 0.2, -0.1))
            
            # Update pole angle
            pole_length = 0.5
            x = cart_pos
            y = 0
            x_end = x + pole_length * np.sin(pole_angle)
            y_end = y + pole_length * np.cos(pole_angle)
            pole.set_data([x, x_end], [y, y_end])
            
            return cart, pole
        
        anim = FuncAnimation(fig, animate, frames=len(cart_positions),
                             init_func=init, blit=True, interval=50)
        
        # Save animation
        anim.save(f'cartpole_metrics/{filename_prefix}_animation.gif', writer='pillow', fps=30)
        plt.close()
    
    def evolve_network(self, generations=5, population_size=20, render_best=True):
        """Evolve a network for CartPole using a simple evolutionary algorithm."""
        print(f"Evolving network for {generations} generations with population size {population_size}")
        
        # Initialize population
        population = [SimpleNetwork() for _ in range(population_size)]
        
        best_fitness_per_gen = []
        avg_fitness_per_gen = []
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = []
            for i, network in enumerate(population):
                # Only render the first individual of each generation
                should_render = render_best and i == 0 and generation > 0
                fitness = self.run_episode(network, render=should_render)
                fitness_scores.append(fitness)
                print(f"Generation {generation+1}, Individual {i+1}: Fitness = {fitness}")
            
            # Get best network
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_network = population[best_idx]
            
            # Record statistics
            best_fitness_per_gen.append(best_fitness)
            avg_fitness_per_gen.append(np.mean(fitness_scores))
            
            print(f"\nGeneration {generation+1} complete")
            print(f"Best fitness: {best_fitness}")
            print(f"Average fitness: {np.mean(fitness_scores)}")
            
            # Save best network
            best_network.save_weights(f"cartpole_metrics/best_network_gen_{generation+1}.json")
            
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
        
        # Plot evolution progress
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, generations+1), best_fitness_per_gen, 'b-', label='Best Fitness')
        plt.plot(range(1, generations+1), avg_fitness_per_gen, 'r-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('cartpole_metrics/evolution_progress.png')
        plt.close()
        
        # Visualize best network
        print("\nVisualizing best network...")
        best_network = population[np.argmax(fitness_scores)]
        final_fitness = self.run_episode(best_network, render=True)
        print(f"Final fitness: {final_fitness}")
        
        # Plot metrics
        self.plot_metrics()
        
        return best_network

def main():
    """Main function to run the enhanced CartPole visualization."""
    visualizer = CartPoleVisualizer()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "visualize" and len(sys.argv) > 2:
            # Visualize a specific network
            network = SimpleNetwork(sys.argv[2])
            fitness = visualizer.run_episode(network, render=True)
            print(f"Fitness: {fitness}")
            visualizer.plot_metrics(filename_prefix=os.path.basename(sys.argv[2]).split('.')[0])
        else:
            # Evolve for specified number of generations
            try:
                generations = int(sys.argv[1])
                visualizer.evolve_network(generations=generations)
            except ValueError:
                print("Invalid argument. Usage: python enhanced_cartpole_viz.py [generations|visualize network_file]")
    else:
        # Default: evolve for 5 generations
        visualizer.evolve_network(generations=5)
    
    visualizer.env.close()

if __name__ == "__main__":
    main()
