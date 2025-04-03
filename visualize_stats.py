import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse
import glob

class StatsRecorder:
    """Helper class to record statistics during simulation for later analysis"""
    
    def __init__(self):
        self.herbivore_counts = []
        self.carnivore_counts = []
        self.grass_counts = []
        self.ticks = []
        self.current_tick = 0
        
        # Expanded statistics for analysis
        self.herbivore_energies = []
        self.carnivore_energies = []
        self.herbivore_ages = []
        self.carnivore_ages = []
        self.reproduction_events = []
        
        # Add grass growth rate tracking
        self.grass_growth_rates = []
        
    def update(self, simulation):
        """Record statistics from the current simulation state"""
        self.current_tick += 1
        
        # Only record every 10 ticks to save memory
        if self.current_tick % 10 != 0:
            return
            
        self.ticks.append(self.current_tick)
        self.herbivore_counts.append(simulation.count_herbivores())
        self.carnivore_counts.append(simulation.count_carnivores())
        self.grass_counts.append(simulation.count_grass())
        
        # Record growth rate if available
        if hasattr(simulation.environment, 'get_current_growth_rate'):
            self.grass_growth_rates.append(simulation.environment.get_current_growth_rate())
        
        # Record more detailed statistics
        if simulation.creatures:
            herbivores = [c for c in simulation.creatures if hasattr(c, '_get_color') and c._get_color()[1] > 0]
            carnivores = [c for c in simulation.creatures if hasattr(c, '_get_color') and c._get_color()[0] > 0]
            
            self.herbivore_energies.append(
                sum([h.energy for h in herbivores]) / len(herbivores) if herbivores else 0
            )
            
            self.carnivore_energies.append(
                sum([c.energy for c in carnivores]) / len(carnivores) if carnivores else 0
            )
            
            self.herbivore_ages.append(
                sum([h.age for h in herbivores]) / len(herbivores) if herbivores else 0
            )
            
            self.carnivore_ages.append(
                sum([c.age for c in carnivores]) / len(carnivores) if carnivores else 0
            )
            
    def save(self, filename='simulation_stats.pkl'):
        """Save statistics to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Statistics saved to {filename}")
    
    @classmethod
    def load(cls, filename='simulation_stats.pkl'):
        """Load statistics from a file"""
        recorder = cls()
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                recorder.__dict__ = pickle.load(f)
            print(f"Statistics loaded from {filename}")
        else:
            print(f"File {filename} not found")
        return recorder
    
    def visualize(self):
        """Generate plots visualizing the statistics"""
        if not self.ticks:
            print("No data to visualize")
            return
            
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Population over time
        plt.subplot(2, 2, 1)
        plt.plot(self.ticks, self.herbivore_counts, 'g-', label='Herbivores')
        plt.plot(self.ticks, self.carnivore_counts, 'r-', label='Carnivores')
        plt.plot(self.ticks, self.grass_counts, 'g--', alpha=0.5, label='Grass')
        plt.title('Population over Time')
        plt.xlabel('Simulation Ticks')
        plt.ylabel('Population')
        plt.legend()
        plt.grid(True)
        
        # Average energy over time
        if self.herbivore_energies and self.carnivore_energies:
            plt.subplot(2, 2, 2)
            plt.plot(self.ticks[:len(self.herbivore_energies)], self.herbivore_energies, 'g-', label='Herbivores')
            plt.plot(self.ticks[:len(self.carnivore_energies)], self.carnivore_energies, 'r-', label='Carnivores')
            plt.title('Average Energy over Time')
            plt.xlabel('Simulation Ticks')
            plt.ylabel('Average Energy')
            plt.legend()
            plt.grid(True)
        
        # Average age over time
        if self.herbivore_ages and self.carnivore_ages:
            plt.subplot(2, 2, 3)
            plt.plot(self.ticks[:len(self.herbivore_ages)], self.herbivore_ages, 'g-', label='Herbivores')
            plt.plot(self.ticks[:len(self.carnivore_ages)], self.carnivore_ages, 'r-', label='Carnivores')
            plt.title('Average Age over Time')
            plt.xlabel('Simulation Ticks')
            plt.ylabel('Average Age')
            plt.legend()
            plt.grid(True)
        
        # Grass count and growth rate
        plt.subplot(2, 2, 4)
        ax1 = plt.gca()
        ax1.plot(self.ticks, self.grass_counts, 'g-', label='Grass Count')
        ax1.set_xlabel('Simulation Ticks')
        ax1.set_ylabel('Grass Count', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        
        if self.grass_growth_rates:
            # Create second Y axis for growth rate
            ax2 = ax1.twinx()
            ax2.plot(self.ticks[:len(self.grass_growth_rates)], self.grass_growth_rates, 'y-', label='Growth Rate')
            ax2.set_ylabel('Growth Rate', color='y')
            ax2.tick_params(axis='y', labelcolor='y')
            
        plt.title('Grass Count and Growth Rate')
        
        plt.tight_layout()
        plt.show()

def analyze_simulation_run(stats_file):
    """Analyze a saved simulation run"""
    stats = StatsRecorder.load(stats_file)
    stats.visualize()

def analyze_batch_simulations(batch_dir):
    """Analyze multiple simulation runs from a batch directory"""
    # Find all stats files
    stats_files = sorted(glob.glob(os.path.join(batch_dir, "stats_*.pkl")))
    
    if not stats_files:
        print(f"No stats files found in {batch_dir}")
        return
    
    print(f"Found {len(stats_files)} stats files to analyze")
    
    # Load all stats
    all_stats = [StatsRecorder.load(file) for file in stats_files]
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 12))
    
    # Population over time (one plot with all iterations)
    plt.subplot(2, 2, 1)
    
    # Calculate overall x-axis (continuing ticks)
    cumulative_ticks = []
    last_tick = 0
    
    for i, stats in enumerate(all_stats):
        # Adjust ticks to be continuous
        ticks = [t + last_tick for t in stats.ticks]
        
        # Plot with different line styles and a label for each iteration
        plt.plot(ticks, stats.herbivore_counts, 'g-', alpha=0.5 + 0.1*i, 
                 linewidth=1, label=f'Herbivores {i+1}')
        plt.plot(ticks, stats.carnivore_counts, 'r-', alpha=0.5 + 0.1*i, 
                 linewidth=1, label=f'Carnivores {i+1}')
        
        # Update the last tick for next iteration
        if stats.ticks:
            last_tick = ticks[-1]
            
        # Store for later use
        cumulative_ticks.append(ticks)
    
    plt.title('Population over Time (All Iterations)')
    plt.xlabel('Simulation Ticks')
    plt.ylabel('Population')
    plt.grid(True)
    
    # Energy over time
    plt.subplot(2, 2, 2)
    for i, stats in enumerate(all_stats):
        if stats.herbivore_energies and stats.carnivore_energies:
            plt.plot(cumulative_ticks[i][:len(stats.herbivore_energies)], 
                     stats.herbivore_energies, 'g-', alpha=0.5 + 0.1*i, 
                     linewidth=1, label=f'Herb Energy {i+1}')
            plt.plot(cumulative_ticks[i][:len(stats.carnivore_energies)], 
                     stats.carnivore_energies, 'r-', alpha=0.5 + 0.1*i, 
                     linewidth=1, label=f'Carn Energy {i+1}')
    
    plt.title('Average Energy over Time')
    plt.xlabel('Simulation Ticks')
    plt.ylabel('Average Energy')
    plt.grid(True)
    
    # Grass count over time
    plt.subplot(2, 2, 3)
    for i, stats in enumerate(all_stats):
        plt.plot(cumulative_ticks[i], stats.grass_counts, 'g-', alpha=0.5 + 0.1*i, 
                 linewidth=1, label=f'Grass {i+1}')
    
    plt.title('Grass Count over Time')
    plt.xlabel('Simulation Ticks')
    plt.ylabel('Grass Count')
    plt.grid(True)
    
    # Grass growth rates
    plt.subplot(2, 2, 4)
    for i, stats in enumerate(all_stats):
        if stats.grass_growth_rates:
            plt.plot(cumulative_ticks[i][:len(stats.grass_growth_rates)], 
                     stats.grass_growth_rates, 'y-', alpha=0.5 + 0.1*i, 
                     linewidth=1, label=f'Growth Rate {i+1}')
    
    plt.title('Grass Growth Rate')
    plt.xlabel('Simulation Ticks')
    plt.ylabel('Growth Rate')
    plt.grid(True)
    
    # Final adjustments
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(batch_dir, 'batch_analysis.png')
    plt.savefig(output_file, dpi=150)
    print(f"Saved batch analysis to {output_file}")
    
    # Show the figure
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize simulation statistics')
    parser.add_argument('--file', type=str, default='simulation_stats.pkl', 
                        help='Statistics file to visualize')
    parser.add_argument('--batch', type=str, help='Directory containing batch simulation results')
    args = parser.parse_args()
    
    if args.batch:
        analyze_batch_simulations(args.batch)
    else:
        analyze_simulation_run(args.file)
