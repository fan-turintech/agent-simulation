"""
Integration tests for the Darwinian Simulation.

These tests run the full simulation from predefined starting states
and collect key metrics to verify simulation consistency.
"""
import sys
import os
import pytest
import random
import numpy as np
import json
from pathlib import Path
import statistics

# Add parent directory to path to import simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation import Simulation
from config import Config
from creature import Herbivore, Carnivore
from environment import Environment, Grass

# Fixed seed for reproducibility
FIXED_SEED = 42

# Test scenarios
SCENARIOS = {
    "balanced": {
        "herbivores": 20,
        "carnivores": 5,
        "grass": 100,
        "steps": 100
    },
    "herbivore_heavy": {
        "herbivores": 30,
        "carnivores": 3,
        "grass": 50,
        "steps": 100
    },
    "carnivore_heavy": {
        "herbivores": 10,
        "carnivores": 10,
        "grass": 150,
        "steps": 100
    }
}

# Reference values file path
REFERENCE_VALUES_PATH = os.path.join(os.path.dirname(__file__), "reference_values.json")

def create_test_simulation(scenario_name):
    """Create a simulation with the specified scenario configuration"""
    # Set fixed random seeds
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    
    # Get scenario configuration
    scenario = SCENARIOS[scenario_name]
    
    # Create new simulation
    simulation = Simulation(scenario["grass"], scenario["herbivores"], scenario["carnivores"])
    
    # # Reset creature list
    # simulation.creatures = []
    #
    # # Create herbivores with fixed positions
    # for i in range(scenario["herbivores"]):
    #     # Deterministic positioning for reproducibility
    #     x = Config.SCREEN_WIDTH * (0.1 + 0.8 * i / max(1, scenario["herbivores"]))
    #     y = Config.SCREEN_HEIGHT * 0.3
    #     direction = (i * 30) % 360  # Deterministic directions
    #     herbivore = Herbivore(x, y, direction=direction)
    #     simulation.creatures.append(herbivore)
    #
    # # Create carnivores with fixed positions
    # for i in range(scenario["carnivores"]):
    #     # Deterministic positioning for reproducibility
    #     x = Config.SCREEN_WIDTH * (0.2 + 0.6 * i / max(1, scenario["carnivores"]))
    #     y = Config.SCREEN_HEIGHT * 0.7
    #     direction = (i * 45) % 360  # Deterministic directions
    #     carnivore = Carnivore(x, y, direction=direction)
    #     simulation.creatures.append(carnivore)
    #
    # # Reset environment
    # simulation.environment.grass = []
    # simulation.environment.grass_positions = set()
    #
    # # Create grass in a deterministic grid pattern
    # grid_size = int(math.sqrt(scenario["grass"]))
    # cell_width = Config.SCREEN_WIDTH / (grid_size + 1)
    # cell_height = Config.SCREEN_HEIGHT / (grid_size + 1)
    #
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         if len(simulation.environment.grass) < scenario["grass"]:
    #             x = cell_width * (i + 1)
    #             y = cell_height * (j + 1)
    #             grass = Grass(x, y)
    #             simulation.environment.grass.append(grass)
    #
    #             # Add to position tracking
    #             pos_key = (int(x) // simulation.environment.grid_size,
    #                        int(y) // simulation.environment.grid_size)
    #             simulation.environment.grass_positions.add(pos_key)
    
    return simulation

def run_simulation_steps(simulation, steps):
    """Run the simulation for a specified number of steps"""
    # Set fixed random seeds before each run
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    
    # Create history lists to track metrics
    herbivore_counts = []
    carnivore_counts = []
    grass_counts = []
    
    # Run simulation for specified steps
    for i in range(steps):
        # Update simulation
        simulation.update()
        
        # Record metrics
        herbivore_counts.append(simulation.count_herbivores())
        carnivore_counts.append(simulation.count_carnivores())
        grass_counts.append(simulation.count_grass())
    
    # Compile results into a metrics dictionary
    metrics = {
        "final_herbivores": herbivore_counts[-1],
        "final_carnivores": carnivore_counts[-1],
        "final_grass": grass_counts[-1],
        "herbivore_peak": max(herbivore_counts),
        "carnivore_peak": max(carnivore_counts),
        "grass_peak": max(grass_counts),
        "herbivore_min": min(herbivore_counts),
        "carnivore_min": min(carnivore_counts),
        "grass_min": min(grass_counts),
        "herbivore_avg": statistics.mean(herbivore_counts),
        "carnivore_avg": statistics.mean(carnivore_counts),
        "grass_avg": statistics.mean(grass_counts)
    }
    
    return metrics

def load_reference_values():
    """Load reference values from file, or create a new file if it doesn't exist"""
    if os.path.exists(REFERENCE_VALUES_PATH):
        with open(REFERENCE_VALUES_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_reference_values(reference_values):
    """Save reference values to file"""
    with open(REFERENCE_VALUES_PATH, 'w') as f:
        json.dump(reference_values, f, indent=2)

def generate_reference_values():
    """Generate reference values for all scenarios"""
    reference_values = {}
    
    for scenario_name in SCENARIOS:
        # Create test simulation for this scenario
        simulation = create_test_simulation(scenario_name)
        
        # Run simulation and collect metrics
        steps = SCENARIOS[scenario_name]["steps"]
        metrics = run_simulation_steps(simulation, steps)
        
        # Store metrics for this scenario
        reference_values[scenario_name] = metrics
    
    # Save reference values
    save_reference_values(reference_values)
    
    print("\nReference values generated and saved to:", REFERENCE_VALUES_PATH)
    print("Integration tests will now use these values for verification.")
    print("Re-run with --generate-references to update reference values.\n")
    
    return reference_values

import math

class TestIntegration:
    """Integration tests for simulation consistency"""
    
    @classmethod
    def setup_class(cls):
        """Set up test class - load or generate reference values"""
        # Check if called with --generate-references
        if any(arg == '--generate-references' for arg in sys.argv):
            cls.reference_values = generate_reference_values()
        else:
            cls.reference_values = load_reference_values()
            
            # If no reference values exist, generate them
            if not cls.reference_values:
                print("\nNo reference values found. Generating new reference values...")
                cls.reference_values = generate_reference_values()
    
    @pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
    def test_simulation_consistency(self, scenario_name):
        """Test that simulation produces consistent results across runs"""
        print(f"\nRunning integration test for scenario: {scenario_name}")
        
        # Get reference values for this scenario
        reference = self.reference_values.get(scenario_name)
        assert reference, f"Missing reference values for {scenario_name}"
        
        # Create test simulation
        simulation = create_test_simulation(scenario_name)
        
        # Run simulation
        steps = SCENARIOS[scenario_name]["steps"]
        metrics = run_simulation_steps(simulation, steps)
        
        # Print metrics for debugging
        print(f"  Reference metrics: {reference}")
        print(f"  Current metrics: {metrics}")
        
        # Check key metrics against reference values
        assert abs(metrics["herbivore_avg"] - reference["herbivore_avg"]) < 2, "Herbivore avg count differs"
        assert abs(metrics["carnivore_avg"] - reference["carnivore_avg"]) < 2, "Carnivore avg count differs"
        assert abs(metrics["grass_avg"] - reference["grass_avg"]) < 10, "Grass avg count differs"
        
        # Check population peaks and lows
        # assert metrics["herbivore_peak"] == reference["herbivore_peak"], "Herbivore peak differs"
        # assert metrics["carnivore_peak"] == reference["carnivore_peak"], "Carnivore peak differs"
        # assert metrics["grass_peak"] == reference["grass_peak"], "Grass peak differs"
        # assert metrics["herbivore_min"] == reference["herbivore_min"], "Herbivore min differs"
        # assert metrics["carnivore_min"] == reference["carnivore_min"], "Carnivore min differs"
        # assert metrics["grass_min"] == reference["grass_min"], "Grass min differs"

if __name__ == "__main__":
    # If run directly, just generate reference values
    generate_reference_values()
