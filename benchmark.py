#!/usr/bin/env python3
"""
Benchmarking tool for the Darwinian Simulation

This script runs the simulation with a fixed random seed and initial state
to produce reproducible results for performance testing.
"""

import os
import sys
import time
import random
import numpy as np
import pickle
import argparse
from pathlib import Path
import statistics
from datetime import datetime
import cProfile
import pstats
import io

# Import project modules
from simulation import Simulation
from config import Config
from environment import Environment
from creature import Herbivore, Carnivore

# Benchmark scenarios
SCENARIOS = {
    "balanced": {
        "description": "Balanced ecosystem with moderate populations",
        "herbivores": 30,
        "carnivores": 10,
        "grass": 150,
        "steps": 1000
    },
    "herbivore_heavy": {
        "description": "Ecosystem dominated by herbivores",
        "herbivores": 60, 
        "carnivores": 5,
        "grass": 100,
        "steps": 1000
    },
    "carnivore_heavy": {
        "description": "Ecosystem dominated by carnivores",
        "herbivores": 30,
        "carnivores": 20,
        "grass": 200,
        "steps": 1000
    },
    "sparse": {
        "description": "Low population density",
        "herbivores": 10,
        "carnivores": 3,
        "grass": 50,
        "steps": 1000
    },
    "dense": {
        "description": "High population density",
        "herbivores": 100,
        "carnivores": 30,
        "grass": 250,
        "steps": 1000
    },
    "long_run": {
        "description": "Extended simulation for stability testing",
        "herbivores": 30,
        "carnivores": 10,
        "grass": 150,
        "steps": 5000
    }
}

def set_random_seed(seed):
    """Set random seeds for all random number generators"""
    random.seed(seed)
    np.random.seed(seed)

def create_benchmark_state(scenario_name, seed=42):
    """Create a fixed initial state for benchmarking"""
    scenario = SCENARIOS[scenario_name]
    
    # Set reproducible random seed
    set_random_seed(seed)
    
    # Create a fresh simulation
    simulation = Simulation()
    
    # Clear existing creatures
    simulation.creatures = []
    
    # Create fixed number of herbivores
    for i in range(scenario["herbivores"]):
        x = Config.SCREEN_WIDTH * (0.1 + 0.8 * i / scenario["herbivores"])
        y = Config.SCREEN_HEIGHT * (0.3 + 0.4 * (i % 3) / 3)
        herbivore = Herbivore(x, y, direction=random.uniform(0, 360))
        simulation.creatures.append(herbivore)
    
    # Create fixed number of carnivores
    for i in range(scenario["carnivores"]):
        x = Config.SCREEN_WIDTH * (0.2 + 0.6 * i / scenario["carnivores"])
        y = Config.SCREEN_HEIGHT * (0.2 + 0.6 * (i % 3) / 3)
        carnivore = Carnivore(x, y, direction=random.uniform(0, 360))
        simulation.creatures.append(carnivore)
    
    # Clear existing grass
    simulation.environment.grass = []
    simulation.environment.grass_positions = set()
    
    # Create fixed amount of grass
    for _ in range(scenario["grass"]):
        simulation.environment.add_grass()
    
    return simulation

def run_benchmark(simulation, steps, seed=42):
    """Run the simulation for a specified number of steps"""
    # Set reproducible random seed
    set_random_seed(seed)
    
    # Reset tick count
    simulation.tick = 0
    
    # Track timing information
    start_time = time.time()
    step_times = []
    
    # Run simulation for specified number of steps
    for i in range(steps):
        step_start = time.time()
        simulation.update()
        step_end = time.time()
        step_times.append(step_end - step_start)
        
        # Progress updates
        if (i+1) % 100 == 0 or i == 0 or i == steps-1:
            elapsed = time.time() - start_time
            herbivores = simulation.count_herbivores()
            carnivores = simulation.count_carnivores()
            grass = simulation.count_grass()
            print(f"Step {i+1}/{steps}: {elapsed:.2f}s elapsed - "
                  f"H: {herbivores}, C: {carnivores}, G: {grass}, "
                  f"Avg step: {statistics.mean(step_times[-100:]) * 1000:.2f}ms")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_step_time = statistics.mean(step_times) * 1000  # Convert to ms
    max_step_time = max(step_times) * 1000
    min_step_time = min(step_times) * 1000
    median_step_time = statistics.median(step_times) * 1000
    
    # Create results dictionary
    results = {
        "total_time": total_time,
        "steps": steps,
        "avg_step_time_ms": avg_step_time,
        "median_step_time_ms": median_step_time,
        "min_step_time_ms": min_step_time,
        "max_step_time_ms": max_step_time,
        "final_herbivores": simulation.count_herbivores(),
        "final_carnivores": simulation.count_carnivores(),
        "final_grass": simulation.count_grass(),
        "step_times": step_times,
        "timestamp": datetime.now().isoformat()
    }
    
    return results

def save_benchmark_results(results, scenario_name, output_dir="benchmark_results"):
    """Save benchmark results to file"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/benchmark_{scenario_name}_{timestamp}.pkl"
    
    # Save results
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filename}")
    return filename

def load_benchmark_results(filename):
    """Load benchmark results from file"""
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def compare_benchmark_results(old_file, new_file):
    """Compare two benchmark results and print differences"""
    old_results = load_benchmark_results(old_file)
    new_results = load_benchmark_results(new_file)
    
    print("\nBenchmark Comparison:")
    print(f"Old: {old_file}")
    print(f"New: {new_file}")
    print("-" * 50)
    
    # Compare timing metrics
    time_diff = (new_results["total_time"] - old_results["total_time"])
    time_pct = (time_diff / old_results["total_time"]) * 100
    
    print(f"Total Time: {old_results['total_time']:.2f}s → {new_results['total_time']:.2f}s "
          f"({time_diff:.2f}s, {time_pct:+.2f}%)")
    
    avg_diff = new_results["avg_step_time_ms"] - old_results["avg_step_time_ms"]
    avg_pct = (avg_diff / old_results["avg_step_time_ms"]) * 100
    
    print(f"Avg Step Time: {old_results['avg_step_time_ms']:.2f}ms → {new_results['avg_step_time_ms']:.2f}ms "
          f"({avg_diff:.2f}ms, {avg_pct:+.2f}%)")
    
    print(f"Median Step: {old_results['median_step_time_ms']:.2f}ms → {new_results['median_step_time_ms']:.2f}ms")
    print(f"Min Step: {old_results['min_step_time_ms']:.2f}ms → {new_results['min_step_time_ms']:.2f}ms")
    print(f"Max Step: {old_results['max_step_time_ms']:.2f}ms → {new_results['max_step_time_ms']:.2f}ms")
    
    # Compare population metrics
    print(f"Final Herbivores: {old_results['final_herbivores']} → {new_results['final_herbivores']}")
    print(f"Final Carnivores: {old_results['final_carnivores']} → {new_results['final_carnivores']}")
    print(f"Final Grass: {old_results['final_grass']} → {new_results['final_grass']}")
    
    # Overall assessment
    if abs(time_pct) < 1:
        status = "EQUIVALENT"
    elif time_pct < -10:
        status = "MAJOR IMPROVEMENT"
    elif time_pct < 0:
        status = "IMPROVEMENT"
    elif time_pct < 10:
        status = "REGRESSION"
    else:
        status = "MAJOR REGRESSION"
    
    print(f"\nOverall Performance Change: {status} ({time_pct:+.2f}%)")

def run_profiling(scenario_name, seed=42, steps=1000):
    """Run profiling on the simulation"""
    # Create benchmark state
    simulation = create_benchmark_state(scenario_name, seed)
    
    # Set up profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Run benchmark
    set_random_seed(seed)
    for _ in range(steps):
        simulation.update()
    
    # Disable profiler
    pr.disable()
    
    # Create output directory if it doesn't exist
    output_dir = "benchmark_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save profiling results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_file = f"{output_dir}/profile_{scenario_name}_{timestamp}.prof"
    
    # Save stats to file
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.dump_stats(profile_file)
    
    # Print top 20 results
    ps = pstats.Stats(pr).sort_stats(sortby)
    print("\nProfiling Results (Top 20):")
    ps.print_stats(20)
    
    print(f"\nDetailed profiling results saved to {profile_file}")
    return profile_file

def main():
    """Main entry point for the benchmark tool"""
    parser = argparse.ArgumentParser(description="Benchmark the Darwinian Simulation")
    parser.add_argument("--scenario", type=str, default="balanced", 
                        choices=list(SCENARIOS.keys()),
                        help="Benchmark scenario to run")
    parser.add_argument("--steps", type=int, 
                        help="Override number of steps to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible results")
    parser.add_argument("--profile", action="store_true",
                        help="Run with profiling enabled")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("OLD", "NEW"),
                        help="Compare two benchmark result files")
    parser.add_argument("--list", action="store_true",
                        help="List available scenarios and exit")
    parser.add_argument("--save", action="store_true",
                        help="Save benchmark results to file")
    parser.add_argument("--output", type=str,
                        help="Specify output filename for benchmark results")
    args = parser.parse_args()
    
    # List scenarios if requested
    if args.list:
        print("Available benchmark scenarios:")
        for name, details in SCENARIOS.items():
            print(f"  {name}: {details['description']}")
            print(f"    Herbivores: {details['herbivores']}, Carnivores: {details['carnivores']}, "
                  f"Grass: {details['grass']}, Steps: {details['steps']}")
        return 0
    
    # Compare results if requested
    if args.compare:
        compare_benchmark_results(args.compare[0], args.compare[1])
        return 0
    
    # Get scenario details
    if args.scenario not in SCENARIOS:
        print(f"Error: Unknown scenario '{args.scenario}'")
        return 1
    
    scenario = SCENARIOS[args.scenario]
    steps = args.steps if args.steps is not None else scenario["steps"]
    
    print(f"Running benchmark: {args.scenario}")
    print(f"Description: {scenario['description']}")
    print(f"Initial state: {scenario['herbivores']} herbivores, "
          f"{scenario['carnivores']} carnivores, {scenario['grass']} grass")
    print(f"Steps: {steps}, Random seed: {args.seed}")
    
    # Run profiling if requested
    if args.profile:
        print("\nRunning with profiling...")
        profile_file = run_profiling(args.scenario, args.seed, steps)
        return 0
    
    # Create benchmark state
    simulation = create_benchmark_state(args.scenario, args.seed)
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = run_benchmark(simulation, steps, args.seed)
    
    # Print results summary
    print("\nBenchmark Results:")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average step time: {results['avg_step_time_ms']:.2f} ms")
    print(f"Median step time: {results['median_step_time_ms']:.2f} ms")
    print(f"Min/Max step time: {results['min_step_time_ms']:.2f}/{results['max_step_time_ms']:.2f} ms")
    print(f"Final populations - Herbivores: {results['final_herbivores']}, "
          f"Carnivores: {results['final_carnivores']}, Grass: {results['final_grass']}")
    
    # Save results only if explicitly requested
    if args.save:
        if args.output:
            # Use custom output filename if provided
            save_path = args.output
            # Add .pkl extension if not already present
            if not save_path.endswith('.pkl'):
                save_path += '.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to {save_path}")
        else:
            # Use default naming convention
            save_benchmark_results(results, args.scenario)
    else:
        print("\nResults not saved. Use --save to save benchmark results.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
