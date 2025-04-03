# Darwinian Evolution Simulation

This project simulates natural evolution in a 2D environment with herbivorous and carnivorous creatures that evolve using neural networks and natural selection.

## Overview

The simulation features:
- Creatures that use neural networks to make decisions based on visual input
- Herbivores that consume grass and avoid predators
- Carnivores that hunt herbivores
- Natural selection through energy dynamics and reproduction
- Mutation of neural networks during reproduction
- Real-time visualization and statistics
- Save and load functionality to continue simulations later

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository or download the source code
2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python main.py
```

Or use the launcher script which checks for required dependencies:
```bash
python run.py
```

### Controls

- **Space**: Pause/Resume simulation
- **Up Arrow**: Increase simulation speed (or navigate in save menu)
- **Down Arrow**: Decrease simulation speed (or navigate in save menu)
- **H**: Toggle help screen
- **F**: Toggle save/load menu
- **S**: Save statistics (or create new save in save menu)
- **L**: Load selected save file (in save menu)
- **Escape**: Quit the simulation (or close menu)

### Saving and Loading

You can save your simulation progress at any time by:
1. Pressing **F** to open the save/load menu
2. Pressing **S** to create a new save file
3. The file will be saved with a timestamp in the current directory

To load a previously saved simulation:
1. Press **F** to open the save/load menu
2. Use **Up/Down** arrows to select a save file
3. Press **L** to load the selected file

## How It Works

### Environment
The environment consists of a 2D canvas where grass grows randomly at a configurable rate. Grass provides energy to herbivores when consumed.

### Creatures
There are two types of creatures:

**Herbivores** (Green):
- Feed on grass for energy
- Try to avoid carnivores
- Move slower than carnivores

**Carnivores** (Red):
- Hunt and consume herbivores for energy
- Move faster than herbivores

### Vision System
Each creature has a 180-degree cone of vision in front of them, divided into "rays" that detect:
- Grass (for herbivores)
- Other creatures
- Boundaries of the environment

### Neural Network
Creatures make decisions using a neural network with:
- Input layer: vision data + internal energy state
- Hidden layer: processes inputs
- Output layer: speed and turning angle

### Reproduction and Mutation
When a creature reaches sufficient energy:
- It has a chance to reproduce
- The offspring inherits a mutated copy of the parent's neural network
- Energy is split between parent and offspring
- Mutations allow for evolution of more successful behaviors over time

### Energy Dynamics
- Creatures consume energy constantly
- Movement costs more energy at higher speeds
- Creatures die when energy reaches zero
- Energy is gained by consuming food (grass or other creatures)

## Configuring the Simulation

You can modify parameters in the `config.py` file to change:
- Screen size and FPS
- Grass growth rate and maximum amount
- Initial creature populations
- Creature properties (speed, vision, energy, etc.)
- Reproduction thresholds and chances
- Mutation rates

## Advanced Usage

### Headless Mode

You can run the simulation without visualization for a specified number of steps:

```bash
python main.py --headless --steps 5000 --output "my_simulation.pkl"
```

Options:
- `--headless`: Run without visualization
- `--steps`: Number of simulation steps to run (default: 1000)
- `--input`: Load initial state from a save file
- `--output`: Save final state to this file

### Batch Simulations

For running multiple sequential simulations:

```bash
python batch_simulate.py --steps 2000 --iterations 5 --continue --name "evolution_experiment"
```

Options:
- `--steps`: Steps per iteration (default: 1000)
- `--iterations`: Number of simulations to run (default: 5)
- `--continue`: Each simulation continues from the previous one's state
- `--name`: Base name for the output directory

Results are saved in a timestamped directory and include:
- Save files for each iteration
- Statistics for each iteration
- Combined visualization of all iterations
