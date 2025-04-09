import sys
import os
import pytest
import numpy as np
import random
import math

# Add parent directory to path to import simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creature import Herbivore, Carnivore
from environment import Grass
from simulation import Simulation
from config import Config

class TestCreatures:
    """Test cases for Creature classes"""
    
    def setup_method(self):
        """Set up test creatures and simulation"""
        self.herbivore = Herbivore(100, 100)
        self.carnivore = Carnivore(200, 200)
        self.simulation = Simulation(10, 0, 0)  # Create simulation with just grass
    
    def test_initialization(self):
        """Test creature initialization"""
        # Test herbivore
        assert self.herbivore.x == 100
        assert self.herbivore.y == 100
        assert 0 <= self.herbivore.direction < 360
        assert self.herbivore.energy == Config.HERBIVORE_INITIAL_ENERGY
        assert self.herbivore.age == 0
        assert hasattr(self.herbivore, 'brain')
        
        # Test carnivore
        assert self.carnivore.x == 200
        assert self.carnivore.y == 200
        assert 0 <= self.carnivore.direction < 360
        assert self.carnivore.energy == Config.CARNIVORE_INITIAL_ENERGY
        assert self.carnivore.age == 0
        assert hasattr(self.carnivore, 'brain')
    
    def test_thinking(self):
        """Test creature neural processing"""
        # Create test inputs
        test_inputs = [0] * (self.herbivore.get_vision_resolution() + 1)
        
        # Get outputs
        outputs = self.herbivore.think(test_inputs)
        
        # Should have 2 outputs (speed and turn)
        assert len(outputs) == 2
        assert 0 <= outputs[0] <= 1
        assert 0 <= outputs[1] <= 1
    
    def test_acting(self):
        """Test creature actions"""
        initial_x = self.herbivore.x
        initial_y = self.herbivore.y
        initial_energy = self.herbivore.energy
        
        # Set test outputs (full speed straight ahead)
        outputs = [1.0, 0.5]  # Max speed, no turn
        
        # Act
        self.herbivore.act(outputs)
        
        # Check movement
        assert self.herbivore.x != initial_x or self.herbivore.y != initial_y
        
        # Check energy consumption
        assert self.herbivore.energy < initial_energy
    
    def test_energy_dynamics(self):
        """Test creature energy dynamics"""
        # First, test energy consumption during act()
        self.herbivore.energy = Config.HERBIVORE_INITIAL_ENERGY + 10
        initial_energy = self.herbivore.energy
        
        # Execute act() to consume energy
        outputs = [0.5, 0.5]  # Half speed, no turn
        self.herbivore.act(outputs)
        
        # Check energy decreased
        assert self.herbivore.energy < initial_energy, "Energy should decrease after act()"
        
        # Now test eating grass via simulation update_creature
        self.herbivore.energy = Config.HERBIVORE_MAX_ENERGY / 2  # Set energy to half max
        before_eating = self.herbivore.energy
        
        # Add herbivore to simulation
        self.simulation.creatures.append(self.herbivore)
        
        # Create grass directly at herbivore's position to ensure it gets eaten
        grass = Grass(self.herbivore.x, self.herbivore.y)
        self.simulation.grass = [grass]
        self.simulation.grass_positions = {(int(grass.x) // self.simulation.grid_size, 
                                      int(grass.y) // self.simulation.grid_size)}
        
        # Use update_creature to trigger grass consumption
        self.simulation.update_creature(self.herbivore)
        
        # Verify energy increased and grass was removed
        assert self.herbivore.energy > before_eating, "Energy should increase after eating grass"
        assert len(self.simulation.grass) == 0, "Grass should be consumed"
    
    def test_reproduction(self):
        """Test creature reproduction"""
        # Give herbivore enough energy to reproduce
        self.herbivore.energy = Config.HERBIVORE_REPRODUCTION_THRESHOLD + 10
        
        # Store the original random function
        original_random_func = random.random
        
        try:
            # Replace random.random with a function that always returns 0
            def always_reproduce():
                return 0.0
            
            random.random = always_reproduce
            
            # Check reproduction ability
            assert self.herbivore.can_reproduce()
            
            # Get offspring
            offspring = self.herbivore.reproduce()
            
            # Verify offspring
            assert offspring is not None
            assert isinstance(offspring, Herbivore)
            assert offspring.energy == self.herbivore.energy  # Equal energy split
            assert offspring.brain is not self.herbivore.brain  # Different neural network
            
            # Parent's energy should be halved
            assert math.isclose(self.herbivore.energy, (Config.HERBIVORE_REPRODUCTION_THRESHOLD + 10) / 2)
            
        finally:
            # Restore the original random function
            random.random = original_random_func
    
    def test_predation(self):
        """Test carnivore hunting herbivores via simulation update_creature"""
        # Place herbivore next to carnivore
        self.herbivore.x = self.carnivore.x
        self.herbivore.y = self.carnivore.y
        
        # Add to simulation
        self.simulation.creatures = [self.herbivore, self.carnivore]
        
        # Track initial energy
        initial_energy = self.carnivore.energy
        herbivore_energy = self.herbivore.energy
        
        # Update carnivore through simulation (should eat herbivore)
        self.simulation.update_creature(self.carnivore)
        
        # Carnivore should gain energy (herbivore energy + bonus)
        expected_gain = min(herbivore_energy + 100,  # Fixed bonus energy is 100
                           Config.CARNIVORE_MAX_ENERGY - initial_energy)
        
        assert self.carnivore.energy == pytest.approx(initial_energy + expected_gain)
        
        # Herbivore should be dead (energy = 0)
        assert self.herbivore.energy == 0
