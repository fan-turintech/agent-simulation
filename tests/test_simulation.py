import sys
import os
import pytest
import tempfile

# Add parent directory to path to import simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation import Simulation
from creature import Herbivore, Carnivore
from config import Config

class TestSimulation:
    """Test cases for the Simulation class"""
    
    def setup_method(self):
        """Set up test simulation"""
        self.simulation = Simulation()
    
    def test_initialization(self):
        """Test simulation initialization"""
        assert hasattr(self.simulation, 'environment')
        assert hasattr(self.simulation, 'creatures')
        assert hasattr(self.simulation, 'tick')
        assert self.simulation.tick == 0
        
        # Check creature initialization
        herbivores = [c for c in self.simulation.creatures if isinstance(c, Herbivore)]
        carnivores = [c for c in self.simulation.creatures if isinstance(c, Carnivore)]
        
        assert len(herbivores) == Config.INITIAL_HERBIVORES
        assert len(carnivores) == Config.INITIAL_CARNIVORES
    
    def test_update(self):
        """Test simulation update"""
        # Track initial state
        initial_tick = self.simulation.tick
        initial_herb_count = self.simulation.count_herbivores()
        initial_carn_count = self.simulation.count_carnivores()
        
        # Perform update
        self.simulation.update()
        
        # Check tick incremented
        assert self.simulation.tick == initial_tick + 1
        
        # Check histories updated
        assert len(self.simulation.herbivore_count_history) == 1
        assert len(self.simulation.carnivore_count_history) == 1
        assert len(self.simulation.grass_count_history) == 1
    
    def test_population_maintenance(self):
        """Test minimum population maintenance"""
        # Remove all herbivores
        self.simulation.creatures = [c for c in self.simulation.creatures 
                                     if not isinstance(c, Herbivore)]
        
        # Check count is zero
        assert self.simulation.count_herbivores() == 0
        
        # Update simulation
        self.simulation.update()
        
        # Check minimum herbivores spawned
        assert self.simulation.count_herbivores() >= Config.MINIMUM_HERBIVORES
        
        # Remove all carnivores
        self.simulation.creatures = [c for c in self.simulation.creatures 
                                     if not isinstance(c, Carnivore)]
        
        # Check count is zero
        assert self.simulation.count_carnivores() == 0
        
        # Update simulation
        self.simulation.update()
        
        # Check minimum carnivores spawned
        assert self.simulation.count_carnivores() >= Config.MINIMUM_CARNIVORES
    
    def test_save_load(self):
        """Test saving and loading simulation state"""
        # Run simulation for a few ticks to get some history
        for _ in range(10):
            self.simulation.update()
        
        # Create temp file for save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp:
            save_path = temp.name
        
        try:
            # Save simulation
            saved = self.simulation.save_simulation(save_path)
            assert saved
            
            # Get state before loading
            tick_before = self.simulation.tick
            herb_count = self.simulation.count_herbivores()
            carn_count = self.simulation.count_carnivores()
            grass_count = self.simulation.count_grass()
            
            # Load simulation
            loaded_sim = Simulation.load_simulation(save_path)
            
            # Check loaded simulation matches original
            assert loaded_sim is not None
            assert loaded_sim.tick == tick_before
            assert loaded_sim.count_herbivores() == herb_count
            assert loaded_sim.count_carnivores() == carn_count
            assert loaded_sim.count_grass() == grass_count
            assert len(loaded_sim.herbivore_count_history) == len(self.simulation.herbivore_count_history)
            
        finally:
            # Clean up temp file
            os.unlink(save_path)
