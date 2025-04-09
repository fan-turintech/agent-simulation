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
        # Check grass initialization
        assert hasattr(self.simulation, 'grass')
        assert isinstance(self.simulation.grass, list)
        
        # Check creature initialization
        assert hasattr(self.simulation, 'creatures')
        assert hasattr(self.simulation, 'tick')
        assert self.simulation.tick == 0
        
        # Check creature counts
        herbivores = [c for c in self.simulation.creatures if isinstance(c, Herbivore)]
        carnivores = [c for c in self.simulation.creatures if isinstance(c, Carnivore)]
        
        assert len(herbivores) == Config.INITIAL_HERBIVORES
        assert len(carnivores) == Config.INITIAL_CARNIVORES
        
        # Check grass management
        assert hasattr(self.simulation, 'grass_positions')
        assert hasattr(self.simulation, 'grid_size')
    
    def test_update(self):
        """Test simulation update"""
        # Track initial state
        initial_tick = self.simulation.tick
        initial_herb_count = self.simulation.count_herbivores()
        initial_carn_count = self.simulation.count_carnivores()
        initial_grass_count = self.simulation.get_grass_count()
        
        # Perform update
        self.simulation.update()
        
        # Check tick incremented
        assert self.simulation.tick == initial_tick + 1
        
        # Check histories updated
        assert len(self.simulation.herbivore_count_history) == 1
        assert len(self.simulation.carnivore_count_history) == 1
        assert len(self.simulation.grass_count_history) == 1
    
    def test_grass_management(self):
        """Test grass management functionalities"""
        # Test adding grass
        initial_grass = len(self.simulation.grass)
        self.simulation.add_grass()
        assert len(self.simulation.grass) > initial_grass
        
        # Test removing grass
        if self.simulation.grass:
            grass_to_remove = self.simulation.grass[0]
            self.simulation.remove_grass(grass_to_remove)
            assert grass_to_remove not in self.simulation.grass
        
        # Test grass growth rate calculation
        growth_rate = self.simulation.calculate_adaptive_growth_rate()
        assert 0 <= growth_rate <= Config.MAX_GRASS_GROWTH_RATE
        assert len(self.simulation.grass_growth_history) > 0
    
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
    
    def test_sensing(self):
        """Test sensing capabilities"""
        # Add a test creature
        herbivore = Herbivore(100, 100, direction=0)  # Facing right
        self.simulation.creatures.append(herbivore)
        
        # Add some grass in front of the herbivore
        for _ in range(3):
            self.simulation.grass = []
            x = herbivore.x + herbivore.get_vision_range() // 2
            y = herbivore.y
            self.simulation.grass.append(self.create_test_grass(x, y))
            
            # Test vision
            inputs = self.simulation.sense_environment_for_creature(herbivore)
            
            # Should have vision_resolution + 1 inputs
            assert len(inputs) == herbivore.get_vision_resolution() + 1
            # Ensure some vision ray detected grass
            assert any(i > 0 for i in inputs[:-1]), "Grass should be detected in vision"
            
            # Test vision data specifics
            vision_data = self.simulation.get_vision_for_creature(herbivore)
            assert len(vision_data) == herbivore.get_vision_resolution()
            
            # Add a carnivore and test the herbivore's sensing to make sure it detects danger
            carnivore = Carnivore(herbivore.x + 30, herbivore.y, direction=180)  # Facing left, towards herbivore
            self.simulation.creatures.append(carnivore)
            
            # Test vision with carnivore
            inputs = self.simulation.sense_environment_for_creature(herbivore)
            vision_data = inputs[:-1]  # Exclude energy input
            
            # Ensure herbivore detects carnivore (with negative value for danger)
            assert any(v < 0 for v in vision_data), "Carnivore should be detected as danger"
    
    def create_test_grass(self, x, y):
        """Create a grass instance at specified position and add to grass_positions"""
        from environment import Grass
        grass = Grass(x, y)
        pos_key = (int(x) // self.simulation.grid_size, int(y) // self.simulation.grid_size)
        self.simulation.grass_positions.add(pos_key)
        return grass
    
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
