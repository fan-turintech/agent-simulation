import sys
import os
import pytest
import numpy as np

# Add parent directory to path to import simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import Environment, Grass
from config import Config

class TestEnvironment:
    """Test cases for the Environment class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.env = Environment()
    
    def test_initialization(self):
        """Test environment initialization"""
        assert hasattr(self.env, 'grass')
        assert isinstance(self.env.grass, list)
        assert hasattr(self.env, 'width')
        assert hasattr(self.env, 'height')
        assert hasattr(self.env, 'grass_positions')
        assert self.env.width == Config.SCREEN_WIDTH
        assert self.env.height == Config.SCREEN_HEIGHT
        
        # Check initial grass population
        assert len(self.env.grass) > 0
        assert len(self.env.grass) <= Config.MAX_GRASS
    
    def test_add_grass(self):
        """Test adding grass to environment"""
        # Clear existing grass
        initial_count = len(self.env.grass)
        self.env.grass = []
        self.env.grass_positions = set()
        
        # Add new grass
        added = self.env.add_grass()
        
        # Check grass was added
        assert added
        assert len(self.env.grass) == 1
        assert isinstance(self.env.grass[0], Grass)
        
        # Check position is within bounds
        assert 0 <= self.env.grass[0].x <= self.env.width
        assert 0 <= self.env.grass[0].y <= self.env.height
        
        # Check grass has correct properties
        assert self.env.grass[0].energy == Config.GRASS_ENERGY
        assert self.env.grass[0].size == Config.GRASS_SIZE
        assert self.env.grass[0].color == Config.GRASS_COLOR
    
    def test_max_grass_limit(self):
        """Test grass doesn't exceed maximum limit"""
        # Set environment to maximum capacity
        self.env.grass = []
        self.env.grass_positions = set()
        
        for _ in range(Config.MAX_GRASS):
            self.env.add_grass()
        
        # Try to add one more grass
        added = self.env.add_grass()
        
        # Should not add more grass
        assert not added
        assert len(self.env.grass) == Config.MAX_GRASS
    
    def test_remove_grass(self):
        """Test removing grass from environment"""
        # Add grass if environment is empty
        if not self.env.grass:
            self.env.add_grass()
        
        # Get first grass
        grass = self.env.grass[0]
        
        # Remember position
        pos_key = (grass.x // self.env.grid_size, grass.y // self.env.grid_size)
        
        # Remove grass
        self.env.remove_grass(grass)
        
        # Check grass was removed
        assert grass not in self.env.grass
        assert pos_key not in self.env.grass_positions
    
    def test_calculate_adaptive_growth_rate(self):
        """Test adaptive growth rate calculation"""
        # Ensure environment has the grass_growth_history attribute
        if not hasattr(self.env, 'grass_growth_history'):
            self.env.grass_growth_history = []
        
        # Test with more manageable grass levels
        # Just test the extremes and a middle value
        grass_levels = [
            0,                      # Empty - should give high growth rate
            Config.MAX_GRASS // 2,  # Half capacity - should give medium growth rate 
            Config.MAX_GRASS,       # Full capacity - should give low growth rate
        ]
        
        growth_rates = []
        for level in grass_levels:
            # Simulate this grass level
            self.env.grass = [Grass(0, 0) for _ in range(level)]
            # Get growth rate
            rate = self.env.calculate_adaptive_growth_rate()
            growth_rates.append(rate)
            
        # Check we got the right number of results
        assert len(growth_rates) == 3
        
        # Check growth rates decrease as grass increases
        assert growth_rates[0] > growth_rates[1]  # Empty > Half
        assert growth_rates[1] > growth_rates[2]  # Half > Full
        
        # Full capacity should give very low growth rate 
        # (not necessarily zero, but much lower than base)
        assert growth_rates[2] < Config.BASE_GRASS_GROWTH_RATE / 2
