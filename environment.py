import random
import pygame
import numpy as np
from config import Config

class Grass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = Config.GRASS_ENERGY
        self.size = Config.GRASS_SIZE
        self.color = Config.GRASS_COLOR
    
    def render(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)

    def __repr__(self):
        return f"grass @ ({self.x}, {self.y})"

class Environment:
    def __init__(self, grass_count=Config.MAX_GRASS//2):
        self.grass = []
        self.width = Config.SCREEN_WIDTH
        self.height = Config.SCREEN_HEIGHT
        
        # Pre-compute grid for faster grass placement - moved before initialize_grass
        self.grid_size = max(Config.GRASS_SIZE * 2, 15)  # Grid cell size based on grass size
        self.grid_width = self.width // self.grid_size
        self.grid_height = self.height // self.grid_size
        self.grid = [[False for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        self.grass_positions = set()  # Fast lookup for existing grass positions
        
        # Now initialize grass after grid variables are set up
        self.initialize_grass(grass_count)
        
        # Track grass stats
        self.grass_growth_history = []  # Track growth rate history
    
    def initialize_grass(self, grass_count):
        # Create initial grass
        initial_count = min(grass_count, Config.MAX_GRASS)
        for _ in range(initial_count):
            self.add_grass()
    
    def add_grass(self):
        """Add a single grass at a random position if under the maximum count"""
        if len(self.grass) >= Config.MAX_GRASS:
            return False
            
        # Find an empty cell using the grid system
        attempts = 0
        max_attempts = 10  # Limit the number of placement attempts
        
        while attempts < max_attempts:
            # Pick random grid cell
            grid_x = random.randint(0, self.grid_width - 1)
            grid_y = random.randint(0, self.grid_height - 1)
            
            # Convert to world coordinates with some randomness within the cell
            x = grid_x * self.grid_size + random.randint(Config.GRASS_SIZE, self.grid_size - Config.GRASS_SIZE)
            y = grid_y * self.grid_size + random.randint(Config.GRASS_SIZE, self.grid_size - Config.GRASS_SIZE)
            
            # Ensure it's within bounds
            x = min(max(x, Config.GRASS_SIZE), self.width - Config.GRASS_SIZE)
            y = min(max(y, Config.GRASS_SIZE), self.height - Config.GRASS_SIZE)
            
            # Check if position is already taken
            pos_key = (x // self.grid_size, y // self.grid_size)
            if pos_key not in self.grass_positions:
                # Add the new grass
                self.grass.append(Grass(x, y))
                self.grass_positions.add(pos_key)
                return True
                
            attempts += 1
            
        return False  # Could not find a suitable position
    
    def calculate_adaptive_growth_rate(self):
        """Calculate growth rate based on current grass count"""
        current_grass = len(self.grass)
        max_grass = Config.MAX_GRASS
        
        # Calculate grass saturation ratio
        saturation_ratio = current_grass / max_grass
        
        # If we're close to max capacity (over 80%), use exponential dropoff
        if saturation_ratio > 0.8:
            # Formula creates a steeper decline as saturation approaches 1
            # At 100% saturation, growth rate will be close to 0
            growth_rate = Config.BASE_GRASS_GROWTH_RATE * (1 - saturation_ratio) ** 3
        # If grass count is below minimum, increase growth rate more aggressively
        elif current_grass < Config.MIN_GRASS:
            min_deficit = 1.0 - (current_grass / Config.MIN_GRASS)
            # Scale from base to max growth rate based on how far below minimum
            growth_rate = Config.BASE_GRASS_GROWTH_RATE + (
                (Config.MAX_GRASS_GROWTH_RATE - Config.BASE_GRASS_GROWTH_RATE) * min_deficit
            )
        else:
            # Normal scaling between base and slightly above base for mid-range
            grass_deficit_ratio = 1.0 - saturation_ratio
            growth_rate = Config.BASE_GRASS_GROWTH_RATE * grass_deficit_ratio * 1.5
        
        # Track history for possible visualization
        self.grass_growth_history.append(growth_rate)
        if len(self.grass_growth_history) > 1000:
            self.grass_growth_history = self.grass_growth_history[-1000:]
            
        return growth_rate
    
    def update(self):
        """Update the environment for a single time step"""
        # Calculate adaptive growth rate
        growth_rate = self.calculate_adaptive_growth_rate()
        
        # Calculate how many grass to try to add this update
        new_grass_count = int(growth_rate)
        
        # Add fractional part probabilistically 
        if random.random() < (growth_rate % 1):
            new_grass_count += 1
        
        # Batch add the calculated number of grass
        for _ in range(new_grass_count):
            self.add_grass()
        
        # Additional random spawn chance
        if random.random() < Config.GRASS_RANDOM_SPAWN_CHANCE:
            self.add_grass()
    
    def render(self, surface):
        # Batch render all grass at once
        for grass in self.grass:
            grass.render(surface)
    
    def remove_grass(self, grass):
        """Remove a grass instance from the environment"""
        if grass in self.grass:
            # Remove from positions set to allow new grass to grow there
            pos_key = (grass.x // self.grid_size, grass.y // self.grid_size)
            if pos_key in self.grass_positions:
                self.grass_positions.remove(pos_key)
            self.grass.remove(grass)
    
    def get_grass_count(self):
        return len(self.grass)
    
    def get_current_growth_rate(self):
        """Return the current adaptive growth rate"""
        if self.grass_growth_history:
            return self.grass_growth_history[-1]
        return Config.BASE_GRASS_GROWTH_RATE
