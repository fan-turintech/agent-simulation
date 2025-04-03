import random
import pygame
import math
import pickle
import os
from config import Config
from environment import Environment
from creature import Herbivore, Carnivore

class Simulation:
    def __init__(self):
        # Initialize environment
        self.environment = Environment()
        
        # Initialize creatures
        self.creatures = []
        self.initialize_creatures()
        
        # Statistics
        self.tick = 0
        self.herbivore_count_history = []
        self.carnivore_count_history = []
        self.grass_count_history = []
        
    def initialize_creatures(self):
        # Create initial herbivores
        for _ in range(Config.INITIAL_HERBIVORES):
            x = random.randint(50, Config.SCREEN_WIDTH - 50)
            y = random.randint(50, Config.SCREEN_HEIGHT - 50)
            herbivore = Herbivore(x, y)
            self.creatures.append(herbivore)
        
        # Create initial carnivores
        for _ in range(Config.INITIAL_CARNIVORES):
            x = random.randint(50, Config.SCREEN_WIDTH - 50)
            y = random.randint(50, Config.SCREEN_HEIGHT - 50)
            carnivore = Carnivore(x, y)
            self.creatures.append(carnivore)
    
    def update(self):
        # Increment simulation tick
        self.tick += 1
        
        # Update environment
        self.environment.update()
        
        # Process creature sensing, thinking, and acting
        for creature in self.creatures:
            # Sense the environment
            inputs = creature.sense_environment(self.environment, self.creatures)
            
            # Think about what to do
            outputs = creature.think(inputs)
            
            # Take action based on thoughts
            creature.act(outputs)
        
        # Process creature interactions, deaths, and reproduction
        new_creatures = []
        
        for creature in self.creatures[:]:  # Make a copy to avoid modification during iteration
            # Update creature and check if it's still alive
            if not creature.update(self.environment, self.creatures):
                self.creatures.remove(creature)
                continue
            
            # Check for reproduction
            if creature.can_reproduce():
                offspring = creature.reproduce()
                if offspring:
                    new_creatures.append(offspring)
        
        # Add new creatures
        self.creatures.extend(new_creatures)
        
        # Check if we need to maintain minimum populations
        self.maintain_minimum_population()
        
        # Track statistics
        self.herbivore_count_history.append(self.count_herbivores())
        self.carnivore_count_history.append(self.count_carnivores())
        self.grass_count_history.append(self.count_grass())
        
        # Keep only recent history
        max_history = 1000
        if len(self.herbivore_count_history) > max_history:
            self.herbivore_count_history = self.herbivore_count_history[-max_history:]
            self.carnivore_count_history = self.carnivore_count_history[-max_history:]
            self.grass_count_history = self.grass_count_history[-max_history:]
    
    def maintain_minimum_population(self):
        """Ensure minimum populations of herbivores and carnivores are maintained"""
        herbivore_count = self.count_herbivores()
        carnivore_count = self.count_carnivores()
        
        # Check and restore herbivore population if needed
        if herbivore_count < Config.MINIMUM_HERBIVORES:
            needed = Config.MINIMUM_HERBIVORES - herbivore_count
            for _ in range(needed):
                self.spawn_new_herbivore()
                
        # Check and restore carnivore population if needed
        if carnivore_count < Config.MINIMUM_CARNIVORES:
            needed = Config.MINIMUM_CARNIVORES - carnivore_count
            for _ in range(needed):
                self.spawn_new_carnivore()
    
    def spawn_new_herbivore(self):
        """Spawn a new herbivore by cloning the best herbivore or creating a random one"""
        # Find the best herbivore based on energy
        herbivores = [c for c in self.creatures if isinstance(c, Herbivore)]
        
        # Calculate initial energy - use max energy minus 80 for population protection
        initial_energy = Config.HERBIVORE_MAX_ENERGY - 80
        
        if herbivores:
            # Sort by energy and get the best one
            best_herbivore = max(herbivores, key=lambda h: h.energy)
            
            # Create a new position away from the best one
            angle = random.uniform(0, 360)
            distance = best_herbivore._get_size() * 5
            new_x = best_herbivore.x + math.cos(math.radians(angle)) * distance
            new_y = best_herbivore.y + math.sin(math.radians(angle)) * distance
            
            # Ensure within boundaries
            new_x = max(0, min(new_x, Config.SCREEN_WIDTH))
            new_y = max(0, min(new_y, Config.SCREEN_HEIGHT))
            
            # Create a new herbivore with a mutated brain from the best one
            mutated_brain = best_herbivore.brain.mutate()
            new_herbivore = Herbivore(new_x, new_y, 
                                     direction=random.uniform(0, 360),
                                     energy=initial_energy,  # Use the calculated energy value
                                     brain=mutated_brain)  # Use a mutated brain
            
            print(f"Spawned new herbivore with mutations from best (energy: {best_herbivore.energy:.1f}, initial energy: {initial_energy})")
        else:
            # No herbivores left, create a random one
            new_x = random.randint(50, Config.SCREEN_WIDTH - 50)
            new_y = random.randint(50, Config.SCREEN_HEIGHT - 50)
            new_herbivore = Herbivore(new_x, new_y, energy=initial_energy)  # Use the calculated energy value
            print(f"Spawned new random herbivore (initial energy: {initial_energy})")
            
        self.creatures.append(new_herbivore)
    
    def spawn_new_carnivore(self):
        """Spawn a new carnivore by cloning the best carnivore or creating a random one"""
        # Find the best carnivore based on energy
        carnivores = [c for c in self.creatures if isinstance(c, Carnivore)]
        
        # Calculate initial energy - use max energy minus 80 for population protection
        initial_energy = Config.CARNIVORE_MAX_ENERGY - 80
        
        if carnivores:
            # Sort by energy and get the best one
            best_carnivore = max(carnivores, key=lambda c: c.energy)
            
            # Create a new position away from the best one
            angle = random.uniform(0, 360)
            distance = best_carnivore._get_size() * 5
            new_x = best_carnivore.x + math.cos(math.radians(angle)) * distance
            new_y = best_carnivore.y + math.sin(math.radians(angle)) * distance
            
            # Ensure within boundaries
            new_x = max(0, min(new_x, Config.SCREEN_WIDTH))
            new_y = max(0, min(new_y, Config.SCREEN_HEIGHT))
            
            # Create a new carnivore with a mutated brain from the best one
            mutated_brain = best_carnivore.brain.mutate()
            new_carnivore = Carnivore(new_x, new_y, 
                                     direction=random.uniform(0, 360),
                                     energy=initial_energy,  # Use the calculated energy value
                                     brain=mutated_brain)  # Use a mutated brain
            
            print(f"Spawned new carnivore with mutations from best (energy: {best_carnivore.energy:.1f}, initial energy: {initial_energy})")
        else:
            # No carnivores left, create a random one
            new_x = random.randint(50, Config.SCREEN_WIDTH - 50)
            new_y = random.randint(50, Config.SCREEN_HEIGHT - 50)
            new_carnivore = Carnivore(new_x, new_y, energy=initial_energy)  # Use the calculated energy value
            print(f"Spawned new random carnivore (initial energy: {initial_energy})")
            
        self.creatures.append(new_carnivore)
    
    def render(self, surface):
        # Render environment
        self.environment.render(surface)
        
        # Render creatures
        for creature in self.creatures:
            creature.render(surface)
            
        # Render statistics graph at the bottom
        self.render_stats(surface)
    
    def render_stats(self, surface):
        # Simplified statistics visualization
        history_width = 200
        history_height = 50
        x_start = Config.SCREEN_WIDTH - history_width - 10
        y_start = Config.SCREEN_HEIGHT - history_height - 10
        
        # Draw background
        pygame.draw.rect(surface, (30, 30, 30), (x_start, y_start, history_width, history_height))
        
        # Max values for scaling
        max_herb = max(self.herbivore_count_history[-history_width:] + [1])
        max_carn = max(self.carnivore_count_history[-history_width:] + [1])
        max_grass = max(self.grass_count_history[-history_width:] + [1])
        max_val = max(max_herb, max_carn, max_grass)
        
        # Draw data points
        for i in range(min(history_width, len(self.herbivore_count_history))):
            idx = -history_width + i if len(self.herbivore_count_history) >= history_width else i
            
            # Herbivores
            if idx < len(self.herbivore_count_history):
                herb_height = int((self.herbivore_count_history[idx] / max_val) * history_height)
                pygame.draw.line(surface, (0, 255, 0), 
                                (x_start + i, y_start + history_height), 
                                (x_start + i, y_start + history_height - herb_height), 1)
            
            # Carnivores
            if idx < len(self.carnivore_count_history):
                carn_height = int((self.carnivore_count_history[idx] / max_val) * history_height)
                pygame.draw.line(surface, (255, 0, 0), 
                                (x_start + i, y_start + history_height), 
                                (x_start + i, y_start + history_height - carn_height), 1)
            
            # Grass (rendered with lower opacity)
            if idx < len(self.grass_count_history):
                grass_height = int((self.grass_count_history[idx] / max_val) * history_height)
                pygame.draw.line(surface, (0, 150, 0), 
                                (x_start + i, y_start + history_height), 
                                (x_start + i, y_start + history_height - grass_height), 1)
        
        # Draw adaptive growth rate if available
        if hasattr(self.environment, 'grass_growth_history') and self.environment.grass_growth_history:
            # Scale growth rate for display
            max_growth = Config.MAX_GRASS_GROWTH_RATE * 1.1  # Slight padding
            recent_history = self.environment.grass_growth_history[-history_width:]
            for i in range(min(history_width, len(recent_history))):
                # Calculate height based on growth rate
                growth_height = int((recent_history[i] / max_growth) * (history_height * 0.3))
                # Draw growth rate as a yellow line at the bottom of the chart
                pygame.draw.line(surface, (255, 255, 0), 
                               (x_start + i, y_start + history_height), 
                               (x_start + i, y_start + history_height - growth_height), 1)
    
    def count_herbivores(self):
        return sum(1 for creature in self.creatures if isinstance(creature, Herbivore))
    
    def count_carnivores(self):
        return sum(1 for creature in self.creatures if isinstance(creature, Carnivore))
    
    def count_grass(self):
        return self.environment.get_grass_count()
    
    def save_simulation(self, filename='simulation_save.pkl'):
        """Save the current state of the simulation to a file"""
        save_data = {
            'tick': self.tick,
            'creatures': self.creatures,
            'grass': self.environment.grass,
            'grass_positions': self.environment.grass_positions,
            'grass_growth_history': getattr(self.environment, 'grass_growth_history', []),
            'herbivore_count_history': self.herbivore_count_history,
            'carnivore_count_history': self.carnivore_count_history,
            'grass_count_history': self.grass_count_history
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"Simulation saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving simulation: {e}")
            return False
    
    @classmethod
    def load_simulation(cls, filename='simulation_save.pkl'):
        """Load a simulation state from a file"""
        if not os.path.exists(filename):
            print(f"Save file {filename} not found")
            return None
            
        try:
            with open(filename, 'rb') as f:
                save_data = pickle.load(f)
                
            # Create a new simulation instance
            simulation = cls()
            
            # Restore saved state
            simulation.tick = save_data['tick']
            simulation.creatures = save_data['creatures']
            simulation.environment.grass = save_data['grass']
            if 'grass_positions' in save_data:
                simulation.environment.grass_positions = save_data['grass_positions']
            if 'grass_growth_history' in save_data:
                simulation.environment.grass_growth_history = save_data['grass_growth_history']
            simulation.herbivore_count_history = save_data['herbivore_count_history']
            simulation.carnivore_count_history = save_data['carnivore_count_history']
            simulation.grass_count_history = save_data['grass_count_history']
            
            print(f"Simulation loaded from {filename}")
            return simulation
        except Exception as e:
            print(f"Error loading simulation: {e}")
            return None
