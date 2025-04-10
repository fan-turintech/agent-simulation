import random
import pygame
import math
import pickle
import os
from config import Config
from creature import Herbivore, Carnivore
from environment import Grass  # Now only importing Grass class

class Simulation:
    def __init__(self, grass_count=Config.MAX_GRASS//2, herbivore_count=Config.INITIAL_HERBIVORES, carnivore_count=Config.INITIAL_CARNIVORES):
        # Initialize grass directly in simulation
        self.grass = []
        self.width = Config.SCREEN_WIDTH
        self.height = Config.SCREEN_HEIGHT
        
        # Pre-compute grid for faster grass placement
        self.grid_size = max(Config.GRASS_SIZE * 2, 15)  # Grid cell size based on grass size
        self.grid_width = self.width // self.grid_size
        self.grid_height = self.height // self.grid_size
        self.grid = [[False for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        self.grass_positions = set()  # Fast lookup for existing grass positions
        
        # Spatial partitioning grids for collision detection
        self.grass_spatial_grid = {}
        self.creature_spatial_grid = {}
        
        # Initialize grass
        self.initialize_grass(grass_count)
        
        # Track grass stats
        self.grass_growth_history = []  # Track growth rate history
        
        # Initialize creatures
        self.creatures = []
        self.initialize_creatures(herbivore_count, carnivore_count)
        
        # Statistics
        self.tick = 0
        self.herbivore_count_history = []
        self.carnivore_count_history = []
        self.grass_count_history = []
        
    def get_grid_cell(self, x, y):
        """Convert world coordinates to grid cell coordinates"""
        grid_x = max(0, min(int(x / self.grid_size), self.grid_width - 1))
        grid_y = max(0, min(int(y / self.grid_size), self.grid_height - 1))
        return (grid_x, grid_y)
        
    def update_spatial_grids(self):
        """Update the spatial partitioning grids with current entity positions"""
        # Clear previous grids
        self.grass_spatial_grid = {}
        self.creature_spatial_grid = {}
        
        # Add grass to spatial grid
        for grass in self.grass:
            grid_pos = self.get_grid_cell(grass.x, grass.y)
            if grid_pos not in self.grass_spatial_grid:
                self.grass_spatial_grid[grid_pos] = []
            self.grass_spatial_grid[grid_pos].append(grass)
        
        # Add creatures to spatial grid
        for creature in self.creatures:
            grid_pos = self.get_grid_cell(creature.x, creature.y)
            if grid_pos not in self.creature_spatial_grid:
                self.creature_spatial_grid[grid_pos] = []
            self.creature_spatial_grid[grid_pos].append(creature)
    
    def get_nearby_entities(self, entity, is_grass=False):
        """Get entities in the same and adjacent grid cells"""
        grid_pos = self.get_grid_cell(entity.x, entity.y)
        nearby = []
        
        # Check current cell and all adjacent cells (including diagonals)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                
                # Skip out-of-bounds cells
                if (check_pos[0] < 0 or check_pos[0] >= self.grid_width or 
                    check_pos[1] < 0 or check_pos[1] >= self.grid_height):
                    continue
                
                # Get entities from appropriate grid
                if is_grass:
                    if check_pos in self.grass_spatial_grid:
                        nearby.extend(self.grass_spatial_grid[check_pos])
                else:
                    if check_pos in self.creature_spatial_grid:
                        nearby.extend(self.creature_spatial_grid[check_pos])
        
        return nearby
    
    def initialize_grass(self, grass_count):
        """Create initial grass"""
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
                new_grass = Grass(x, y)
                self.grass.append(new_grass)
                self.grass_positions.add(pos_key)
                
                # Add to spatial grid
                if pos_key not in self.grass_spatial_grid:
                    self.grass_spatial_grid[pos_key] = []
                self.grass_spatial_grid[pos_key].append(new_grass)
                return True
                
            attempts += 1
            
        return False  # Could not find a suitable position
    
    def remove_grass(self, grass):
        """Remove a grass instance from the environment"""
        if grass in self.grass:
            # Remove from positions set to allow new grass to grow there
            pos_key = (grass.x // self.grid_size, grass.y // self.grid_size)
            if pos_key in self.grass_positions:
                self.grass_positions.remove(pos_key)
            
            # Remove from grass list
            self.grass.remove(grass)
            
            # Remove from spatial grid
            if pos_key in self.grass_spatial_grid and grass in self.grass_spatial_grid[pos_key]:
                self.grass_spatial_grid[pos_key].remove(grass)
                if not self.grass_spatial_grid[pos_key]:  # If no more grass in this cell
                    del self.grass_spatial_grid[pos_key]
    
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
    
    def update_grass(self):
        """Update grass growth for a single time step"""
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
    
    def get_grass_count(self):
        """Return the current number of grass patches"""
        return len(self.grass)
    
    def get_current_growth_rate(self):
        """Return the current adaptive growth rate"""
        if self.grass_growth_history:
            return self.grass_growth_history[-1]
        return Config.BASE_GRASS_GROWTH_RATE
        
    def initialize_creatures(self, herbivore_count, carnivore_count):
        # Create initial herbivores
        for _ in range(herbivore_count):
            x = random.randint(50, Config.SCREEN_WIDTH - 50)
            y = random.randint(50, Config.SCREEN_HEIGHT - 50)
            herbivore = Herbivore(x, y)
            self.creatures.append(herbivore)
            # Add to spatial grid
            grid_pos = self.get_grid_cell(x, y)
            if grid_pos not in self.creature_spatial_grid:
                self.creature_spatial_grid[grid_pos] = []
            self.creature_spatial_grid[grid_pos].append(herbivore)
        
        # Create initial carnivores
        for _ in range(carnivore_count):
            x = random.randint(50, Config.SCREEN_WIDTH - 50)
            y = random.randint(50, Config.SCREEN_HEIGHT - 50)
            carnivore = Carnivore(x, y)
            self.creatures.append(carnivore)
            # Add to spatial grid
            grid_pos = self.get_grid_cell(x, y)
            if grid_pos not in self.creature_spatial_grid:
                self.creature_spatial_grid[grid_pos] = []
            self.creature_spatial_grid[grid_pos].append(carnivore)
    
    def update(self):
        # Increment simulation tick
        self.tick += 1
        
        # Update grass
        self.update_grass()
        
        # Update spatial grids with current positions
        self.update_spatial_grids()
        
        # Process creature sensing, thinking, and acting
        for creature in self.creatures:
            # Sense the environment
            inputs = self.sense_environment_for_creature(creature)
            
            # Think about what to do
            outputs = creature.think(inputs)
            
            # Take action based on thoughts
            creature.act(outputs)
        
        # Process creature interactions, deaths, and reproduction
        new_creatures = []
        
        for creature in self.creatures[:]:  # Make a copy to avoid modification during iteration
            # Update creature and check if it's still alive
            if not self.update_creature(creature):
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
        self.grass_count_history.append(self.get_grass_count())
        
        # Keep only recent history
        max_history = 1000
        if len(self.herbivore_count_history) > max_history:
            self.herbivore_count_history = self.herbivore_count_history[-max_history:]
            self.carnivore_count_history = self.carnivore_count_history[-max_history:]
            self.grass_count_history = self.grass_count_history[-max_history:]
    
    def update_creature(self, creature):
        """Update a creature and handle its interactions with the environment and other creatures"""
        # Increase age
        creature.age += 1
        
        # If energy is depleted, creature dies
        if creature.energy <= 0:
            return False  # Dead
        
        # Handle feeding behavior based on creature type using spatial partitioning
        if isinstance(creature, Herbivore):
            # Check for grass consumption using nearby grass only
            nearby_grass = self.get_nearby_entities(creature, is_grass=True)
            for grass in nearby_grass:
                distance = math.sqrt((creature.x - grass.x)**2 + (creature.y - grass.y)**2)
                if distance < creature.get_size() + grass.size:
                    creature.energy = min(creature.energy + grass.energy, creature.get_max_energy())
                    self.remove_grass(grass)
        
        elif isinstance(creature, Carnivore):
            # Check for herbivore hunting using nearby creatures only
            nearby_creatures = self.get_nearby_entities(creature, is_grass=False)
            for target in nearby_creatures:
                if isinstance(target, Herbivore):
                    distance = math.sqrt((creature.x - target.x)**2 + (creature.y - target.y)**2)
                    if distance < creature.get_size() + target.get_size():
                        # Consume the herbivore with additional energy bonus
                        bonus_energy = 100  # Fixed bonus energy for consuming a herbivore
                        total_energy = target.energy + bonus_energy
                        creature.energy = min(creature.energy + total_energy, creature.get_max_energy())
                        target.energy = 0  # Kill the herbivore
        
        return True  # Still alive
    
    def sense_environment_for_creature(self, creature):
        """Get sensory inputs for a creature based on its environment and other creatures"""
        # Get vision input from the environment
        vision_data = self.get_vision_for_creature(creature)
        
        # Add energy level as an internal state input
        normalized_energy = creature.energy / creature.get_max_energy()
        
        # Combine all inputs for the neural network
        inputs = vision_data + [normalized_energy]
        return inputs
    
    def get_vision_for_creature(self, creature):
        """Calculate what a creature can see in its vision cone"""
        vision_resolution = creature.get_vision_resolution()
        vision_data = [0] * vision_resolution
        
        # Calculate the starting angle (left edge of vision cone)
        half_angle = creature.get_vision_angle() / 2
        start_angle = creature.direction - half_angle
        
        # Angle step for each "ray" of vision
        angle_step = creature.get_vision_angle() / (vision_resolution - 1)
        
        # Cast rays for each angle in the vision cone
        for i in range(vision_resolution):
            angle = start_angle + i * angle_step
            angle_rad = math.radians(angle)
            
            # Cast ray and check what the creature sees
            vision_data[i] = self.cast_ray_for_creature(creature, angle_rad)
            
        return vision_data
    
    def cast_ray_for_creature(self, creature, angle_rad):
        """Cast a ray from the creature and check what it sees"""
        # Default: nothing detected
        detection = 0
        
        # Get the direction vector
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        
        # Precompute grid cells that the ray passes through
        max_distance = creature.get_vision_range()
        ray_points = []
        for distance in range(0, max_distance, 5):  # Step by 5 for efficiency
            check_x = creature.x + dx * distance
            check_y = creature.y + dy * distance
            
            # Stop if out of bounds
            if (check_x < 0 or check_x >= Config.SCREEN_WIDTH or 
                check_y < 0 or check_y >= Config.SCREEN_HEIGHT):
                detection = 0.5  # Edge of screen
                break
                
            # Add point to ray path
            ray_points.append((check_x, check_y))
            
        # Check grid cells along ray path
        visited_cells = set()
        for point_x, point_y in ray_points:
            grid_cell = self.get_grid_cell(point_x, point_y)
            
            # Skip already checked cells
            if grid_cell in visited_cells:
                continue
            visited_cells.add(grid_cell)
            
            # Check for creatures in this cell
            if grid_cell in self.creature_spatial_grid:
                for other in self.creature_spatial_grid[grid_cell]:
                    if other != creature:  # Don't detect self
                        dist = math.sqrt((other.x - point_x)**2 + (other.y - point_y)**2)
                        if dist < other.get_size():
                            if isinstance(creature, Herbivore):
                                if isinstance(other, Carnivore):
                                    detection = -1  # Carnivore: danger
                                else:
                                    detection = 0.2  # Other herbivores: neutral
                            else:  # I'm a carnivore
                                if isinstance(other, Herbivore):
                                    detection = 1  # Herbivore: food
                                else:
                                    detection = 0.3  # Other carnivores: neutral/competition
                            return detection
            
            # For herbivores, check for grass in this cell
            if isinstance(creature, Herbivore) and grid_cell in self.grass_spatial_grid:
                for grass in self.grass_spatial_grid[grid_cell]:
                    dist = math.sqrt((grass.x - point_x)**2 + (grass.y - point_y)**2)
                    if dist < grass.size:
                        detection = 0.8  # Grass: food for herbivores
                        return detection
                
        return detection
    
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
            distance = best_herbivore.get_size() * 5
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
        
        # Add to spatial grid
        grid_pos = self.get_grid_cell(new_herbivore.x, new_herbivore.y)
        if grid_pos not in self.creature_spatial_grid:
            self.creature_spatial_grid[grid_pos] = []
        self.creature_spatial_grid[grid_pos].append(new_herbivore)
    
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
            distance = best_carnivore.get_size() * 5
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
        
        # Add to spatial grid
        grid_pos = self.get_grid_cell(new_carnivore.x, new_carnivore.y)
        if grid_pos not in self.creature_spatial_grid:
            self.creature_spatial_grid[grid_pos] = []
        self.creature_spatial_grid[grid_pos].append(new_carnivore)
    
    def render(self, surface):
        # Render grass
        for grass in self.grass:
            grass.render(surface)
        
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
        if self.grass_growth_history:
            # Scale growth rate for display
            max_growth = Config.MAX_GRASS_GROWTH_RATE * 1.1  # Slight padding
            recent_history = self.grass_growth_history[-history_width:]
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
        return self.get_grass_count()
    
    def save_simulation(self, filename='simulation_save.pkl'):
        """Save the current state of the simulation to a file"""
        save_data = {
            'tick': self.tick,
            'creatures': self.creatures,
            'grass': self.grass,
            'grass_positions': self.grass_positions,
            'grass_growth_history': self.grass_growth_history,
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
            simulation = cls(0, 0, 0)  # Create empty simulation
            
            # Restore saved state
            simulation.tick = save_data['tick']
            simulation.creatures = save_data['creatures']
            simulation.grass = save_data['grass']
            if 'grass_positions' in save_data:
                simulation.grass_positions = save_data['grass_positions']
            if 'grass_growth_history' in save_data:
                simulation.grass_growth_history = save_data['grass_growth_history']
            simulation.herbivore_count_history = save_data['herbivore_count_history']
            simulation.carnivore_count_history = save_data['carnivore_count_history']
            simulation.grass_count_history = save_data['grass_count_history']
            
            # Rebuild spatial grids after loading
            simulation.update_spatial_grids()
            
            print(f"Simulation loaded from {filename}")
            return simulation
        except Exception as e:
            print(f"Error loading simulation: {e}")
            return None