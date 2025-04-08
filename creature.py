import pygame
import random
import math
import numpy as np
from config import Config
from neural_network import NeuralNetwork

class Creature:
    def __init__(self, x, y, direction=None, energy=None, brain=None):
        self.x = x
        self.y = y
        
        # Direction is in degrees (0 is right, 90 is down)
        self.direction = direction if direction is not None else random.uniform(0, 360)
        
        # Internal state
        self.energy = energy
        self.age = 0
        
        # Neural network setup
        input_size = self._get_vision_resolution() + 1  # Vision + energy level
        hidden_size = Config.HIDDEN_LAYER_SIZE
        output_size = 2  # Speed and turn
        
        self.brain = brain if brain is not None else NeuralNetwork(input_size, hidden_size, output_size)
        
        # Movement properties
        self.speed = 0

    def __repr__(self):
        return f"location: ({self.x}, {self.y}), direction: {self.direction}, energy: {self.energy}"
    
    def _get_vision_resolution(self):
        # Will be overridden by subclasses
        pass
    
    def _get_vision_range(self):
        # Will be overridden by subclasses
        pass
    
    def _get_vision_angle(self):
        # Will be overridden by subclasses
        pass
    
    def _get_max_speed(self):
        # Will be overridden by subclasses
        pass
    
    def _get_max_turn(self):
        # Will be overridden by subclasses
        pass
    
    def _get_color(self):
        # Will be overridden by subclasses
        pass
    
    def _get_size(self):
        # Will be overridden by subclasses
        pass
    
    def _get_energy_consumption(self):
        # Will be overridden by subclasses
        pass
    
    def _get_max_energy(self):
        # Will be overridden by subclasses
        pass
    
    def get_position(self):
        return (self.x, self.y)
    
    def sense_environment(self, environment, creatures):
        # Get vision input from the environment
        vision_data = self._get_vision(environment, creatures)
        # Add energy level as an internal state input
        normalized_energy = self.energy / self._get_max_energy()
        
        # Combine all inputs for the neural network
        inputs = vision_data + [normalized_energy]
        return inputs
    
    def _get_vision(self, environment, creatures):
        vision_data = [0] * self._get_vision_resolution()
        
        # Calculate the starting angle (left edge of vision cone)
        half_angle = self._get_vision_angle() / 2
        start_angle = self.direction - half_angle
        
        # Angle step for each "ray" of vision
        angle_step = self._get_vision_angle() / (self._get_vision_resolution() - 1)
        
        # Cast rays for each angle in the vision cone
        for i in range(self._get_vision_resolution()):
            angle = start_angle + i * angle_step
            angle_rad = math.radians(angle)
            
            # Cast ray and check what the creature sees
            vision_data[i] = self._cast_ray(angle_rad, environment, creatures)
            
        return vision_data
    
    def _cast_ray(self, angle_rad, environment, creatures):
        # Default: nothing detected
        detection = 0
        
        # Get the direction vector
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        
        # Cast the ray to check for objects
        for distance in range(1, self._get_vision_range(), 2):  # Step by 5 for efficiency
            check_x = self.x + dx * distance
            check_y = self.y + dy * distance
            
            # Check boundaries
            if check_x < 0 or check_x >= Config.SCREEN_WIDTH or check_y < 0 or check_y >= Config.SCREEN_HEIGHT:
                detection = 0.5  # Edge of screen
                break
            
            # Check creatures first - they have priority in vision
            for creature in creatures:
                if creature != self:  # Don't detect self
                    dist = math.sqrt((creature.x - check_x)**2 + (creature.y - check_y)**2)
                    if dist < creature._get_size():
                        if isinstance(self, Herbivore):
                            if isinstance(creature, Carnivore):
                                detection = -1  # Carnivore: danger
                            else:
                                detection = 0.2  # Other herbivores: neutral
                        else:  # I'm a carnivore
                            if isinstance(creature, Herbivore):
                                detection = 1  # Herbivore: food
                            else:
                                detection = 0.3  # Other carnivores: neutral/competition
                        break
            
            # Check grass - only for herbivores
            if detection == 0 and isinstance(self, Herbivore):
                for grass in environment.grass:
                    dist = math.sqrt((grass.x - check_x)**2 + (grass.y - check_y)**2)
                    if dist < grass.size:
                        detection = 0.8  # Grass: food for herbivores
                        break
            
            if detection != 0:  # If something was detected, stop the ray
                break
                
        return detection
    
    def think(self, inputs):
        # Process inputs through the neural network
        outputs = self.brain.forward(inputs)
        return outputs
    
    def act(self, outputs):
        # Extract actions from neural network outputs
        speed_output = outputs[0]  # Between 0 and 1
        turn_output = outputs[1]   # Between 0 and 1
        
        # Convert to actual speed and turn values
        self.speed = speed_output * self._get_max_speed()
        
        # Convert turn from [0,1] to [-max_turn, max_turn]
        turn = (turn_output * 2 - 1) * self._get_max_turn()
        self.direction += turn
        
        # Keep direction in [0, 360)
        self.direction = self.direction % 360
        
        # Move the creature
        rad_direction = math.radians(self.direction)
        self.x += math.cos(rad_direction) * self.speed
        self.y += math.sin(rad_direction) * self.speed
        
        # Boundary checking
        self.x = max(0, min(self.x, Config.SCREEN_WIDTH))
        self.y = max(0, min(self.y, Config.SCREEN_HEIGHT))
        
        # Energy consumption based on actions
        energy_use = self._get_energy_consumption() * (0.5 + 0.5 * self.speed / self._get_max_speed())
        self.energy -= energy_use
    
    def update(self, environment, creatures):
        # Increase age
        self.age += 1
        
        # If energy is depleted, creature dies
        if self.energy <= 0:
            return False  # Dead
        
        return True  # Still alive
    
    def can_reproduce(self):
        # Check if the creature has enough energy to reproduce
        return False  # Will be overridden by subclasses
    
    def reproduce(self):
        # Create a new creature with mutated brain
        return None  # Will be overridden by subclasses
    
    def render(self, surface):
        # Draw the vision cone
        self.render_vision_cone(surface)
        
        # Draw the creature
        pygame.draw.circle(surface, self._get_color(), (int(self.x), int(self.y)), self._get_size())
    
    def render_vision_cone(self, surface):
        # Calculate the starting and ending angles of the vision cone
        half_angle = self._get_vision_angle() / 2
        start_angle = self.direction - half_angle
        end_angle = self.direction + half_angle
        
        # Convert angles to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        
        # Calculate the points for vision cone edges
        start_x = self.x + math.cos(start_rad) * self._get_vision_range()
        start_y = self.y + math.sin(start_rad) * self._get_vision_range()
        
        end_x = self.x + math.cos(end_rad) * self._get_vision_range()
        end_y = self.y + math.sin(end_rad) * self._get_vision_range()
        
        # Get cone color based on creature type (with transparency)
        if isinstance(self, Herbivore):
            cone_color = (0, 255, 0, 30)  # Semi-transparent green
        else:  # Carnivore
            cone_color = (255, 0, 0, 30)  # Semi-transparent red
        
        # Create a semi-transparent surface for the cone
        cone_surface = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Draw the vision cone with filled polygon
        points = [(int(self.x), int(self.y)), 
                 (int(start_x), int(start_y))]
        
        # Add arc points to create a smooth cone
        steps = 10
        for i in range(steps + 1):
            angle = start_rad + (end_rad - start_rad) * i / steps
            x = self.x + math.cos(angle) * self._get_vision_range()
            y = self.y + math.sin(angle) * self._get_vision_range()
            points.append((int(x), int(y)))
            
        points.append((int(end_x), int(end_y)))
        
        # Draw the polygon on the transparent surface
        pygame.draw.polygon(cone_surface, cone_color, points)
        
        # Draw the outline of the vision cone
        for i in range(len(points) - 1):
            pygame.draw.line(cone_surface, (cone_color[0], cone_color[1], cone_color[2], 100), 
                            points[i], points[i+1], 1)
        
        # Blit the transparent surface onto the main surface
        surface.blit(cone_surface, (0, 0))

class Herbivore(Creature):
    def __init__(self, x, y, direction=None, energy=None, brain=None):
        energy = Config.HERBIVORE_INITIAL_ENERGY if energy is None else energy
        super().__init__(x, y, direction, energy, brain)
    
    def _get_vision_resolution(self):
        return Config.HERBIVORE_VISION_RESOLUTION
    
    def _get_vision_range(self):
        return Config.HERBIVORE_VISION_RANGE
    
    def _get_vision_angle(self):
        return Config.HERBIVORE_VISION_ANGLE
    
    def _get_max_speed(self):
        return Config.HERBIVORE_MAX_SPEED
    
    def _get_max_turn(self):
        return Config.HERBIVORE_MAX_TURN
    
    def _get_color(self):
        # Adjust color based on energy level
        energy_ratio = self.energy / Config.HERBIVORE_MAX_ENERGY
        green = min(255, int(Config.HERBIVORE_COLOR[1] * energy_ratio * 1.5))
        return (0, green, 0)
    
    def _get_size(self):
        return Config.HERBIVORE_SIZE
    
    def _get_energy_consumption(self):
        return Config.HERBIVORE_ENERGY_CONSUMPTION
    
    def _get_max_energy(self):
        return Config.HERBIVORE_MAX_ENERGY
    
    def update(self, environment, creatures):
        if not super().update(environment, creatures):
            return False
        
        # Check for grass consumption
        for grass in environment.grass[:]:  # Create a copy of the list for safe iteration
            distance = math.sqrt((self.x - grass.x)**2 + (self.y - grass.y)**2)
            if distance < self._get_size() + grass.size:
                self.energy = min(self.energy + grass.energy, self._get_max_energy())
                environment.remove_grass(grass)
        
        return True
    
    def can_reproduce(self):
        return (self.energy > Config.HERBIVORE_REPRODUCTION_THRESHOLD and 
                random.random() < Config.HERBIVORE_REPRODUCTION_CHANCE)
    
    def reproduce(self):
        if not self.can_reproduce():
            return None
            
        # Create offspring with mutation
        mutated_brain = self.brain.mutate()
        
        # Calculate position nearby parent
        angle = random.uniform(0, 360)
        distance = self._get_size() * 3
        new_x = self.x + math.cos(math.radians(angle)) * distance
        new_y = self.y + math.sin(math.radians(angle)) * distance
        
        # Ensure offspring is within boundaries
        new_x = max(0, min(new_x, Config.SCREEN_WIDTH))
        new_y = max(0, min(new_y, Config.SCREEN_HEIGHT))
        
        # Split energy between parent and offspring
        offspring_energy = self.energy / 2
        self.energy = self.energy / 2
        
        return Herbivore(new_x, new_y, random.uniform(0, 360), offspring_energy, mutated_brain)
    
    def render_vision_cone(self, surface):
        # Custom color for herbivore vision cone
        vision_color = (0, 255, 0, 30)  # Semi-transparent green
        self._render_vision_cone_with_color(surface, vision_color)
    
    def _render_vision_cone_with_color(self, surface, vision_color):
        # Same implementation as parent class but with specific color
        
        # Calculate the starting and ending angles of the vision cone
        half_angle = self._get_vision_angle() / 2
        start_angle = self.direction - half_angle
        end_angle = self.direction + half_angle
        
        # Convert angles to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        
        # Calculate the points for vision cone edges
        start_x = self.x + math.cos(start_rad) * self._get_vision_range()
        start_y = self.y + math.sin(start_rad) * self._get_vision_range()
        
        end_x = self.x + math.cos(end_rad) * self._get_vision_range()
        end_y = self.y + math.sin(end_rad) * self._get_vision_range()
        
        # Create a semi-transparent surface for the cone
        cone_surface = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Draw the vision cone with filled polygon
        points = [(int(self.x), int(self.y)), 
                 (int(start_x), int(start_y))]
        
        # Add arc points to create a smooth cone
        steps = 10
        for i in range(steps + 1):
            angle = start_rad + (end_rad - start_rad) * i / steps
            x = self.x + math.cos(angle) * self._get_vision_range()
            y = self.y + math.sin(angle) * self._get_vision_range()
            points.append((int(x), int(y)))
        
        # Draw the polygon on the transparent surface
        pygame.draw.polygon(cone_surface, vision_color, points)
        
        # Draw the outline of the vision cone (slightly more opaque)
        outline_color = (vision_color[0], vision_color[1], vision_color[2], 100)
        for i in range(len(points) - 1):
            pygame.draw.line(cone_surface, outline_color, points[i], points[i+1], 1)
        
        # Blit the transparent surface onto the main surface
        surface.blit(cone_surface, (0, 0))

class Carnivore(Creature):
    def __init__(self, x, y, direction=None, energy=None, brain=None):
        energy = Config.CARNIVORE_INITIAL_ENERGY if energy is None else energy
        super().__init__(x, y, direction, energy, brain)
    
    def _get_vision_resolution(self):
        return Config.CARNIVORE_VISION_RESOLUTION
    
    def _get_vision_range(self):
        return Config.CARNIVORE_VISION_RANGE
    
    def _get_vision_angle(self):
        return Config.CARNIVORE_VISION_ANGLE
    
    def _get_max_speed(self):
        return Config.CARNIVORE_MAX_SPEED
    
    def _get_max_turn(self):
        return Config.CARNIVORE_MAX_TURN
    
    def _get_color(self):
        # Adjust color based on energy level
        energy_ratio = self.energy / Config.CARNIVORE_MAX_ENERGY
        red = min(255, int(Config.CARNIVORE_COLOR[0] * energy_ratio * 1.5))
        return (red, 0, 0)
    
    def _get_size(self):
        return Config.CARNIVORE_SIZE
    
    def _get_energy_consumption(self):
        return Config.CARNIVORE_ENERGY_CONSUMPTION
    
    def _get_max_energy(self):
        return Config.CARNIVORE_MAX_ENERGY
    
    def update(self, environment, creatures):
        if not super().update(environment, creatures):
            return False
        
        # Check for herbivore hunting
        for creature in creatures[:]:  # Create a copy of the list for safe iteration
            if isinstance(creature, Herbivore):
                distance = math.sqrt((self.x - creature.x)**2 + (self.y - creature.y)**2)
                if distance < self._get_size() + creature._get_size():
                    # Consume the herbivore with additional energy bonus
                    bonus_energy = 100  # Fixed bonus energy for consuming a herbivore
                    total_energy = creature.energy + bonus_energy
                    self.energy = min(self.energy + total_energy, self._get_max_energy())
                    creature.energy = 0  # Kill the herbivore
        
        return True
    
    def can_reproduce(self):
        return (self.energy > Config.CARNIVORE_REPRODUCTION_THRESHOLD and 
                random.random() < Config.CARNIVORE_REPRODUCTION_CHANCE)
    
    def reproduce(self):
        if not self.can_reproduce():
            return None
            
        # Create offspring with mutation
        mutated_brain = self.brain.mutate()
        
        # Calculate position nearby parent
        angle = random.uniform(0, 360)
        distance = self._get_size() * 3
        new_x = self.x + math.cos(math.radians(angle)) * distance
        new_y = self.y + math.sin(math.radians(angle)) * distance
        
        # Ensure offspring is within boundaries
        new_x = max(0, min(new_x, Config.SCREEN_WIDTH))
        new_y = max(0, min(new_y, Config.SCREEN_HEIGHT))
        
        # Split energy between parent and offspring
        offspring_energy = self.energy / 2
        self.energy = self.energy / 2
        
        return Carnivore(new_x, new_y, random.uniform(0, 360), offspring_energy, mutated_brain)
    
    def render_vision_cone(self, surface):
        # Custom color for carnivore vision cone
        vision_color = (255, 0, 0, 30)  # Semi-transparent red
        self._render_vision_cone_with_color(surface, vision_color)
    
    def _render_vision_cone_with_color(self, surface, vision_color):
        # Same implementation as Herbivore class
        
        # Calculate the starting and ending angles of the vision cone
        half_angle = self._get_vision_angle() / 2
        start_angle = self.direction - half_angle
        end_angle = self.direction + half_angle
        
        # Convert angles to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        
        # Calculate the points for vision cone edges
        start_x = self.x + math.cos(start_rad) * self._get_vision_range()
        start_y = self.y + math.sin(start_rad) * self._get_vision_range()
        
        end_x = self.x + math.cos(end_rad) * self._get_vision_range()
        end_y = self.y + math.sin(end_rad) * self._get_vision_range()
        
        # Create a semi-transparent surface for the cone
        cone_surface = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Draw the vision cone with filled polygon
        points = [(int(self.x), int(self.y)), 
                 (int(start_x), int(start_y))]
        
        # Add arc points to create a smooth cone
        steps = 10
        for i in range(steps + 1):
            angle = start_rad + (end_rad - start_rad) * i / steps
            x = self.x + math.cos(angle) * self._get_vision_range()
            y = self.y + math.sin(angle) * self._get_vision_range()
            points.append((int(x), int(y)))
        
        # Draw the polygon on the transparent surface
        pygame.draw.polygon(cone_surface, vision_color, points)
        
        # Draw the outline of the vision cone (slightly more opaque)
        outline_color = (vision_color[0], vision_color[1], vision_color[2], 100)
        for i in range(len(points) - 1):
            pygame.draw.line(cone_surface, outline_color, points[i], points[i+1], 1)
        
        # Blit the transparent surface onto the main surface
        surface.blit(cone_surface, (0, 0))
