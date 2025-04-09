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
        input_size = self.get_vision_resolution() + 1  # Vision + energy level
        hidden_size = Config.HIDDEN_LAYER_SIZE
        output_size = 2  # Speed and turn
        
        self.brain = brain if brain is not None else NeuralNetwork(input_size, hidden_size, output_size)
        
        # Movement properties
        self.speed = 0
        
        # Vision property for rendering
        self.show_vision = Config.SHOW_VISION_CONES

    def __repr__(self):
        return f"location: ({self.x}, {self.y}), direction: {self.direction}, energy: {self.energy}"
    
    # Vision property getters
    def get_vision_resolution(self):
        # Will be overridden by subclasses
        pass
    
    def get_vision_range(self):
        # Will be overridden by subclasses
        pass
    
    def get_vision_angle(self):
        # Will be overridden by subclasses
        pass
    
    # Movement and physical property getters
    def get_max_speed(self):
        # Will be overridden by subclasses
        pass
    
    def get_max_turn(self):
        # Will be overridden by subclasses
        pass
    
    def get_color(self):
        # Will be overridden by subclasses
        pass
    
    def get_size(self):
        # Will be overridden by subclasses
        pass
    
    def get_energy_consumption(self):
        # Will be overridden by subclasses
        pass
    
    def get_max_energy(self):
        # Will be overridden by subclasses
        pass
    
    def get_position(self):
        return (self.x, self.y)
    
    def think(self, inputs):
        # Process inputs through the neural network
        outputs = self.brain.forward(inputs)
        return outputs
    
    def act(self, outputs):
        # Extract actions from neural network outputs
        speed_output = outputs[0]  # Between 0 and 1
        turn_output = outputs[1]   # Between 0 and 1
        
        # Convert to actual speed and turn values
        self.speed = speed_output * self.get_max_speed()
        
        # Convert turn from [0,1] to [-max_turn, max_turn]
        turn = (turn_output * 2 - 1) * self.get_max_turn()
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
        energy_use = self.get_energy_consumption() * (0.5 + 0.5 * self.speed / self.get_max_speed())
        self.energy -= energy_use
    
    def can_reproduce(self):
        # Check if the creature has enough energy to reproduce
        return False  # Will be overridden by subclasses
    
    def reproduce(self):
        # Create a new creature with mutated brain
        return None  # Will be overridden by subclasses
    
    def render(self, surface):
        # Draw the vision cone if enabled
        if self.show_vision:
            self.render_vision_cone(surface)
        
        # Draw the creature
        pygame.draw.circle(surface, self.get_color(), (int(self.x), int(self.y)), self.get_size())
    
    def render_vision_cone(self, surface):
        # Calculate the starting and ending angles of the vision cone
        half_angle = self.get_vision_angle() / 2
        start_angle = self.direction - half_angle
        end_angle = self.direction + half_angle
        
        # Convert angles to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        
        # Calculate the points for vision cone edges
        start_x = self.x + math.cos(start_rad) * self.get_vision_range()
        start_y = self.y + math.sin(start_rad) * self.get_vision_range()
        
        end_x = self.x + math.cos(end_rad) * self.get_vision_range()
        end_y = self.y + math.sin(end_rad) * self.get_vision_range()
        
        # Get cone color based on creature type (with transparency)
        if isinstance(self, Herbivore):
            cone_color = (0, 255, 0, Config.VISION_CONE_OPACITY)  # Semi-transparent green
        else:  # Carnivore
            cone_color = (255, 0, 0, Config.VISION_CONE_OPACITY)  # Semi-transparent red
        
        # Create a semi-transparent surface for the cone
        cone_surface = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Draw the vision cone with filled polygon
        points = [(int(self.x), int(self.y)), 
                 (int(start_x), int(start_y))]
        
        # Add arc points to create a smooth cone
        steps = 10
        for i in range(steps + 1):
            angle = start_rad + (end_rad - start_rad) * i / steps
            x = self.x + math.cos(angle) * self.get_vision_range()
            y = self.y + math.sin(angle) * self.get_vision_range()
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
    
    def get_vision_resolution(self):
        return Config.HERBIVORE_VISION_RESOLUTION
    
    def get_vision_range(self):
        return Config.HERBIVORE_VISION_RANGE
    
    def get_vision_angle(self):
        return Config.HERBIVORE_VISION_ANGLE
    
    def get_max_speed(self):
        return Config.HERBIVORE_MAX_SPEED
    
    def get_max_turn(self):
        return Config.HERBIVORE_MAX_TURN
    
    def get_color(self):
        # Adjust color based on energy level
        energy_ratio = self.energy / Config.HERBIVORE_MAX_ENERGY
        green = min(255, int(Config.HERBIVORE_COLOR[1] * energy_ratio * 1.5))
        return (0, green, 0)
    
    def get_size(self):
        return Config.HERBIVORE_SIZE
    
    def get_energy_consumption(self):
        return Config.HERBIVORE_ENERGY_CONSUMPTION
    
    def get_max_energy(self):
        return Config.HERBIVORE_MAX_ENERGY
    
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
        distance = self.get_size() * 3
        new_x = self.x + math.cos(math.radians(angle)) * distance
        new_y = self.y + math.sin(math.radians(angle)) * distance
        
        # Ensure offspring is within boundaries
        new_x = max(0, min(new_x, Config.SCREEN_WIDTH))
        new_y = max(0, min(new_y, Config.SCREEN_HEIGHT))
        
        # Split energy between parent and offspring
        offspring_energy = self.energy / 2
        self.energy = self.energy / 2
        
        return Herbivore(new_x, new_y, random.uniform(0, 360), offspring_energy, mutated_brain)

class Carnivore(Creature):
    def __init__(self, x, y, direction=None, energy=None, brain=None):
        energy = Config.CARNIVORE_INITIAL_ENERGY if energy is None else energy
        super().__init__(x, y, direction, energy, brain)
    
    def get_vision_resolution(self):
        return Config.CARNIVORE_VISION_RESOLUTION
    
    def get_vision_range(self):
        return Config.CARNIVORE_VISION_RANGE
    
    def get_vision_angle(self):
        return Config.CARNIVORE_VISION_ANGLE
    
    def get_max_speed(self):
        return Config.CARNIVORE_MAX_SPEED
    
    def get_max_turn(self):
        return Config.CARNIVORE_MAX_TURN
    
    def get_color(self):
        # Adjust color based on energy level
        energy_ratio = self.energy / Config.CARNIVORE_MAX_ENERGY
        red = min(255, int(Config.CARNIVORE_COLOR[0] * energy_ratio * 1.5))
        return (red, 0, 0)
    
    def get_size(self):
        return Config.CARNIVORE_SIZE
    
    def get_energy_consumption(self):
        return Config.CARNIVORE_ENERGY_CONSUMPTION
    
    def get_max_energy(self):
        return Config.CARNIVORE_MAX_ENERGY
    
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
        distance = self.get_size() * 3
        new_x = self.x + math.cos(math.radians(angle)) * distance
        new_y = self.y + math.sin(math.radians(angle)) * distance
        
        # Ensure offspring is within boundaries
        new_x = max(0, min(new_x, Config.SCREEN_WIDTH))
        new_y = max(0, min(new_y, Config.SCREEN_HEIGHT))
        
        # Split energy between parent and offspring
        offspring_energy = self.energy / 2
        self.energy = self.energy / 2
        
        return Carnivore(new_x, new_y, random.uniform(0, 360), offspring_energy, mutated_brain)
