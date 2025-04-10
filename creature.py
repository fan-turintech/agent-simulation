import pygame
import random
import math
import numpy as np
from config import Config
from neural_network import NeuralNetwork
from functools import lru_cache
from typing import Optional, Tuple, Type, Union, List, Dict, Any
import weakref
from enum import Enum, auto

class CreatureType(Enum):
    HERBIVORE = auto()
    CARNIVORE = auto()

class Creature:
    __slots__ = ('x', 'y', 'direction', 'energy', 'age', 'brain', 'speed', 
                 'show_vision', '_cached_vision_cone', '_cached_direction', 
                 '_last_energy_ratio', '_cached_color', 'stats')
    
    def __init__(self, x: float, y: float, direction: Optional[float]=None, 
                 energy: Optional[float]=None, brain: Optional[NeuralNetwork]=None):
        """Initialize a new creature with given parameters."""
        self.x = x
        self.y = y
        
        # Direction is in degrees (0 is right, 90 is down)
        self.direction = direction if direction is not None else random.uniform(0, 360)
        
        # Internal state
        self.energy = energy  # Will be set by subclass if None
        self.age = 0
        
        # Neural network will be initialized in subclasses
        self.brain = brain
        
        # Movement properties
        self.speed = 0
        
        # Vision property for rendering
        self.show_vision = Config.SHOW_VISION_CONES
        
        # Caching for better performance
        self._cached_vision_cone = None
        self._cached_direction = None
        self._last_energy_ratio = -1  # Invalid value to ensure first update
        self._cached_color = None
        
        # Statistics tracking
        self.stats = {
            'distance_moved': 0.0,
            'energy_consumed': 0.0,
            'offspring': 0,
            'food_eaten': 0
        }

    def __repr__(self) -> str:
        """String representation of creature."""
        return f"{self.__class__.__name__}: pos=({self.x:.1f}, {self.y:.1f}), dir={self.direction:.1f}°, energy={self.energy:.1f}"
    
    # Vision property getters
    def get_vision_resolution(self) -> int:
        """Get the number of vision rays."""
        # Will be overridden by subclasses
        pass
    
    def get_vision_range(self) -> float:
        """Get the distance a creature can see."""
        # Will be overridden by subclasses
        pass
    
    def get_vision_angle(self) -> float:
        """Get the width of vision in degrees."""
        # Will be overridden by subclasses
        pass
    
    # Movement and physical property getters
    def get_max_speed(self) -> float:
        """Get the maximum speed the creature can move."""
        # Will be overridden by subclasses
        pass
    
    def get_max_turn(self) -> float:
        """Get the maximum turn rate in degrees."""
        # Will be overridden by subclasses
        pass
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get the color representing the creature, with energy-based variation."""
        # Check if we need to recalculate color
        energy_ratio = self.energy / self.get_max_energy()
        if abs(energy_ratio - self._last_energy_ratio) > 0.05 or self._cached_color is None:
            self._last_energy_ratio = energy_ratio
            self._cached_color = self._calculate_color(energy_ratio)
        return self._cached_color
    
    def _calculate_color(self, energy_ratio: float) -> Tuple[int, int, int]:
        """Calculate color based on energy level - overridden by subclasses."""
        pass
    
    def get_size(self) -> int:
        """Get the radius of the creature."""
        # Will be overridden by subclasses
        pass
    
    def get_energy_consumption(self) -> float:
        """Get the base energy consumption rate per tick."""
        # Will be overridden by subclasses
        pass
    
    def get_max_energy(self) -> float:
        """Get the maximum energy capacity."""
        # Will be overridden by subclasses
        pass
    
    def get_position(self) -> Tuple[float, float]:
        """Get the current position of the creature."""
        return (self.x, self.y)
    
    def think(self, inputs: List[float]) -> List[float]:
        """Process inputs through the neural network."""
        return self.brain.forward(inputs)
    
    @staticmethod
    @lru_cache(maxsize=360)
    def _get_movement_vector(direction_degrees: int, speed: float) -> Tuple[float, float]:
        """Calculate movement vector from direction and speed (cached for performance)."""
        rad_direction = math.radians(direction_degrees)
        return (
            math.cos(rad_direction) * speed,
            math.sin(rad_direction) * speed
        )
    
    def act(self, outputs: List[float]) -> None:
        """Execute actions based on neural network outputs."""
        # Extract actions from neural network outputs
        speed_output = min(max(outputs[0], 0.0), 1.0)  # Clamp between 0 and 1
        turn_output = min(max(outputs[1], 0.0), 1.0)   # Clamp between 0 and 1
        
        # Convert to actual speed and turn values
        max_speed = self.get_max_speed()
        self.speed = speed_output * max_speed
        
        # Convert turn from [0,1] to [-max_turn, max_turn]
        max_turn = self.get_max_turn()
        turn = (turn_output * 2 - 1) * max_turn
        
        # Update direction efficiently, keeping it in [0, 360)
        old_direction = self.direction
        self.direction = (self.direction + turn) % 360
        
        # Only invalidate vision cone cache if direction changed significantly
        if abs(old_direction - self.direction) > 1.0:
            self._cached_vision_cone = None
            self._cached_direction = self.direction
        
        # Move the creature using pre-calculated and cached vectors for common directions
        # Round direction to nearest degree for cache lookup
        direction_int = round(self.direction)
        dx, dy = self._get_movement_vector(direction_int, self.speed)
        
        # Store old position to calculate distance moved
        old_x, old_y = self.x, self.y
        
        # Update position with boundary checking in one step
        self.x = max(0, min(self.x + dx, Config.SCREEN_WIDTH))
        self.y = max(0, min(self.y + dy, Config.SCREEN_HEIGHT))
        
        # Track distance moved for statistics
        actual_dx = self.x - old_x
        actual_dy = self.y - old_y
        distance_moved = math.sqrt(actual_dx**2 + actual_dy**2)
        self.stats['distance_moved'] += distance_moved
        
        # Energy consumption based on actions with variable cost depending on speed
        # This creates a more realistic energy model where moving costs energy
        base_consumption = self.get_energy_consumption()
        speed_factor = 0.5 + 0.5 * (self.speed / max_speed)
        energy_used = base_consumption * speed_factor
        
        self.energy -= energy_used
        self.stats['energy_consumed'] += energy_used
    
    def can_reproduce(self) -> bool:
        """Check if the creature has enough energy to reproduce."""
        return False  # Will be overridden by subclasses
    
    def reproduce(self):
        """Create a new creature with mutated brain."""
        return None  # Will be overridden by subclasses
    
    def _generate_vision_cone_points(self) -> List[Tuple[int, int]]:
        """Generate the points for the vision cone."""
        half_angle = self.get_vision_angle() / 2
        vision_range = self.get_vision_range()
        x, y = int(self.x), int(self.y)
        
        # Calculate the starting and ending angles of the vision cone
        start_angle = self.direction - half_angle
        end_angle = self.direction + half_angle
        
        # Convert angles to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        
        # Create points for the vision cone (center + arc points)
        points = [(x, y)]
        
        # Add arc points efficiently
        steps = 10  # Number of points along arc
        step_angle = (end_rad - start_rad) / steps
        
        # Generate all points at once
        for i in range(steps + 1):
            angle = start_rad + step_angle * i
            arc_x = x + int(math.cos(angle) * vision_range)
            arc_y = y + int(math.sin(angle) * vision_range)
            points.append((arc_x, arc_y))
        
        return points
    
    def render(self, surface: pygame.Surface) -> None:
        """Render the creature and its vision cone if enabled."""
        # Draw the vision cone if enabled
        if self.show_vision:
            self.render_vision_cone(surface)
        
        # Draw the creature
        pygame.draw.circle(surface, self.get_color(), (int(self.x), int(self.y)), self.get_size())
    
    def render_vision_cone(self, surface: pygame.Surface) -> None:
        """Render the creature's vision cone with optimized rendering."""
        # Check if we can use cached vision cone
        if (self._cached_vision_cone is not None and 
            abs(self._cached_direction - self.direction) < 1.0):
            # Use cached vision cone
            cone_surface, pos = self._cached_vision_cone
            surface.blit(cone_surface, pos)
            return
            
        # Generate vision cone points
        points = self._generate_vision_cone_points()
        
        # Calculate bounding box for efficient surface size
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        
        # Add padding
        padding = 5
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(Config.SCREEN_WIDTH, max_x + padding)
        max_y = min(Config.SCREEN_HEIGHT, max_y + padding)
        
        # Create surface sized to bounding box
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return
            
        # Determine cone color based on creature type
        is_herbivore = isinstance(self, Herbivore)
        cone_color = (0, 255, 0, Config.VISION_CONE_OPACITY) if is_herbivore else (255, 0, 0, Config.VISION_CONE_OPACITY)
        outline_color = (cone_color[0], cone_color[1], cone_color[2], 100)
        
        cone_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Adjust points for the local surface
        local_points = [(p[0] - min_x, p[1] - min_y) for p in points]
        
        # Draw the polygon on the transparent surface
        pygame.draw.polygon(cone_surface, cone_color, local_points)
        
        # Draw the outline with optimized approach (fewer line segments)
        pygame.draw.lines(cone_surface, outline_color, False, local_points, 1)
        
        # Cache the vision cone
        self._cached_vision_cone = (cone_surface, (min_x, min_y))
        self._cached_direction = self.direction
        
        # Blit the transparent surface onto the main surface
        surface.blit(cone_surface, (min_x, min_y))


class Herbivore(Creature):
    __slots__ = ('_food_detection_mask', 'generation')
    
    # Class-level energy configuration cache for performance
    _energy_config = None
    
    def __init__(self, x: float, y: float, direction: Optional[float]=None, 
                 energy: Optional[float]=None, brain: Optional[NeuralNetwork]=None,
                 generation: int=1):
        """Initialize a herbivore with specific properties."""
        # Cache energy configuration if not already cached
        if Herbivore._energy_config is None:
            Herbivore._energy_config = {
                'initial': Config.HERBIVORE_INITIAL_ENERGY,
                'reproduction': Config.HERBIVORE_REPRODUCTION_THRESHOLD,
                'max': Config.HERBIVORE_MAX_ENERGY,
                'consumption': Config.HERBIVORE_ENERGY_CONSUMPTION
            }
        
        energy = Herbivore._energy_config['initial'] if energy is None else energy
        super().__init__(x, y, direction, energy, brain)
        
        # Initialize brain if not provided
        if self.brain is None:
            input_size = self.get_vision_resolution() + 1  # Vision + energy level
            hidden_size = Config.HIDDEN_LAYER_SIZE
            output_size = 2  # Speed and turn
            self.brain = NeuralNetwork(input_size, hidden_size, output_size)
        
        # Specialized properties for herbivores
        self._food_detection_mask = None  # For efficient food detection
        self.generation = generation
    
    def get_vision_resolution(self) -> int:
        return Config.HERBIVORE_VISION_RESOLUTION
    
    def get_vision_range(self) -> float:
        return Config.HERBIVORE_VISION_RANGE
    
    def get_vision_angle(self) -> float:
        return Config.HERBIVORE_VISION_ANGLE
    
    def get_max_speed(self) -> float:
        return Config.HERBIVORE_MAX_SPEED
    
    def get_max_turn(self) -> float:
        return Config.HERBIVORE_MAX_TURN
    
    def _calculate_color(self, energy_ratio: float) -> Tuple[int, int, int]:
        # More nuanced color based on energy and generation
        gen_factor = min(1.0, self.generation / 10)  # Darker for later generations
        base_green = Config.HERBIVORE_COLOR[1]
        green = min(255, int(base_green * energy_ratio * 1.5))
        blue = min(100, int(50 * gen_factor))  # Add blue tint for later generations
        return (0, green, blue)
    
    def get_size(self) -> int:
        return Config.HERBIVORE_SIZE
    
    def get_energy_consumption(self) -> float:
        return Herbivore._energy_config['consumption']
    
    def get_max_energy(self) -> float:
        return Herbivore._energy_config['max']
    
    def can_reproduce(self) -> bool:
        # Enhanced reproduction logic with energy check and random chance
        reproduction_energy = Herbivore._energy_config['reproduction']
        
        if self.energy > reproduction_energy:
            # Higher energy levels increase reproduction chance
            energy_factor = min(1.0, (self.energy - reproduction_energy) / 
                               (Herbivore._energy_config['max'] - reproduction_energy))
            chance_modifier = 0.5 + 0.5 * energy_factor
            return random.random() < Config.HERBIVORE_REPRODUCTION_CHANCE * chance_modifier
        return False
    
    def reproduce(self) -> Optional['Herbivore']:
        """Create a new herbivore with mutated traits."""
        if not self.can_reproduce():
            return None
            
        # Create offspring with mutation
        mutated_brain = self.brain.mutate()
        
        # Calculate position nearby parent using efficient placement
        angle = random.uniform(0, 360)
        distance = self.get_size() * 3
        angle_rad = math.radians(angle)
        
        # Calculate new position with boundary checking
        new_x = min(max(0, self.x + math.cos(angle_rad) * distance), Config.SCREEN_WIDTH)
        new_y = min(max(0, self.y + math.sin(angle_rad) * distance), Config.SCREEN_HEIGHT)
        
        # Split energy between parent and offspring
        offspring_energy = self.energy / 2
        self.energy = self.energy / 2
        
        # Update stats
        self.stats['offspring'] += 1
        
        # Increment generation for offspring
        new_generation = self.generation + 1
        
        return Herbivore(new_x, new_y, random.uniform(0, 360), offspring_energy, mutated_brain, new_generation)


class Carnivore(Creature):
    __slots__ = ('hunt_cooldown', 'kill_count', 'generation')
    
    # Class-level energy configuration cache for performance
    _energy_config = None
    
    def __init__(self, x: float, y: float, direction: Optional[float]=None, 
                 energy: Optional[float]=None, brain: Optional[NeuralNetwork]=None,
                 generation: int=1):
        """Initialize a carnivore with specific properties."""
        # Cache energy configuration if not already cached
        if Carnivore._energy_config is None:
            Carnivore._energy_config = {
                'initial': Config.CARNIVORE_INITIAL_ENERGY,
                'reproduction': Config.CARNIVORE_REPRODUCTION_THRESHOLD,
                'max': Config.CARNIVORE_MAX_ENERGY,
                'consumption': Config.CARNIVORE_ENERGY_CONSUMPTION
            }
        
        energy = Carnivore._energy_config['initial'] if energy is None else energy
        super().__init__(x, y, direction, energy, brain)
        
        # Initialize brain if not provided
        if self.brain is None:
            input_size = self.get_vision_resolution() + 1  # Vision + energy level
            hidden_size = Config.HIDDEN_LAYER_SIZE
            output_size = 2  # Speed and turn
            self.brain = NeuralNetwork(input_size, hidden_size, output_size)
        
        # Specialized properties for carnivores
        self.hunt_cooldown = 0
        self.kill_count = 0
        self.generation = generation
    
    def get_vision_resolution(self) -> int:
        return Config.CARNIVORE_VISION_RESOLUTION
    
    def get_vision_range(self) -> float:
        return Config.CARNIVORE_VISION_RANGE
    
    def get_vision_angle(self) -> float:
        return Config.CARNIVORE_VISION_ANGLE
    
    def get_max_speed(self) -> float:
        return Config.CARNIVORE_MAX_SPEED
    
    def get_max_turn(self) -> float:
        return Config.CARNIVORE_MAX_TURN
    
    def _calculate_color(self, energy_ratio: float) -> Tuple[int, int, int]:
        # More nuanced color based on energy, kills, and generation
        kill_factor = min(1.0, self.kill_count / 10)  # More kills = more purple tint
        gen_factor = min(1.0, self.generation / 10)  # Later generations get darker
        
        red = min(255, int(Config.CARNIVORE_COLOR[0] * energy_ratio * 1.5))
        green = 0
        blue = min(100, int(80 * kill_factor))  # Purple tint for successful hunters
        
        # Darken with generations
        if gen_factor > 0.5:
            red = int(red * (1.5 - gen_factor))
        
        return (red, green, blue)
    
    def get_size(self) -> int:
        return Config.CARNIVORE_SIZE
    
    def get_energy_consumption(self) -> float:
        return Carnivore._energy_config['consumption']
    
    def get_max_energy(self) -> float:
        return Carnivore._energy_config['max']
    
    def can_reproduce(self) -> bool:
        # Enhanced reproduction logic with energy check and random chance
        reproduction_energy = Carnivore._energy_config['reproduction']
        
        if self.energy > reproduction_energy:
            # Higher energy levels and kill count increase reproduction chance
            energy_factor = min(1.0, (self.energy - reproduction_energy) / 
                               (Carnivore._energy_config['max'] - reproduction_energy))
            kill_bonus = min(0.5, self.kill_count * 0.05)  # Successful hunters more likely to reproduce
            chance_modifier = 0.5 + 0.5 * energy_factor + kill_bonus
            return random.random() < Config.CARNIVORE_REPRODUCTION_CHANCE * chance_modifier
        return False
    
    def reproduce(self) -> Optional['Carnivore']:
        """Create a new carnivore with mutated traits."""
        if not self.can_reproduce():
            return None
            
        # Create offspring with mutation
        mutated_brain = self.brain.mutate()
        
        # Calculate position nearby parent using efficient placement
        angle = random.uniform(0, 360)
        distance = self.get_size() * 3
        angle_rad = math.radians(angle)
        
        # Calculate new position with boundary checking
        new_x = min(max(0, self.x + math.cos(angle_rad) * distance), Config.SCREEN_WIDTH)
        new_y = min(max(0, self.y + math.sin(angle_rad) * distance), Config.SCREEN_HEIGHT)
        
        # Split energy between parent and offspring
        offspring_energy = self.energy / 2
        self.energy = self.energy / 2
        
        # Update stats
        self.stats['offspring'] += 1
        
        # Increment generation for offspring
        new_generation = self.generation + 1
        
        return Carnivore(new_x, new_y, random.uniform(0, 360), offspring_energy, mutated_brain, new_generation)