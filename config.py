class Config:
    # Screen settings
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    BACKGROUND_COLOR = (0, 0, 0)
    
    # Environment settings
    BASE_GRASS_GROWTH_RATE = 0.15  # Reduced from 0.2
    MAX_GRASS_GROWTH_RATE = 0.5   # Adjusted from 0.6
    GRASS_RANDOM_SPAWN_CHANCE = 0.03  # Reduced from 0.05
    MAX_GRASS = 300
    MIN_GRASS = 50  # Minimum desired grass count
    
    # Creature settings
    INITIAL_HERBIVORES = 40  # Increased from 20
    INITIAL_CARNIVORES = 10  # Increased from 5
    MINIMUM_HERBIVORES = 5  # Minimum herbivore population
    MINIMUM_CARNIVORES = 5   # Minimum carnivore population
    
    # Herbivore settings
    HERBIVORE_MAX_SPEED = 5.0  # Increased from 2.0
    HERBIVORE_MAX_TURN = 20  # degrees per frame
    HERBIVORE_COLOR = (0, 255, 0)
    HERBIVORE_SIZE = 8
    HERBIVORE_VISION_RANGE = 50  # Increased from 40
    HERBIVORE_VISION_ANGLE = 180  # degrees
    HERBIVORE_VISION_RESOLUTION = 10  # number of "rays"
    HERBIVORE_INITIAL_ENERGY = 100
    HERBIVORE_MAX_ENERGY = 200
    HERBIVORE_ENERGY_CONSUMPTION = 0.2
    HERBIVORE_REPRODUCTION_THRESHOLD = 150
    HERBIVORE_REPRODUCTION_CHANCE = 0.05  # Increased from 0.02
    
    # Carnivore settings
    CARNIVORE_MAX_SPEED = 4.0
    CARNIVORE_MAX_TURN = 30  # degrees per frame
    CARNIVORE_COLOR = (255, 0, 0)
    CARNIVORE_SIZE = 10
    CARNIVORE_VISION_RANGE = 60  # Reduced from 150
    CARNIVORE_VISION_ANGLE = 180  # degrees
    CARNIVORE_VISION_RESOLUTION = 10  # number of "rays"
    CARNIVORE_INITIAL_ENERGY = 150
    CARNIVORE_MAX_ENERGY = 300
    CARNIVORE_ENERGY_CONSUMPTION = 0.35  # Increased by 20% from 0.2
    CARNIVORE_REPRODUCTION_THRESHOLD = 240  # Increased from 200
    CARNIVORE_REPRODUCTION_CHANCE = 0.02  # Increased from 0.01
    CARNIVORE_HUNT_BONUS_ENERGY = 100  # Bonus energy for successful hunt
    
    # Grass settings
    GRASS_COLOR = (0, 200, 0)
    GRASS_SIZE = 5
    GRASS_ENERGY = 50
    
    # Neural network settings
    MUTATION_RATE = 0.1  # Probability of a weight being mutated
    MUTATION_RANGE = 0.2  # Maximum amount a weight can be mutated
    SPAWNED_MUTATION_RATE = 0.15  # Higher mutation rate for spawned creatures
    SPAWNED_MUTATION_RANGE = 0.3  # Higher mutation range for spawned creatures
    HIDDEN_LAYER_SIZE = 16
    
    # Visualization settings
    SHOW_VISION_CONES = True  # Can be toggled to show/hide vision cones
    VISION_CONE_OPACITY = 30  # 0-255 where 0 is transparent and 255 is opaque
