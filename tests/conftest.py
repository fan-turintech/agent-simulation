import pytest
import sys
import os

# Add parent directory to path to import simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from config import Config

@pytest.fixture(scope="session", autouse=True)
def setup_test_config():
    """Configure environment for testing"""
    # Save original values (if they exist)
    original_values = {}
    for attr in ['INITIAL_HERBIVORES', 'INITIAL_CARNIVORES', 
                'MINIMUM_HERBIVORES', 'MINIMUM_CARNIVORES', 
                'MAX_GRASS', 'HERBIVORE_INITIAL_ENERGY', 
                'HERBIVORE_MAX_ENERGY', 'GRASS_ENERGY']:
        if hasattr(Config, attr):
            original_values[attr] = getattr(Config, attr)
    
    # Set required test values
    Config.INITIAL_HERBIVORES = 5
    Config.INITIAL_CARNIVORES = 3
    Config.MINIMUM_HERBIVORES = 3
    Config.MINIMUM_CARNIVORES = 2
    Config.MAX_GRASS = 50
    Config.HERBIVORE_INITIAL_ENERGY = 100
    Config.HERBIVORE_MAX_ENERGY = 200
    Config.GRASS_ENERGY = 50
    
    # Return config for potential use in tests
    yield Config
    
    # Restore original values after tests
    for key, value in original_values.items():
        setattr(Config, key, value)
