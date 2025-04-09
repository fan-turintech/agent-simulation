import pygame
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
