"""
Terrain system - ground tiles with different properties.
"""
import pygame
import random
from enum import Enum
from typing import Tuple, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE,
    MUD_SPEED_MODIFIER, MUD_COVERAGE,
    COLOR_GROUND, COLOR_MUD, COLOR_DIRT
)


class TileType(Enum):
    """Types of terrain tiles."""
    NORMAL = "normal"
    MUD = "mud"
    DIRT = "dirt"


class Terrain:
    """
    Manages the terrain grid with different tile types.
    Affects robot movement speed based on tile type.
    """

    TILE_COLORS = {
        TileType.NORMAL: COLOR_GROUND,
        TileType.MUD: COLOR_MUD,
        TileType.DIRT: COLOR_DIRT,
    }

    SPEED_MODIFIERS = {
        TileType.NORMAL: 1.0,
        TileType.MUD: MUD_SPEED_MODIFIER,
        TileType.DIRT: 0.85,  # Slight slowdown on dirt
    }

    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self.tile_size = TILE_SIZE

        self.cols = width // TILE_SIZE
        self.rows = height // TILE_SIZE

        # 2D grid of tile types
        self.grid: List[List[TileType]] = []

        # Pre-rendered surface for performance
        self.surface: pygame.Surface = None

        self.generate_random()

    def generate_random(self):
        """Generate a random terrain with patches of mud and dirt."""
        # Initialize all as normal
        self.grid = [[TileType.NORMAL for _ in range(self.cols)] for _ in range(self.rows)]

        # Add mud patches
        num_mud_patches = int(self.cols * self.rows * MUD_COVERAGE / 10)
        for _ in range(num_mud_patches):
            self._add_patch(TileType.MUD, random.randint(2, 5))

        # Add dirt patches (less than mud)
        num_dirt_patches = int(self.cols * self.rows * 0.1 / 8)
        for _ in range(num_dirt_patches):
            self._add_patch(TileType.DIRT, random.randint(3, 6))

        # Pre-render the terrain
        self._render_surface()

    def _add_patch(self, tile_type: TileType, size: int):
        """Add a patch of a specific tile type."""
        # Random center point
        center_row = random.randint(0, self.rows - 1)
        center_col = random.randint(0, self.cols - 1)

        # Fill a rough circular area
        for dr in range(-size, size + 1):
            for dc in range(-size, size + 1):
                # Circular-ish shape with some randomness
                if dr * dr + dc * dc <= size * size * random.uniform(0.5, 1.2):
                    row = center_row + dr
                    col = center_col + dc
                    if 0 <= row < self.rows and 0 <= col < self.cols:
                        self.grid[row][col] = tile_type

    def _render_surface(self):
        """Pre-render the entire terrain to a surface."""
        self.surface = pygame.Surface((self.width, self.height))

        for row in range(self.rows):
            for col in range(self.cols):
                tile_type = self.grid[row][col]
                color = self.TILE_COLORS[tile_type]

                rect = pygame.Rect(
                    col * self.tile_size,
                    row * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                pygame.draw.rect(self.surface, color, rect)

                # Add some texture variation
                if tile_type == TileType.MUD:
                    # Draw some darker spots
                    for _ in range(3):
                        spot_x = rect.x + random.randint(2, self.tile_size - 2)
                        spot_y = rect.y + random.randint(2, self.tile_size - 2)
                        darker = (color[0] - 15, color[1] - 15, color[2] - 10)
                        pygame.draw.circle(self.surface, darker, (spot_x, spot_y), 2)
                elif tile_type == TileType.DIRT:
                    # Draw some specks
                    for _ in range(2):
                        spot_x = rect.x + random.randint(2, self.tile_size - 2)
                        spot_y = rect.y + random.randint(2, self.tile_size - 2)
                        lighter = (min(255, color[0] + 20), min(255, color[1] + 15), min(255, color[2] + 10))
                        pygame.draw.circle(self.surface, lighter, (spot_x, spot_y), 1)

    def get_tile_at(self, x: float, y: float) -> TileType:
        """Get the tile type at a world position."""
        col = int(x // self.tile_size)
        row = int(y // self.tile_size)

        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return TileType.NORMAL  # Default for out of bounds

    def get_speed_modifier(self, x: float, y: float) -> float:
        """Get the speed modifier at a world position."""
        tile_type = self.get_tile_at(x, y)
        return self.SPEED_MODIFIERS.get(tile_type, 1.0)

    def draw(self, screen: pygame.Surface):
        """Draw the terrain."""
        screen.blit(self.surface, (0, 0))

    def __repr__(self) -> str:
        return f"Terrain({self.cols}x{self.rows} tiles)"
