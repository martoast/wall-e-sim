"""
Obstacle entity - rocks, walls, and other impassable objects.
"""
import pygame
import random
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX,
    COLOR_OBSTACLE
)


class Obstacle(pygame.sprite.Sprite):
    """An impassable obstacle in the environment."""

    TYPES = ['rock', 'wall']

    def __init__(
        self,
        position: Tuple[float, float],
        size: Optional[Tuple[int, int]] = None,
        obstacle_type: Optional[str] = None
    ):
        super().__init__()

        self.x, self.y = position
        self.obstacle_type = obstacle_type if obstacle_type else random.choice(self.TYPES)

        # Set size
        if size:
            self.width, self.height = size
        else:
            base_size = random.randint(OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX)
            if self.obstacle_type == 'wall':
                # Walls are elongated
                self.width = base_size * 2
                self.height = base_size // 2
            else:
                # Rocks are roughly square
                self.width = base_size
                self.height = base_size

        # Create the sprite image
        self._create_image()

    def _create_image(self):
        """Create the visual representation of the obstacle."""
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        if self.obstacle_type == 'wall':
            # Wall - solid rectangle
            pygame.draw.rect(
                self.image, COLOR_OBSTACLE,
                (0, 0, self.width, self.height)
            )
            # Add some texture lines
            for i in range(0, self.width, 20):
                pygame.draw.line(
                    self.image, (COLOR_OBSTACLE[0] - 20, COLOR_OBSTACLE[1] - 20, COLOR_OBSTACLE[2] - 20),
                    (i, 0), (i, self.height), 1
                )
        else:
            # Rock - irregular polygon
            # Create a rough circular shape with some randomness
            center_x = self.width // 2
            center_y = self.height // 2
            points = []
            num_points = 8

            for i in range(num_points):
                angle = (i / num_points) * 360
                radius = min(center_x, center_y) * random.uniform(0.7, 1.0)
                import math
                x = center_x + radius * math.cos(math.radians(angle))
                y = center_y + radius * math.sin(math.radians(angle))
                points.append((x, y))

            pygame.draw.polygon(self.image, COLOR_OBSTACLE, points)

            # Add some shading
            darker = (COLOR_OBSTACLE[0] - 30, COLOR_OBSTACLE[1] - 30, COLOR_OBSTACLE[2] - 30)
            # Draw a smaller darker polygon offset
            offset_points = [(p[0] + 2, p[1] + 2) for p in points[:len(points)//2]]
            if len(offset_points) >= 3:
                pygame.draw.polygon(self.image, darker, offset_points + [points[len(points)//2]])

        self.rect = self.image.get_rect()
        self.rect.center = (int(self.x), int(self.y))

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position."""
        return (self.x, self.y)

    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle."""
        return self.rect

    def draw(self, screen: pygame.Surface):
        """Draw the obstacle on screen."""
        screen.blit(self.image, self.rect)

    def __repr__(self) -> str:
        return f"Obstacle({self.obstacle_type}, pos={self.position}, size=({self.width}, {self.height}))"
