"""
Trash entity - collectible items for the robot.
"""
import pygame
import random
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRASH_SIZE_MIN, TRASH_SIZE_MAX,
    COLOR_TRASH, COLOR_TRASH_CAN, COLOR_TRASH_BOTTLE, COLOR_TRASH_PAPER
)

# Global counter for unique trash IDs
_next_trash_id = 0


class Trash(pygame.sprite.Sprite):
    """A piece of trash that can be picked up by a robot."""

    TYPES = ['general', 'can', 'bottle', 'paper']
    TYPE_COLORS = {
        'general': COLOR_TRASH,
        'can': COLOR_TRASH_CAN,
        'bottle': COLOR_TRASH_BOTTLE,
        'paper': COLOR_TRASH_PAPER,
    }

    def __init__(
        self,
        position: Tuple[float, float],
        size: Optional[int] = None,
        trash_type: Optional[str] = None
    ):
        super().__init__()

        # Assign unique ID
        global _next_trash_id
        self.id = _next_trash_id
        _next_trash_id += 1

        self.x, self.y = position
        self.size = size if size else random.randint(TRASH_SIZE_MIN, TRASH_SIZE_MAX)
        self.trash_type = trash_type if trash_type else random.choice(self.TYPES)
        self.is_picked = False
        self.held_by = None  # Reference to robot holding this trash

        # Physics - velocity for being pushed
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.mass = self.size  # Larger trash is heavier

        # Create the sprite image
        self._create_image()

    def _create_image(self):
        """Create the visual representation of the trash."""
        diameter = self.size * 2
        self.image = pygame.Surface((diameter, diameter), pygame.SRCALPHA)

        color = self.TYPE_COLORS.get(self.trash_type, COLOR_TRASH)

        # Draw based on type
        if self.trash_type == 'can':
            # Cylindrical can shape (rectangle with rounded ends)
            rect = pygame.Rect(2, self.size // 2, diameter - 4, self.size)
            pygame.draw.rect(self.image, color, rect, border_radius=3)
            pygame.draw.ellipse(self.image, color, (2, 2, diameter - 4, self.size // 2))
        elif self.trash_type == 'bottle':
            # Bottle shape (oval with neck)
            pygame.draw.ellipse(self.image, color, (4, self.size // 2, diameter - 8, self.size))
            neck_width = diameter // 4
            pygame.draw.rect(
                self.image, color,
                (diameter // 2 - neck_width // 2, 2, neck_width, self.size // 2)
            )
        elif self.trash_type == 'paper':
            # Crumpled paper (irregular polygon)
            points = [
                (self.size, 2),
                (diameter - 4, self.size // 2),
                (diameter - 2, self.size),
                (self.size + 2, diameter - 2),
                (4, self.size + 2),
                (2, self.size // 2),
            ]
            pygame.draw.polygon(self.image, color, points)
        else:
            # General trash (circle)
            pygame.draw.circle(
                self.image, color,
                (self.size, self.size),
                self.size
            )

        self.rect = self.image.get_rect()
        self.rect.center = (int(self.x), int(self.y))

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position."""
        return (self.x, self.y)

    @position.setter
    def position(self, pos: Tuple[float, float]):
        """Set position."""
        self.x, self.y = pos
        self.rect.center = (int(self.x), int(self.y))

    def pick_up(self, robot) -> bool:
        """
        Mark this trash as picked up.

        Args:
            robot: The robot picking up this trash

        Returns:
            True if successfully picked up, False if already picked
        """
        if self.is_picked:
            return False

        self.is_picked = True
        self.held_by = robot
        return True

    def release(self) -> bool:
        """
        Release this trash (drop it).

        Returns:
            True if successfully released
        """
        self.is_picked = False
        self.held_by = None
        return True

    def update(self):
        """Update trash state."""
        # If being held, position is managed by the arm
        if self.is_picked and self.held_by:
            # Position will be set by the arm holding it
            pass

        self.rect.center = (int(self.x), int(self.y))

    def draw(self, screen: pygame.Surface):
        """Draw the trash on screen."""
        if not self.is_picked:
            screen.blit(self.image, self.rect)

    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle."""
        return self.rect

    def __repr__(self) -> str:
        return f"Trash(id={self.id}, {self.trash_type}, pos={self.position}, picked={self.is_picked})"
