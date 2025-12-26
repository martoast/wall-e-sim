"""
Nest entity - central bin with ramp for trash disposal.
"""
import pygame
import math
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    NEST_WIDTH, NEST_HEIGHT, NEST_POSITION, NEST_CAPACITY,
    RAMP_WIDTH, RAMP_LENGTH, RAMP_ANGLE,
    COLOR_NEST, COLOR_NEST_FILL, COLOR_RAMP
)


class Nest(pygame.sprite.Sprite):
    """
    Central collection bin with a ramp for robots to climb and dump trash.

    The nest consists of:
    - Main bin (tall rectangle)
    - Ramp leading up to dumping position
    - One-way flap (visual only)
    """

    def __init__(self, position: Tuple[float, float] = None):
        super().__init__()

        if position is None:
            position = NEST_POSITION

        self.x, self.y = position
        self.width = NEST_WIDTH
        self.height = NEST_HEIGHT

        # Ramp properties
        self.ramp_width = RAMP_WIDTH
        self.ramp_length = RAMP_LENGTH
        self.ramp_angle = RAMP_ANGLE  # degrees incline

        # Capacity
        self.capacity = NEST_CAPACITY
        self.fill_level = 0

        # Calculate key positions
        self._calculate_positions()

        # Create rect for collision
        self.rect = pygame.Rect(
            int(self.x - self.width // 2),
            int(self.y - self.height // 2),
            self.width,
            self.height
        )

    def _calculate_positions(self):
        """Calculate important positions for navigation."""
        # Ramp starts to the left of the bin
        self.ramp_start = (
            self.x - self.width // 2 - self.ramp_length,
            self.y + self.height // 2 - 20  # Near bottom
        )

        # Ramp ends at the bin
        self.ramp_end = (
            self.x - self.width // 2,
            self.y + self.height // 2 - 20 - math.tan(math.radians(self.ramp_angle)) * self.ramp_length
        )

        # Dock position is at top of ramp
        self.dock_position = (
            self.x - self.width // 2 + 10,
            self.ramp_end[1]
        )

        # Dump position is where trash goes
        self.dump_position = (
            self.x,
            self.y - self.height // 4
        )

        # Ramp collision rect
        self.ramp_rect = pygame.Rect(
            int(self.ramp_start[0]),
            int(self.ramp_end[1]) - 30,
            int(self.ramp_length),
            60
        )

    @property
    def position(self) -> Tuple[float, float]:
        """Get center position."""
        return (self.x, self.y)

    @property
    def is_full(self) -> bool:
        """Check if the bin is full."""
        return self.fill_level >= self.capacity

    @property
    def fill_percentage(self) -> float:
        """Get fill level as percentage."""
        return self.fill_level / self.capacity

    def get_ramp_entry(self) -> Tuple[float, float]:
        """Get the position where robots should start climbing the ramp."""
        return self.ramp_start

    def get_dock_position(self) -> Tuple[float, float]:
        """Get the position where robots dock to dump."""
        return self.dock_position

    def get_dump_position(self) -> Tuple[float, float]:
        """Get the position above the bin where trash is released."""
        return self.dump_position

    def get_waiting_position(self, queue_position: int = 0) -> Tuple[float, float]:
        """
        Get a waiting position for robots in queue, away from the ramp.

        Args:
            queue_position: Position in queue (0 = first waiting, 1 = second, etc.)

        Returns:
            Position where robot should wait
        """
        # Wait to the left and below the ramp entry, staggered by queue position
        base_x = self.ramp_start[0] - 80  # Left of ramp
        base_y = self.ramp_start[1] + 60  # Below ramp

        # Stagger waiting positions so robots don't stack on each other
        offset_x = (queue_position % 3) * 60  # 3 columns
        offset_y = (queue_position // 3) * 60  # Multiple rows if needed

        return (base_x - offset_x, base_y + offset_y)

    def is_robot_at_ramp_entry(self, robot_pos: Tuple[float, float], threshold: float = 30) -> bool:
        """Check if robot is at the ramp entry point."""
        dx = robot_pos[0] - self.ramp_start[0]
        dy = robot_pos[1] - self.ramp_start[1]
        return math.sqrt(dx * dx + dy * dy) < threshold

    def is_robot_docked(self, robot_pos: Tuple[float, float], threshold: float = 20) -> bool:
        """Check if robot is properly docked for dumping."""
        dx = robot_pos[0] - self.dock_position[0]
        dy = robot_pos[1] - self.dock_position[1]
        return math.sqrt(dx * dx + dy * dy) < threshold

    def receive_trash(self, count: int) -> int:
        """
        Receive trash from a robot.

        Args:
            count: Number of trash items being dumped

        Returns:
            Number actually received (might be less if nearly full)
        """
        space_available = self.capacity - self.fill_level
        received = min(count, space_available)
        self.fill_level += received
        return received

    def update(self):
        """Update nest state."""
        pass

    def draw(self, screen: pygame.Surface):
        """Draw the nest."""
        # Draw ramp
        ramp_points = [
            self.ramp_start,
            self.ramp_end,
            (self.ramp_end[0], self.ramp_end[1] + 15),
            (self.ramp_start[0], self.ramp_start[1] + 15),
        ]
        pygame.draw.polygon(screen, COLOR_RAMP, [(int(p[0]), int(p[1])) for p in ramp_points])

        # Ramp side rails
        pygame.draw.line(
            screen, (COLOR_RAMP[0] - 30, COLOR_RAMP[1] - 30, COLOR_RAMP[2] - 30),
            (int(self.ramp_start[0]), int(self.ramp_start[1]) - 5),
            (int(self.ramp_end[0]), int(self.ramp_end[1]) - 5),
            3
        )
        pygame.draw.line(
            screen, (COLOR_RAMP[0] - 30, COLOR_RAMP[1] - 30, COLOR_RAMP[2] - 30),
            (int(self.ramp_start[0]), int(self.ramp_start[1]) + 20),
            (int(self.ramp_end[0]), int(self.ramp_end[1]) + 20),
            3
        )

        # Draw main bin
        bin_rect = pygame.Rect(
            int(self.x - self.width // 2),
            int(self.y - self.height // 2),
            self.width,
            self.height
        )
        pygame.draw.rect(screen, COLOR_NEST, bin_rect, border_radius=5)

        # Draw fill level
        fill_height = int(self.height * self.fill_percentage * 0.8)
        if fill_height > 0:
            fill_rect = pygame.Rect(
                int(self.x - self.width // 2) + 5,
                int(self.y + self.height // 2) - fill_height - 5,
                self.width - 10,
                fill_height
            )
            pygame.draw.rect(screen, COLOR_NEST_FILL, fill_rect, border_radius=3)

        # Draw bin opening (top)
        opening_rect = pygame.Rect(
            int(self.x - self.width // 2) + 10,
            int(self.y - self.height // 2) - 5,
            self.width - 20,
            15
        )
        pygame.draw.rect(screen, (40, 40, 50), opening_rect, border_radius=3)

        # Draw one-way flap (cosmetic)
        flap_y = int(self.y - self.height // 2) + 5
        pygame.draw.line(
            screen, (60, 60, 70),
            (int(self.x - self.width // 4), flap_y),
            (int(self.x + self.width // 4), flap_y),
            2
        )

        # Draw bin border
        pygame.draw.rect(screen, (60, 65, 80), bin_rect, 2, border_radius=5)

    def draw_debug(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw debug information."""
        # Draw dock position marker
        pygame.draw.circle(
            screen, (0, 255, 255),
            (int(self.dock_position[0]), int(self.dock_position[1])),
            8, 2
        )

        # Draw ramp entry marker
        pygame.draw.circle(
            screen, (255, 255, 0),
            (int(self.ramp_start[0]), int(self.ramp_start[1])),
            8, 2
        )

        # Draw fill level text
        fill_text = f"{self.fill_level}/{self.capacity}"
        label = font.render(fill_text, True, (255, 255, 255))
        label_rect = label.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(label, label_rect)

    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle."""
        return self.rect

    def __repr__(self) -> str:
        return f"Nest(pos={self.position}, fill={self.fill_level}/{self.capacity})"
