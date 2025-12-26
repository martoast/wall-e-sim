"""
Robot entity - the main WALL-E style garbage collection robot.
"""
import pygame
import math
from typing import Tuple, Optional, List, TYPE_CHECKING
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ROBOT_WIDTH, ROBOT_HEIGHT, ROBOT_SPEED, ROBOT_TURN_SPEED,
    ROBOT_BIN_CAPACITY, ROBOT_SENSOR_RANGE, ROBOT_GRAB_RANGE,
    COLOR_ROBOT_BODY, COLOR_ROBOT_TRACKS,
    LED_PATROL, LED_SEEKING, LED_APPROACHING, LED_PICKING,
    LED_STORING, LED_RETURNING, LED_WAITING, LED_DOCKING, LED_DUMPING, LED_IDLE
)
from utils.math_helpers import (
    distance, normalize, angle_to, angle_diff, lerp_angle, rotate_point
)

if TYPE_CHECKING:
    from .arm import Arm
    from .trash import Trash


class RobotState(Enum):
    """States the robot can be in."""
    PATROL = "patrol"
    SEEKING = "seeking"
    APPROACHING = "approaching"
    PICKING = "picking"
    STORING = "storing"
    RETURNING = "returning"
    WAITING = "waiting"  # Waiting in queue to dump
    DOCKING = "docking"
    DUMPING = "dumping"
    UNDOCKING = "undocking"  # Driving back down the ramp
    IDLE = "idle"


# LED colors by state
STATE_LED_COLORS = {
    RobotState.PATROL: LED_PATROL,
    RobotState.SEEKING: LED_SEEKING,
    RobotState.APPROACHING: LED_APPROACHING,
    RobotState.PICKING: LED_PICKING,
    RobotState.STORING: LED_STORING,
    RobotState.RETURNING: LED_RETURNING,
    RobotState.WAITING: LED_WAITING,
    RobotState.DOCKING: LED_DOCKING,
    RobotState.DUMPING: LED_DUMPING,
    RobotState.UNDOCKING: LED_DOCKING,  # Same as docking
    RobotState.IDLE: LED_IDLE,
}


class Robot(pygame.sprite.Sprite):
    """
    A WALL-E style garbage collection robot.

    Features:
    - Boxy body with tank tracks
    - LED eyes that change color based on state
    - Internal bin for storing collected trash
    - Articulated arm (managed separately)
    """

    def __init__(self, position: Tuple[float, float], robot_id: int = 0):
        super().__init__()

        self.id = robot_id
        self.x, self.y = position
        self.angle = 0.0  # Facing direction in degrees (0 = right)
        self.base_speed = ROBOT_SPEED
        self.speed_modifier = 1.0  # Affected by terrain
        self.turn_speed = ROBOT_TURN_SPEED

        self.width = ROBOT_WIDTH
        self.height = ROBOT_HEIGHT

        # State
        self.state = RobotState.PATROL
        self.led_color = STATE_LED_COLORS[self.state]

        # Internal bin
        self.bin_capacity = ROBOT_BIN_CAPACITY
        self.internal_bin: List['Trash'] = []

        # Arm reference (set by Arm class)
        self.arm: Optional['Arm'] = None

        # Movement
        self.target_position: Optional[Tuple[float, float]] = None
        self.target_angle: Optional[float] = None
        self.is_moving = False

        # Physics - velocity for physics system
        self.velocity: Tuple[float, float] = (0.0, 0.0)

        # Stuck detection
        self._position_history: List[Tuple[float, float]] = []
        self._stuck_counter = 0

        # Create collision rect
        self.rect = pygame.Rect(0, 0, self.width, self.height)
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

    @property
    def current_speed(self) -> float:
        """Get current speed including terrain modifier."""
        return self.base_speed * self.speed_modifier

    @property
    def bin_count(self) -> int:
        """Number of items in the bin."""
        return len(self.internal_bin)

    @property
    def bin_full(self) -> bool:
        """Check if bin is at capacity."""
        return len(self.internal_bin) >= self.bin_capacity

    @property
    def bin_empty(self) -> bool:
        """Check if bin is empty."""
        return len(self.internal_bin) == 0

    def set_state(self, new_state: RobotState):
        """Change the robot state and update LED color."""
        self.state = new_state
        self.led_color = STATE_LED_COLORS.get(new_state, LED_IDLE)

    def set_speed_modifier(self, modifier: float):
        """Set speed modifier (e.g., from terrain)."""
        self.speed_modifier = max(0.1, min(1.0, modifier))

    def set_velocity(self, vx: float, vy: float):
        """Set intended velocity for this frame (physics will apply it)."""
        self.velocity = (vx, vy)

    def add_to_bin(self, trash: 'Trash') -> bool:
        """
        Add trash to internal bin.

        Args:
            trash: The trash item to add

        Returns:
            True if added successfully, False if bin is full
        """
        if self.bin_full:
            return False

        self.internal_bin.append(trash)
        return True

    def empty_bin(self) -> int:
        """
        Empty the internal bin.

        Returns:
            Number of items emptied
        """
        count = len(self.internal_bin)
        self.internal_bin.clear()
        return count

    def move_toward(self, target: Tuple[float, float], dt: float = 1.0) -> bool:
        """
        Move toward a target position (sets velocity for physics).

        Args:
            target: Target position (x, y)
            dt: Delta time multiplier

        Returns:
            True if reached target
        """
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 2:  # Close enough
            return True

        # First rotate toward target
        target_angle = angle_to(self.position, target)
        angle_difference = abs(angle_diff(self.angle, target_angle))

        if angle_difference > 5:
            # Need to rotate first
            self.rotate_toward(target_angle, dt)
            # Only move if roughly facing target
            if angle_difference > 30:
                return False

        # Set velocity toward target (physics will apply movement)
        move_dist = min(self.current_speed * dt, dist)
        vx = (dx / dist) * move_dist
        vy = (dy / dist) * move_dist
        self.set_velocity(vx, vy)

        return dist <= move_dist

    def rotate_toward(self, target_angle: float, dt: float = 1.0):
        """
        Rotate toward a target angle.

        Args:
            target_angle: Target angle in degrees
            dt: Delta time multiplier
        """
        diff = angle_diff(self.angle, target_angle)
        rotate_amount = min(abs(diff), self.turn_speed * dt)

        if diff > 0:
            self.angle += rotate_amount
        else:
            self.angle -= rotate_amount

        # Normalize angle
        self.angle = self.angle % 360

    def get_front_position(self) -> Tuple[float, float]:
        """Get the position in front of the robot."""
        front_offset = self.width / 2 + 5
        rad = math.radians(self.angle)
        return (
            self.x + math.cos(rad) * front_offset,
            self.y + math.sin(rad) * front_offset
        )

    def get_arm_mount_position(self) -> Tuple[float, float]:
        """Get the position where the arm is mounted."""
        # Arm mounts at front-center of robot
        mount_offset = self.width / 2 - 5
        rad = math.radians(self.angle)
        return (
            self.x + math.cos(rad) * mount_offset,
            self.y + math.sin(rad) * mount_offset
        )

    def update(self, dt: float = 1.0):
        """Update robot state."""
        self.rect.center = (int(self.x), int(self.y))

        # Track position for stuck detection
        self._position_history.append((self.x, self.y))
        if len(self._position_history) > 30:  # Keep last 30 frames
            self._position_history.pop(0)

    def is_stuck(self) -> bool:
        """Check if robot hasn't moved significantly in recent frames."""
        if len(self._position_history) < 15:
            return False

        # Check if we've moved less than 10 pixels in last 15 frames
        # More sensitive to catch jittering sooner
        oldest = self._position_history[0]
        newest = self._position_history[-1]
        dist_moved = math.sqrt((newest[0] - oldest[0])**2 + (newest[1] - oldest[1])**2)

        return dist_moved < 10.0

    def clear_stuck(self):
        """Clear stuck detection history."""
        self._position_history.clear()
        self._stuck_counter = 0

    def draw(self, screen: pygame.Surface):
        """Draw the robot."""
        # Create a surface for the robot that we can rotate
        surf_size = max(self.width, self.height) + 20
        robot_surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        center = surf_size // 2

        # Calculate positions relative to center
        body_rect = pygame.Rect(
            center - self.width // 2,
            center - self.height // 2,
            self.width,
            self.height
        )

        # Draw tracks (two rectangles on sides)
        track_height = 8
        track_width = self.width - 4

        # Top track
        track_top = pygame.Rect(
            center - track_width // 2,
            center - self.height // 2 - track_height // 2,
            track_width,
            track_height
        )
        pygame.draw.rect(robot_surface, COLOR_ROBOT_TRACKS, track_top, border_radius=2)

        # Bottom track
        track_bottom = pygame.Rect(
            center - track_width // 2,
            center + self.height // 2 - track_height // 2,
            track_width,
            track_height
        )
        pygame.draw.rect(robot_surface, COLOR_ROBOT_TRACKS, track_bottom, border_radius=2)

        # Draw body
        pygame.draw.rect(robot_surface, COLOR_ROBOT_BODY, body_rect, border_radius=4)

        # Draw LED eyes
        eye_size = 5
        eye_spacing = 10
        eye_y = center - 2

        # Left eye
        pygame.draw.circle(
            robot_surface, self.led_color,
            (center + self.width // 4 - eye_spacing // 2, eye_y),
            eye_size
        )
        # Right eye
        pygame.draw.circle(
            robot_surface, self.led_color,
            (center + self.width // 4 + eye_spacing // 2, eye_y),
            eye_size
        )

        # Draw direction indicator (small triangle at front)
        front_x = center + self.width // 2
        pygame.draw.polygon(
            robot_surface, COLOR_ROBOT_TRACKS,
            [
                (front_x, center),
                (front_x - 6, center - 4),
                (front_x - 6, center + 4),
            ]
        )

        # Rotate the surface
        rotated = pygame.transform.rotate(robot_surface, -self.angle)
        rotated_rect = rotated.get_rect(center=(int(self.x), int(self.y)))

        screen.blit(rotated, rotated_rect)

    def draw_debug(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw debug information."""
        # State label
        label = font.render(self.state.value.upper(), True, (255, 255, 255))
        label_rect = label.get_rect(center=(int(self.x), int(self.y) - self.height - 15))
        screen.blit(label, label_rect)

        # Bin level bar
        bar_width = 40
        bar_height = 6
        bar_x = int(self.x) - bar_width // 2
        bar_y = int(self.y) + self.height // 2 + 5

        # Background
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        # Fill
        fill_width = int(bar_width * (self.bin_count / self.bin_capacity))
        fill_color = (100, 200, 100) if not self.bin_full else (200, 100, 100)
        pygame.draw.rect(screen, fill_color, (bar_x, bar_y, fill_width, bar_height))
        # Border
        pygame.draw.rect(screen, (150, 150, 150), (bar_x, bar_y, bar_width, bar_height), 1)

    def get_collision_rect(self) -> pygame.Rect:
        """Get collision rectangle."""
        return self.rect

    def __repr__(self) -> str:
        return f"Robot(id={self.id}, pos={self.position}, state={self.state.value}, bin={self.bin_count}/{self.bin_capacity})"
