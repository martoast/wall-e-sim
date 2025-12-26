"""
Navigation system - patrol paths and obstacle avoidance.
"""
import pygame
import math
import random
from typing import List, Tuple, Optional, TYPE_CHECKING

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCREEN_WIDTH, SCREEN_HEIGHT, ROBOT_WIDTH, ROBOT_HEIGHT
from utils.math_helpers import (
    distance, angle_to, angle_diff, normalize, rotate_point,
    vector_from_angle, add_vectors, scale_vector
)

if TYPE_CHECKING:
    from entities.robot import Robot
    from entities.obstacle import Obstacle
    from entities.nest import Nest


class Navigation:
    """
    Handles robot navigation including:
    - Patrol path generation
    - Waypoint following
    - Obstacle avoidance
    - Path to target navigation
    """

    def __init__(
        self,
        bounds: Tuple[int, int, int, int] = None,
        margin: int = 80
    ):
        """
        Args:
            bounds: (x, y, width, height) of navigable area
            margin: Distance to keep from edges
        """
        if bounds is None:
            bounds = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

        self.bounds = bounds
        self.margin = margin

        # Define safe navigation area
        self.nav_bounds = (
            bounds[0] + margin,
            bounds[1] + margin,
            bounds[2] - margin * 2,
            bounds[3] - margin * 2
        )

    def generate_patrol_loop(
        self,
        center: Tuple[float, float] = None,
        radius: float = None,
        point_count: int = 6
    ) -> List[Tuple[float, float]]:
        """
        Generate a circular patrol path.

        Args:
            center: Center of patrol area (defaults to screen center)
            radius: Radius of patrol (defaults to fit in bounds)
            point_count: Number of waypoints

        Returns:
            List of waypoint positions
        """
        if center is None:
            center = (
                self.nav_bounds[0] + self.nav_bounds[2] / 2,
                self.nav_bounds[1] + self.nav_bounds[3] / 2
            )

        if radius is None:
            # Fit within navigation bounds
            radius = min(self.nav_bounds[2], self.nav_bounds[3]) / 2 - 50

        waypoints = []
        for i in range(point_count):
            angle = (i / point_count) * 360
            rad = math.radians(angle)

            # Add some variation to make it less perfect
            r = radius * random.uniform(0.8, 1.0)

            x = center[0] + math.cos(rad) * r
            y = center[1] + math.sin(rad) * r

            # Clamp to bounds
            x = max(self.nav_bounds[0], min(x, self.nav_bounds[0] + self.nav_bounds[2]))
            y = max(self.nav_bounds[1], min(y, self.nav_bounds[1] + self.nav_bounds[3]))

            waypoints.append((x, y))

        return waypoints

    def generate_random_patrol(
        self,
        point_count: int = 6,
        nest_position: Tuple[float, float] = None,
        zone_bounds: Tuple[int, int, int, int] = None
    ) -> List[Tuple[float, float]]:
        """
        Generate a random patrol path that avoids the nest area.

        Args:
            point_count: Number of waypoints
            nest_position: Position of nest to avoid
            zone_bounds: Optional (x, y, width, height) to constrain patrol area

        Returns:
            List of waypoint positions
        """
        waypoints = []
        nest_avoid_radius = 200  # Stay this far from nest during patrol

        # Use zone bounds if provided, otherwise use full nav bounds
        if zone_bounds:
            bounds = zone_bounds
        else:
            bounds = self.nav_bounds

        attempts = 0
        while len(waypoints) < point_count and attempts < 100:
            x = random.uniform(bounds[0], bounds[0] + bounds[2])
            y = random.uniform(bounds[1], bounds[1] + bounds[3])

            # Check if too close to nest
            if nest_position:
                dist_to_nest = distance((x, y), nest_position)
                if dist_to_nest < nest_avoid_radius:
                    attempts += 1
                    continue

            # Check if too close to other waypoints
            too_close = False
            for wp in waypoints:
                if distance((x, y), wp) < 100:
                    too_close = True
                    break

            if not too_close:
                waypoints.append((x, y))

            attempts += 1

        return waypoints

    def get_next_waypoint(
        self,
        current_index: int,
        waypoints: List[Tuple[float, float]]
    ) -> Tuple[int, Tuple[float, float]]:
        """
        Get the next waypoint in sequence.

        Args:
            current_index: Current waypoint index
            waypoints: List of waypoints

        Returns:
            (new_index, waypoint_position)
        """
        next_index = (current_index + 1) % len(waypoints)
        return (next_index, waypoints[next_index])

    def navigate_to_target(
        self,
        robot: 'Robot',
        target: Tuple[float, float],
        obstacles: pygame.sprite.Group,
        other_robots: List['Robot'] = None,
        trash_group: pygame.sprite.Group = None,
        target_trash: 'Trash' = None,
        dt: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate movement vector toward target.

        Simple approach: just move toward target.
        Physics handles collision response, stuck detection handles re-pathing.

        Args:
            robot: The robot
            target: Target position
            obstacles: Group of static obstacles (unused - physics handles)
            other_robots: Other robots (unused - physics handles)
            trash_group: Trash items (unused - physics handles)
            target_trash: The specific trash we're targeting (unused)
            dt: Delta time

        Returns:
            Movement vector (dx, dy)
        """
        # Just go toward the target - physics will handle collisions
        target_angle = angle_to(robot.position, target)

        # Create movement vector
        speed = robot.current_speed * dt
        move_vec = vector_from_angle(target_angle, speed)

        return move_vec

    def is_at_waypoint(
        self,
        robot: 'Robot',
        waypoint: Tuple[float, float],
        threshold: float = 20.0
    ) -> bool:
        """Check if robot has reached a waypoint."""
        return distance(robot.position, waypoint) < threshold

    def is_in_bounds(self, position: Tuple[float, float]) -> bool:
        """Check if a position is within navigation bounds."""
        x, y = position
        return (
            self.nav_bounds[0] <= x <= self.nav_bounds[0] + self.nav_bounds[2] and
            self.nav_bounds[1] <= y <= self.nav_bounds[1] + self.nav_bounds[3]
        )

    def clamp_to_bounds(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Clamp a position to within navigation bounds."""
        x = max(self.nav_bounds[0], min(position[0], self.nav_bounds[0] + self.nav_bounds[2]))
        y = max(self.nav_bounds[1], min(position[1], self.nav_bounds[1] + self.nav_bounds[3]))
        return (x, y)

    def draw_debug(
        self,
        screen: pygame.Surface,
        waypoints: List[Tuple[float, float]],
        current_index: int
    ):
        """Draw debug visualization of patrol path."""
        if not waypoints:
            return

        # Draw lines between waypoints
        for i in range(len(waypoints)):
            start = waypoints[i]
            end = waypoints[(i + 1) % len(waypoints)]

            color = (100, 100, 100)
            pygame.draw.line(
                screen, color,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                1
            )

        # Draw waypoint markers
        for i, wp in enumerate(waypoints):
            if i == current_index:
                color = (0, 255, 0)  # Current waypoint
                size = 8
            else:
                color = (100, 100, 100)
                size = 5

            pygame.draw.circle(
                screen, color,
                (int(wp[0]), int(wp[1])),
                size
            )

            # Draw waypoint number
            # font = pygame.font.Font(None, 20)
            # label = font.render(str(i), True, (200, 200, 200))
            # screen.blit(label, (int(wp[0]) + 10, int(wp[1]) - 10))
