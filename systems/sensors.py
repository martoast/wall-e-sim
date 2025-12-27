"""
Sensor system - collision detection, vision, and distance checks.
"""
import pygame
import math
from typing import List, Tuple, Optional, TYPE_CHECKING

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ROBOT_SENSOR_RANGE, ROBOT_VISION_CONE, ROBOT_GRAB_RANGE,
    COLOR_DEBUG_SENSOR, COLOR_DEBUG_VISION
)
from utils.math_helpers import distance, angle_to, angle_diff, point_in_cone, normalize

if TYPE_CHECKING:
    from entities.robot import Robot
    from entities.trash import Trash
    from entities.obstacle import Obstacle
    from entities.nest import Nest


class SensorSystem:
    """
    Handles all sensor-related functionality for robots.

    Includes:
    - Collision detection
    - Trash detection (range and vision cone)
    - Nest detection
    - Raycast for obstacle avoidance
    """

    def __init__(self, sensor_range: float = ROBOT_SENSOR_RANGE):
        self.sensor_range = sensor_range
        self.vision_cone = ROBOT_VISION_CONE
        self.grab_range = ROBOT_GRAB_RANGE

    def check_collision(
        self,
        robot: 'Robot',
        obstacles: pygame.sprite.Group
    ) -> bool:
        """
        Check if robot collides with any obstacle.

        Args:
            robot: The robot to check
            obstacles: Group of obstacles

        Returns:
            True if collision detected
        """
        robot_rect = robot.get_collision_rect()
        for obstacle in obstacles:
            if robot_rect.colliderect(obstacle.get_rect()):
                return True
        return False

    def check_collision_at(
        self,
        position: Tuple[float, float],
        size: Tuple[float, float],
        obstacles: pygame.sprite.Group
    ) -> bool:
        """
        Check collision at a specific position.

        Args:
            position: Position to check (center)
            size: Size of the collision box (width, height)
            obstacles: Group of obstacles

        Returns:
            True if collision would occur
        """
        test_rect = pygame.Rect(
            position[0] - size[0] / 2,
            position[1] - size[1] / 2,
            size[0],
            size[1]
        )

        for obstacle in obstacles:
            if test_rect.colliderect(obstacle.get_rect()):
                return True
        return False

    def detect_trash_in_range(
        self,
        robot: 'Robot',
        trash_group: pygame.sprite.Group,
        range_override: float = None
    ) -> List['Trash']:
        """
        Detect all trash within sensor range.

        Args:
            robot: The robot doing the sensing
            trash_group: Group of trash items
            range_override: Optional custom range

        Returns:
            List of detected trash items, sorted by distance
        """
        detection_range = range_override or self.sensor_range
        detected = []

        for trash in trash_group:
            if trash.is_picked:
                continue

            dist = distance(robot.position, trash.position)
            if dist <= detection_range:
                detected.append((dist, trash))

        # Sort by distance and return just the trash objects
        detected.sort(key=lambda x: x[0])
        return [t for _, t in detected]

    def detect_trash_in_vision(
        self,
        robot: 'Robot',
        trash_group: pygame.sprite.Group
    ) -> List['Trash']:
        """
        Detect trash within the robot's vision cone.

        Args:
            robot: The robot doing the sensing
            trash_group: Group of trash items

        Returns:
            List of visible trash items, sorted by distance
        """
        detected = []

        for trash in trash_group:
            if trash.is_picked:
                continue

            if point_in_cone(
                robot.position,
                robot.angle,
                trash.position,
                self.vision_cone,
                self.sensor_range
            ):
                dist = distance(robot.position, trash.position)
                detected.append((dist, trash))

        detected.sort(key=lambda x: x[0])
        return [t for _, t in detected]

    def find_nearest_trash(
        self,
        robot: 'Robot',
        trash_group: pygame.sprite.Group,
        use_vision_cone: bool = False
    ) -> Optional['Trash']:
        """
        Find the nearest trash item.

        Args:
            robot: The robot doing the sensing
            trash_group: Group of trash items
            use_vision_cone: If True, only check vision cone

        Returns:
            Nearest trash or None
        """
        if use_vision_cone:
            detected = self.detect_trash_in_vision(robot, trash_group)
        else:
            detected = self.detect_trash_in_range(robot, trash_group)

        return detected[0] if detected else None

    def is_trash_in_grab_range(
        self,
        robot: 'Robot',
        trash: 'Trash'
    ) -> bool:
        """
        Check if a specific trash item is within grabbing range.

        Args:
            robot: The robot
            trash: The trash to check

        Returns:
            True if within grab range
        """
        # Check distance from robot's arm claw position
        if robot.arm:
            claw_pos = robot.arm.get_claw_position()
            return distance(claw_pos, trash.position) <= self.grab_range
        else:
            # Fallback to robot front position
            front_pos = robot.get_front_position()
            return distance(front_pos, trash.position) <= self.grab_range

    def is_nest_visible(
        self,
        robot: 'Robot',
        nest: 'Nest',
        max_distance: float = None
    ) -> bool:
        """
        Check if the nest is visible to the robot.

        Args:
            robot: The robot
            nest: The nest
            max_distance: Maximum detection distance

        Returns:
            True if nest is visible
        """
        max_dist = max_distance or self.sensor_range * 2  # Nest is easier to see
        return distance(robot.position, nest.position) <= max_dist

    def get_distance_to_nest(
        self,
        robot: 'Robot',
        nest: 'Nest'
    ) -> float:
        """Get distance from robot to nest."""
        return distance(robot.position, nest.position)

    def raycast(
        self,
        start: Tuple[float, float],
        angle: float,
        max_distance: float,
        obstacles: pygame.sprite.Group,
        step_size: float = 5.0
    ) -> Optional[Tuple[float, float]]:
        """
        Cast a ray and find first obstacle hit.

        Args:
            start: Starting position
            angle: Direction angle in degrees
            max_distance: Maximum ray length
            obstacles: Group of obstacles to check
            step_size: Step size for ray marching

        Returns:
            Hit position or None if no hit
        """
        rad = math.radians(angle)
        dx = math.cos(rad) * step_size
        dy = math.sin(rad) * step_size

        x, y = start
        traveled = 0

        while traveled < max_distance:
            x += dx
            y += dy
            traveled += step_size

            point = pygame.Rect(x - 1, y - 1, 2, 2)
            for obstacle in obstacles:
                if point.colliderect(obstacle.get_rect()):
                    return (x, y)

        return None

    def check_path_clear(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        obstacles: pygame.sprite.Group
    ) -> bool:
        """
        Check if a direct path is clear of obstacles.

        Args:
            start: Starting position
            end: End position
            obstacles: Group of obstacles

        Returns:
            True if path is clear
        """
        angle = angle_to(start, end)
        dist = distance(start, end)
        hit = self.raycast(start, angle, dist, obstacles)
        return hit is None

    def draw_debug(
        self,
        screen: pygame.Surface,
        robot: 'Robot',
        show_range: bool = True,
        show_vision: bool = True,
        scan_angle_offset: float = 0.0  # Offset for scanning "head"
    ):
        """Draw debug visualization of sensors."""
        robot_pos = (int(robot.position[0]), int(robot.position[1]))

        if show_range:
            # Draw sensor range circle
            range_surface = pygame.Surface(
                (self.sensor_range * 2, self.sensor_range * 2),
                pygame.SRCALPHA
            )
            pygame.draw.circle(
                range_surface,
                (0, 255, 0, 30),
                (self.sensor_range, self.sensor_range),
                int(self.sensor_range)
            )
            pygame.draw.circle(
                range_surface,
                (0, 255, 0, 100),
                (self.sensor_range, self.sensor_range),
                int(self.sensor_range),
                1
            )
            screen.blit(
                range_surface,
                (robot_pos[0] - self.sensor_range, robot_pos[1] - self.sensor_range)
            )

        if show_vision:
            # Draw vision cone at body direction (FOV = where robot faces)
            # Robot must physically rotate to see different directions
            look_angle = robot.angle + scan_angle_offset

            cone_surface = pygame.Surface(
                (self.sensor_range * 2, self.sensor_range * 2),
                pygame.SRCALPHA
            )

            # Vision cone as a pie slice
            half_cone = self.vision_cone / 2

            points = [(self.sensor_range, self.sensor_range)]
            for a in range(int(-look_angle - half_cone), int(-look_angle + half_cone) + 1, 5):
                rad = math.radians(a)
                points.append((
                    self.sensor_range + math.cos(rad) * self.sensor_range,
                    self.sensor_range + math.sin(rad) * self.sensor_range
                ))
            points.append((self.sensor_range, self.sensor_range))

            if len(points) >= 3:
                pygame.draw.polygon(cone_surface, (255, 200, 0, 40), points)
                pygame.draw.polygon(cone_surface, (255, 200, 0, 100), points, 1)

            screen.blit(
                cone_surface,
                (robot_pos[0] - self.sensor_range, robot_pos[1] - self.sensor_range)
            )
