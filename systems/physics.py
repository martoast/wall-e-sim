"""
Physics system - collision detection and response.
"""
import pygame
import math
from typing import List, Tuple, Optional, TYPE_CHECKING

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_MARGIN,
    PHYSICS_PUSH_FORCE, PHYSICS_FRICTION, PHYSICS_SEPARATION_FORCE
)

if TYPE_CHECKING:
    from entities.robot import Robot
    from entities.trash import Trash
    from entities.obstacle import Obstacle
    from entities.nest import Nest


class PhysicsSystem:
    """
    Handles collision detection and response for all entities.

    Collision Types:
    - Robot vs Obstacle: Stop robot, slide along surface
    - Robot vs Trash: Allow approach if targeting that trash, otherwise stop
    - Robot vs Robot: Both stop, push apart
    - Trash vs Obstacle: Trash stops
    - Trash vs Trash: Both stop
    - All vs Screen bounds: Clamp to screen
    """

    def __init__(self):
        self.debug_collisions: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        # Track which trash each robot is targeting (robot_id -> trash_id)
        self.robot_targets: dict = {}

    def set_robot_target(self, robot_id: int, trash_id: int = None):
        """Set or clear the trash target for a robot."""
        if trash_id is None:
            self.robot_targets.pop(robot_id, None)
        else:
            self.robot_targets[robot_id] = trash_id

    def update(
        self,
        robots: List['Robot'],
        trash_group: pygame.sprite.Group,
        obstacles: pygame.sprite.Group,
        nest: 'Nest',
        dt: float = 1.0
    ):
        """
        Main physics update loop.

        Uses substeps at high speeds to prevent tunneling through objects.
        """
        self.debug_collisions.clear()

        # At high speeds (dt > 1), use substeps to prevent tunneling
        num_substeps = max(1, int(dt))
        substep_dt = dt / num_substeps

        for _ in range(num_substeps):
            # Push robots out of any obstacles they're stuck inside
            self._push_robots_out_of_obstacles(robots, obstacles, nest)

            # Separate overlapping robots
            self._separate_all_overlapping_robots(robots)

            # CRITICAL: Separate robots from trash they're overlapping with
            self._separate_robots_from_trash(robots, trash_group)

            # Process each robot with swept collision
            for robot in robots:
                self._process_robot_physics(robot, robots, trash_group, obstacles, nest, substep_dt)

            # Process trash physics
            for trash in trash_group:
                if not trash.is_picked:
                    self._process_trash_physics(trash, trash_group, obstacles, nest, substep_dt)

    def _process_robot_physics(
        self,
        robot: 'Robot',
        all_robots: List['Robot'],
        trash_group: pygame.sprite.Group,
        obstacles: pygame.sprite.Group,
        nest: 'Nest',
        dt: float
    ):
        """Process physics for a single robot."""
        if not hasattr(robot, 'velocity'):
            return

        vx, vy = robot.velocity
        if abs(vx) < 0.001 and abs(vy) < 0.001:
            return

        # Calculate new position
        new_x = robot.x + vx * dt
        new_y = robot.y + vy * dt

        # Create test rect at new position
        test_rect = pygame.Rect(
            new_x - robot.width / 2,
            new_y - robot.height / 2,
            robot.width,
            robot.height
        )

        # Check obstacle collisions
        blocked = False
        for obstacle in obstacles:
            if test_rect.colliderect(obstacle.get_rect()):
                # Resolve collision - slide along obstacle
                new_x, new_y = self._resolve_aabb_collision(
                    robot.x, robot.y, robot.width, robot.height,
                    obstacle.get_rect(), vx * dt, vy * dt
                )
                blocked = True
                self.debug_collisions.append((robot.position, obstacle.position))

        # Check nest collision (main body, not ramp)
        nest_body_rect = pygame.Rect(
            nest.x - nest.width / 2,
            nest.y - nest.height / 2,
            nest.width,
            nest.height
        )
        if test_rect.colliderect(nest_body_rect):
            # Allow if robot is docking/dumping (near ramp), otherwise block
            if robot.state.value not in ['docking', 'dumping']:
                new_x, new_y = self._resolve_aabb_collision(
                    robot.x, robot.y, robot.width, robot.height,
                    nest_body_rect, vx * dt, vy * dt
                )
                blocked = True

        # Check trash collisions - SWEPT collision to prevent tunneling
        target_trash_id = self.robot_targets.get(robot.id, None)
        for trash in trash_group:
            if trash.is_picked:
                continue

            # Allow robot to approach its target trash for pickup
            if target_trash_id is not None and trash.id == target_trash_id:
                continue

            # Use circle collision with robot's effective radius
            robot_radius = max(robot.width, robot.height) / 2
            min_dist = robot_radius + trash.size + 2  # +2 buffer

            # Check if movement would cause collision
            # Use swept circle-circle collision
            collision_pos = self._swept_circle_collision(
                robot.x, robot.y, new_x, new_y, robot_radius,
                trash.x, trash.y, trash.size
            )

            if collision_pos is not None:
                # Stop at the collision point
                new_x, new_y = collision_pos
                blocked = True
                self.debug_collisions.append((robot.position, trash.position))

        # Check robot-robot collisions - DEFLECT instead of stop
        for other in all_robots:
            if other.id == robot.id:
                continue

            other_rect = pygame.Rect(
                other.x - other.width / 2,
                other.y - other.height / 2,
                other.width,
                other.height
            )
            if test_rect.colliderect(other_rect):
                # Calculate deflection - slide around the other robot
                deflected_x, deflected_y = self._deflect_around_robot(
                    robot, other, new_x, new_y, vx, vy, dt
                )
                new_x, new_y = deflected_x, deflected_y
                self.debug_collisions.append((robot.position, other.position))

        # Clamp to screen bounds
        new_x = max(SCREEN_MARGIN + robot.width / 2,
                    min(SCREEN_WIDTH - SCREEN_MARGIN - robot.width / 2, new_x))
        new_y = max(SCREEN_MARGIN + robot.height / 2,
                    min(SCREEN_HEIGHT - SCREEN_MARGIN - robot.height / 2, new_y))

        # Apply final position
        robot.x = new_x
        robot.y = new_y
        robot.rect.center = (int(robot.x), int(robot.y))

        # Clear velocity after applying
        robot.velocity = (0.0, 0.0)

    def _process_trash_physics(
        self,
        trash: 'Trash',
        all_trash: pygame.sprite.Group,
        obstacles: pygame.sprite.Group,
        nest: 'Nest',
        dt: float
    ):
        """Process physics for a single trash item."""
        if not hasattr(trash, 'velocity'):
            return

        vx, vy = trash.velocity

        # Apply friction
        trash.velocity = (vx * PHYSICS_FRICTION, vy * PHYSICS_FRICTION)

        # Skip if velocity is negligible
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            trash.velocity = (0.0, 0.0)
            return

        # Calculate new position
        new_x = trash.x + vx * dt
        new_y = trash.y + vy * dt

        # Check obstacle collisions
        for obstacle in obstacles:
            if self._circle_rect_collision(new_x, new_y, trash.size, obstacle.get_rect()):
                # Stop trash
                trash.velocity = (0.0, 0.0)
                return

        # Check nest collision
        nest_rect = pygame.Rect(
            nest.x - nest.width / 2,
            nest.y - nest.height / 2,
            nest.width,
            nest.height
        )
        if self._circle_rect_collision(new_x, new_y, trash.size, nest_rect):
            trash.velocity = (0.0, 0.0)
            return

        # Check trash-trash collisions
        for other in all_trash:
            if other is trash or other.is_picked:
                continue

            dist = math.sqrt((new_x - other.x) ** 2 + (new_y - other.y) ** 2)
            min_dist = trash.size + other.size

            if dist < min_dist:
                # Stop this trash
                trash.velocity = (0.0, 0.0)
                return

        # Clamp to screen bounds
        new_x = max(SCREEN_MARGIN + trash.size,
                    min(SCREEN_WIDTH - SCREEN_MARGIN - trash.size, new_x))
        new_y = max(SCREEN_MARGIN + trash.size,
                    min(SCREEN_HEIGHT - SCREEN_MARGIN - trash.size, new_y))

        # Apply position
        trash.x = new_x
        trash.y = new_y
        trash.rect.center = (int(trash.x), int(trash.y))

    def _push_trash(
        self,
        robot: 'Robot',
        trash: 'Trash',
        robot_vx: float,
        robot_vy: float,
        obstacles: pygame.sprite.Group,
        nest: 'Nest',
        all_trash: pygame.sprite.Group,
        dt: float
    ) -> bool:
        """
        Attempt to push trash.

        Returns:
            True if trash was successfully pushed
        """
        # Calculate push direction (from robot to trash)
        dx = trash.x - robot.x
        dy = trash.y - robot.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.001:
            return False

        # Normalize and apply push force
        push_x = (dx / dist) * abs(robot_vx) * PHYSICS_PUSH_FORCE * 2
        push_y = (dy / dist) * abs(robot_vy) * PHYSICS_PUSH_FORCE * 2

        # Check if trash can move to new position
        new_trash_x = trash.x + push_x * dt
        new_trash_y = trash.y + push_y * dt

        # Check obstacle collisions for trash
        for obstacle in obstacles:
            if self._circle_rect_collision(new_trash_x, new_trash_y, trash.size, obstacle.get_rect()):
                return False  # Trash blocked by obstacle

        # Check nest collision
        nest_rect = pygame.Rect(
            nest.x - nest.width / 2,
            nest.y - nest.height / 2,
            nest.width,
            nest.height
        )
        if self._circle_rect_collision(new_trash_x, new_trash_y, trash.size, nest_rect):
            return False  # Trash blocked by nest

        # Check screen bounds
        if (new_trash_x < SCREEN_MARGIN + trash.size or
            new_trash_x > SCREEN_WIDTH - SCREEN_MARGIN - trash.size or
            new_trash_y < SCREEN_MARGIN + trash.size or
            new_trash_y > SCREEN_HEIGHT - SCREEN_MARGIN - trash.size):
            return False  # Trash at screen edge

        # Check other trash
        for other in all_trash:
            if other is trash or other.is_picked:
                continue

            other_dist = math.sqrt((new_trash_x - other.x) ** 2 + (new_trash_y - other.y) ** 2)
            if other_dist < trash.size + other.size:
                return False  # Would hit other trash

        # Push succeeded - set trash velocity
        if hasattr(trash, 'velocity'):
            trash.velocity = (push_x, push_y)

        return True

    def _resolve_aabb_collision(
        self,
        x: float, y: float,
        width: float, height: float,
        static_rect: pygame.Rect,
        vx: float, vy: float
    ) -> Tuple[float, float]:
        """
        Resolve AABB collision by sliding along the obstacle.

        Returns:
            New (x, y) position after collision resolution
        """
        # Try moving only on X axis
        test_x = x + vx
        test_rect_x = pygame.Rect(test_x - width / 2, y - height / 2, width, height)

        # Try moving only on Y axis
        test_y = y + vy
        test_rect_y = pygame.Rect(x - width / 2, test_y - height / 2, width, height)

        new_x, new_y = x, y

        # Check if X movement is valid
        if not test_rect_x.colliderect(static_rect):
            new_x = test_x

        # Check if Y movement is valid
        if not test_rect_y.colliderect(static_rect):
            new_y = test_y

        return new_x, new_y

    def _circle_rect_collision(
        self,
        cx: float, cy: float, radius: float,
        rect: pygame.Rect
    ) -> bool:
        """Check if circle overlaps with rectangle."""
        # Find closest point on rectangle to circle center
        closest_x = max(rect.left, min(cx, rect.right))
        closest_y = max(rect.top, min(cy, rect.bottom))

        # Calculate distance from circle center to closest point
        dist_x = cx - closest_x
        dist_y = cy - closest_y
        dist_sq = dist_x * dist_x + dist_y * dist_y

        return dist_sq < radius * radius

    def _swept_circle_collision(
        self,
        start_x: float, start_y: float,
        end_x: float, end_y: float,
        moving_radius: float,
        static_x: float, static_y: float,
        static_radius: float
    ) -> Optional[Tuple[float, float]]:
        """
        Swept collision between a moving circle and a static circle.

        Checks if a circle moving from (start_x, start_y) to (end_x, end_y)
        would collide with a static circle at (static_x, static_y).

        Returns:
            Position just before collision, or None if no collision
        """
        # Combined radius (sum of both radii)
        combined_r = moving_radius + static_radius + 3  # +3 buffer

        # Vector from start to end
        dx = end_x - start_x
        dy = end_y - start_y
        move_dist_sq = dx * dx + dy * dy

        if move_dist_sq < 0.0001:
            # Not moving, just check current overlap
            dist_sq = (start_x - static_x) ** 2 + (start_y - static_y) ** 2
            if dist_sq < combined_r * combined_r:
                # Already overlapping - push out
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 1
                nx = (start_x - static_x) / dist if dist > 0 else 1
                ny = (start_y - static_y) / dist if dist > 0 else 0
                return (static_x + nx * combined_r, static_y + ny * combined_r)
            return None

        # Vector from start to static circle center
        fx = start_x - static_x
        fy = start_y - static_y

        # Quadratic equation: at^2 + bt + c = 0
        # where t is the parameter along the movement vector (0 to 1)
        a = move_dist_sq
        b = 2 * (fx * dx + fy * dy)
        c = (fx * fx + fy * fy) - combined_r * combined_r

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            # No collision along this path
            return None

        # Find the earliest collision time
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # We want the first intersection in range [0, 1]
        t = None
        if 0 <= t1 <= 1:
            t = t1
        elif 0 <= t2 <= 1:
            t = t2
        elif t1 < 0 and t2 > 1:
            # We're already inside - push out
            dist = math.sqrt(fx * fx + fy * fy)
            if dist > 0:
                nx, ny = fx / dist, fy / dist
            else:
                nx, ny = 1, 0
            return (static_x + nx * combined_r, static_y + ny * combined_r)

        if t is None:
            return None

        # Back off slightly from collision point
        safe_t = max(0, t - 0.05)
        collision_x = start_x + dx * safe_t
        collision_y = start_y + dy * safe_t

        return (collision_x, collision_y)

    def _deflect_around_robot(
        self,
        robot: 'Robot',
        other: 'Robot',
        intended_x: float,
        intended_y: float,
        vx: float,
        vy: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Calculate deflection to slide around another robot instead of stopping.

        Returns:
            (new_x, new_y) - deflected position
        """
        # Vector from this robot to the other
        dx = other.x - robot.x
        dy = other.y - robot.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.001:
            # Exactly overlapping - push in random direction
            return robot.x + 5, robot.y

        # Normalize direction to other robot
        nx = dx / dist
        ny = dy / dist

        # Calculate perpendicular direction (for sliding)
        # Choose the perpendicular that's more aligned with our velocity
        perp1_x, perp1_y = -ny, nx
        perp2_x, perp2_y = ny, -nx

        # Dot product with velocity to choose best perpendicular
        dot1 = perp1_x * vx + perp1_y * vy
        dot2 = perp2_x * vx + perp2_y * vy

        if abs(dot1) >= abs(dot2):
            perp_x, perp_y = perp1_x, perp1_y
        else:
            perp_x, perp_y = perp2_x, perp2_y

        # Calculate how much we can move in the perpendicular direction
        speed = math.sqrt(vx * vx + vy * vy)
        if speed < 0.001:
            speed = robot.current_speed * dt

        # Move perpendicular to slide around
        slide_x = perp_x * speed * 0.8
        slide_y = perp_y * speed * 0.8

        new_x = robot.x + slide_x
        new_y = robot.y + slide_y

        # Also push apart slightly to prevent overlap
        min_dist = (robot.width + other.width) / 2 + 5
        if dist < min_dist:
            push_amount = (min_dist - dist) * 0.3
            new_x -= nx * push_amount
            new_y -= ny * push_amount

        return new_x, new_y

    def _push_robots_out_of_obstacles(
        self,
        robots: List['Robot'],
        obstacles: pygame.sprite.Group,
        nest: 'Nest'
    ):
        """Push any robots that are inside obstacles back out."""
        for robot in robots:
            robot_rect = pygame.Rect(
                robot.x - robot.width / 2,
                robot.y - robot.height / 2,
                robot.width,
                robot.height
            )

            # Check each obstacle
            for obstacle in obstacles:
                obs_rect = obstacle.get_rect()
                if robot_rect.colliderect(obs_rect):
                    # Robot is inside obstacle - push it out
                    self._push_out_of_rect(robot, obs_rect)

            # Check nest body
            nest_rect = pygame.Rect(
                nest.x - nest.width / 2,
                nest.y - nest.height / 2,
                nest.width,
                nest.height
            )
            # Only push out if not docking/dumping
            if robot.state.value not in ['docking', 'dumping']:
                if robot_rect.colliderect(nest_rect):
                    self._push_out_of_rect(robot, nest_rect)

    def _push_out_of_rect(self, robot: 'Robot', rect: pygame.Rect):
        """Push robot out of a rectangle to the nearest edge."""
        # Calculate overlap on each side
        robot_left = robot.x - robot.width / 2
        robot_right = robot.x + robot.width / 2
        robot_top = robot.y - robot.height / 2
        robot_bottom = robot.y + robot.height / 2

        # How far we'd need to push in each direction
        push_left = rect.left - robot_right  # negative = need to go left
        push_right = rect.right - robot_left  # positive = need to go right
        push_up = rect.top - robot_bottom  # negative = need to go up
        push_down = rect.bottom - robot_top  # positive = need to go down

        # Find the smallest push distance (easiest escape)
        pushes = [
            (abs(push_left), push_left - 5, 0),   # Push left
            (abs(push_right), push_right + 5, 0),  # Push right
            (abs(push_up), 0, push_up - 5),        # Push up
            (abs(push_down), 0, push_down + 5),    # Push down
        ]

        # Sort by smallest distance and apply
        pushes.sort(key=lambda x: x[0])
        _, push_x, push_y = pushes[0]

        robot.x += push_x
        robot.y += push_y
        robot.rect.center = (int(robot.x), int(robot.y))

    def _separate_all_overlapping_robots(self, robots: List['Robot']):
        """Proactively push apart any overlapping robots."""
        for i, robot1 in enumerate(robots):
            for robot2 in robots[i+1:]:
                dx = robot2.x - robot1.x
                dy = robot2.y - robot1.y
                dist = math.sqrt(dx * dx + dy * dy)

                # Minimum distance to not overlap (with buffer)
                min_dist = (robot1.width + robot2.width) / 2 + 10

                if dist < min_dist and dist > 0.001:
                    # Push apart
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist

                    push = overlap * 0.6  # Push enough to separate
                    robot1.x -= nx * push
                    robot1.y -= ny * push
                    robot2.x += nx * push
                    robot2.y += ny * push

                    robot1.rect.center = (int(robot1.x), int(robot1.y))
                    robot2.rect.center = (int(robot2.x), int(robot2.y))

    def _separate_robots_from_trash(self, robots: List['Robot'], trash_group: pygame.sprite.Group):
        """
        CRITICAL: Ensure robots NEVER overlap with trash.

        This is called every frame and forcibly separates any overlapping pairs.
        Uses multiple iterations to handle chain reactions.
        """
        # Multiple passes to handle chain separations
        for _ in range(3):
            for robot in robots:
                target_trash_id = self.robot_targets.get(robot.id, None)

                for trash in trash_group:
                    if trash.is_picked:
                        continue

                    # Skip target trash - robot is supposed to approach it
                    if target_trash_id is not None and trash.id == target_trash_id:
                        continue

                    # Check overlap using effective radius
                    dx = trash.x - robot.x
                    dy = trash.y - robot.y
                    dist = math.sqrt(dx * dx + dy * dy)

                    # Minimum distance: robot radius + trash radius + buffer
                    robot_radius = max(robot.width, robot.height) / 2
                    min_dist = robot_radius + trash.size + 5

                    if dist < min_dist:
                        if dist < 0.001:
                            # Exactly overlapping - push in arbitrary direction
                            dx, dy, dist = 1.0, 0.0, 1.0

                        # Calculate separation needed
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist

                        # Push FULLY apart - no partial separation
                        # Robot moves back, trash moves forward
                        robot.x -= nx * (overlap * 0.4 + 2)
                        robot.y -= ny * (overlap * 0.4 + 2)
                        trash.x += nx * (overlap * 0.6 + 2)
                        trash.y += ny * (overlap * 0.6 + 2)

                        robot.rect.center = (int(robot.x), int(robot.y))
                        trash.rect.center = (int(trash.x), int(trash.y))

    def _separate_robots(self, robot1: 'Robot', robot2: 'Robot'):
        """Push two overlapping robots apart."""
        dx = robot2.x - robot1.x
        dy = robot2.y - robot1.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.001:
            # Exactly overlapping, push in random direction
            dx, dy = 1.0, 0.0
            dist = 1.0

        # Calculate overlap
        min_dist = (robot1.width + robot2.width) / 2
        overlap = min_dist - dist

        if overlap > 0:
            # Push apart
            push_x = (dx / dist) * overlap * 0.5 * PHYSICS_SEPARATION_FORCE
            push_y = (dy / dist) * overlap * 0.5 * PHYSICS_SEPARATION_FORCE

            robot1.x -= push_x
            robot1.y -= push_y
            robot2.x += push_x
            robot2.y += push_y

            robot1.rect.center = (int(robot1.x), int(robot1.y))
            robot2.rect.center = (int(robot2.x), int(robot2.y))

    def draw_debug(self, screen: pygame.Surface, robots: List['Robot'], trash_group: pygame.sprite.Group):
        """Draw debug visualization of collision boxes."""
        # Draw robot collision boxes
        for robot in robots:
            rect = pygame.Rect(
                robot.x - robot.width / 2,
                robot.y - robot.height / 2,
                robot.width,
                robot.height
            )
            pygame.draw.rect(screen, (0, 255, 0), rect, 2)

            # Draw velocity vector if moving
            if hasattr(robot, 'velocity'):
                vx, vy = robot.velocity
                if abs(vx) > 0.1 or abs(vy) > 0.1:
                    end_x = robot.x + vx * 10
                    end_y = robot.y + vy * 10
                    pygame.draw.line(screen, (255, 0, 255),
                                   (int(robot.x), int(robot.y)),
                                   (int(end_x), int(end_y)), 2)

        # Draw trash collision circles
        for trash in trash_group:
            if not trash.is_picked:
                pygame.draw.circle(screen, (255, 255, 0),
                                 (int(trash.x), int(trash.y)),
                                 trash.size, 1)

                # Draw velocity if moving
                if hasattr(trash, 'velocity'):
                    vx, vy = trash.velocity
                    if abs(vx) > 0.1 or abs(vy) > 0.1:
                        end_x = trash.x + vx * 5
                        end_y = trash.y + vy * 5
                        pygame.draw.line(screen, (255, 100, 100),
                                       (int(trash.x), int(trash.y)),
                                       (int(end_x), int(end_y)), 2)

        # Draw collision indicators
        for pos1, pos2 in self.debug_collisions:
            pygame.draw.line(screen, (255, 0, 0),
                           (int(pos1[0]), int(pos1[1])),
                           (int(pos2[0]), int(pos2[1])), 1)
