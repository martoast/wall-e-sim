"""
Behavior system - robot state machine and decision making.

Design principles:
1. Simple: Move toward goal, physics handles collisions
2. Graceful failure: If stuck, give up and try something else
3. No cheating: All movement is smooth and visible
"""
import pygame
import math
from typing import Optional, List, Tuple, TYPE_CHECKING

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROBOT_GRAB_RANGE
from entities.robot import RobotState
from entities.arm import ArmState
from systems.sensors import SensorSystem
from systems.navigation import Navigation
from utils.math_helpers import distance, angle_to

if TYPE_CHECKING:
    from entities.robot import Robot
    from entities.trash import Trash
    from entities.nest import Nest
    from entities.obstacle import Obstacle
    from systems.coordinator import Coordinator
    from systems.telemetry import Telemetry
    from systems.shared_map import SharedMap

from systems.telemetry import EventType


class BehaviorController:
    """
    Controls robot behavior through a state machine.

    Simple, elegant design:
    - Robot moves toward its goal
    - If blocked, physics stops it
    - If no progress for a while, give up and try something else
    """

    def __init__(
        self,
        robot: 'Robot',
        nest: 'Nest',
        sensors: SensorSystem = None,
        navigation: Navigation = None
    ):
        self.robot = robot
        self.nest = nest
        self.sensors = sensors or SensorSystem()
        self.navigation = navigation or Navigation()

        # State
        self.current_state = RobotState.PATROL
        robot.set_state(self.current_state)

        # Patrol
        self.patrol_waypoints: List[Tuple[float, float]] = []
        self.current_waypoint_index = 0

        # Target tracking
        self.target_trash: Optional['Trash'] = None

        # Timers
        self.pickup_timer = 0
        self.dump_timer = 0

        # Exploration - track how long since we found trash
        self._frames_since_found_trash = 0
        self._wander_target: Optional[Tuple[float, float]] = None
        self.WANDER_THRESHOLD = 150  # ~5 seconds at 30fps - time before wandering to new area
        self._patrol_loops_without_trash = 0  # Track full patrol loops without finding anything

        # ==============================================
        # ANT-LIKE NAVIGATION - Simple rules that always work
        # ==============================================

        # Position history - for detecting TRUE stuck (robot not moving)
        self._position_history: List[Tuple[float, float]] = []
        self._position_history_max = 10  # Track last 10 positions

        # Wall-following mode (ant behavior when blocked)
        self._wall_follow_direction: Optional[str] = None  # None, 'left', or 'right'
        self._wall_follow_frames = 0
        self._wall_follow_max_frames = 90  # Try one direction for ~3 seconds before switching
        self._tried_both_directions = False

        # Simple stuck detection
        self._stuck_frames = 0
        self._stuck_threshold = 15  # Frames without movement to consider stuck

        # Goal tracking
        self._current_goal: Optional[Tuple[float, float]] = None

        # Legacy compatibility (keeping for return path logic)
        self._return_stuck_count = 0
        self._return_waypoint: Optional[Tuple[float, float]] = None
        self._navigation_waypoint: Optional[Tuple[float, float]] = None

        # References (set during update)
        self._coordinator: Optional['Coordinator'] = None
        self._telemetry: Optional['Telemetry'] = None
        self._obstacles: Optional[pygame.sprite.Group] = None
        self._shared_map: Optional['SharedMap'] = None

        # ==============================================
        # SCANNING SYSTEM - Robot "head" looks around
        # ==============================================
        # Like WALL-E's head moving to scan the environment
        self._scan_angle = 0.0           # Current scan offset from body angle
        self._scan_target = 0.0          # Target scan offset (smooth interpolation)
        self._scan_speed = 4.0           # Degrees per frame to turn "head"
        self._scan_hold_timer = 0        # Frames held at current position
        self._scan_hold_duration = 25    # Frames to hold at each scan position
        # Scan pattern: left, center, right, center (smooth sweep)
        self._scan_positions = [-55, 0, 55, 0]
        self._scan_index = 0

        # Generate initial patrol path
        self._generate_patrol()

    def _generate_patrol(self):
        """Generate a new patrol path."""
        zone = None
        if self._coordinator:
            zone = self._coordinator.get_patrol_zone(self.robot.id)

        self.patrol_waypoints = self.navigation.generate_random_patrol(
            point_count=6,
            nest_position=self.nest.position,
            zone_bounds=zone,
            shared_map=self._shared_map,
            start_position=self.robot.position
        )
        self.current_waypoint_index = 0
        self._reset_progress_tracking()
        self._navigation_waypoint = None  # Clear any intermediate waypoint
        self._patrol_loops_without_trash = 0  # Reset loop counter for new patrol

    def _reset_progress_tracking(self):
        """Reset navigation state when changing goals."""
        self._position_history.clear()
        self._stuck_frames = 0
        self._wall_follow_direction = None
        self._wall_follow_frames = 0
        self._tried_both_directions = False

    # ==============================================
    # SCANNING SYSTEM - Active visual search
    # ==============================================

    def _update_scan(self):
        """
        Update the scanning "head" movement.

        The robot's camera/head smoothly sweeps left-center-right-center
        while patrolling, like WALL-E looking around for interesting things.
        """
        # Smoothly move scan angle toward target
        angle_diff = self._scan_target - self._scan_angle
        if abs(angle_diff) > self._scan_speed:
            # Still moving toward target
            self._scan_angle += self._scan_speed if angle_diff > 0 else -self._scan_speed
        else:
            # Reached target - hold position then move to next
            self._scan_angle = self._scan_target
            self._scan_hold_timer += 1

            if self._scan_hold_timer >= self._scan_hold_duration:
                # Move to next scan position
                self._scan_hold_timer = 0
                self._scan_index = (self._scan_index + 1) % len(self._scan_positions)
                self._scan_target = self._scan_positions[self._scan_index]

    def _get_scan_look_angle(self) -> float:
        """Get the current direction the robot is 'looking' (body angle + scan offset)."""
        return self.robot.angle + self._scan_angle

    def _scan_for_trash(self, trash_group: pygame.sprite.Group) -> Optional['Trash']:
        """
        Scan for trash at the current scan angle.

        Uses the robot's vision cone but oriented at the scan angle,
        not the body angle. This simulates a camera/head that can
        turn independently of the body.

        Returns:
            Nearest visible trash, or None
        """
        from utils.math_helpers import point_in_cone

        look_angle = self._get_scan_look_angle()
        detected = []

        for trash in trash_group:
            if trash.is_picked:
                continue

            # Check if trash is in vision cone at SCAN angle (not body angle)
            if point_in_cone(
                self.robot.position,
                look_angle,
                trash.position,
                self.sensors.vision_cone,
                self.sensors.sensor_range
            ):
                dist = distance(self.robot.position, trash.position)
                detected.append((dist, trash))

        # Return nearest
        if detected:
            detected.sort(key=lambda x: x[0])
            return detected[0][1]
        return None

    def _reset_scan_to_center(self):
        """Reset scanning to look forward (used when trash found)."""
        self._scan_target = 0
        self._scan_index = 1  # Index of center position
        self._scan_hold_timer = 0

    def _get_current_goal(self) -> Optional[Tuple[float, float]]:
        """Get the current goal position based on state."""
        if self.current_state == RobotState.PATROL:
            if self.patrol_waypoints:
                return self.patrol_waypoints[self.current_waypoint_index]
        elif self.current_state in [RobotState.SEEKING, RobotState.APPROACHING]:
            if self.target_trash:
                return self.target_trash.position
        elif self.current_state == RobotState.RETURNING:
            # If we have a waypoint to navigate around obstacle, go there first
            if self._return_waypoint:
                return self._return_waypoint
            return self.nest.get_ramp_entry()
        elif self.current_state == RobotState.DOCKING:
            return self.nest.get_dock_position()
        elif self.current_state == RobotState.UNDOCKING:
            return self.nest.get_ramp_entry()
        return None

    def _check_stuck(self) -> bool:
        """
        Check if robot is ACTUALLY stuck (position not changing).

        Simple ant-like detection: if we're not moving, we're stuck.
        This is more reliable than checking distance to goal.
        """
        # Only check in movement states
        if self.current_state not in [RobotState.PATROL, RobotState.APPROACHING,
                                       RobotState.SEEKING, RobotState.RETURNING,
                                       RobotState.UNDOCKING]:
            self._position_history.clear()
            self._stuck_frames = 0
            return False

        current_pos = self.robot.position

        # Add to position history
        self._position_history.append(current_pos)
        if len(self._position_history) > self._position_history_max:
            self._position_history.pop(0)

        # Need at least a few frames of history
        if len(self._position_history) < 5:
            return False

        # Check if we've moved significantly from oldest position in history
        oldest_pos = self._position_history[0]
        movement = distance(current_pos, oldest_pos)

        # If we've moved less than 5 pixels over the history window, we're stuck
        if movement < 5:
            self._stuck_frames += 1
            if self._stuck_frames >= self._stuck_threshold:
                return True
        else:
            # We're moving - reset stuck counter and clear wall-follow if making progress
            self._stuck_frames = 0

            # Check if we're making progress toward goal (can exit wall-follow mode)
            if self._wall_follow_direction:
                goal = self._get_current_goal()
                if goal and self._is_path_to_goal_clear(goal):
                    # Path is clear! Exit wall-following mode
                    self._wall_follow_direction = None
                    self._wall_follow_frames = 0
                    self._tried_both_directions = False

        return False

    def _handle_stuck(self):
        """
        Handle being stuck - SURVIVAL INSTINCTS!

        SMART RULES:
        1. First check: Is trash blocking us? If bin not full, GRAB IT!
        2. If not, try wall-following (turn 90°)
        3. Keep checking if direct path to goal is clear
        4. If wall-following fails, try the other direction
        5. If both fail, AGGRESSIVE ESCAPE: back up and flee!
        """
        # Reset stuck counter
        self._stuck_frames = 0
        self._position_history.clear()

        # ==============================================
        # SURVIVAL INSTINCT #1: Opportunistic trash grab
        # ==============================================
        # If we're stuck and there's trash nearby, just grab it!
        # It's probably what's blocking us anyway.
        if not self.robot.bin_full and self._trash_group:
            nearby_trash = self._find_nearby_trash_to_grab()
            if nearby_trash:
                # Switch to grabbing this trash!
                if self._coordinator:
                    # Release old claim if any
                    if self.target_trash:
                        self._coordinator.release_claim(self.target_trash.id)
                    # Claim new trash
                    if self._coordinator.claim_trash(self.robot.id, nearby_trash):
                        self.target_trash = nearby_trash
                        self._wall_follow_direction = None
                        self._wall_follow_frames = 0
                        self._tried_both_directions = False
                        self.transition_to(RobotState.SEEKING)
                        return
                else:
                    self.target_trash = nearby_trash
                    self._wall_follow_direction = None
                    self.transition_to(RobotState.SEEKING)
                    return

        # ==============================================
        # Wall-following mode
        # ==============================================
        # If not already wall-following, start
        if self._wall_follow_direction is None:
            # Pick initial direction (prefer left, like many ant species)
            self._wall_follow_direction = 'left'
            self._wall_follow_frames = 0
            self._tried_both_directions = False
            return

        # Already wall-following but still stuck - the direction isn't working
        self._wall_follow_frames += self._stuck_threshold  # Count stuck time

        # If we've been trying this direction too long, switch
        if self._wall_follow_frames >= self._wall_follow_max_frames:
            if not self._tried_both_directions:
                # Try the other direction
                self._wall_follow_direction = 'right' if self._wall_follow_direction == 'left' else 'left'
                self._wall_follow_frames = 0
                self._tried_both_directions = True
                return
            else:
                # ==============================================
                # SURVIVAL INSTINCT #2: AGGRESSIVE ESCAPE!
                # ==============================================
                # Both directions failed - time for drastic action!
                self._wall_follow_direction = None
                self._wall_follow_frames = 0
                self._tried_both_directions = False
                self._aggressive_escape()
                return

    def _find_nearby_trash_to_grab(self) -> Optional['Trash']:
        """
        Find trash very close to the robot that might be blocking us.

        When stuck, check if there's grabbable trash nearby - it's probably
        what's blocking us! Grab it to clear our path.
        """
        if not self._trash_group:
            return None

        robot_pos = self.robot.position

        # Look for trash within grabbing range (closer than normal detection)
        GRAB_RANGE = 80  # Close enough to grab

        best_trash = None
        best_dist = GRAB_RANGE

        for trash in self._trash_group:
            if trash.is_picked:
                continue

            # Skip if claimed by another robot
            if self._coordinator and self._coordinator.is_claimed_by_other(trash, self.robot.id):
                continue

            trash_dist = distance(robot_pos, trash.position)

            if trash_dist < best_dist:
                best_trash = trash
                best_dist = trash_dist

        return best_trash

    def _aggressive_escape(self):
        """
        SURVIVAL INSTINCT: Aggressive escape when truly stuck!

        Smart about screen edges - prioritizes escaping toward screen center.
        """
        import random
        from config import SCREEN_WIDTH, SCREEN_HEIGHT

        robot_pos = self.robot.position
        margin = 60
        treat_trash_as_obstacles = self.robot.bin_full

        # Calculate direction toward screen center (escape away from edges!)
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        to_center_x = center_x - robot_pos[0]
        to_center_y = center_y - robot_pos[1]
        to_center_dist = math.sqrt(to_center_x**2 + to_center_y**2)

        # Normalize direction to center
        if to_center_dist > 1:
            to_center_x /= to_center_dist
            to_center_y /= to_center_dist

        # Calculate angle toward center
        center_angle = math.atan2(to_center_y, to_center_x)

        # PRIORITY 1: Try escaping toward screen center first (best for edge cases)
        # Try angles biased toward center: center, center±30°, center±60°, center±90°
        center_biased_angles = [
            center_angle,
            center_angle + math.pi/6, center_angle - math.pi/6,  # ±30°
            center_angle + math.pi/3, center_angle - math.pi/3,  # ±60°
            center_angle + math.pi/2, center_angle - math.pi/2,  # ±90°
        ]

        escape_distances = [60, 100, 150, 200]

        for escape_dist in escape_distances:
            for angle in center_biased_angles:
                escape_x = robot_pos[0] + math.cos(angle) * escape_dist
                escape_y = robot_pos[1] + math.sin(angle) * escape_dist

                # Clamp to screen bounds
                escape_x = max(margin, min(SCREEN_WIDTH - margin, escape_x))
                escape_y = max(margin, min(SCREEN_HEIGHT - margin, escape_y))

                # Skip if clamping made target too close (we're at an edge)
                actual_dist = distance(robot_pos, (escape_x, escape_y))
                if actual_dist < 40:
                    continue

                escape_target = (escape_x, escape_y)

                if self._has_line_of_sight(escape_target, account_for_width=True, treat_trash_as_obstacles=treat_trash_as_obstacles):
                    self._wander_target = escape_target
                    self._finalize_escape()
                    return

        # PRIORITY 2: Try all 8 compass directions
        all_angles = [i * math.pi / 4 for i in range(8)]
        random.shuffle(all_angles)

        for escape_dist in escape_distances:
            for angle in all_angles:
                escape_x = robot_pos[0] + math.cos(angle) * escape_dist
                escape_y = robot_pos[1] + math.sin(angle) * escape_dist
                escape_x = max(margin, min(SCREEN_WIDTH - margin, escape_x))
                escape_y = max(margin, min(SCREEN_HEIGHT - margin, escape_y))

                actual_dist = distance(robot_pos, (escape_x, escape_y))
                if actual_dist < 40:
                    continue

                escape_target = (escape_x, escape_y)

                if self._has_line_of_sight(escape_target, account_for_width=True, treat_trash_as_obstacles=treat_trash_as_obstacles):
                    self._wander_target = escape_target
                    self._finalize_escape()
                    return

        # DESPERATE MODE: Squeeze through tight gaps (no width check)
        for escape_dist in [50, 80, 120]:
            for angle in center_biased_angles + all_angles:
                escape_x = robot_pos[0] + math.cos(angle) * escape_dist
                escape_y = robot_pos[1] + math.sin(angle) * escape_dist
                escape_x = max(margin, min(SCREEN_WIDTH - margin, escape_x))
                escape_y = max(margin, min(SCREEN_HEIGHT - margin, escape_y))

                actual_dist = distance(robot_pos, (escape_x, escape_y))
                if actual_dist < 30:
                    continue

                escape_target = (escape_x, escape_y)

                if self._has_line_of_sight(escape_target, account_for_width=False, treat_trash_as_obstacles=treat_trash_as_obstacles):
                    self._wander_target = escape_target
                    self._finalize_escape()
                    return

        # LAST RESORT: Head toward screen center no matter what
        # This at least gets us away from edges
        self._wander_target = (
            robot_pos[0] + to_center_x * 150,
            robot_pos[1] + to_center_y * 150
        )
        self._finalize_escape()

    def _finalize_escape(self):
        """Common cleanup after setting escape target."""

        # Clear current goals - fresh start!
        if self.target_trash:
            if self._coordinator:
                self._coordinator.release_claim(self.target_trash.id)
            self.target_trash = None

        self.patrol_waypoints = []
        self._return_waypoint = None

        # If we were returning to dump, we'll try again after escaping
        if self.current_state in [RobotState.RETURNING, RobotState.WAITING]:
            self._return_stuck_count += 1

        # Transition to patrol (will follow wander_target first)
        self.transition_to(RobotState.PATROL)

    def _skip_current_goal(self):
        """
        Skip the current goal - legacy method, now uses aggressive escape.
        """
        self._aggressive_escape()

    def _is_path_to_goal_clear(self, goal: Tuple[float, float]) -> bool:
        """Check if direct path to goal is now clear (for exiting wall-follow mode)."""
        # When bin is full (RETURNING/WAITING states), treat trash as obstacles
        # because we can't pick them up anyway
        treat_trash_as_obstacles = self.robot.bin_full

        # Check line of sight to obstacles (and trash if bin full)
        if not self._has_line_of_sight(goal, treat_trash_as_obstacles=treat_trash_as_obstacles):
            return False

        return True

    def _get_wall_follow_direction_vector(self, goal: Tuple[float, float]) -> Tuple[float, float]:
        """
        Get movement vector for wall-following behavior.

        Instead of going directly to goal, move perpendicular to it
        (along the "wall") while biasing slightly toward the goal.
        """
        robot_pos = self.robot.position

        # Vector to goal
        dx = goal[0] - robot_pos[0]
        dy = goal[1] - robot_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1:
            return (0, 0)

        # Normalize
        dx /= dist
        dy /= dist

        # Perpendicular vectors (wall-follow directions)
        if self._wall_follow_direction == 'left':
            perp_x, perp_y = -dy, dx  # 90° left
        else:
            perp_x, perp_y = dy, -dx  # 90° right

        # Blend: mostly perpendicular (wall-follow) with slight goal bias
        # This creates a curved path that follows obstacles while trending toward goal
        blend = 0.3  # 30% toward goal, 70% along wall
        final_x = perp_x * (1 - blend) + dx * blend
        final_y = perp_y * (1 - blend) + dy * blend

        # Normalize
        mag = math.sqrt(final_x * final_x + final_y * final_y)
        if mag > 0:
            final_x /= mag
            final_y /= mag

        return (final_x * self.robot.current_speed, final_y * self.robot.current_speed)

    def _pick_wander_target(self):
        """
        Pick a location to wander to - prioritize directions with clear line of sight!

        When stuck, we need to go somewhere we can ACTUALLY reach.
        When bin is full, treat trash as obstacles (we can't pick them up anyway).
        """
        import random
        from config import SCREEN_WIDTH, SCREEN_HEIGHT

        robot_pos = self.robot.position
        margin = 80

        # When bin is full, trash blocks us too!
        treat_trash_as_obstacles = self.robot.bin_full

        # FIRST: Try to find a target we have clear line of sight to
        # This ensures we can actually GET there
        for _ in range(30):
            x = random.uniform(margin, SCREEN_WIDTH - margin)
            y = random.uniform(margin, SCREEN_HEIGHT - margin)

            target = (x, y)

            # Must be at least 150px away
            dist = distance(robot_pos, target)
            if dist < 150:
                continue

            # Avoid nest area
            nest_dist = distance(target, self.nest.position)
            if nest_dist < 150:
                continue

            # KEY: Check if we have clear line of sight to this target!
            if self._has_line_of_sight(target, account_for_width=True, treat_trash_as_obstacles=treat_trash_as_obstacles):
                self._wander_target = target
                return

        # SECOND: If no clear path found, try closer targets
        for _ in range(20):
            # Try random directions at various distances
            angle = random.uniform(0, 360)
            for dist in [80, 120, 180, 250]:
                rad = math.radians(angle)
                target = (
                    robot_pos[0] + math.cos(rad) * dist,
                    robot_pos[1] + math.sin(rad) * dist
                )

                if not self.navigation.is_in_bounds(target):
                    continue

                if self._has_line_of_sight(target, account_for_width=True, treat_trash_as_obstacles=treat_trash_as_obstacles):
                    self._wander_target = target
                    return

        # Fallback: just pick a random spot and hope wall-following helps
        self._wander_target = (
            random.uniform(margin, SCREEN_WIDTH - margin),
            random.uniform(margin, SCREEN_HEIGHT - margin)
        )

    def _has_line_of_sight(self, target_pos: Tuple[float, float], account_for_width: bool = True, treat_trash_as_obstacles: bool = False) -> bool:
        """
        Check if there's a clear line of sight from robot to target.
        Returns False if any obstacle blocks the path.

        When account_for_width is True, inflates obstacle rects by robot width
        to ensure the robot's body can actually fit through.

        When treat_trash_as_obstacles is True, also checks for trash blocking the path.
        Use this when robot can't pick up trash (e.g., bin is full, returning to dump).
        """
        if self._obstacles is None:
            return True

        robot_pos = self.robot.position

        # Inflate amount to account for robot body width
        inflate_amount = int(self.robot.width * 0.6) if account_for_width else 0

        # Check against each obstacle (inflated to account for robot width)
        for obstacle in self._obstacles:
            rect = obstacle.get_rect()
            # Inflate rect so we check if the robot's BODY would fit, not just center line
            inflated_rect = rect.inflate(inflate_amount, inflate_amount)
            if self._line_intersects_rect(robot_pos, target_pos, inflated_rect):
                return False

        # Also check against nest (inflated for robot width)
        nest_rect = self.nest.get_rect()
        inflated_nest = nest_rect.inflate(inflate_amount, inflate_amount)
        if self._line_intersects_rect(robot_pos, target_pos, inflated_nest):
            return False

        # When bin is full or explicitly requested, treat trash as obstacles
        if treat_trash_as_obstacles and self._trash_group:
            for trash in self._trash_group:
                if trash.is_picked:
                    continue
                # Create rect around trash
                trash_rect = pygame.Rect(
                    trash.x - trash.size - 10,
                    trash.y - trash.size - 10,
                    (trash.size + 10) * 2,
                    (trash.size + 10) * 2
                )
                if self._line_intersects_rect(robot_pos, target_pos, trash_rect):
                    return False

        return True

    def _line_intersects_rect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        rect: pygame.Rect
    ) -> bool:
        """Check if a line segment intersects a rectangle."""
        # Get rect edges
        left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom

        # Check if line intersects any edge of the rectangle
        edges = [
            ((left, top), (right, top)),      # Top
            ((right, top), (right, bottom)),  # Right
            ((right, bottom), (left, bottom)), # Bottom
            ((left, bottom), (left, top)),    # Left
        ]

        for edge_start, edge_end in edges:
            if self._lines_intersect(p1, p2, edge_start, edge_end):
                return True

        # Also check if line is entirely inside rect
        if rect.collidepoint(p1) or rect.collidepoint(p2):
            return True

        return False

    def _lines_intersect(
        self,
        p1: Tuple[float, float], p2: Tuple[float, float],
        p3: Tuple[float, float], p4: Tuple[float, float]
    ) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
                ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def transition_to(self, new_state: RobotState):
        """Transition to a new state."""
        old_state = self.current_state
        self.current_state = new_state
        self.robot.set_state(new_state)
        self._reset_progress_tracking()

        # Release trash claims when going back to patrol
        if self.target_trash and self._coordinator:
            if new_state == RobotState.PATROL:
                self._coordinator.release_claim(self.target_trash.id)

        # Leave dump queue if abandoning dump process
        if self._coordinator:
            if old_state in [RobotState.WAITING, RobotState.RETURNING]:
                if new_state == RobotState.PATROL:
                    self._coordinator.leave_queue(self.robot.id)

            # Release ramp if leaving dock-related states unexpectedly
            if old_state in [RobotState.DOCKING, RobotState.DUMPING, RobotState.UNDOCKING]:
                if new_state == RobotState.PATROL:
                    self._coordinator.release_ramp(self.robot.id)

        # State entry actions
        if new_state == RobotState.PATROL:
            self.target_trash = None
            if self.robot.arm:
                self.robot.arm.retract()

        elif new_state == RobotState.PICKING:
            self.pickup_timer = 0

        elif new_state == RobotState.DUMPING:
            self.dump_timer = 0

    def update(
        self,
        dt: float,
        trash_group: pygame.sprite.Group,
        obstacles: pygame.sprite.Group,
        all_robots: List['Robot'] = None,
        coordinator: 'Coordinator' = None,
        telemetry: 'Telemetry' = None,
        shared_map: 'SharedMap' = None
    ):
        """Update behavior based on current state."""
        self._all_robots = all_robots or []
        self._telemetry = telemetry
        self._trash_group = trash_group
        self._coordinator = coordinator
        self._obstacles = obstacles
        self._shared_map = shared_map

        # Check if stuck and handle gracefully
        if self._check_stuck():
            self._handle_stuck()
            return

        # Execute current state
        state_handlers = {
            RobotState.PATROL: lambda: self._execute_patrol(dt, trash_group, obstacles),
            RobotState.SEEKING: lambda: self._execute_seeking(dt),
            RobotState.APPROACHING: lambda: self._execute_approaching(dt, obstacles),
            RobotState.PICKING: lambda: self._execute_picking(dt, trash_group, obstacles),
            RobotState.STORING: lambda: self._execute_storing(dt),
            RobotState.RETURNING: lambda: self._execute_returning(dt, obstacles),
            RobotState.WAITING: lambda: self._execute_waiting(dt, obstacles),
            RobotState.DOCKING: lambda: self._execute_docking(dt),
            RobotState.DUMPING: lambda: self._execute_dumping(dt),
            RobotState.UNDOCKING: lambda: self._execute_undocking(dt),
            RobotState.IDLE: lambda: None,
        }

        handler = state_handlers.get(self.current_state)
        if handler:
            handler()

        # Update robot and arm
        self.robot.update(dt)
        if self.robot.arm:
            self.robot.arm.update(dt)

    def _execute_patrol(self, dt: float, trash_group: pygame.sprite.Group, obstacles: pygame.sprite.Group):
        """Patrol: follow waypoints, actively scan and look for trash."""
        # ==============================================
        # ACTIVE SCANNING - Robot "head" sweeps looking for trash
        # ==============================================
        self._update_scan()  # Update scanning head movement

        # Check for trash using scanning camera system
        if not self.robot.bin_full:
            # Scan for trash at current scan angle
            spotted_trash = self._scan_for_trash(trash_group)

            if spotted_trash:
                # Check if claimed by another robot
                if self._coordinator and self._coordinator.is_claimed_by_other(spotted_trash, self.robot.id):
                    spotted_trash = None

                # Check line of sight (obstacles blocking)
                elif not self._has_line_of_sight(spotted_trash.position):
                    spotted_trash = None

                # Try to claim it
                elif self._coordinator:
                    if not self._coordinator.claim_trash(self.robot.id, spotted_trash):
                        spotted_trash = None

            if spotted_trash:
                # Found trash! Reset scan and go get it
                self._reset_scan_to_center()
                self._frames_since_found_trash = 0
                self._patrol_loops_without_trash = 0
                self._wander_target = None  # Cancel wandering
                self.target_trash = spotted_trash
                self.transition_to(RobotState.SEEKING)
                return

        # If wandering to a new area
        if self._wander_target:
            dist_to_wander = distance(self.robot.position, self._wander_target)
            if dist_to_wander < 50:
                # Reached wander target - generate new patrol here
                self._wander_target = None
                self._frames_since_found_trash = 0
                self._generate_patrol()
            else:
                self._move_toward(self._wander_target, dt)
                return

        # Increment frames since found trash
        self._frames_since_found_trash += 1

        # If we haven't found trash in a while, wander to a new area
        if self._frames_since_found_trash > self.WANDER_THRESHOLD:
            self._pick_wander_target()
            return

        # Return to dump if bin full
        if self.robot.bin_full:
            self._request_dump_or_wait()
            return

        # If no trash left on ground and we have items in bin, go dump to finish the job
        if len(trash_group) == 0 and self.robot.bin_count > 0:
            self._request_dump_or_wait()
            return

        # Follow patrol path
        if not self.patrol_waypoints:
            self._generate_patrol()
            return

        waypoint = self.patrol_waypoints[self.current_waypoint_index]

        if self.navigation.is_at_waypoint(self.robot, waypoint):
            # Check if we're completing a full loop (going from last waypoint back to first)
            old_index = self.current_waypoint_index
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.patrol_waypoints)

            # Detect full loop completion
            if self.current_waypoint_index == 0 and old_index == len(self.patrol_waypoints) - 1:
                self._patrol_loops_without_trash += 1

                # If we've done a full loop without finding trash, try a new area
                if self._patrol_loops_without_trash >= 1:
                    self._pick_wander_target()
                    return

            self._reset_progress_tracking()
        else:
            # Move toward waypoint
            self._move_toward(waypoint, dt)

    def _execute_seeking(self, dt: float):
        """Seeking: turn toward target trash."""
        if not self.target_trash or self.target_trash.is_picked:
            self.transition_to(RobotState.PATROL)
            return

        target_angle = angle_to(self.robot.position, self.target_trash.position)
        self.robot.rotate_toward(target_angle, dt)

        # Check if facing target (within 15 degrees)
        angle_diff = abs(self.robot.angle - target_angle) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff < 15:
            self.transition_to(RobotState.APPROACHING)

    def _execute_approaching(self, dt: float, obstacles: pygame.sprite.Group):
        """Approaching: move toward target trash while facing it."""
        if not self.target_trash or self.target_trash.is_picked:
            self.transition_to(RobotState.PATROL)
            return

        # OPPORTUNISTIC: Check if there's closer trash we should grab instead
        closer_trash = self._find_closer_trash_on_path()
        if closer_trash:
            # Switch to the closer trash!
            if self._coordinator:
                self._coordinator.release_claim(self.target_trash.id)
                if self._coordinator.claim_trash(self.robot.id, closer_trash):
                    self.target_trash = closer_trash

        trash_pos = self.target_trash.position
        dist = distance(self.robot.position, trash_pos)

        # Check if we're facing the trash (important for pickup!)
        target_angle = angle_to(self.robot.position, trash_pos)
        angle_difference = abs(self.robot.angle - target_angle) % 360
        if angle_difference > 180:
            angle_difference = 360 - angle_difference
        facing_trash = angle_difference < 25  # Slightly more lenient

        # Transition to PICKING when close enough
        # Arm reach is ~75px total, but we want to start picking when reasonably close
        if dist < 80 and facing_trash:
            self.transition_to(RobotState.PICKING)
            return

        # Move toward trash (this also rotates to face it)
        self._move_toward(trash_pos, dt)

    def _find_closer_trash_on_path(self) -> Optional['Trash']:
        """
        Check if there's unclaimed trash closer than our target that we should grab first.

        Smart opportunistic behavior: if we pass by other trash, grab it!
        """
        if not self._trash_group or not self.target_trash:
            return None

        robot_pos = self.robot.position
        target_dist = distance(robot_pos, self.target_trash.position)

        # Only consider trash that's significantly closer (at least 30px closer)
        # and within pickup range consideration (< 100px away)
        best_trash = None
        best_dist = target_dist - 30  # Must be at least 30px closer

        for trash in self._trash_group:
            if trash == self.target_trash or trash.is_picked:
                continue

            # Check if claimed by another robot
            if self._coordinator and self._coordinator.is_claimed_by_other(trash, self.robot.id):
                continue

            trash_dist = distance(robot_pos, trash.position)

            # Must be closer than our threshold AND reasonably close to us
            if trash_dist < best_dist and trash_dist < 100:
                # Check we can actually see it (no obstacles blocking)
                if self._has_line_of_sight(trash.position):
                    best_trash = trash
                    best_dist = trash_dist

        return best_trash

    def _execute_picking(self, dt: float, trash_group: pygame.sprite.Group, obstacles: pygame.sprite.Group):
        """
        Picking: extend arm and grab trash.

        COMPLETELY REWRITTEN FOR RELIABILITY:
        - Aggressively move toward trash while extending arm
        - Continuously attempt pickup
        - Self-correcting: always face and approach trash
        """
        if not self.target_trash or self.target_trash.is_picked:
            if self.robot.arm:
                self.robot.arm.retract()
            self.transition_to(RobotState.PATROL)
            return

        self.pickup_timer += dt

        # Timeout after ~5 seconds (increased from 3)
        if self.pickup_timer > 150:
            if self.robot.arm:
                self.robot.arm.retract()
            if self._coordinator:
                self._coordinator.release_claim(self.target_trash.id)
            self.target_trash = None
            self.transition_to(RobotState.PATROL)
            return

        trash_pos = self.target_trash.position
        dist = distance(self.robot.position, trash_pos)

        # CAP the dt for picking operations - this is a delicate operation
        # that shouldn't be affected by simulation speed multiplier
        # This prevents overshooting at high speeds (4x etc)
        picking_dt = min(dt, 1.5)

        # ALWAYS face the trash - this is critical!
        target_angle = angle_to(self.robot.position, trash_pos)
        self.robot.rotate_toward(target_angle, picking_dt)

        if self.robot.arm:
            # Check if we succeeded on a previous frame
            if self.robot.arm.state == ArmState.HOLDING:
                self.transition_to(RobotState.STORING)
                return

            # Extend arm toward trash
            self.robot.arm.reach_toward(trash_pos)

            # Try to grab continuously once arm has any extension
            if self.robot.arm.extension >= 0.3:
                if self.robot.arm.pick_up_trash(self.target_trash):
                    trash_group.remove(self.target_trash)
                    if self._telemetry:
                        self._telemetry.log(EventType.TRASH_PICKUP, self.robot.id,
                                          {'trash_id': self.target_trash.id})
                    self.transition_to(RobotState.STORING)
                    return

            # AGGRESSIVE SELF-CORRECTION: If we can't reach, move closer!
            # Use capped picking_dt to prevent overshooting at high speeds
            if dist > 50:
                # Move toward trash at moderate speed
                self.robot.move_toward(trash_pos, picking_dt)
            elif dist > 35 and self.robot.arm.extension >= 0.6:
                # Creep forward when moderately close
                self.robot.move_toward(trash_pos, picking_dt * 0.5)
            elif dist > 25 and self.robot.arm.extension >= 0.9:
                # Final approach - very slow
                self.robot.move_toward(trash_pos, picking_dt * 0.3)

        else:
            # No arm - walk up and grab directly (also use capped dt)
            if dist < 25:
                if self.target_trash.pick_up(self.robot):
                    self.robot.add_to_bin(self.target_trash)
                    trash_group.remove(self.target_trash)
                    if self._telemetry:
                        self._telemetry.log(EventType.TRASH_PICKUP, self.robot.id,
                                          {'trash_id': self.target_trash.id})
                self.target_trash = None
                self.transition_to(RobotState.PATROL)
            else:
                self.robot.move_toward(trash_pos, picking_dt)

    def _execute_storing(self, dt: float):
        """Storing: retract arm and add trash to bin."""
        if self.robot.arm:
            if self.robot.arm.state == ArmState.HOLDING:
                self.robot.arm.retract()

            elif self.robot.arm.state == ArmState.RETRACTING:
                if self.robot.arm.extension <= 0.1:
                    if self.robot.arm.holding:
                        self.robot.add_to_bin(self.robot.arm.holding)
                        self.robot.arm.release()
                    self.target_trash = None
                    self._decide_after_storing()

            elif self.robot.arm.state == ArmState.IDLE:
                self.target_trash = None
                self._decide_after_storing()
        else:
            self.target_trash = None
            self._decide_after_storing()

    def _decide_after_storing(self):
        """Decide what to do after storing trash."""
        if self.robot.bin_full:
            self._request_dump_or_wait()
        else:
            self.transition_to(RobotState.PATROL)

    def _request_dump_or_wait(self):
        """Request to dump - queue if needed."""
        if self._coordinator:
            if self._coordinator.request_dump(self.robot.id):
                self.transition_to(RobotState.RETURNING)
            else:
                self.transition_to(RobotState.WAITING)
        else:
            self.transition_to(RobotState.RETURNING)

    def _execute_returning(self, dt: float, obstacles: pygame.sprite.Group):
        """Returning: navigate to nest ramp, going around obstacles if needed."""
        ramp_entry = self.nest.get_ramp_entry()
        dist_to_ramp = distance(self.robot.position, ramp_entry)

        # When close to the ramp area, check if we can proceed or need to wait
        if dist_to_ramp < 80:
            can_proceed = True
            if self._coordinator:
                # Must be at front of queue AND ramp must be clear
                can_proceed = self._coordinator.can_dump(self.robot.id) and self._coordinator.is_ramp_clear()

            if can_proceed:
                # We can dock - claim ramp and proceed
                if self._coordinator:
                    self._coordinator.claim_ramp(self.robot.id)
                self._return_waypoint = None
                self._return_stuck_count = 0
                self.transition_to(RobotState.DOCKING)
            else:
                # Ramp busy or not our turn - go to waiting area
                self.transition_to(RobotState.WAITING)
            return

        # If we have a waypoint to navigate around an obstacle
        if self._return_waypoint:
            dist_to_waypoint = distance(self.robot.position, self._return_waypoint)
            if dist_to_waypoint < 30:
                # Reached the waypoint - this is progress!
                self._return_waypoint = None
                self._return_stuck_count = max(0, self._return_stuck_count - 1)
                self._reset_progress_tracking()
            else:
                self._move_toward(self._return_waypoint, dt)
                return

        # Check if direct path to nest is blocked
        if not self._has_line_of_sight(ramp_entry, account_for_width=True, treat_trash_as_obstacles=True):
            waypoint = self._find_return_waypoint(ramp_entry)
            if waypoint:
                self._return_waypoint = waypoint
                self._move_toward(self._return_waypoint, dt)
                return

        # Direct path is clear - move toward nest
        self._move_toward(ramp_entry, dt)

    def _find_return_waypoint(self, target: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Find a waypoint to navigate around obstacles when returning to nest.

        Tries perpendicular offsets from the direct path to find a clear route.
        """
        robot_pos = self.robot.position

        # Direction to target
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 50:
            return None  # Close enough, just go direct

        # Normalize
        dx /= dist
        dy /= dist

        # Perpendicular directions
        perp_left = (-dy, dx)
        perp_right = (dy, -dx)

        # Try waypoints at various perpendicular offsets
        # Start closer, then try farther
        offsets = [80, 120, 160, 200]

        # Try waypoints at 1/3 and 2/3 of the way to target
        for path_fraction in [0.33, 0.5, 0.66]:
            mid_x = robot_pos[0] + dx * dist * path_fraction
            mid_y = robot_pos[1] + dy * dist * path_fraction

            for offset in offsets:
                for perp in [perp_left, perp_right]:
                    waypoint = (
                        mid_x + perp[0] * offset,
                        mid_y + perp[1] * offset
                    )

                    # Check if waypoint is in bounds
                    from config import SCREEN_WIDTH, SCREEN_HEIGHT
                    margin = 60
                    if not (margin < waypoint[0] < SCREEN_WIDTH - margin and
                            margin < waypoint[1] < SCREEN_HEIGHT - margin):
                        continue

                    # Check if we can reach the waypoint AND waypoint can reach target
                    if (self._has_line_of_sight(waypoint, account_for_width=True, treat_trash_as_obstacles=True) and
                        self._has_line_of_sight_from(waypoint, target, treat_trash_as_obstacles=True)):
                        return waypoint

        return None

    def _has_line_of_sight_from(self, start: Tuple[float, float], target: Tuple[float, float], treat_trash_as_obstacles: bool = False) -> bool:
        """Check line of sight from an arbitrary point (not robot position)."""
        if self._obstacles is None:
            return True

        inflate_amount = int(self.robot.width * 0.6)

        for obstacle in self._obstacles:
            rect = obstacle.get_rect()
            inflated_rect = rect.inflate(inflate_amount, inflate_amount)
            if self._line_intersects_rect(start, target, inflated_rect):
                return False

        # Check nest
        nest_rect = self.nest.get_rect()
        inflated_nest = nest_rect.inflate(inflate_amount, inflate_amount)
        if self._line_intersects_rect(start, target, inflated_nest):
            return False

        # Check trash if needed
        if treat_trash_as_obstacles and self._trash_group:
            for trash in self._trash_group:
                if trash.is_picked:
                    continue
                trash_rect = pygame.Rect(
                    trash.x - trash.size - 10,
                    trash.y - trash.size - 10,
                    (trash.size + 10) * 2,
                    (trash.size + 10) * 2
                )
                if self._line_intersects_rect(start, target, trash_rect):
                    return False

        return True

    def _execute_waiting(self, dt: float, obstacles: pygame.sprite.Group):
        """Waiting: stay where you are until it's your turn to dump."""
        # Check if it's our turn AND ramp is clear
        if self._coordinator:
            if self._coordinator.can_dump(self.robot.id) and self._coordinator.is_ramp_clear():
                # Our turn! Go to ramp
                self.transition_to(RobotState.RETURNING)
                return

        # Just stop and wait - don't go anywhere
        self.robot.set_velocity(0, 0)

    def _execute_docking(self, dt: float):
        """Docking: climb up the ramp (we already have exclusive ramp access)."""
        dock_pos = self.nest.get_dock_position()

        # Check if we're docked
        if self.nest.is_robot_docked(self.robot.position):
            self.robot.set_velocity(0, 0)
            self.transition_to(RobotState.DUMPING)
            return

        # Climb up the ramp at steady speed
        self.robot.move_toward(dock_pos, dt * 0.7)

    def _execute_dumping(self, dt: float):
        """Dumping: empty bin into nest."""
        self.dump_timer += dt
        self.robot.set_velocity(0, 0)

        # Wait a moment, then dump
        if self.dump_timer > 15 and not self.robot.bin_empty:
            dumped = self.robot.empty_bin()
            self.nest.receive_trash(dumped)
            if self._telemetry and dumped > 0:
                self._telemetry.log(EventType.TRASH_DUMP, self.robot.id, {'count': dumped})

        # After dumping, wait a bit then undock
        if self.dump_timer > 40:
            if self._coordinator:
                self._coordinator.finish_dump(self.robot.id)
            self.transition_to(RobotState.UNDOCKING)

    def _execute_undocking(self, dt: float):
        """Undocking: back down the ramp and get out of the way."""
        ramp_entry = self.nest.get_ramp_entry()

        # Phase 1: Back down past ramp entry and release it
        if self._coordinator and self._coordinator.get_ramp_owner() == self.robot.id:
            clear_pos = (ramp_entry[0] - 50, ramp_entry[1])
            dist_to_clear = distance(self.robot.position, clear_pos)

            if dist_to_clear < 25:
                # Clear of ramp - release it
                self._coordinator.release_ramp(self.robot.id)
            else:
                self.robot.move_toward(clear_pos, dt * 0.8)
                return

        # Phase 2: Move to departure zone (above the nest, out of the way)
        # This ensures we don't block robots approaching from the waiting area
        departure_pos = (self.nest.x - 100, self.nest.y - self.nest.height // 2 - 60)
        dist_to_departure = distance(self.robot.position, departure_pos)

        if dist_to_departure < 40:
            # Far enough away - now patrol
            self._generate_patrol()
            self.transition_to(RobotState.PATROL)
            return

        self.robot.move_toward(departure_pos, dt * 0.9)

    def _get_robot_repulsion_vector(self) -> Tuple[float, float]:
        """
        Calculate a repulsion vector away from nearby robots.

        Like opposing magnets - the closer another robot is, the stronger
        the push to steer away from it. This prevents robots from colliding
        and creates smooth avoidance behavior.

        Returns:
            (vx, vy) repulsion vector to blend with movement
        """
        if not self._all_robots:
            return (0.0, 0.0)

        repulsion_x = 0.0
        repulsion_y = 0.0

        robot_pos = self.robot.position

        # Repulsion parameters
        REPULSION_RANGE = 120  # Start steering away at this distance
        MIN_DISTANCE = 30  # Minimum distance to prevent division issues
        REPULSION_STRENGTH = 150  # Base strength of repulsion

        for other in self._all_robots:
            if other.id == self.robot.id:
                continue

            other_pos = other.position
            dx = robot_pos[0] - other_pos[0]
            dy = robot_pos[1] - other_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < REPULSION_RANGE and dist > 1:
                # Normalize the direction away from other robot
                dx /= dist
                dy /= dist

                # Inverse square falloff - stronger when closer
                # (REPULSION_RANGE - dist) makes it 0 at edge, max at MIN_DISTANCE
                effective_dist = max(dist, MIN_DISTANCE)
                strength = REPULSION_STRENGTH * (1 - dist / REPULSION_RANGE) ** 2 / effective_dist * 10

                repulsion_x += dx * strength
                repulsion_y += dy * strength

        return (repulsion_x, repulsion_y)

    def _move_toward(self, target: Tuple[float, float], dt: float):
        """
        Move toward a target using simple ant-like navigation.

        If in wall-following mode, move along the obstacle edge instead of direct.
        Periodically check if direct path is clear to exit wall-following.
        """
        # Calculate robot repulsion (magnetic avoidance of other robots)
        apply_repulsion = self.current_state not in [
            RobotState.DOCKING, RobotState.DUMPING, RobotState.UNDOCKING
        ]
        repulsion = self._get_robot_repulsion_vector() if apply_repulsion else (0.0, 0.0)

        # =====================================================
        # ANT-LIKE WALL FOLLOWING MODE
        # =====================================================
        if self._wall_follow_direction:
            self._wall_follow_frames += 1

            # Periodically check if direct path to goal is now clear
            if self._wall_follow_frames % 10 == 0:  # Check every 10 frames
                if self._is_path_to_goal_clear(target):
                    # Path is clear! Exit wall-following
                    self._wall_follow_direction = None
                    self._wall_follow_frames = 0
                    self._tried_both_directions = False
                    # Fall through to normal movement

            # Still wall-following - move perpendicular to goal direction
            if self._wall_follow_direction:
                wall_vec = self._get_wall_follow_direction_vector(target)
                final_vx = wall_vec[0] + repulsion[0]
                final_vy = wall_vec[1] + repulsion[1]
                self.robot.set_velocity(final_vx, final_vy)

                # Face the direction we're moving
                if abs(final_vx) > 0.1 or abs(final_vy) > 0.1:
                    move_angle = math.degrees(math.atan2(final_vy, final_vx))
                    self.robot.rotate_toward(move_angle, dt)
                return

        # =====================================================
        # NORMAL DIRECT MOVEMENT
        # =====================================================

        # Go straight to target
        move_vec = self.navigation.navigate_to_target(
            self.robot, target, self._obstacles,
            other_robots=self._all_robots,
            trash_group=self._trash_group,
            target_trash=self.target_trash,
            dt=dt
        )

        # Blend in repulsion
        final_vx = move_vec[0] + repulsion[0]
        final_vy = move_vec[1] + repulsion[1]
        self.robot.set_velocity(final_vx, final_vy)
        self.robot.rotate_toward(angle_to(self.robot.position, target), dt)

    def get_current_waypoint(self) -> Optional[Tuple[float, float]]:
        """Get current patrol waypoint."""
        if self.patrol_waypoints and 0 <= self.current_waypoint_index < len(self.patrol_waypoints):
            return self.patrol_waypoints[self.current_waypoint_index]
        return None

    def draw_debug(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw debug visualization."""
        self.navigation.draw_debug(screen, self.patrol_waypoints, self.current_waypoint_index)

        if self.target_trash and not self.target_trash.is_picked:
            pygame.draw.circle(screen, (255, 0, 0),
                             (int(self.target_trash.x), int(self.target_trash.y)),
                             self.target_trash.size + 5, 2)

    def __repr__(self) -> str:
        return f"BehaviorController(state={self.current_state.value})"
