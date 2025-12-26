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

        # Goal progress tracking (for stuck detection)
        self._goal_distance_at_check = float('inf')
        self._frames_since_progress = 0

        # Return path - waypoint to navigate around obstacles when returning
        self._return_waypoint: Optional[Tuple[float, float]] = None

        # References (set during update)
        self._coordinator: Optional['Coordinator'] = None
        self._telemetry: Optional['Telemetry'] = None
        self._obstacles: Optional[pygame.sprite.Group] = None

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
            zone_bounds=zone
        )
        self.current_waypoint_index = 0
        self._reset_progress_tracking()

    def _reset_progress_tracking(self):
        """Reset the goal progress tracker."""
        self._goal_distance_at_check = float('inf')
        self._frames_since_progress = 0

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
        Check if robot is stuck (not making progress toward goal).

        Simple logic: If we haven't gotten closer to goal in N frames, we're stuck.
        """
        # Only check in movement states
        if self.current_state not in [RobotState.PATROL, RobotState.APPROACHING,
                                       RobotState.SEEKING, RobotState.RETURNING,
                                       RobotState.UNDOCKING]:
            return False

        goal = self._get_current_goal()
        if goal is None:
            return False

        current_dist = distance(self.robot.position, goal)

        # Check every 30 frames (~1 second)
        self._frames_since_progress += 1
        if self._frames_since_progress >= 30:
            # Have we gotten closer?
            if current_dist < self._goal_distance_at_check - 5:
                # Yes, making progress
                self._goal_distance_at_check = current_dist
                self._frames_since_progress = 0
                return False
            else:
                # No progress - we're stuck
                return True

        return False

    def _handle_stuck(self):
        """Handle being stuck - give up current goal gracefully."""
        self._reset_progress_tracking()

        # If targeting trash, give up on it
        if self.target_trash:
            if self._coordinator:
                self._coordinator.release_claim(self.target_trash.id)
            self.target_trash = None
            self._generate_patrol()
            self.transition_to(RobotState.PATROL)
            return

        # If returning to nest and stuck, try to go around the obstacle
        if self.current_state == RobotState.RETURNING:
            self._find_return_waypoint()
            return

        # For patrol/other states, just get new patrol path
        self._generate_patrol()
        self.transition_to(RobotState.PATROL)

    def _find_return_waypoint(self):
        """Find a waypoint to navigate around obstacles when returning."""
        import random

        ramp_entry = self.nest.get_ramp_entry()
        robot_pos = self.robot.position

        # If we already have a waypoint and got stuck going to it, clear it and try a different one
        if self._return_waypoint:
            self._return_waypoint = None

        # Try to find a clear path by checking perpendicular directions
        # Calculate vector to nest
        dx = ramp_entry[0] - robot_pos[0]
        dy = ramp_entry[1] - robot_pos[1]
        dist_to_nest = math.sqrt(dx * dx + dy * dy)

        if dist_to_nest < 1:
            return

        # Normalize
        dx /= dist_to_nest
        dy /= dist_to_nest

        # Perpendicular vectors (left and right of current direction)
        perp_left = (-dy, dx)
        perp_right = (dy, -dx)

        # Try waypoints at various distances perpendicular to the path
        offsets = [100, 150, 200]
        directions = [perp_left, perp_right]
        random.shuffle(directions)  # Randomize which side we try first

        for direction in directions:
            for offset in offsets:
                waypoint = (
                    robot_pos[0] + direction[0] * offset,
                    robot_pos[1] + direction[1] * offset
                )

                # Check if this waypoint is in bounds
                if not self.navigation.is_in_bounds(waypoint):
                    continue

                # Check if we have line of sight to this waypoint
                if self._has_line_of_sight(waypoint):
                    self._return_waypoint = waypoint
                    return

        # If no good waypoint found, try a random position in our patrol zone
        zone = None
        if self._coordinator:
            zone = self._coordinator.get_patrol_zone(self.robot.id)

        if zone:
            # Pick a random point in our zone
            self._return_waypoint = (
                random.uniform(zone[0], zone[0] + zone[2]),
                random.uniform(zone[1], zone[1] + zone[3])
            )
        else:
            # Just pick a random nearby point
            self._return_waypoint = (
                robot_pos[0] + random.uniform(-100, 100),
                robot_pos[1] + random.uniform(-100, 100)
            )

    def _has_line_of_sight(self, target_pos: Tuple[float, float]) -> bool:
        """
        Check if there's a clear line of sight from robot to target.
        Returns False if any obstacle blocks the path.
        """
        if self._obstacles is None:
            return True

        robot_pos = self.robot.position

        # Check against each obstacle
        for obstacle in self._obstacles:
            rect = obstacle.get_rect()
            if self._line_intersects_rect(robot_pos, target_pos, rect):
                return False

        # Also check against nest (can't grab through nest)
        nest_rect = self.nest.get_rect()
        if self._line_intersects_rect(robot_pos, target_pos, nest_rect):
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
        telemetry: 'Telemetry' = None
    ):
        """Update behavior based on current state."""
        self._all_robots = all_robots or []
        self._telemetry = telemetry
        self._trash_group = trash_group
        self._coordinator = coordinator
        self._obstacles = obstacles

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
        """Patrol: follow waypoints, look for trash."""
        # Look for trash if bin not full
        if not self.robot.bin_full:
            visible_trash = self.sensors.detect_trash_in_vision(self.robot, trash_group)

            for trash in visible_trash:
                # Skip if claimed by another robot
                if self._coordinator and self._coordinator.is_claimed_by_other(trash, self.robot.id):
                    continue

                # Check line of sight - can we actually see this trash?
                if not self._has_line_of_sight(trash.position):
                    continue

                # Try to claim
                if self._coordinator:
                    if not self._coordinator.claim_trash(self.robot.id, trash):
                        continue

                self.target_trash = trash
                self.transition_to(RobotState.SEEKING)
                return

        # Return to dump if bin full
        if self.robot.bin_full:
            self._request_dump_or_wait()
            return

        # Follow patrol path
        if not self.patrol_waypoints:
            self._generate_patrol()
            return

        waypoint = self.patrol_waypoints[self.current_waypoint_index]

        if self.navigation.is_at_waypoint(self.robot, waypoint):
            # Next waypoint
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.patrol_waypoints)
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
        """Approaching: move toward target trash."""
        if not self.target_trash or self.target_trash.is_picked:
            self.transition_to(RobotState.PATROL)
            return

        front_pos = self.robot.get_front_position()
        dist = distance(front_pos, self.target_trash.position)

        # Close enough to pick?
        if dist < 50:
            self.transition_to(RobotState.PICKING)
            return

        # Move toward trash
        self._move_toward(self.target_trash.position, dt)

    def _execute_picking(self, dt: float, trash_group: pygame.sprite.Group, obstacles: pygame.sprite.Group):
        """Picking: extend arm and grab trash."""
        if not self.target_trash or self.target_trash.is_picked:
            self.transition_to(RobotState.PATROL)
            return

        self.pickup_timer += dt

        # Check line of sight before grabbing
        if not self._has_line_of_sight(self.target_trash.position):
            # Can't see trash (blocked by obstacle) - give up
            if self._coordinator:
                self._coordinator.release_claim(self.target_trash.id)
            self.target_trash = None
            if self.robot.arm:
                self.robot.arm.retract()
            self.transition_to(RobotState.PATROL)
            return

        if self.robot.arm:
            if self.robot.arm.state == ArmState.IDLE:
                self.robot.arm.reach_toward(self.target_trash.position)

            elif self.robot.arm.state == ArmState.EXTENDING:
                self.robot.arm.reach_toward(self.target_trash.position)

                if self.robot.arm.extension >= 0.7:
                    if self.robot.arm.pick_up_trash(self.target_trash):
                        trash_group.remove(self.target_trash)
                        if self._telemetry:
                            self._telemetry.log(EventType.TRASH_PICKUP, self.robot.id,
                                              {'trash_id': self.target_trash.id})
                        self.transition_to(RobotState.STORING)
                    elif self.pickup_timer > 45:
                        # Timeout - can't reach
                        self.robot.arm.retract()
                        if self._coordinator:
                            self._coordinator.release_claim(self.target_trash.id)
                        self.target_trash = None
                        self.transition_to(RobotState.PATROL)

            elif self.robot.arm.state == ArmState.HOLDING:
                self.transition_to(RobotState.STORING)
        else:
            # No arm - direct pickup
            if self.target_trash.pick_up(self.robot):
                self.robot.add_to_bin(self.target_trash)
                trash_group.remove(self.target_trash)
                if self._telemetry:
                    self._telemetry.log(EventType.TRASH_PICKUP, self.robot.id,
                                      {'trash_id': self.target_trash.id})
            self.target_trash = None
            self.transition_to(RobotState.PATROL)

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

        # Check if we've reached the nest
        if self.nest.is_robot_at_ramp_entry(self.robot.position):
            self._return_waypoint = None
            self.transition_to(RobotState.DOCKING)
            return

        # If we have a waypoint to navigate around an obstacle
        if self._return_waypoint:
            dist_to_waypoint = distance(self.robot.position, self._return_waypoint)
            if dist_to_waypoint < 30:
                # Reached the waypoint, clear it and continue to nest
                self._return_waypoint = None
                self._reset_progress_tracking()
            else:
                # Move toward the waypoint
                self._move_toward(self._return_waypoint, dt)
                return

        # Move toward nest
        self._move_toward(ramp_entry, dt)

    def _execute_waiting(self, dt: float, obstacles: pygame.sprite.Group):
        """Waiting: wait for turn to dump."""
        if self._coordinator and self._coordinator.can_dump(self.robot.id):
            self.transition_to(RobotState.RETURNING)
            return

        # Just stop and wait
        self.robot.set_velocity(0, 0)

    def _execute_docking(self, dt: float):
        """Docking: climb up the ramp."""
        dock_pos = self.nest.get_dock_position()

        if self.nest.is_robot_docked(self.robot.position):
            self.transition_to(RobotState.DUMPING)
            return

        # Move up ramp
        self.robot.move_toward(dock_pos, dt)

    def _execute_dumping(self, dt: float):
        """Dumping: empty bin into nest."""
        self.dump_timer += dt

        # Wait for dump animation
        if self.dump_timer < 30:
            return

        # Dump trash
        if not self.robot.bin_empty:
            dumped = self.robot.empty_bin()
            self.nest.receive_trash(dumped)
            if self._telemetry and dumped > 0:
                self._telemetry.log(EventType.TRASH_DUMP, self.robot.id, {'count': dumped})

        # After dump animation, drive back down
        if self.dump_timer > 60:
            if self._coordinator:
                self._coordinator.finish_dump(self.robot.id)
            self.transition_to(RobotState.UNDOCKING)

    def _execute_undocking(self, dt: float):
        """Undocking: drive back down the ramp smoothly."""
        ramp_entry = self.nest.get_ramp_entry()

        # Check if we've reached the bottom of the ramp
        dist = distance(self.robot.position, ramp_entry)
        if dist < 20:
            # Done undocking - start patrol
            self._generate_patrol()
            self.transition_to(RobotState.PATROL)
            return

        # Drive down the ramp
        self.robot.move_toward(ramp_entry, dt)

    def _move_toward(self, target: Tuple[float, float], dt: float):
        """Simple movement toward a target."""
        move_vec = self.navigation.navigate_to_target(
            self.robot, target, self._obstacles,
            other_robots=self._all_robots,
            trash_group=self._trash_group,
            target_trash=self.target_trash,
            dt=dt
        )
        self.robot.set_velocity(move_vec[0], move_vec[1])
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
