"""
Arm entity - articulated robot arm with claw.
"""
import pygame
import math
from typing import Tuple, Optional, TYPE_CHECKING
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ARM_SEGMENT_1_LENGTH, ARM_SEGMENT_2_LENGTH,
    ARM_EXTEND_SPEED, ARM_RETRACT_SPEED, CLAW_SIZE,
    COLOR_ARM, COLOR_CLAW, COLOR_CLAW_OPEN
)
from utils.math_helpers import distance, angle_to, lerp, clamp

if TYPE_CHECKING:
    from .robot import Robot
    from .trash import Trash


class ArmState(Enum):
    """States the arm can be in."""
    IDLE = "idle"
    EXTENDING = "extending"
    GRABBING = "grabbing"
    RETRACTING = "retracting"
    HOLDING = "holding"


class Arm:
    """
    A 2-segment articulated arm with a claw.

    Uses simple point-at-target logic rather than full inverse kinematics.
    """

    def __init__(self, robot: 'Robot'):
        self.robot = robot
        robot.arm = self  # Back-reference

        # Segment lengths
        self.segment1_length = ARM_SEGMENT_1_LENGTH
        self.segment2_length = ARM_SEGMENT_2_LENGTH

        # Joint angles (relative to previous segment)
        self.joint1_angle = 0.0  # Base joint, relative to robot facing
        self.joint2_angle = 0.0  # Elbow joint

        # Extension state (0 = retracted, 1 = fully extended)
        self.extension = 0.0
        self.target_extension = 0.0

        # Claw
        self.claw_open = True
        self.claw_angle = 30.0  # Angle of claw opening

        # State
        self.state = ArmState.IDLE
        self.target_position: Optional[Tuple[float, float]] = None

        # Held item
        self.holding: Optional['Trash'] = None

    @property
    def total_length(self) -> float:
        """Total arm reach."""
        return self.segment1_length + self.segment2_length

    def get_mount_position(self) -> Tuple[float, float]:
        """Get where the arm is attached to the robot."""
        return self.robot.get_arm_mount_position()

    def get_joint_position(self) -> Tuple[float, float]:
        """Get the position of the elbow joint."""
        mount = self.get_mount_position()

        # Joint1 angle is relative to robot facing
        absolute_angle = self.robot.angle + self.joint1_angle

        # Scale by extension
        length = self.segment1_length * self.extension

        rad = math.radians(absolute_angle)
        return (
            mount[0] + math.cos(rad) * length,
            mount[1] + math.sin(rad) * length
        )

    def get_claw_position(self) -> Tuple[float, float]:
        """Get the position of the claw end."""
        joint = self.get_joint_position()

        # Joint2 angle is relative to joint1
        absolute_angle = self.robot.angle + self.joint1_angle + self.joint2_angle

        # Scale by extension
        length = self.segment2_length * self.extension

        rad = math.radians(absolute_angle)
        return (
            joint[0] + math.cos(rad) * length,
            joint[1] + math.sin(rad) * length
        )

    def reach_toward(self, target: Tuple[float, float]):
        """
        Set the arm to reach toward a target position.

        Uses simple point-at-target logic.
        """
        self.target_position = target
        self.state = ArmState.EXTENDING
        self.target_extension = 1.0
        self.claw_open = True

        # Calculate angles to point at target
        mount = self.get_mount_position()
        target_angle = angle_to(mount, target)

        # Joint1 points toward target (relative to robot)
        self.joint1_angle = target_angle - self.robot.angle

        # Joint2 continues in same direction
        self.joint2_angle = 0

    def grab(self) -> bool:
        """
        Close the claw to grab.

        Returns:
            True if currently holding something
        """
        self.claw_open = False
        self.state = ArmState.GRABBING
        return self.holding is not None

    def release(self) -> Optional['Trash']:
        """
        Open the claw and release held item.

        Returns:
            The released trash item, if any
        """
        self.claw_open = True
        released = self.holding
        self.holding = None
        return released

    def retract(self):
        """Retract the arm back to idle position."""
        self.state = ArmState.RETRACTING
        self.target_extension = 0.0
        self.target_position = None

    def pick_up_trash(self, trash: 'Trash') -> bool:
        """
        Attempt to pick up a piece of trash.

        IMPROVED: Multiple pickup zones for maximum reliability:
        1. Near the claw tip
        2. Near the elbow joint
        3. Along the arm segment
        4. Near the robot front (for trash that slipped under)
        """
        if self.holding is not None:
            return False

        if trash.is_picked:
            return False

        claw_pos = self.get_claw_position()
        joint_pos = self.get_joint_position()
        mount_pos = self.get_mount_position()

        # Check distance from multiple points to catch all scenarios
        claw_dist = distance(claw_pos, trash.position)
        joint_dist = distance(joint_pos, trash.position)
        mount_dist = distance(mount_pos, trash.position)

        # Also check the midpoint of segment 2 (between joint and claw)
        mid_seg2 = (
            (joint_pos[0] + claw_pos[0]) / 2,
            (joint_pos[1] + claw_pos[1]) / 2
        )
        mid_dist = distance(mid_seg2, trash.position)

        # Find the minimum distance from any arm part
        min_dist = min(claw_dist, joint_dist, mid_dist, mount_dist)

        # VERY generous pickup range - prioritize success over precision
        # This ensures the robot can actually grab trash reliably
        pickup_range = CLAW_SIZE + trash.size + 40  # Increased from 30

        if min_dist <= pickup_range:
            if trash.pick_up(self.robot):
                self.holding = trash
                self.claw_open = False
                self.state = ArmState.HOLDING
                return True

        return False

    def update(self, dt: float = 1.0):
        """Update arm state."""
        # Cap dt so arm doesn't extend too fast at high simulation speeds
        # This ensures reliable pickup behavior at any speed multiplier
        arm_dt = min(dt, 1.5)

        # Arm extension/retraction rates
        extend_rate = 0.12
        retract_rate = 0.15

        if self.extension < self.target_extension:
            self.extension = min(
                self.extension + extend_rate * arm_dt,
                self.target_extension
            )
        elif self.extension > self.target_extension:
            self.extension = max(
                self.extension - retract_rate * arm_dt,
                self.target_extension
            )

        # Check if done extending/retracting
        if abs(self.extension - self.target_extension) < 0.01:
            self.extension = self.target_extension
            if self.state == ArmState.RETRACTING and self.extension == 0:
                self.state = ArmState.IDLE
                self.joint1_angle = 0
                self.joint2_angle = 0

        # Update held item position
        if self.holding:
            self.holding.position = self.get_claw_position()

    def draw(self, screen: pygame.Surface):
        """Draw the arm."""
        if self.extension < 0.05:
            return  # Arm is retracted, don't draw

        mount = self.get_mount_position()
        joint = self.get_joint_position()
        claw = self.get_claw_position()

        # Draw segment 1
        pygame.draw.line(
            screen, COLOR_ARM,
            (int(mount[0]), int(mount[1])),
            (int(joint[0]), int(joint[1])),
            4
        )

        # Draw joint circle
        pygame.draw.circle(
            screen, COLOR_ARM,
            (int(joint[0]), int(joint[1])),
            5
        )

        # Draw segment 2
        pygame.draw.line(
            screen, COLOR_ARM,
            (int(joint[0]), int(joint[1])),
            (int(claw[0]), int(claw[1])),
            3
        )

        # Draw claw
        claw_color = COLOR_CLAW_OPEN if self.claw_open else COLOR_CLAW

        # Claw is two arcs
        claw_angle = self.robot.angle + self.joint1_angle + self.joint2_angle
        claw_spread = self.claw_angle if self.claw_open else 5

        # Left claw finger
        left_angle = claw_angle - claw_spread
        left_rad = math.radians(left_angle)
        left_end = (
            claw[0] + math.cos(left_rad) * CLAW_SIZE,
            claw[1] + math.sin(left_rad) * CLAW_SIZE
        )
        pygame.draw.line(
            screen, claw_color,
            (int(claw[0]), int(claw[1])),
            (int(left_end[0]), int(left_end[1])),
            3
        )

        # Right claw finger
        right_angle = claw_angle + claw_spread
        right_rad = math.radians(right_angle)
        right_end = (
            claw[0] + math.cos(right_rad) * CLAW_SIZE,
            claw[1] + math.sin(right_rad) * CLAW_SIZE
        )
        pygame.draw.line(
            screen, claw_color,
            (int(claw[0]), int(claw[1])),
            (int(right_end[0]), int(right_end[1])),
            3
        )

        # Draw held item if any (it draws itself at claw position)
        if self.holding:
            self.holding.draw(screen)

    def __repr__(self) -> str:
        return f"Arm(state={self.state.value}, ext={self.extension:.2f}, holding={self.holding is not None})"
