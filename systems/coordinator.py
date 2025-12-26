"""
Coordinator system - centralized multi-robot task allocation.
"""
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import pygame

if TYPE_CHECKING:
    from entities.trash import Trash


class Coordinator:
    """
    Centralized coordinator for multi-robot task allocation.

    Responsibilities:
    - Track which trash is claimed by which robot
    - Prevent multiple robots from pursuing the same trash
    - Assign patrol zones to spread robots across the area
    - Manage dump queue so robots take turns at the nest
    """

    def __init__(self):
        # Trash claims: trash.id -> robot.id
        self.trash_claims: Dict[int, int] = {}

        # Patrol zones: robot.id -> (x, y, width, height)
        self.patrol_zones: Dict[int, Tuple[int, int, int, int]] = {}

        # Dump queue: list of robot IDs waiting to dump
        # First in list = currently dumping or next to dump
        self.dump_queue: List[int] = []

        # Ramp occupancy: which robot (if any) is currently on the ramp
        # None = ramp is clear, robot_id = that robot is on the ramp
        self.robot_on_ramp: Optional[int] = None

    def claim_trash(self, robot_id: int, trash: 'Trash') -> bool:
        """
        Attempt to claim trash for a robot.

        Args:
            robot_id: The robot attempting to claim
            trash: The trash item to claim

        Returns:
            True if claim successful (or already owned by this robot),
            False if claimed by another robot
        """
        if trash.id in self.trash_claims:
            # Already claimed - check if it's ours
            return self.trash_claims[trash.id] == robot_id

        # Claim it
        self.trash_claims[trash.id] = robot_id
        return True

    def release_claim(self, trash_id: int):
        """
        Release a trash claim.

        Args:
            trash_id: ID of the trash to release
        """
        self.trash_claims.pop(trash_id, None)

    def release_claims_for_robot(self, robot_id: int):
        """
        Release all claims held by a specific robot.

        Args:
            robot_id: The robot whose claims to release
        """
        self.trash_claims = {
            tid: rid for tid, rid in self.trash_claims.items()
            if rid != robot_id
        }

    def is_claimed(self, trash: 'Trash') -> bool:
        """
        Check if trash is claimed by any robot.

        Args:
            trash: The trash to check

        Returns:
            True if claimed
        """
        return trash.id in self.trash_claims

    def is_claimed_by_other(self, trash: 'Trash', robot_id: int) -> bool:
        """
        Check if trash is claimed by a different robot.

        Args:
            trash: The trash to check
            robot_id: The robot asking

        Returns:
            True if claimed by someone else
        """
        if trash.id not in self.trash_claims:
            return False
        return self.trash_claims[trash.id] != robot_id

    def get_claimer(self, trash: 'Trash') -> Optional[int]:
        """
        Get the robot ID that claimed this trash.

        Args:
            trash: The trash to check

        Returns:
            Robot ID or None if unclaimed
        """
        return self.trash_claims.get(trash.id)

    def cleanup_claims(self, trash_group: pygame.sprite.Group):
        """
        Remove claims for trash that no longer exists or is picked up.

        Args:
            trash_group: Current trash group to validate against
        """
        valid_ids = {t.id for t in trash_group if not t.is_picked}
        self.trash_claims = {
            tid: rid for tid, rid in self.trash_claims.items()
            if tid in valid_ids
        }

    def assign_patrol_zones(
        self,
        robot_count: int,
        screen_width: int,
        screen_height: int,
        margin: int = 80
    ):
        """
        Divide the screen into patrol zones, one per robot.

        Uses vertical strips so each robot covers a different area.

        Args:
            robot_count: Number of robots
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            margin: Margin from screen edges
        """
        if robot_count <= 0:
            return

        usable_width = screen_width - margin * 2
        zone_width = usable_width // robot_count

        for i in range(robot_count):
            self.patrol_zones[i] = (
                margin + i * zone_width,
                margin,
                zone_width,
                screen_height - margin * 2
            )

    def get_patrol_zone(self, robot_id: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the patrol zone bounds for a robot.

        Args:
            robot_id: The robot's ID

        Returns:
            (x, y, width, height) tuple or None if no zone assigned
        """
        return self.patrol_zones.get(robot_id)

    def get_claim_count(self) -> int:
        """Get the number of active claims."""
        return len(self.trash_claims)

    # ========== Dump Queue Management ==========

    def request_dump(self, robot_id: int) -> bool:
        """
        Robot requests to dump trash at the nest.
        Adds to queue if not already in it.

        Args:
            robot_id: The robot requesting to dump

        Returns:
            True if robot is at front of queue (can proceed),
            False if robot must wait
        """
        if robot_id not in self.dump_queue:
            self.dump_queue.append(robot_id)

        # Return True if this robot is at the front
        return self.dump_queue[0] == robot_id

    def can_dump(self, robot_id: int) -> bool:
        """
        Check if robot is allowed to proceed to dump.

        Args:
            robot_id: The robot to check

        Returns:
            True if robot is at front of queue
        """
        if not self.dump_queue:
            return True  # No queue, go ahead
        return self.dump_queue[0] == robot_id

    def finish_dump(self, robot_id: int):
        """
        Robot finished dumping, remove from queue.

        Args:
            robot_id: The robot that finished
        """
        if robot_id in self.dump_queue:
            self.dump_queue.remove(robot_id)

    def leave_queue(self, robot_id: int):
        """
        Robot leaves the dump queue (e.g., got stuck, changed state).

        Args:
            robot_id: The robot leaving
        """
        if robot_id in self.dump_queue:
            self.dump_queue.remove(robot_id)

    def get_queue_position(self, robot_id: int) -> int:
        """
        Get robot's position in dump queue.

        Args:
            robot_id: The robot to check

        Returns:
            Position (0 = front), or -1 if not in queue
        """
        if robot_id in self.dump_queue:
            return self.dump_queue.index(robot_id)
        return -1

    def get_queue_length(self) -> int:
        """Get number of robots waiting to dump."""
        return len(self.dump_queue)

    # ========== Ramp Occupancy Management ==========

    def claim_ramp(self, robot_id: int) -> bool:
        """
        Robot claims the ramp for exclusive access.

        Args:
            robot_id: The robot claiming the ramp

        Returns:
            True if claim successful (ramp was clear or already owned),
            False if another robot is on the ramp
        """
        if self.robot_on_ramp is None:
            self.robot_on_ramp = robot_id
            return True
        return self.robot_on_ramp == robot_id

    def release_ramp(self, robot_id: int):
        """
        Robot releases the ramp after fully clearing it.

        Args:
            robot_id: The robot releasing the ramp
        """
        if self.robot_on_ramp == robot_id:
            self.robot_on_ramp = None

    def is_ramp_clear(self) -> bool:
        """Check if the ramp is clear for use."""
        return self.robot_on_ramp is None

    def get_ramp_owner(self) -> Optional[int]:
        """Get the robot ID currently on the ramp, or None."""
        return self.robot_on_ramp

    def __repr__(self) -> str:
        return f"Coordinator(claims={len(self.trash_claims)}, zones={len(self.patrol_zones)}, queue={len(self.dump_queue)}, ramp={self.robot_on_ramp})"
