"""
Telemetry system - logging and debug visualization.
"""
import pygame
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG_FONT_SIZE

if TYPE_CHECKING:
    from entities.robot import Robot
    from entities.nest import Nest


class EventType(Enum):
    """Types of events that can be logged."""
    STATE_CHANGE = "state_change"
    TRASH_PICKUP = "trash_pickup"
    TRASH_DUMP = "trash_dump"
    COLLISION = "collision"
    WAYPOINT_REACHED = "waypoint_reached"
    ERROR = "error"


@dataclass
class TelemetryEvent:
    """A single logged event."""
    timestamp: float
    event_type: EventType
    robot_id: int
    data: Dict[str, Any] = field(default_factory=dict)


class Telemetry:
    """
    Tracks and logs simulation events.

    Features:
    - Event logging with timestamps
    - Statistics calculation
    - Debug overlay rendering
    - Export to file
    """

    def __init__(self):
        self.events: List[TelemetryEvent] = []
        self.start_time = time.time()

        # Statistics
        self.stats: Dict[str, Any] = {
            'total_trash_collected': 0,
            'total_trash_dumped': 0,
            'total_distance_traveled': 0,
            'state_changes': 0,
            'collisions': 0,
        }

        # Per-robot stats
        self.robot_stats: Dict[int, Dict[str, Any]] = {}

        # For distance tracking
        self.last_positions: Dict[int, tuple] = {}

        # Font for rendering (debug_mode is controlled by Simulation, not here)
        self.font: Optional[pygame.font.Font] = None

    def init_font(self):
        """Initialize the font for rendering."""
        if self.font is None:
            self.font = pygame.font.Font(None, DEBUG_FONT_SIZE)

    def get_elapsed_time(self) -> float:
        """Get elapsed time since simulation start."""
        return time.time() - self.start_time

    def log(
        self,
        event_type: EventType,
        robot_id: int,
        data: Dict[str, Any] = None
    ):
        """
        Log an event.

        Args:
            event_type: Type of event
            robot_id: ID of the robot involved
            data: Additional event data
        """
        event = TelemetryEvent(
            timestamp=self.get_elapsed_time(),
            event_type=event_type,
            robot_id=robot_id,
            data=data or {}
        )
        self.events.append(event)

        # Update statistics
        self._update_stats(event)

    def _update_stats(self, event: TelemetryEvent):
        """Update statistics based on event."""
        if event.event_type == EventType.TRASH_PICKUP:
            self.stats['total_trash_collected'] += 1
            self._get_robot_stats(event.robot_id)['trash_collected'] += 1

        elif event.event_type == EventType.TRASH_DUMP:
            count = event.data.get('count', 0)
            self.stats['total_trash_dumped'] += count
            self._get_robot_stats(event.robot_id)['trash_dumped'] += count

        elif event.event_type == EventType.STATE_CHANGE:
            self.stats['state_changes'] += 1

        elif event.event_type == EventType.COLLISION:
            self.stats['collisions'] += 1
            self._get_robot_stats(event.robot_id)['collisions'] += 1

    def _get_robot_stats(self, robot_id: int) -> Dict[str, Any]:
        """Get or create stats dict for a robot."""
        if robot_id not in self.robot_stats:
            self.robot_stats[robot_id] = {
                'trash_collected': 0,
                'trash_dumped': 0,
                'distance_traveled': 0,
                'collisions': 0,
            }
        return self.robot_stats[robot_id]

    def update_distance(self, robot_id: int, position: tuple):
        """Update distance traveled for a robot."""
        if robot_id in self.last_positions:
            last_pos = self.last_positions[robot_id]
            dx = position[0] - last_pos[0]
            dy = position[1] - last_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5

            self.stats['total_distance_traveled'] += dist
            self._get_robot_stats(robot_id)['distance_traveled'] += dist

        self.last_positions[robot_id] = position

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self.stats,
            'elapsed_time': self.get_elapsed_time(),
            'events_logged': len(self.events),
            'robot_stats': self.robot_stats.copy(),
        }

    def get_recent_events(self, count: int = 10) -> List[TelemetryEvent]:
        """Get the most recent events."""
        return self.events[-count:]

    def draw_overlay(
        self,
        screen: pygame.Surface,
        robots: List['Robot'],
        nest: 'Nest',
        fps: float
    ):
        """Draw debug overlay on screen. Only call when debug_mode is True."""
        self.init_font()

        # Background panel
        panel_width = 250
        panel_height = 200
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(
            panel_surface, (0, 0, 0, 180),
            (0, 0, panel_width, panel_height),
            border_radius=5
        )
        screen.blit(panel_surface, (10, 10))

        # Stats text
        y = 20
        line_height = 18

        lines = [
            f"FPS: {fps:.1f}",
            f"Time: {self.get_elapsed_time():.1f}s",
            f"",
            f"Trash Collected: {self.stats['total_trash_collected']}",
            f"Trash Dumped: {self.stats['total_trash_dumped']}",
            f"Nest Fill: {nest.fill_level}/{nest.capacity}",
            f"",
            f"Distance: {self.stats['total_distance_traveled']:.0f}px",
            f"Events: {len(self.events)}",
        ]

        # Add robot-specific info
        for robot in robots:
            lines.append(f"")
            lines.append(f"Robot {robot.id}: {robot.state.value}")
            lines.append(f"  Bin: {robot.bin_count}/{robot.bin_capacity}")

        for line in lines[:12]:  # Limit lines to fit panel
            if line:
                text = self.font.render(line, True, (255, 255, 255))
                screen.blit(text, (20, y))
            y += line_height

    def draw_robot_debug(
        self,
        screen: pygame.Surface,
        robot: 'Robot',
        behavior_controller=None
    ):
        """Draw debug info for a specific robot. Only call when debug_mode is True."""
        self.init_font()

        # Robot state label is drawn by robot itself
        robot.draw_debug(screen, self.font)

        # Draw sensor visualization
        if behavior_controller:
            # Pass scan angle so vision cone shows where robot is "looking"
            scan_offset = getattr(behavior_controller, '_scan_angle', 0.0)
            behavior_controller.sensors.draw_debug(screen, robot, scan_angle_offset=scan_offset)
            behavior_controller.draw_debug(screen, self.font)

    def export_log(self, filepath: str):
        """Export event log to a file."""
        with open(filepath, 'w') as f:
            f.write("WALL-E Simulation Telemetry Log\n")
            f.write(f"Duration: {self.get_elapsed_time():.2f}s\n")
            f.write(f"Total Events: {len(self.events)}\n")
            f.write("\n--- Statistics ---\n")
            for key, value in self.stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n--- Events ---\n")
            for event in self.events:
                f.write(
                    f"[{event.timestamp:.2f}] "
                    f"Robot {event.robot_id} - "
                    f"{event.event_type.value}: {event.data}\n"
                )

    def __repr__(self) -> str:
        return f"Telemetry(events={len(self.events)}, elapsed={self.get_elapsed_time():.1f}s)"
