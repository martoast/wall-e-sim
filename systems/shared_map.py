"""
Shared spatial memory - robots learn where obstacles are.

Like ant pheromones: mark bad areas, others avoid them.
Simple, elegant, and emergent.
"""
import pygame
from typing import Tuple


class SharedMap:
    """
    Shared spatial memory for robot navigation.

    A 2D grid where each cell has a "risk" value:
    - 0.0 = unknown/clear
    - 1.0 = definitely blocked

    Robots mark cells when they get stuck, and clear them when passing through.
    All robots share this map, so one robot's mistake helps others avoid it.
    """

    def __init__(self, width: int, height: int, cell_size: int = 25):
        """
        Args:
            width: World width in pixels
            height: World height in pixels
            cell_size: Size of each grid cell (smaller = more precision, more memory)
        """
        self.cell_size = cell_size
        self.cols = width // cell_size + 1
        self.rows = height // cell_size + 1
        self.width = width
        self.height = height

        # Risk grid: 0.0 = clear, 1.0 = blocked
        self.risk = [[0.0] * self.cols for _ in range(self.rows)]

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world position to grid cell."""
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        return (c, r)

    def _is_valid_cell(self, c: int, r: int) -> bool:
        """Check if cell coordinates are valid."""
        return 0 <= c < self.cols and 0 <= r < self.rows

    def mark_blocked(self, x: float, y: float, strength: float = 0.3):
        """
        Mark a position as risky.

        Args:
            x, y: World position
            strength: How much to increase risk (default 0.3)
        """
        c, r = self._get_cell(x, y)
        if self._is_valid_cell(c, r):
            self.risk[r][c] = min(1.0, self.risk[r][c] + strength)

    def mark_obstacle(self, rect: 'pygame.Rect', buffer: int = 10):
        """
        Mark an entire obstacle's area as fully blocked.

        Called when robot detects collision with a static obstacle.
        Marks all cells covered by the obstacle (plus buffer) as max risk.

        Args:
            rect: The obstacle's bounding rectangle
            buffer: Extra pixels around the obstacle to also mark (default 10)
        """
        # Inflate rect by buffer to create safe margin
        inflated = rect.inflate(buffer * 2, buffer * 2)

        # Get all cells that the inflated rectangle covers
        left_c = int(inflated.left / self.cell_size)
        right_c = int(inflated.right / self.cell_size)
        top_r = int(inflated.top / self.cell_size)
        bottom_r = int(inflated.bottom / self.cell_size)

        # Mark all cells as fully blocked (risk = 1.0)
        for r in range(top_r, bottom_r + 1):
            for c in range(left_c, right_c + 1):
                if self._is_valid_cell(c, r):
                    self.risk[r][c] = 1.0

    def mark_clear(self, x: float, y: float):
        """
        Robot passed through successfully - decrease risk.

        Called during successful movement. Decreases risk slowly (0.02)
        so it takes many successful passes to clear a risky area.

        Note: Does NOT clear cells marked as full obstacles (risk = 1.0).
        Those are confirmed static obstacles and should stay blocked.
        """
        c, r = self._get_cell(x, y)
        if self._is_valid_cell(c, r):
            # Don't clear confirmed obstacles (risk = 1.0)
            if self.risk[r][c] < 1.0:
                self.risk[r][c] = max(0.0, self.risk[r][c] - 0.02)

    def get_risk(self, x: float, y: float) -> float:
        """Get risk value at position (0.0 to 1.0)."""
        c, r = self._get_cell(x, y)
        if self._is_valid_cell(c, r):
            return self.risk[r][c]
        return 0.0

    def is_risky(self, x: float, y: float, threshold: float = 0.5) -> bool:
        """
        Check if position is likely blocked.

        Args:
            x, y: World position
            threshold: Risk level to consider "risky" (default 0.5)

        Returns:
            True if risk >= threshold
        """
        return self.get_risk(x, y) >= threshold

    def is_path_clear(self, start: Tuple[float, float], end: Tuple[float, float], threshold: float = 0.5, robot_width: float = 50.0) -> bool:
        """
        Check if a straight-line CORRIDOR between two points is clear of risky cells.

        Checks not just the center line, but also parallel lines offset by robot width
        to ensure the robot's body can fit through the gap.

        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            threshold: Risk level to consider blocked
            robot_width: Width of the robot (checks corridor this wide)

        Returns:
            True if corridor is clear, False if any part is blocked
        """
        import math

        x1, y1 = start
        x2, y2 = end

        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1:
            return True

        # Normalize
        dx /= dist
        dy /= dist

        # Perpendicular vector for width offset
        perp_x = -dy
        perp_y = dx

        # Check multiple lines: center, left offset, right offset
        half_width = robot_width / 2
        offsets = [0, half_width * 0.8, -half_width * 0.8]  # Center and sides (80% of half-width for some margin)

        for offset in offsets:
            # Offset start and end points perpendicular to path direction
            ox1 = x1 + perp_x * offset
            oy1 = y1 + perp_y * offset
            ox2 = x2 + perp_x * offset
            oy2 = y2 + perp_y * offset

            # Convert to cell coordinates
            c1, r1 = self._get_cell(ox1, oy1)
            c2, r2 = self._get_cell(ox2, oy2)

            # Walk the line using simple interpolation
            steps = max(abs(c2 - c1), abs(r2 - r1), 1)

            for i in range(steps + 1):
                t = i / steps
                c = int(c1 + (c2 - c1) * t)
                r = int(r1 + (r2 - r1) * t)

                if self._is_valid_cell(c, r):
                    if self.risk[r][c] >= threshold:
                        return False

        return True

    def find_clear_path_waypoint(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        threshold: float = 0.5
    ) -> Tuple[float, float]:
        """
        Find an intermediate waypoint to navigate around blocked cells.

        If direct path is blocked, tries perpendicular offsets to find a clear route.

        Args:
            start: Starting position
            end: Target position
            threshold: Risk threshold

        Returns:
            Intermediate waypoint position, or end if path is already clear
        """
        import math

        # If path is already clear, just go direct
        if self.is_path_clear(start, end, threshold):
            return end

        # Calculate perpendicular direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1:
            return end

        # Normalize
        dx /= dist
        dy /= dist

        # Perpendicular vectors
        perp_left = (-dy, dx)
        perp_right = (dy, -dx)

        # Try waypoints at various perpendicular offsets
        offsets = [50, 100, 150, 200]
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        for offset in offsets:
            for perp in [perp_left, perp_right]:
                waypoint = (
                    mid_x + perp[0] * offset,
                    mid_y + perp[1] * offset
                )

                # Check if waypoint itself is clear
                if self.is_risky(waypoint[0], waypoint[1], threshold):
                    continue

                # Check if path to waypoint is clear AND path from waypoint to end is clear
                if (self.is_path_clear(start, waypoint, threshold) and
                    self.is_path_clear(waypoint, end, threshold)):
                    return waypoint

        # Couldn't find clear path - return end anyway (stuck detection will handle it)
        return end

    def draw_debug(self, screen: pygame.Surface):
        """
        Draw risk heatmap overlay.

        Red = risky (robots got stuck there)
        Transparent = clear/unknown
        """
        # Create a transparent surface for the overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        for r in range(self.rows):
            for c in range(self.cols):
                risk = self.risk[r][c]
                if risk > 0.05:  # Only draw if there's some risk
                    # Red color with alpha based on risk level
                    alpha = int(risk * 120)  # Max alpha 120 (semi-transparent)
                    color = (255, 50, 50, alpha)

                    rect = pygame.Rect(
                        c * self.cell_size,
                        r * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(overlay, color, rect)

        screen.blit(overlay, (0, 0))
