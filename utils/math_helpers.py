"""
Math helper utilities for 2D vector operations.
"""
import math
from typing import Tuple

Point = Tuple[float, float]
Vector = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def normalize(v: Vector) -> Vector:
    """Return unit vector. Returns (0, 0) if zero vector."""
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2)
    if mag == 0:
        return (0.0, 0.0)
    return (v[0] / mag, v[1] / mag)


def angle_to(a: Point, b: Point) -> float:
    """
    Calculate angle from point a to point b in degrees.
    0 degrees = right, 90 = down (Pygame coordinates).
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def angle_diff(angle1: float, angle2: float) -> float:
    """
    Calculate the shortest angular difference between two angles.
    Returns value in range [-180, 180].
    """
    diff = (angle2 - angle1) % 360
    if diff > 180:
        diff -= 360
    return diff


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by factor t."""
    return a + (b - a) * t


def lerp_angle(a: float, b: float, t: float) -> float:
    """
    Lerp between angles, taking the shortest path.
    """
    diff = angle_diff(a, b)
    return a + diff * t


def rotate_point(point: Point, angle: float, origin: Point = (0, 0)) -> Point:
    """
    Rotate a point around an origin by angle (degrees).
    """
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Translate to origin
    px = point[0] - origin[0]
    py = point[1] - origin[1]

    # Rotate
    rx = px * cos_a - py * sin_a
    ry = px * sin_a + py * cos_a

    # Translate back
    return (rx + origin[0], ry + origin[1])


def point_in_cone(
    origin: Point,
    direction_angle: float,
    target: Point,
    cone_angle: float,
    max_distance: float
) -> bool:
    """
    Check if target point is within a vision cone.

    Args:
        origin: The source point of the cone
        direction_angle: The direction the cone is facing (degrees)
        target: The point to check
        cone_angle: Total cone angle in degrees (e.g., 120 = 60 on each side)
        max_distance: Maximum range of the cone

    Returns:
        True if target is within the cone
    """
    # Check distance first
    dist = distance(origin, target)
    if dist > max_distance or dist == 0:
        return False

    # Check angle
    target_angle = angle_to(origin, target)
    diff = abs(angle_diff(direction_angle, target_angle))

    return diff <= cone_angle / 2


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def vector_from_angle(angle: float, magnitude: float = 1.0) -> Vector:
    """Create a vector from an angle (degrees) and magnitude."""
    rad = math.radians(angle)
    return (math.cos(rad) * magnitude, math.sin(rad) * magnitude)


def add_vectors(a: Vector, b: Vector) -> Vector:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1])


def scale_vector(v: Vector, scalar: float) -> Vector:
    """Scale a vector by a scalar."""
    return (v[0] * scalar, v[1] * scalar)


def dot_product(a: Vector, b: Vector) -> float:
    """Calculate dot product of two vectors."""
    return a[0] * b[0] + a[1] * b[1]
