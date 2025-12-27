"""
Perception System - What the robot actually "sees".

This system simulates realistic sensor limitations:
- Distance degrades perception quality
- Angle to object affects visibility (edge of FOV is worse)
- Occlusion reduces confidence
- Small objects are harder to perceive
- Noise is added to all measurements

The robot ONLY has access to PerceptionResult, never to ground truth.
"""
import math
import random
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROBOT_SENSOR_RANGE, ROBOT_VISION_CONE
from entities.world_object import (
    WorldObject, VisualFeatures, Shape, Texture, Material
)


# =============================================================================
# PERCEPTION RESULT
# =============================================================================

@dataclass
class PerceivedFeatures:
    """
    Features as perceived by the robot (with noise).
    This is what the classifier receives - NOT ground truth!
    """
    # Perceived position (with noise)
    position: Tuple[float, float]
    position_confidence: float  # How sure are we about position?

    # Perceived size (with noise)
    size: float
    size_confidence: float

    # Perceived color (with noise)
    color: Tuple[int, int, int]
    color_confidence: float

    # Perceived shape (may be misidentified)
    shape: Shape
    shape_confidence: float

    # Perceived texture (may be misidentified)
    texture: Texture
    texture_confidence: float

    # Perceived material (inferred, may be wrong)
    apparent_material: Material
    material_confidence: float

    # Other perceived features
    has_text: bool
    has_branding: bool
    is_damaged: bool
    reflectivity: float
    transparency: float


@dataclass
class PerceptionResult:
    """
    Complete perception result for a single object.
    """
    # Reference to the object (for tracking, NOT for reading ground truth!)
    object_id: int

    # Perceived features
    features: PerceivedFeatures

    # Overall perception quality
    overall_confidence: float  # 0.0 = basically guessing, 1.0 = crystal clear

    # Perception conditions
    distance: float
    angle_offset: float  # Degrees from center of vision
    is_occluded: bool
    occlusion_amount: float  # 0.0 = fully visible, 1.0 = fully hidden

    # Raw sensor data (for debugging)
    raw_distance: float
    raw_angle: float


# =============================================================================
# PERCEPTION PARAMETERS
# =============================================================================

@dataclass
class PerceptionParams:
    """Configurable perception parameters."""
    # Range and FOV
    max_range: float = ROBOT_SENSOR_RANGE
    fov_degrees: float = ROBOT_VISION_CONE

    # Distance effects
    distance_noise_base: float = 2.0  # Base position noise in pixels
    distance_noise_scale: float = 0.05  # Additional noise per unit distance
    distance_confidence_falloff: float = 0.0015  # Confidence loss per unit distance (0.55 conf at 300px)

    # Angle effects (edge of FOV)
    angle_confidence_falloff: float = 0.01  # Confidence loss per degree from center

    # Size effects
    min_perceivable_size: float = 1.0  # Objects smaller than this are invisible (very low for cigarette butts)
    size_confidence_scale: float = 0.15  # Larger objects are easier to see (boosted for small object detection)

    # Color perception
    color_noise_base: int = 5  # Base RGB noise
    color_noise_distance_scale: float = 0.1  # Additional noise per unit distance

    # Shape misidentification
    shape_misid_base_chance: float = 0.05  # Base chance to misidentify shape
    shape_misid_distance_scale: float = 0.001  # Additional misid chance per distance

    # Texture perception
    texture_min_distance: float = 80.0  # Can't perceive texture beyond this
    texture_misid_chance: float = 0.1

    # Material inference
    material_base_confidence: float = 0.6  # Materials are always somewhat uncertain

    # Overall perception
    min_confidence: float = 0.1  # Never go below this confidence
    max_confidence: float = 0.95  # Never exceed this confidence


# =============================================================================
# PERCEPTION SYSTEM
# =============================================================================

class PerceptionSystem:
    """
    Simulates realistic robot perception of world objects.

    Key principles:
    1. Distance degrades everything
    2. Angle from center affects quality
    3. Small objects are harder to perceive
    4. All measurements have noise
    5. Some features can be misidentified
    """

    def __init__(self, params: Optional[PerceptionParams] = None):
        self.params = params or PerceptionParams()

    def perceive(
        self,
        obj: WorldObject,
        robot_pos: Tuple[float, float],
        robot_angle: float,  # Degrees, 0 = right, 90 = down
    ) -> Optional[PerceptionResult]:
        """
        Perceive a single object from the robot's perspective.

        Args:
            obj: The world object to perceive
            robot_pos: Robot's (x, y) position
            robot_angle: Robot's heading in degrees

        Returns:
            PerceptionResult if object is visible, None otherwise
        """
        # Skip picked objects
        if obj.is_picked:
            return None

        # Calculate distance and angle
        dx = obj.x - robot_pos[0]
        dy = obj.y - robot_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Check if within range
        if distance > self.params.max_range:
            return None

        # Calculate angle to object
        angle_to_obj = math.degrees(math.atan2(dy, dx))
        angle_diff = self._normalize_angle(angle_to_obj - robot_angle)

        # Check if within FOV
        half_fov = self.params.fov_degrees / 2
        if abs(angle_diff) > half_fov:
            return None

        # Check if object is too small to perceive at this distance
        # Very small penalty (0.003) to allow tiny trash detection at max sensor range
        # At 150px range, a 3px object becomes 2.55px effective - still visible
        effective_size = obj.features.size - (distance * 0.003)
        if effective_size < self.params.min_perceivable_size:
            return None

        # Calculate base confidence factors
        distance_confidence = self._calc_distance_confidence(distance)
        angle_confidence = self._calc_angle_confidence(angle_diff)
        size_confidence = self._calc_size_confidence(obj.features.size, distance)

        # Combined base confidence
        base_confidence = distance_confidence * angle_confidence * size_confidence

        # Perceive features with noise
        perceived_features = self._perceive_features(
            obj.features, distance, angle_diff, base_confidence
        )

        # Calculate overall confidence
        overall_confidence = self._calc_overall_confidence(
            perceived_features, base_confidence
        )

        return PerceptionResult(
            object_id=obj.id,
            features=perceived_features,
            overall_confidence=overall_confidence,
            distance=distance,
            angle_offset=angle_diff,
            is_occluded=False,  # TODO: Implement occlusion
            occlusion_amount=0.0,
            raw_distance=distance,
            raw_angle=angle_to_obj,
        )

    def perceive_all(
        self,
        objects: List[WorldObject],
        robot_pos: Tuple[float, float],
        robot_angle: float,
    ) -> List[PerceptionResult]:
        """
        Perceive all objects and return visible ones sorted by distance.
        """
        results = []
        for obj in objects:
            result = self.perceive(obj, robot_pos, robot_angle)
            if result is not None:
                results.append(result)

        # Sort by distance (closest first)
        results.sort(key=lambda r: r.distance)
        return results

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180]."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _calc_distance_confidence(self, distance: float) -> float:
        """Calculate confidence factor based on distance."""
        falloff = self.params.distance_confidence_falloff * distance
        confidence = max(0.0, 1.0 - falloff)
        return confidence

    def _calc_angle_confidence(self, angle_offset: float) -> float:
        """Calculate confidence factor based on angle from center of vision."""
        falloff = self.params.angle_confidence_falloff * abs(angle_offset)
        confidence = max(0.0, 1.0 - falloff)
        return confidence

    def _calc_size_confidence(self, size: float, distance: float) -> float:
        """Calculate confidence factor based on object size."""
        # Apparent size decreases with distance
        apparent_size = size / (1 + distance * 0.01)
        confidence = min(1.0, apparent_size * self.params.size_confidence_scale)
        # Higher floor (0.5) to prevent tiny objects from tanking overall confidence
        return max(0.5, confidence)

    def _perceive_features(
        self,
        true_features: VisualFeatures,
        distance: float,
        angle_offset: float,
        base_confidence: float
    ) -> PerceivedFeatures:
        """
        Generate perceived features with appropriate noise.
        """
        # Position noise
        pos_noise = (
            self.params.distance_noise_base +
            self.params.distance_noise_scale * distance
        )
        perceived_x = true_features.size + random.gauss(0, pos_noise)  # Placeholder
        perceived_y = true_features.size + random.gauss(0, pos_noise)

        # Size noise
        size_noise = 0.05 + 0.001 * distance
        perceived_size = true_features.size * (1 + random.gauss(0, size_noise))
        perceived_size = max(self.params.min_perceivable_size, perceived_size)

        # Color noise
        color_noise = int(
            self.params.color_noise_base +
            self.params.color_noise_distance_scale * distance
        )
        r, g, b = true_features.color
        perceived_color = (
            max(0, min(255, r + random.randint(-color_noise, color_noise))),
            max(0, min(255, g + random.randint(-color_noise, color_noise))),
            max(0, min(255, b + random.randint(-color_noise, color_noise))),
        )

        # Shape perception (may misidentify)
        shape_misid_chance = (
            self.params.shape_misid_base_chance +
            self.params.shape_misid_distance_scale * distance
        )
        if random.random() < shape_misid_chance:
            perceived_shape = self._misidentify_shape(true_features.shape)
            shape_confidence = base_confidence * 0.5
        else:
            perceived_shape = true_features.shape
            shape_confidence = base_confidence * 0.9

        # Texture perception (limited by distance)
        if distance > self.params.texture_min_distance:
            # Too far to perceive texture
            perceived_texture = Texture.UNKNOWN if hasattr(Texture, 'UNKNOWN') else Texture.MATTE
            texture_confidence = 0.2
        elif random.random() < self.params.texture_misid_chance:
            perceived_texture = self._misidentify_texture(true_features.texture)
            texture_confidence = base_confidence * 0.6
        else:
            perceived_texture = true_features.texture
            texture_confidence = base_confidence * 0.8

        # Material inference (always uncertain)
        perceived_material = true_features.apparent_material
        if random.random() < 0.15:  # 15% chance to get material wrong
            perceived_material = self._misidentify_material(perceived_material)
        material_confidence = base_confidence * self.params.material_base_confidence

        # Other features (may not perceive at distance)
        has_text = true_features.has_text and distance < 60 and random.random() < 0.8
        has_branding = true_features.has_branding and distance < 80 and random.random() < 0.7
        is_damaged = true_features.is_damaged and random.random() < (0.9 - distance * 0.005)

        # Reflectivity and transparency (with noise)
        reflectivity = true_features.reflectivity + random.gauss(0, 0.1)
        reflectivity = max(0.0, min(1.0, reflectivity))
        transparency = true_features.transparency + random.gauss(0, 0.1)
        transparency = max(0.0, min(1.0, transparency))

        return PerceivedFeatures(
            position=(perceived_x, perceived_y),  # Note: actual position set elsewhere
            position_confidence=base_confidence * 0.95,
            size=perceived_size,
            size_confidence=base_confidence * 0.85,
            color=perceived_color,
            color_confidence=base_confidence * 0.8,
            shape=perceived_shape,
            shape_confidence=shape_confidence,
            texture=perceived_texture,
            texture_confidence=texture_confidence,
            apparent_material=perceived_material,
            material_confidence=material_confidence,
            has_text=has_text,
            has_branding=has_branding,
            is_damaged=is_damaged,
            reflectivity=reflectivity,
            transparency=transparency,
        )

    def _misidentify_shape(self, true_shape: Shape) -> Shape:
        """Return a plausible wrong shape."""
        # Similar shapes get confused
        confusion_map = {
            Shape.ROUND: [Shape.IRREGULAR, Shape.CRUMPLED],
            Shape.RECTANGULAR: [Shape.FLAT, Shape.CYLINDRICAL],
            Shape.CYLINDRICAL: [Shape.RECTANGULAR, Shape.ELONGATED],
            Shape.IRREGULAR: [Shape.CRUMPLED, Shape.ROUND],
            Shape.CRUMPLED: [Shape.IRREGULAR, Shape.ROUND],
            Shape.FLAT: [Shape.RECTANGULAR, Shape.IRREGULAR],
            Shape.ELONGATED: [Shape.CYLINDRICAL, Shape.FLAT],
        }
        options = confusion_map.get(true_shape, list(Shape))
        return random.choice(options)

    def _misidentify_texture(self, true_texture: Texture) -> Texture:
        """Return a plausible wrong texture."""
        confusion_map = {
            Texture.SMOOTH: [Texture.SHINY, Texture.MATTE],
            Texture.ROUGH: [Texture.MATTE, Texture.FIBROUS],
            Texture.SHINY: [Texture.SMOOTH, Texture.TRANSLUCENT],
            Texture.MATTE: [Texture.ROUGH, Texture.SMOOTH],
            Texture.CRINKLED: [Texture.ROUGH, Texture.FIBROUS],
            Texture.TRANSLUCENT: [Texture.SHINY, Texture.SMOOTH],
            Texture.FIBROUS: [Texture.ROUGH, Texture.CRINKLED],
        }
        options = confusion_map.get(true_texture, list(Texture))
        return random.choice(options)

    def _misidentify_material(self, true_material: Material) -> Material:
        """Return a plausible wrong material."""
        confusion_map = {
            Material.PLASTIC: [Material.GLASS, Material.UNKNOWN],
            Material.PAPER: [Material.FABRIC, Material.ORGANIC],
            Material.METAL: [Material.PLASTIC, Material.GLASS],
            Material.GLASS: [Material.PLASTIC, Material.METAL],
            Material.ORGANIC: [Material.PAPER, Material.FABRIC],
            Material.FABRIC: [Material.PAPER, Material.ORGANIC],
            Material.UNKNOWN: list(Material),
        }
        options = confusion_map.get(true_material, list(Material))
        return random.choice(options)

    def _calc_overall_confidence(
        self,
        features: PerceivedFeatures,
        base_confidence: float
    ) -> float:
        """Calculate overall perception confidence."""
        # Weighted average of feature confidences
        weights = {
            'position': 0.2,
            'size': 0.15,
            'color': 0.2,
            'shape': 0.25,
            'texture': 0.1,
            'material': 0.1,
        }

        confidence = (
            weights['position'] * features.position_confidence +
            weights['size'] * features.size_confidence +
            weights['color'] * features.color_confidence +
            weights['shape'] * features.shape_confidence +
            weights['texture'] * features.texture_confidence +
            weights['material'] * features.material_confidence
        )

        # Clamp to valid range
        confidence = max(self.params.min_confidence, min(self.params.max_confidence, confidence))

        return confidence


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_perception_system(difficulty: int = 2) -> PerceptionSystem:
    """Create perception system with difficulty-adjusted parameters."""
    params = PerceptionParams()

    if difficulty == 1:  # Easy - better perception
        params.distance_confidence_falloff *= 0.5
        params.shape_misid_base_chance *= 0.5
        params.color_noise_base = 3

    elif difficulty == 3:  # Hard
        params.distance_confidence_falloff *= 1.3
        params.shape_misid_base_chance *= 1.5
        params.color_noise_base = 8

    elif difficulty == 4:  # Very Hard
        params.distance_confidence_falloff *= 1.6
        params.shape_misid_base_chance *= 2.0
        params.color_noise_base = 12

    elif difficulty == 5:  # Chaos
        params.distance_confidence_falloff *= 2.0
        params.shape_misid_base_chance *= 3.0
        params.color_noise_base = 15
        params.texture_min_distance = 50.0

    return PerceptionSystem(params)
