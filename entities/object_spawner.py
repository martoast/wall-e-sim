"""
ObjectSpawner - Generates diverse world objects with realistic distributions.

Handles:
- Weighted random selection based on real-world litter prevalence
- Spatial distribution avoiding obstacles and nests
- Difficulty-based trash/non-trash ratios
- Batch spawning for simulation initialization
"""
import random
import math
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_MARGIN
from entities.world_object import (
    WorldObject, ObjectTemplate, OBJECT_TEMPLATES,
    create_object_from_template
)


@dataclass
class SpawnConfig:
    """Configuration for object spawning."""
    # Trash vs non-trash balance
    trash_ratio: float = 0.7  # 70% trash by default

    # Category weights (multiplied with template weights)
    plastic_weight: float = 1.0
    paper_weight: float = 1.0
    metal_weight: float = 1.0
    glass_weight: float = 0.5  # Less common
    organic_weight: float = 1.0
    natural_weight: float = 1.0
    valuable_weight: float = 0.1  # Rare
    ambiguous_weight: float = 0.3

    # Size modifiers
    size_variance: float = 0.2  # +/- 20% size variation

    # Spatial settings
    min_spacing: float = 25.0  # Minimum distance between objects
    spawn_margin: float = 50.0  # Distance from screen edges

    # Difficulty scaling
    ambiguity_boost: float = 0.0  # Add to ambiguity (higher = harder)


class ObjectSpawner:
    """
    Spawns diverse world objects for the simulation.

    Uses weighted random selection based on:
    1. Template weights (realistic litter prevalence)
    2. Category weights from SpawnConfig
    3. Trash/non-trash ratio
    """

    def __init__(self, config: Optional[SpawnConfig] = None):
        self.config = config or SpawnConfig()

        # Separate trash and non-trash templates
        self.trash_templates: List[ObjectTemplate] = []
        self.non_trash_templates: List[ObjectTemplate] = []

        for template in OBJECT_TEMPLATES:
            if template.is_trash:
                self.trash_templates.append(template)
            else:
                self.non_trash_templates.append(template)

        # Pre-calculate weights
        self._update_weights()

    def _update_weights(self):
        """Calculate spawn weights for all templates."""
        self.trash_weights: List[float] = []
        self.non_trash_weights: List[float] = []

        for template in self.trash_templates:
            weight = template.weight * self._get_category_weight(template.super_category)
            self.trash_weights.append(weight)

        for template in self.non_trash_templates:
            weight = template.weight * self._get_category_weight(template.super_category)
            self.non_trash_weights.append(weight)

    def _get_category_weight(self, super_category: str) -> float:
        """Get weight multiplier for a category."""
        cfg = self.config

        # Trash categories
        if super_category in ['plastic_bottle', 'plastic_bag', 'plastic_straw',
                              'disposable_plastic_cup', 'plastic_bottle_cap',
                              'other_plastic_wrapper', 'other_plastic',
                              'plastic_container', 'plastic_film', 'plastic_lid']:
            return cfg.plastic_weight

        if super_category in ['paper', 'paper_cup', 'paper_bag', 'cardboard', 'carton']:
            return cfg.paper_weight

        if super_category in ['drink_can', 'food_can', 'metal_bottle_cap',
                              'aerosol', 'aluminium_foil', 'metal_container']:
            return cfg.metal_weight

        if super_category in ['glass_bottle', 'glass_jar', 'broken_glass']:
            return cfg.glass_weight

        if super_category in ['cigarette', 'food_waste', 'styrofoam']:
            return cfg.organic_weight

        # Non-trash categories
        if super_category == 'natural':
            return cfg.natural_weight

        if super_category == 'valuable':
            return cfg.valuable_weight

        if super_category == 'ambiguous':
            return cfg.ambiguous_weight

        return 1.0  # Default weight

    def set_difficulty(self, level: int):
        """
        Adjust spawning parameters based on difficulty level (1-5).

        Higher difficulty means:
        - More ambiguous objects
        - More non-trash that looks like trash
        - Lower trash ratio (more false positive traps)
        """
        if level == 1:  # Easy
            self.config.trash_ratio = 0.9
            self.config.valuable_weight = 0.0
            self.config.ambiguous_weight = 0.0
            self.config.natural_weight = 0.3
            self.config.ambiguity_boost = -0.1

        elif level == 2:  # Normal
            self.config.trash_ratio = 0.75
            self.config.valuable_weight = 0.05
            self.config.ambiguous_weight = 0.2
            self.config.natural_weight = 0.8
            self.config.ambiguity_boost = 0.0

        elif level == 3:  # Hard
            self.config.trash_ratio = 0.65
            self.config.valuable_weight = 0.15
            self.config.ambiguous_weight = 0.4
            self.config.natural_weight = 1.0
            self.config.ambiguity_boost = 0.1

        elif level == 4:  # Very Hard
            self.config.trash_ratio = 0.55
            self.config.valuable_weight = 0.25
            self.config.ambiguous_weight = 0.6
            self.config.natural_weight = 1.2
            self.config.ambiguity_boost = 0.2

        elif level == 5:  # Chaos
            self.config.trash_ratio = 0.45
            self.config.valuable_weight = 0.4
            self.config.ambiguous_weight = 0.8
            self.config.natural_weight = 1.5
            self.config.ambiguity_boost = 0.3

        self._update_weights()

    def spawn_single(
        self,
        position: Tuple[float, float],
        force_trash: Optional[bool] = None
    ) -> WorldObject:
        """
        Spawn a single object at the given position.

        Args:
            position: (x, y) coordinates
            force_trash: If set, forces trash (True) or non-trash (False)

        Returns:
            A new WorldObject instance
        """
        # Decide trash vs non-trash
        if force_trash is not None:
            is_trash = force_trash
        else:
            is_trash = random.random() < self.config.trash_ratio

        # Select template
        if is_trash:
            template = random.choices(
                self.trash_templates,
                weights=self.trash_weights,
                k=1
            )[0]
        else:
            template = random.choices(
                self.non_trash_templates,
                weights=self.non_trash_weights,
                k=1
            )[0]

        # Create object
        obj = create_object_from_template(template, position)

        # Apply difficulty ambiguity boost
        if self.config.ambiguity_boost != 0:
            gt = obj._ground_truth
            gt.ambiguity = max(0.0, min(1.0, gt.ambiguity + self.config.ambiguity_boost))

        return obj

    def spawn_batch(
        self,
        count: int,
        avoid_positions: Optional[List[Tuple[float, float]]] = None,
        avoid_radii: Optional[List[float]] = None,
        avoid_rects: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[WorldObject]:
        """
        Spawn multiple objects with proper spacing.

        Args:
            count: Number of objects to spawn
            avoid_positions: List of (x, y) positions to avoid
            avoid_radii: Corresponding radii for each position
            avoid_rects: List of (x, y, width, height) rectangles to avoid

        Returns:
            List of spawned WorldObjects
        """
        objects: List[WorldObject] = []
        placed_positions: List[Tuple[float, float]] = []
        placed_sizes: List[float] = []

        avoid_positions = avoid_positions or []
        avoid_radii = avoid_radii or []
        avoid_rects = avoid_rects or []

        # Pad radii list if needed
        while len(avoid_radii) < len(avoid_positions):
            avoid_radii.append(50.0)

        attempts = 0
        max_attempts = count * 50  # Prevent infinite loop

        while len(objects) < count and attempts < max_attempts:
            attempts += 1

            # Random position within spawn area
            margin = self.config.spawn_margin
            x = random.uniform(margin, SCREEN_WIDTH - margin)
            y = random.uniform(margin, SCREEN_HEIGHT - margin)

            # Check against avoid positions
            valid = True
            for (ax, ay), radius in zip(avoid_positions, avoid_radii):
                dist = math.sqrt((x - ax) ** 2 + (y - ay) ** 2)
                if dist < radius + self.config.min_spacing:
                    valid = False
                    break

            if not valid:
                continue

            # Check against avoid rectangles
            for (rx, ry, rw, rh) in avoid_rects:
                # Add buffer around rectangle
                buffer = self.config.min_spacing
                if (rx - buffer < x < rx + rw + buffer and
                    ry - buffer < y < ry + rh + buffer):
                    valid = False
                    break

            if not valid:
                continue

            # Check against already placed objects
            for (px, py), psize in zip(placed_positions, placed_sizes):
                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                if dist < psize + self.config.min_spacing:
                    valid = False
                    break

            if not valid:
                continue

            # Spawn object
            obj = self.spawn_single((x, y))
            objects.append(obj)
            placed_positions.append((x, y))
            placed_sizes.append(obj.features.size)

        return objects

    def spawn_cluster(
        self,
        center: Tuple[float, float],
        count: int,
        radius: float = 100.0,
        category_filter: Optional[str] = None
    ) -> List[WorldObject]:
        """
        Spawn a cluster of objects around a center point.

        Useful for simulating trash accumulation areas.
        """
        objects: List[WorldObject] = []
        placed_positions: List[Tuple[float, float]] = []

        for _ in range(count * 10):  # Limited attempts
            if len(objects) >= count:
                break

            # Random position in cluster
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, radius)
            x = center[0] + dist * math.cos(angle)
            y = center[1] + dist * math.sin(angle)

            # Clamp to screen
            margin = self.config.spawn_margin
            x = max(margin, min(SCREEN_WIDTH - margin, x))
            y = max(margin, min(SCREEN_HEIGHT - margin, y))

            # Check spacing
            valid = True
            for px, py in placed_positions:
                if math.sqrt((x - px)**2 + (y - py)**2) < self.config.min_spacing:
                    valid = False
                    break

            if valid:
                obj = self.spawn_single((x, y), force_trash=True)
                objects.append(obj)
                placed_positions.append((x, y))

        return objects

    def spawn_at_position(
        self,
        position: Tuple[float, float],
        force_trash: Optional[bool] = None,
        template_name: Optional[str] = None
    ) -> Optional[WorldObject]:
        """
        Spawn a specific object at an exact position.

        Args:
            position: (x, y) coordinates
            force_trash: If set, forces trash (True) or non-trash (False)
            template_name: Specific template to use (by name)

        Returns:
            A new WorldObject instance, or None if template not found
        """
        if template_name:
            # Find specific template
            template = None
            for t in OBJECT_TEMPLATES:
                if t.name == template_name:
                    template = t
                    break

            if template is None:
                return None

            return create_object_from_template(template, position)
        else:
            return self.spawn_single(position, force_trash)

    def get_spawn_statistics(self) -> dict:
        """Get statistics about spawn configuration."""
        total_trash_weight = sum(self.trash_weights)
        total_non_trash_weight = sum(self.non_trash_weights)

        trash_probs = {}
        for template, weight in zip(self.trash_templates, self.trash_weights):
            prob = (weight / total_trash_weight) * self.config.trash_ratio
            trash_probs[template.name] = prob

        non_trash_probs = {}
        for template, weight in zip(self.non_trash_templates, self.non_trash_weights):
            prob = (weight / total_non_trash_weight) * (1 - self.config.trash_ratio)
            non_trash_probs[template.name] = prob

        return {
            'trash_ratio': self.config.trash_ratio,
            'trash_probabilities': trash_probs,
            'non_trash_probabilities': non_trash_probs,
            'total_trash_templates': len(self.trash_templates),
            'total_non_trash_templates': len(self.non_trash_templates),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_spawner(difficulty: int = 2) -> ObjectSpawner:
    """Create a spawner with default configuration for given difficulty."""
    spawner = ObjectSpawner()
    spawner.set_difficulty(difficulty)
    return spawner


def spawn_initial_objects(
    count: int,
    difficulty: int = 2,
    nest_rect: Optional[Tuple[float, float, float, float]] = None,
    obstacle_positions: Optional[List[Tuple[float, float]]] = None,
    obstacle_radii: Optional[List[float]] = None,
    robot_positions: Optional[List[Tuple[float, float]]] = None,
) -> List[WorldObject]:
    """
    Convenience function to spawn initial world objects.

    Args:
        count: Number of objects to spawn
        difficulty: Difficulty level (1-5)
        nest_rect: (x, y, width, height) of nest to avoid
        obstacle_positions: List of obstacle centers to avoid
        obstacle_radii: List of obstacle radii
        robot_positions: List of robot positions to avoid

    Returns:
        List of spawned WorldObjects
    """
    spawner = create_default_spawner(difficulty)

    # Build avoidance lists
    avoid_positions: List[Tuple[float, float]] = []
    avoid_radii: List[float] = []
    avoid_rects: List[Tuple[float, float, float, float]] = []

    if obstacle_positions:
        avoid_positions.extend(obstacle_positions)
        if obstacle_radii:
            avoid_radii.extend(obstacle_radii)
        else:
            avoid_radii.extend([50.0] * len(obstacle_positions))

    if robot_positions:
        for pos in robot_positions:
            avoid_positions.append(pos)
            avoid_radii.append(60.0)  # Buffer around robots

    if nest_rect:
        avoid_rects.append(nest_rect)

    return spawner.spawn_batch(
        count,
        avoid_positions=avoid_positions,
        avoid_radii=avoid_radii,
        avoid_rects=avoid_rects
    )
