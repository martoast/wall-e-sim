"""
WorldObject - Feature-based objects for perception-driven classification.

This replaces the old Trash class. Objects are defined by their PERCEIVABLE
features (what a robot camera could see), not by labels. The robot must
CLASSIFY objects based on features to decide if they're trash.

Ground truth (is_actually_trash) is ONLY used for scoring, never for robot decisions.

Uses TACO (Trash Annotations in Context) dataset taxonomy:
https://github.com/pedropro/TACO
"""
import pygame
import random
import math
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCREEN_WIDTH, SCREEN_HEIGHT


# =============================================================================
# VISUAL FEATURE ENUMS
# =============================================================================

class Shape(Enum):
    """Perceivable shape categories."""
    ROUND = "round"
    RECTANGULAR = "rectangular"
    CYLINDRICAL = "cylindrical"
    IRREGULAR = "irregular"
    CRUMPLED = "crumpled"
    FLAT = "flat"
    ELONGATED = "elongated"


class Texture(Enum):
    """Perceivable texture categories."""
    SMOOTH = "smooth"
    ROUGH = "rough"
    SHINY = "shiny"
    MATTE = "matte"
    CRINKLED = "crinkled"
    TRANSLUCENT = "translucent"
    FIBROUS = "fibrous"


class Material(Enum):
    """Inferred material (what it looks like, not guaranteed)."""
    PLASTIC = "plastic"
    PAPER = "paper"
    METAL = "metal"
    GLASS = "glass"
    ORGANIC = "organic"
    FABRIC = "fabric"
    UNKNOWN = "unknown"


# =============================================================================
# TACO CATEGORY TAXONOMY
# =============================================================================

# From TACO dataset - 28 super-categories mapped to specific types
TACO_CATEGORIES: Dict[str, List[str]] = {
    # PLASTIC
    "plastic_bottle": ["clear_plastic_bottle", "other_plastic_bottle"],
    "plastic_bottle_cap": ["plastic_bottle_cap"],
    "plastic_bag": ["single_use_carrier_bag", "polypropylene_bag", "other_plastic_bag"],
    "plastic_container": ["plastic_container", "plastic_tupperware"],
    "plastic_film": ["plastic_film", "six_pack_rings"],
    "plastic_straw": ["plastic_straw"],
    "disposable_plastic_cup": ["disposable_plastic_cup", "plastic_cup_lid"],
    "plastic_lid": ["plastic_lid", "plastic_cup_lid"],
    "other_plastic_wrapper": ["crisp_packet", "sweet_wrapper", "other_plastic_wrapper"],
    "other_plastic": ["plastic_utensil", "other_plastic"],

    # PAPER / CARDBOARD
    "paper": ["normal_paper", "magazine_paper", "tissues", "toilet_tube"],
    "paper_cup": ["paper_cup"],
    "paper_bag": ["paper_bag"],
    "cardboard": ["cardboard", "corrugated_carton"],
    "carton": ["drink_carton", "egg_carton", "meal_carton", "pizza_box"],

    # METAL
    "drink_can": ["drink_can"],
    "food_can": ["food_can"],
    "metal_bottle_cap": ["metal_bottle_cap"],
    "aerosol": ["aerosol"],
    "aluminium_foil": ["aluminium_foil", "foil_food_wrapping"],
    "metal_container": ["metal_container", "scrap_metal"],

    # GLASS
    "glass_bottle": ["clear_glass_bottle", "other_glass_bottle"],
    "glass_jar": ["glass_jar"],
    "broken_glass": ["broken_glass"],

    # OTHER
    "cigarette": ["cigarette"],
    "food_waste": ["food_waste", "food_scraps"],
    "styrofoam": ["styrofoam_piece", "foam_cup", "foam_container"],
    "battery": ["battery"],
    "rope_and_strings": ["rope", "string"],
    "shoe": ["shoe"],
    "squeezable_tube": ["squeezable_tube"],
    "unlabeled_litter": ["unlabeled_litter"],
}

# All trash categories (anything that should be picked up)
ALL_TRASH_CATEGORIES = list(TACO_CATEGORIES.keys())

# Non-trash categories (should NOT be picked up)
NON_TRASH_CATEGORIES: Dict[str, List[str]] = {
    # Natural objects
    "natural": ["leaf", "rock", "pinecone", "stick", "flower", "acorn", "bark", "moss"],

    # Valuable items (lost property - DO NOT pick up!)
    "valuable": ["phone", "wallet", "keys", "toy", "jewelry", "watch", "sunglasses", "headphones"],

    # Ambiguous (could be trash, could be not)
    "ambiguous": ["worn_paper", "fabric_scrap", "unknown_object", "cardboard_box"],
}

# TACO-10 simplified categories for experiments
TACO_10 = [
    "cigarette",
    "drink_can",
    "plastic_bottle",
    "plastic_bag",
    "bottle_cap",  # plastic + metal combined
    "paper_cup",
    "carton",
    "styrofoam",
    "glass_bottle",
    "other_litter",
]


# =============================================================================
# VISUAL FEATURE DEFINITIONS
# =============================================================================

@dataclass
class VisualFeatures:
    """
    Features that a robot camera COULD perceive.
    These are what classification decisions are based on.
    """
    shape: Shape
    size: float  # radius in pixels
    color: Tuple[int, int, int]  # RGB
    texture: Texture
    apparent_material: Material

    # Additional visual cues
    has_text: bool = False  # Visible text/labels
    has_branding: bool = False  # Corporate logos
    is_damaged: bool = False  # Crushed, torn, broken
    reflectivity: float = 0.0  # 0.0 = matte, 1.0 = mirror-like
    transparency: float = 0.0  # 0.0 = opaque, 1.0 = fully transparent


@dataclass
class GroundTruth:
    """
    Ground truth information - ONLY for scoring/metrics.
    The robot should NEVER access this during operation!
    """
    is_actually_trash: bool
    category: str  # Specific TACO category or non-trash category
    super_category: str  # Parent category

    # How hard is this to classify correctly?
    ambiguity: float = 0.0  # 0.0 = obvious, 1.0 = very ambiguous

    # For metrics
    description: str = ""  # Human-readable description


# =============================================================================
# OBJECT TEMPLATES
# =============================================================================

@dataclass
class ObjectTemplate:
    """Template for spawning consistent object types."""
    name: str
    category: str
    super_category: str
    is_trash: bool

    # Visual feature ranges
    shape: Shape
    size_range: Tuple[float, float]
    color_base: Tuple[int, int, int]
    color_variance: int
    texture: Texture
    material: Material

    # Optional features
    has_text_chance: float = 0.0
    has_branding_chance: float = 0.0
    damage_chance: float = 0.0
    reflectivity_range: Tuple[float, float] = (0.0, 0.0)
    transparency_range: Tuple[float, float] = (0.0, 0.0)

    # Classification difficulty
    ambiguity: float = 0.0
    weight: float = 1.0  # Spawn weight


# Define common object templates
OBJECT_TEMPLATES: List[ObjectTemplate] = [
    # =========================================================================
    # TRASH - Plastic
    # =========================================================================
    ObjectTemplate(
        name="Clear Plastic Bottle",
        category="clear_plastic_bottle",
        super_category="plastic_bottle",
        is_trash=True,
        shape=Shape.CYLINDRICAL,
        size_range=(12, 25),
        color_base=(180, 220, 240),  # Light blue tint
        color_variance=20,
        texture=Texture.SMOOTH,
        material=Material.PLASTIC,
        has_branding_chance=0.7,
        damage_chance=0.3,
        reflectivity_range=(0.3, 0.6),
        transparency_range=(0.4, 0.8),
        ambiguity=0.1,
        weight=3.0,
    ),
    ObjectTemplate(
        name="Crushed Plastic Bottle",
        category="other_plastic_bottle",
        super_category="plastic_bottle",
        is_trash=True,
        shape=Shape.CRUMPLED,
        size_range=(10, 20),
        color_base=(160, 200, 220),
        color_variance=30,
        texture=Texture.CRINKLED,
        material=Material.PLASTIC,
        has_branding_chance=0.3,
        damage_chance=1.0,
        reflectivity_range=(0.1, 0.3),
        transparency_range=(0.2, 0.5),
        ambiguity=0.2,
        weight=2.0,
    ),
    ObjectTemplate(
        name="Plastic Bag",
        category="single_use_carrier_bag",
        super_category="plastic_bag",
        is_trash=True,
        shape=Shape.IRREGULAR,
        size_range=(15, 35),
        color_base=(220, 220, 220),  # White-ish
        color_variance=30,
        texture=Texture.CRINKLED,
        material=Material.PLASTIC,
        has_branding_chance=0.4,
        damage_chance=0.5,
        reflectivity_range=(0.1, 0.3),
        transparency_range=(0.1, 0.4),
        ambiguity=0.15,
        weight=2.5,
    ),
    ObjectTemplate(
        name="Plastic Wrapper",
        category="crisp_packet",
        super_category="other_plastic_wrapper",
        is_trash=True,
        shape=Shape.CRUMPLED,
        size_range=(8, 18),
        color_base=(200, 50, 50),  # Bright colors
        color_variance=100,  # High variance - wrappers are colorful
        texture=Texture.SHINY,
        material=Material.PLASTIC,
        has_branding_chance=0.9,
        damage_chance=0.6,
        reflectivity_range=(0.5, 0.8),
        transparency_range=(0.0, 0.1),
        ambiguity=0.1,
        weight=4.0,  # Very common
    ),
    ObjectTemplate(
        name="Plastic Straw",
        category="plastic_straw",
        super_category="plastic_straw",
        is_trash=True,
        shape=Shape.ELONGATED,
        size_range=(5, 10),
        color_base=(255, 255, 255),
        color_variance=50,
        texture=Texture.SMOOTH,
        material=Material.PLASTIC,
        reflectivity_range=(0.2, 0.4),
        ambiguity=0.2,  # Small, easy to miss
        weight=1.5,
    ),
    ObjectTemplate(
        name="Plastic Cup",
        category="disposable_plastic_cup",
        super_category="disposable_plastic_cup",
        is_trash=True,
        shape=Shape.CYLINDRICAL,
        size_range=(10, 20),
        color_base=(240, 240, 240),
        color_variance=20,
        texture=Texture.SMOOTH,
        material=Material.PLASTIC,
        has_branding_chance=0.5,
        damage_chance=0.4,
        reflectivity_range=(0.2, 0.4),
        transparency_range=(0.3, 0.6),
        ambiguity=0.1,
        weight=2.0,
    ),
    ObjectTemplate(
        name="Bottle Cap",
        category="plastic_bottle_cap",
        super_category="plastic_bottle_cap",
        is_trash=True,
        shape=Shape.ROUND,
        size_range=(5, 10),
        color_base=(50, 100, 200),  # Often blue, red, white
        color_variance=100,
        texture=Texture.SMOOTH,
        material=Material.PLASTIC,
        reflectivity_range=(0.3, 0.5),
        ambiguity=0.25,  # Small, could be mistaken
        weight=2.5,
    ),

    # =========================================================================
    # TRASH - Metal
    # =========================================================================
    ObjectTemplate(
        name="Aluminum Can",
        category="drink_can",
        super_category="drink_can",
        is_trash=True,
        shape=Shape.CYLINDRICAL,
        size_range=(12, 18),
        color_base=(180, 180, 190),  # Silver
        color_variance=40,
        texture=Texture.SHINY,
        material=Material.METAL,
        has_branding_chance=0.95,
        damage_chance=0.4,
        reflectivity_range=(0.6, 0.9),
        ambiguity=0.05,  # Very recognizable
        weight=3.5,
    ),
    ObjectTemplate(
        name="Crushed Can",
        category="drink_can",
        super_category="drink_can",
        is_trash=True,
        shape=Shape.CRUMPLED,
        size_range=(8, 15),
        color_base=(170, 170, 180),
        color_variance=40,
        texture=Texture.SHINY,
        material=Material.METAL,
        has_branding_chance=0.6,
        damage_chance=1.0,
        reflectivity_range=(0.4, 0.7),
        ambiguity=0.15,
        weight=2.0,
    ),
    ObjectTemplate(
        name="Aluminum Foil",
        category="aluminium_foil",
        super_category="aluminium_foil",
        is_trash=True,
        shape=Shape.CRUMPLED,
        size_range=(6, 15),
        color_base=(200, 200, 210),
        color_variance=20,
        texture=Texture.CRINKLED,
        material=Material.METAL,
        damage_chance=1.0,
        reflectivity_range=(0.5, 0.8),
        ambiguity=0.2,
        weight=1.5,
    ),
    ObjectTemplate(
        name="Metal Bottle Cap",
        category="metal_bottle_cap",
        super_category="metal_bottle_cap",
        is_trash=True,
        shape=Shape.ROUND,
        size_range=(5, 8),
        color_base=(180, 160, 50),  # Gold/bronze color
        color_variance=50,
        texture=Texture.SHINY,
        material=Material.METAL,
        reflectivity_range=(0.5, 0.8),
        ambiguity=0.3,  # Small, could be coin
        weight=1.0,
    ),

    # =========================================================================
    # TRASH - Paper/Cardboard
    # =========================================================================
    ObjectTemplate(
        name="Paper Cup",
        category="paper_cup",
        super_category="paper_cup",
        is_trash=True,
        shape=Shape.CYLINDRICAL,
        size_range=(10, 18),
        color_base=(240, 230, 210),  # Off-white
        color_variance=20,
        texture=Texture.MATTE,
        material=Material.PAPER,
        has_branding_chance=0.8,
        damage_chance=0.5,
        ambiguity=0.1,
        weight=2.5,
    ),
    ObjectTemplate(
        name="Crumpled Paper",
        category="normal_paper",
        super_category="paper",
        is_trash=True,
        shape=Shape.CRUMPLED,
        size_range=(8, 20),
        color_base=(250, 250, 245),
        color_variance=15,
        texture=Texture.MATTE,
        material=Material.PAPER,
        has_text_chance=0.6,
        damage_chance=1.0,
        ambiguity=0.25,  # Could be important document
        weight=2.0,
    ),
    ObjectTemplate(
        name="Tissue",
        category="tissues",
        super_category="paper",
        is_trash=True,
        shape=Shape.CRUMPLED,
        size_range=(5, 12),
        color_base=(255, 255, 255),
        color_variance=10,
        texture=Texture.FIBROUS,
        material=Material.PAPER,
        damage_chance=1.0,
        ambiguity=0.1,
        weight=2.0,
    ),
    ObjectTemplate(
        name="Cardboard Piece",
        category="cardboard",
        super_category="cardboard",
        is_trash=True,
        shape=Shape.FLAT,
        size_range=(15, 35),
        color_base=(180, 140, 100),  # Brown
        color_variance=30,
        texture=Texture.ROUGH,
        material=Material.PAPER,
        damage_chance=0.7,
        ambiguity=0.3,  # Could be part of something
        weight=1.5,
    ),
    ObjectTemplate(
        name="Food Carton",
        category="meal_carton",
        super_category="carton",
        is_trash=True,
        shape=Shape.RECTANGULAR,
        size_range=(12, 25),
        color_base=(200, 180, 150),
        color_variance=40,
        texture=Texture.MATTE,
        material=Material.PAPER,
        has_branding_chance=0.7,
        damage_chance=0.6,
        ambiguity=0.15,
        weight=1.5,
    ),

    # =========================================================================
    # TRASH - Glass
    # =========================================================================
    ObjectTemplate(
        name="Glass Bottle",
        category="clear_glass_bottle",
        super_category="glass_bottle",
        is_trash=True,
        shape=Shape.CYLINDRICAL,
        size_range=(15, 30),
        color_base=(200, 230, 200),  # Slight green tint
        color_variance=30,
        texture=Texture.SMOOTH,
        material=Material.GLASS,
        has_branding_chance=0.6,
        reflectivity_range=(0.4, 0.7),
        transparency_range=(0.5, 0.9),
        ambiguity=0.1,
        weight=1.5,
    ),
    ObjectTemplate(
        name="Broken Glass",
        category="broken_glass",
        super_category="broken_glass",
        is_trash=True,
        shape=Shape.IRREGULAR,
        size_range=(5, 15),
        color_base=(210, 240, 210),
        color_variance=30,
        texture=Texture.SMOOTH,
        material=Material.GLASS,
        damage_chance=1.0,
        reflectivity_range=(0.5, 0.8),
        transparency_range=(0.3, 0.7),
        ambiguity=0.2,  # Dangerous!
        weight=0.5,  # Less common
    ),

    # =========================================================================
    # TRASH - Other
    # =========================================================================
    ObjectTemplate(
        name="Cigarette Butt",
        category="cigarette",
        super_category="cigarette",
        is_trash=True,
        shape=Shape.ELONGATED,
        size_range=(5, 8),  # Increased for reliable detection at max sensor range
        color_base=(200, 180, 150),  # Tan/orange
        color_variance=30,
        texture=Texture.FIBROUS,
        material=Material.ORGANIC,
        damage_chance=1.0,
        ambiguity=0.15,  # Small but recognizable
        weight=5.0,  # Very common - #1 litter item
    ),
    ObjectTemplate(
        name="Styrofoam Piece",
        category="styrofoam_piece",
        super_category="styrofoam",
        is_trash=True,
        shape=Shape.IRREGULAR,
        size_range=(8, 25),
        color_base=(255, 255, 255),
        color_variance=10,
        texture=Texture.ROUGH,
        material=Material.PLASTIC,  # Technically plastic
        damage_chance=0.8,
        ambiguity=0.2,
        weight=1.5,
    ),
    ObjectTemplate(
        name="Food Waste",
        category="food_waste",
        super_category="food_waste",
        is_trash=True,
        shape=Shape.IRREGULAR,
        size_range=(5, 20),
        color_base=(100, 80, 50),  # Brown organic
        color_variance=50,
        texture=Texture.ROUGH,
        material=Material.ORGANIC,
        damage_chance=1.0,
        ambiguity=0.4,  # Could be natural
        weight=1.0,
    ),

    # =========================================================================
    # NON-TRASH - Natural
    # =========================================================================
    ObjectTemplate(
        name="Leaf",
        category="leaf",
        super_category="natural",
        is_trash=False,
        shape=Shape.FLAT,
        size_range=(8, 20),
        color_base=(80, 120, 50),  # Green to brown
        color_variance=60,
        texture=Texture.MATTE,
        material=Material.ORGANIC,
        damage_chance=0.5,
        ambiguity=0.2,  # Could look like paper
        weight=3.0,
    ),
    ObjectTemplate(
        name="Rock",
        category="rock",
        super_category="natural",
        is_trash=False,
        shape=Shape.IRREGULAR,
        size_range=(10, 30),
        color_base=(120, 110, 100),
        color_variance=30,
        texture=Texture.ROUGH,
        material=Material.UNKNOWN,
        ambiguity=0.05,  # Very obviously not trash
        weight=2.0,
    ),
    ObjectTemplate(
        name="Pinecone",
        category="pinecone",
        super_category="natural",
        is_trash=False,
        shape=Shape.IRREGULAR,
        size_range=(10, 20),
        color_base=(100, 70, 40),
        color_variance=20,
        texture=Texture.ROUGH,
        material=Material.ORGANIC,
        ambiguity=0.1,
        weight=1.0,
    ),
    ObjectTemplate(
        name="Stick",
        category="stick",
        super_category="natural",
        is_trash=False,
        shape=Shape.ELONGATED,
        size_range=(8, 25),
        color_base=(90, 60, 30),
        color_variance=30,
        texture=Texture.ROUGH,
        material=Material.ORGANIC,
        ambiguity=0.15,
        weight=2.0,
    ),
    ObjectTemplate(
        name="Flower",
        category="flower",
        super_category="natural",
        is_trash=False,
        shape=Shape.IRREGULAR,
        size_range=(8, 18),
        color_base=(220, 100, 150),  # Pink/colorful
        color_variance=80,
        texture=Texture.MATTE,
        material=Material.ORGANIC,
        ambiguity=0.3,  # Colorful like trash wrappers
        weight=0.5,
    ),

    # =========================================================================
    # NON-TRASH - Valuable (MUST NOT PICK UP!)
    # =========================================================================
    ObjectTemplate(
        name="Phone",
        category="phone",
        super_category="valuable",
        is_trash=False,
        shape=Shape.RECTANGULAR,
        size_range=(12, 18),
        color_base=(30, 30, 30),  # Black
        color_variance=20,
        texture=Texture.SMOOTH,
        material=Material.PLASTIC,
        reflectivity_range=(0.3, 0.6),
        ambiguity=0.4,  # Could look like black plastic trash
        weight=0.3,  # Rare
    ),
    ObjectTemplate(
        name="Keys",
        category="keys",
        super_category="valuable",
        is_trash=False,
        shape=Shape.IRREGULAR,
        size_range=(6, 12),
        color_base=(180, 180, 190),  # Metallic
        color_variance=30,
        texture=Texture.SHINY,
        material=Material.METAL,
        reflectivity_range=(0.5, 0.8),
        ambiguity=0.35,  # Metallic like cans
        weight=0.3,
    ),
    ObjectTemplate(
        name="Wallet",
        category="wallet",
        super_category="valuable",
        is_trash=False,
        shape=Shape.RECTANGULAR,
        size_range=(10, 15),
        color_base=(60, 40, 30),  # Brown leather
        color_variance=40,
        texture=Texture.SMOOTH,
        material=Material.FABRIC,
        ambiguity=0.35,
        weight=0.2,
    ),
    ObjectTemplate(
        name="Toy",
        category="toy",
        super_category="valuable",
        is_trash=False,
        shape=Shape.IRREGULAR,
        size_range=(8, 25),
        color_base=(200, 50, 50),  # Bright colors
        color_variance=100,
        texture=Texture.SMOOTH,
        material=Material.PLASTIC,
        has_branding_chance=0.5,
        reflectivity_range=(0.2, 0.5),
        ambiguity=0.5,  # Looks like colorful trash!
        weight=0.3,
    ),

    # =========================================================================
    # NON-TRASH - Ambiguous
    # =========================================================================
    ObjectTemplate(
        name="Worn Paper",
        category="worn_paper",
        super_category="ambiguous",
        is_trash=False,  # Important document!
        shape=Shape.FLAT,
        size_range=(10, 25),
        color_base=(230, 220, 200),
        color_variance=30,
        texture=Texture.MATTE,
        material=Material.PAPER,
        has_text_chance=0.8,
        damage_chance=0.7,
        ambiguity=0.7,  # Very confusable with trash paper
        weight=0.5,
    ),
    ObjectTemplate(
        name="Fabric Scrap",
        category="fabric_scrap",
        super_category="ambiguous",
        is_trash=False,
        shape=Shape.IRREGULAR,
        size_range=(10, 30),
        color_base=(150, 100, 80),
        color_variance=60,
        texture=Texture.FIBROUS,
        material=Material.FABRIC,
        damage_chance=0.8,
        ambiguity=0.6,  # Could be trash, could be lost item
        weight=0.4,
    ),
]


# =============================================================================
# WORLD OBJECT CLASS
# =============================================================================

# Global counter for unique IDs
_next_object_id = 0


class WorldObject(pygame.sprite.Sprite):
    """
    A world object with perceivable features and hidden ground truth.

    The robot can ONLY access features through the perception system.
    Ground truth is ONLY used for scoring/metrics.
    """

    def __init__(
        self,
        position: Tuple[float, float],
        features: VisualFeatures,
        ground_truth: GroundTruth,
        template_name: str = "Unknown"
    ):
        super().__init__()

        # Assign unique ID
        global _next_object_id
        self.id = _next_object_id
        _next_object_id += 1

        # Position
        self.x, self.y = position

        # Perceivable features (what the robot can see)
        self.features = features

        # Ground truth (ONLY for scoring - robot cannot access!)
        self._ground_truth = ground_truth

        # Template name for debugging
        self.template_name = template_name

        # Pickup state
        self.is_picked = False
        self.held_by = None

        # Physics
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.mass = features.size  # Mass based on size

        # Classification tracking (what the robot decided)
        self.was_classified = False
        self.classification_result: Optional[bool] = None  # What robot decided
        self.classification_confidence: float = 0.0

        # Create visual representation
        self._create_image()

    def _create_image(self):
        """Create visual representation based on features."""
        size = int(self.features.size)
        diameter = size * 2

        self.image = pygame.Surface((diameter + 4, diameter + 4), pygame.SRCALPHA)

        # Base color with some variation
        r, g, b = self.features.color
        color = (
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b))
        )

        # Draw based on shape
        center = (diameter // 2 + 2, diameter // 2 + 2)

        if self.features.shape == Shape.ROUND:
            pygame.draw.circle(self.image, color, center, size)
            # Add highlight for shiny objects
            if self.features.reflectivity > 0.5:
                highlight = (min(255, r + 60), min(255, g + 60), min(255, b + 60))
                pygame.draw.circle(self.image, highlight,
                                 (center[0] - size//3, center[1] - size//3), size//4)

        elif self.features.shape == Shape.CYLINDRICAL:
            rect = pygame.Rect(center[0] - size//2, center[1] - size, size, size * 2)
            pygame.draw.rect(self.image, color, rect, border_radius=size//3)
            # Top ellipse
            pygame.draw.ellipse(self.image, color,
                              (center[0] - size//2, center[1] - size - size//4, size, size//2))

        elif self.features.shape == Shape.RECTANGULAR:
            rect = pygame.Rect(center[0] - size, center[1] - size//2, size * 2, size)
            pygame.draw.rect(self.image, color, rect, border_radius=2)

        elif self.features.shape == Shape.CRUMPLED:
            # Irregular polygon
            points = []
            for i in range(7):
                angle = (i / 7) * 2 * math.pi
                r_offset = size * (0.6 + 0.4 * random.random())
                px = center[0] + int(r_offset * math.cos(angle))
                py = center[1] + int(r_offset * math.sin(angle))
                points.append((px, py))
            pygame.draw.polygon(self.image, color, points)

        elif self.features.shape == Shape.IRREGULAR:
            # More random shape
            points = []
            for i in range(6):
                angle = (i / 6) * 2 * math.pi + random.uniform(-0.3, 0.3)
                r_offset = size * (0.5 + 0.5 * random.random())
                px = center[0] + int(r_offset * math.cos(angle))
                py = center[1] + int(r_offset * math.sin(angle))
                points.append((px, py))
            pygame.draw.polygon(self.image, color, points)

        elif self.features.shape == Shape.FLAT:
            # Flat ellipse
            pygame.draw.ellipse(self.image, color,
                              (center[0] - size, center[1] - size//3, size * 2, size * 2 // 3))

        elif self.features.shape == Shape.ELONGATED:
            # Long thin shape
            rect = pygame.Rect(center[0] - size, center[1] - size//4, size * 2, size//2)
            pygame.draw.rect(self.image, color, rect, border_radius=size//4)

        else:
            # Default circle
            pygame.draw.circle(self.image, color, center, size)

        # Add texture effects
        if self.features.texture == Texture.ROUGH:
            # Add some noise dots
            for _ in range(int(size)):
                dx = random.randint(-size, size)
                dy = random.randint(-size, size)
                if dx*dx + dy*dy < size*size:
                    dark = (max(0, r-30), max(0, g-30), max(0, b-30))
                    self.image.set_at((center[0]+dx, center[1]+dy), dark)

        elif self.features.texture == Texture.CRINKLED:
            # Add crinkle lines
            for _ in range(3):
                x1 = center[0] + random.randint(-size//2, size//2)
                y1 = center[1] + random.randint(-size//2, size//2)
                x2 = x1 + random.randint(-size//2, size//2)
                y2 = y1 + random.randint(-size//2, size//2)
                dark = (max(0, r-40), max(0, g-40), max(0, b-40))
                pygame.draw.line(self.image, dark, (x1, y1), (x2, y2), 1)

        self.rect = self.image.get_rect()
        self.rect.center = (int(self.x), int(self.y))

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @position.setter
    def position(self, pos: Tuple[float, float]):
        self.x, self.y = pos
        self.rect.center = (int(self.x), int(self.y))

    @property
    def size(self) -> float:
        """Convenience property for collision detection."""
        return self.features.size

    # =========================================================================
    # GROUND TRUTH ACCESS (for scoring system only!)
    # =========================================================================

    def get_ground_truth(self) -> GroundTruth:
        """
        Get ground truth for scoring.
        WARNING: Robot behavior code should NEVER call this!
        """
        return self._ground_truth

    def is_actually_trash(self) -> bool:
        """
        Returns whether this is actually trash.
        WARNING: Robot behavior code should NEVER call this!
        """
        return self._ground_truth.is_actually_trash

    # =========================================================================
    # PICKUP/RELEASE
    # =========================================================================

    def pick_up(self, robot) -> bool:
        """Mark this object as picked up by a robot."""
        if self.is_picked:
            return False
        self.is_picked = True
        self.held_by = robot
        return True

    def release(self) -> bool:
        """Release this object."""
        self.is_picked = False
        self.held_by = None
        return True

    # =========================================================================
    # UPDATE/DRAW
    # =========================================================================

    def update(self):
        """Update object state."""
        self.rect.center = (int(self.x), int(self.y))

    def draw(self, screen: pygame.Surface):
        """Draw the object on screen."""
        if not self.is_picked:
            screen.blit(self.image, self.rect)

    def draw_debug(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw debug information."""
        if self.is_picked:
            return

        # Show ground truth (for debugging only!)
        gt = self._ground_truth

        # Color code: green = trash, red = not trash
        if gt.is_actually_trash:
            color = (0, 200, 0)
            label = "T"
        else:
            color = (200, 0, 0)
            label = "N"

        # Show ambiguity level
        ambiguity_str = f"{gt.ambiguity:.1f}"

        text = f"{label} {ambiguity_str}"
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.centerx = int(self.x)
        text_rect.bottom = int(self.y - self.features.size - 2)
        screen.blit(text_surface, text_rect)

    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle."""
        return self.rect

    def __repr__(self) -> str:
        gt = self._ground_truth
        return (f"WorldObject(id={self.id}, {self.template_name}, "
                f"trash={gt.is_actually_trash}, pos={self.position})")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_object_from_template(
    template: ObjectTemplate,
    position: Tuple[float, float]
) -> WorldObject:
    """Create a WorldObject instance from a template."""

    # Generate size within range
    size = random.uniform(template.size_range[0], template.size_range[1])

    # Generate color with variance
    base_r, base_g, base_b = template.color_base
    var = template.color_variance
    color = (
        max(0, min(255, base_r + random.randint(-var, var))),
        max(0, min(255, base_g + random.randint(-var, var))),
        max(0, min(255, base_b + random.randint(-var, var))),
    )

    # Generate optional features
    has_text = random.random() < template.has_text_chance
    has_branding = random.random() < template.has_branding_chance
    is_damaged = random.random() < template.damage_chance

    reflectivity = random.uniform(*template.reflectivity_range)
    transparency = random.uniform(*template.transparency_range)

    # Create features
    features = VisualFeatures(
        shape=template.shape,
        size=size,
        color=color,
        texture=template.texture,
        apparent_material=template.material,
        has_text=has_text,
        has_branding=has_branding,
        is_damaged=is_damaged,
        reflectivity=reflectivity,
        transparency=transparency,
    )

    # Create ground truth
    ground_truth = GroundTruth(
        is_actually_trash=template.is_trash,
        category=template.category,
        super_category=template.super_category,
        ambiguity=template.ambiguity,
        description=template.name,
    )

    return WorldObject(position, features, ground_truth, template.name)
