"""
Classifier System - Probabilistic trash classification.

Takes PERCEIVED features (not ground truth!) and produces:
- Classification: is this trash or not?
- Confidence: how sure are we?
- Category guess: what type of trash/object?

The classifier uses rule-based heuristics that can later be replaced
with a trained ML model calibrated on TACO dataset.
"""
import math
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entities.world_object import Shape, Texture, Material
from systems.perception import PerceptionResult, PerceivedFeatures


# =============================================================================
# CLASSIFICATION RESULT
# =============================================================================

class Decision(Enum):
    """Robot's decision about an object."""
    TRASH = "trash"           # Confident it's trash - pick it up
    NOT_TRASH = "not_trash"   # Confident it's not trash - ignore
    UNCERTAIN = "uncertain"   # Not sure - may need investigation


@dataclass
class ClassificationResult:
    """
    Result of classifying a perceived object.
    """
    # Binary classification
    is_trash: bool

    # Decision with uncertainty
    decision: Decision

    # Confidence scores
    trash_probability: float  # 0.0 = definitely not trash, 1.0 = definitely trash
    confidence: float  # How confident are we in this classification?

    # Category predictions (for detailed analysis)
    predicted_category: str
    category_confidence: float

    # Feature scores (for debugging/explanation)
    feature_scores: Dict[str, float]

    # Raw perception confidence (passed through)
    perception_confidence: float


# =============================================================================
# CLASSIFICATION THRESHOLDS
# =============================================================================

@dataclass
class ClassifierParams:
    """Configurable classification parameters."""
    # Decision thresholds
    confident_trash_threshold: float = 0.75  # Above this = definitely trash
    uncertain_high_threshold: float = 0.55   # Above this = might be trash
    uncertain_low_threshold: float = 0.35    # Below this = probably not trash
    confident_not_trash_threshold: float = 0.25  # Below this = definitely not trash

    # Confidence thresholds for final decision
    high_confidence_threshold: float = 0.7   # Above this = confident decision
    investigation_threshold: float = 0.4     # Below this = needs investigation

    # Feature weights
    shape_weight: float = 0.25
    color_weight: float = 0.20
    material_weight: float = 0.20
    size_weight: float = 0.10
    texture_weight: float = 0.10
    branding_weight: float = 0.10
    damage_weight: float = 0.05


# =============================================================================
# CLASSIFIER
# =============================================================================

class Classifier:
    """
    Rule-based trash classifier.

    Analyzes perceived features to estimate probability that an object is trash.
    Uses heuristics based on:
    - Shape (crumpled/irregular = more likely trash)
    - Color (artificial colors = more likely trash)
    - Material (plastic = more likely trash)
    - Size (small = more likely trash)
    - Texture (crinkled = more likely trash)
    - Branding/text (branded = more likely packaging trash)
    - Damage (damaged = more likely trash)
    """

    def __init__(self, params: Optional[ClassifierParams] = None):
        self.params = params or ClassifierParams()

        # Color classification
        self._natural_colors = self._define_natural_colors()
        self._artificial_colors = self._define_artificial_colors()

    def classify(self, perception: PerceptionResult) -> ClassificationResult:
        """
        Classify a perceived object.

        Args:
            perception: Result from perception system

        Returns:
            ClassificationResult with classification and confidence
        """
        features = perception.features
        feature_scores = {}

        # Score each feature
        shape_score = self._score_shape(features.shape)
        feature_scores['shape'] = shape_score

        color_score = self._score_color(features.color)
        feature_scores['color'] = color_score

        material_score = self._score_material(features.apparent_material)
        feature_scores['material'] = material_score

        size_score = self._score_size(features.size)
        feature_scores['size'] = size_score

        texture_score = self._score_texture(features.texture)
        feature_scores['texture'] = texture_score

        branding_score = self._score_branding(features.has_branding, features.has_text)
        feature_scores['branding'] = branding_score

        damage_score = self._score_damage(features.is_damaged)
        feature_scores['damage'] = damage_score

        # Weighted combination
        p = self.params
        trash_probability = (
            p.shape_weight * shape_score +
            p.color_weight * color_score +
            p.material_weight * material_score +
            p.size_weight * size_score +
            p.texture_weight * texture_score +
            p.branding_weight * branding_score +
            p.damage_weight * damage_score
        )

        # Clamp to [0, 1]
        trash_probability = max(0.0, min(1.0, trash_probability))

        # Apply perception confidence as a modifier
        # Low perception confidence = less certain about classification
        confidence = perception.overall_confidence * self._calc_classification_confidence(
            feature_scores, features
        )

        # Make decision
        decision, is_trash = self._make_decision(trash_probability, confidence)

        # Predict category
        predicted_category, category_confidence = self._predict_category(features, trash_probability)

        return ClassificationResult(
            is_trash=is_trash,
            decision=decision,
            trash_probability=trash_probability,
            confidence=confidence,
            predicted_category=predicted_category,
            category_confidence=category_confidence,
            feature_scores=feature_scores,
            perception_confidence=perception.overall_confidence,
        )

    def _score_shape(self, shape: Shape) -> float:
        """Score shape for trash likelihood."""
        shape_scores = {
            Shape.CRUMPLED: 0.85,     # Very likely trash
            Shape.IRREGULAR: 0.65,    # Somewhat likely trash
            Shape.CYLINDRICAL: 0.60,  # Cans, bottles - could be trash
            Shape.ROUND: 0.50,        # Could be bottle cap or natural
            Shape.RECTANGULAR: 0.50,  # Boxes, phones - mixed
            Shape.FLAT: 0.50,         # Paper, leaves - mixed
            Shape.ELONGATED: 0.65,    # Straws, cigarettes - often litter!
        }
        return shape_scores.get(shape, 0.5)

    def _score_color(self, color: Tuple[int, int, int]) -> float:
        """Score color for trash likelihood."""
        r, g, b = color

        # Check if it's a natural color
        if self._is_natural_color(r, g, b):
            return 0.3  # Less likely trash

        # Check if it's an artificial color
        if self._is_artificial_color(r, g, b):
            return 0.75  # More likely trash

        # Check for white/clear (common in packaging)
        if r > 220 and g > 220 and b > 220:
            return 0.6  # Somewhat likely trash (packaging)

        # Check for metallic (silver/aluminum)
        avg = (r + g + b) / 3
        variance = abs(r - avg) + abs(g - avg) + abs(b - avg)
        if variance < 30 and 150 < avg < 210:
            return 0.65  # Likely metal trash

        return 0.5  # Neutral

    def _is_natural_color(self, r: int, g: int, b: int) -> bool:
        """Check if color looks natural (earth tones, greens, browns)."""
        # Brown/earth tones
        if 60 < r < 160 and 40 < g < 120 and 20 < b < 80:
            if r > g > b:  # Brown gradient
                return True

        # Natural greens
        if g > r and g > b and 40 < g < 150:
            return True

        # Gray/stone
        avg = (r + g + b) / 3
        variance = abs(r - avg) + abs(g - avg) + abs(b - avg)
        if variance < 20 and 80 < avg < 150:
            return True

        return False

    def _is_artificial_color(self, r: int, g: int, b: int) -> bool:
        """Check if color looks artificial (bright, saturated)."""
        # Bright red packaging
        if r > 180 and g < 100 and b < 100:
            return True

        # Bright blue
        if b > 180 and r < 100 and g < 150:
            return True

        # Bright yellow
        if r > 200 and g > 200 and b < 100:
            return True

        # Neon/fluorescent
        max_channel = max(r, g, b)
        if max_channel > 220:
            other_avg = (r + g + b - max_channel) / 2
            if max_channel - other_avg > 150:
                return True

        # Pink/magenta
        if r > 180 and b > 150 and g < 120:
            return True

        return False

    def _define_natural_colors(self) -> List[Tuple[int, int, int]]:
        """Define typical natural colors."""
        return [
            (80, 100, 50),   # Green vegetation
            (100, 80, 50),   # Brown earth
            (120, 110, 100), # Stone gray
            (60, 45, 30),    # Dark soil
            (140, 120, 80),  # Dry grass
        ]

    def _define_artificial_colors(self) -> List[Tuple[int, int, int]]:
        """Define typical artificial colors."""
        return [
            (255, 0, 0),     # Bright red
            (0, 0, 255),     # Bright blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
        ]

    def _score_material(self, material: Material) -> float:
        """Score material for trash likelihood."""
        material_scores = {
            Material.PLASTIC: 0.75,   # Plastic is usually trash
            Material.PAPER: 0.60,     # Could be trash or document
            Material.METAL: 0.60,     # Cans are trash, but could be valuable
            Material.GLASS: 0.55,     # Bottles are trash, but careful
            Material.ORGANIC: 0.50,   # Could be natural, food waste, or cigarettes
            Material.FABRIC: 0.40,    # Less likely trash but could be
            Material.UNKNOWN: 0.50,   # Neutral
        }
        return material_scores.get(material, 0.5)

    def _score_size(self, size: float) -> float:
        """Score size for trash likelihood."""
        # Tiny (< 6): very likely cigarette butt - #1 litter item!
        if size < 6:
            return 0.80

        # Very small (< 10): likely cigarette butt, bottle cap
        if size < 10:
            return 0.70

        # Small (10-20): typical trash size
        if size < 20:
            return 0.60

        # Medium (20-30): could be anything
        if size < 30:
            return 0.50

        # Large (30-40): less likely casual litter
        if size < 40:
            return 0.40

        # Very large (> 40): probably not typical litter
        return 0.30

    def _score_texture(self, texture: Texture) -> float:
        """Score texture for trash likelihood."""
        texture_scores = {
            Texture.CRINKLED: 0.80,    # Very likely packaging/wrapper
            Texture.SHINY: 0.65,       # Plastic or metal packaging
            Texture.SMOOTH: 0.55,      # Could be plastic or other
            Texture.TRANSLUCENT: 0.60, # Likely plastic
            Texture.MATTE: 0.45,       # Neutral
            Texture.ROUGH: 0.40,       # Could be natural or damaged trash
            Texture.FIBROUS: 0.55,     # Paper, cigarette filters - often litter
        }
        return texture_scores.get(texture, 0.5)

    def _score_branding(self, has_branding: bool, has_text: bool) -> float:
        """Score branding/text for trash likelihood."""
        if has_branding:
            return 0.85  # Corporate branding = packaging trash
        if has_text:
            return 0.60  # Text could be packaging or document
        return 0.45  # No visible text/branding

    def _score_damage(self, is_damaged: bool) -> float:
        """Score damage for trash likelihood."""
        if is_damaged:
            return 0.70  # Damaged = more likely discarded
        return 0.45  # Intact = could be lost item

    def _calc_classification_confidence(
        self,
        feature_scores: Dict[str, float],
        features: PerceivedFeatures
    ) -> float:
        """Calculate confidence in the classification based on feature consistency."""
        scores = list(feature_scores.values())

        # If all features agree, high confidence
        # If features disagree, lower confidence
        score_variance = sum((s - 0.5) ** 2 for s in scores) / len(scores)

        # Higher variance from neutral = more confident (features agree on something)
        base_confidence = 0.5 + score_variance * 2  # Boosted variance impact

        # Weight by feature confidences - use sqrt to reduce penalty from low confidences
        avg_feature_conf = (
            features.shape_confidence * 0.3 +
            features.color_confidence * 0.25 +
            features.material_confidence * 0.2 +
            features.size_confidence * 0.15 +
            features.texture_confidence * 0.1
        )

        # Use sqrt to reduce harsh multiplication penalty, and boost by 2.5x
        return min(1.0, base_confidence * (avg_feature_conf ** 0.5) * 2.5)

    def _make_decision(
        self,
        trash_probability: float,
        confidence: float
    ) -> Tuple[Decision, bool]:
        """Make a decision based on probability and confidence."""
        p = self.params

        # High confidence decisions
        if confidence > p.high_confidence_threshold:
            if trash_probability > p.confident_trash_threshold:
                return Decision.TRASH, True
            if trash_probability < p.confident_not_trash_threshold:
                return Decision.NOT_TRASH, False

        # Low confidence = uncertain
        if confidence < p.investigation_threshold:
            return Decision.UNCERTAIN, trash_probability > 0.5

        # Medium confidence in ambiguous range
        if p.uncertain_low_threshold < trash_probability < p.uncertain_high_threshold:
            return Decision.UNCERTAIN, trash_probability > 0.5

        # Make a call based on probability
        if trash_probability > 0.5:
            return Decision.TRASH, True
        else:
            return Decision.NOT_TRASH, False

    def _predict_category(
        self,
        features: PerceivedFeatures,
        trash_probability: float
    ) -> Tuple[str, float]:
        """Predict specific category based on features."""
        if trash_probability < 0.4:
            # Not trash - try to identify what it is
            if features.apparent_material == Material.ORGANIC:
                if features.texture == Texture.ROUGH:
                    return "natural_object", 0.6
                return "unknown_organic", 0.4
            if features.apparent_material == Material.METAL and features.shape == Shape.IRREGULAR:
                return "keys", 0.3  # Wild guess
            if features.apparent_material == Material.PLASTIC and features.shape == Shape.RECTANGULAR:
                return "phone_or_device", 0.3
            return "unknown_non_trash", 0.3

        # Likely trash - predict type
        if features.apparent_material == Material.PLASTIC:
            if features.shape == Shape.CYLINDRICAL:
                if features.transparency > 0.4:
                    return "plastic_bottle", 0.7
                return "plastic_cup", 0.5
            if features.shape == Shape.CRUMPLED:
                return "plastic_wrapper", 0.6
            if features.shape == Shape.IRREGULAR and features.size > 20:
                return "plastic_bag", 0.5
            if features.size < 10:
                return "bottle_cap", 0.5
            return "other_plastic", 0.4

        if features.apparent_material == Material.METAL:
            if features.shape == Shape.CYLINDRICAL:
                return "drink_can", 0.8
            if features.shape == Shape.CRUMPLED:
                return "aluminium_foil", 0.6
            if features.size < 10:
                return "bottle_cap", 0.5
            return "other_metal", 0.4

        if features.apparent_material == Material.PAPER:
            if features.shape == Shape.CYLINDRICAL:
                return "paper_cup", 0.6
            if features.shape == Shape.CRUMPLED:
                return "crumpled_paper", 0.5
            if features.shape == Shape.FLAT:
                return "cardboard", 0.4
            return "other_paper", 0.4

        if features.apparent_material == Material.GLASS:
            if features.shape == Shape.CYLINDRICAL:
                return "glass_bottle", 0.7
            if features.shape == Shape.IRREGULAR:
                return "broken_glass", 0.6
            return "glass_container", 0.4

        if features.apparent_material == Material.ORGANIC:
            if features.size < 8:
                return "cigarette", 0.5
            return "food_waste", 0.4

        return "unknown_trash", 0.3


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_classifier(difficulty: int = 2) -> Classifier:
    """Create classifier with difficulty-adjusted parameters."""
    params = ClassifierParams()

    if difficulty == 1:  # Easy - clear thresholds
        params.confident_trash_threshold = 0.65
        params.confident_not_trash_threshold = 0.35
        params.high_confidence_threshold = 0.5

    elif difficulty == 3:  # Hard - narrower confidence band
        params.confident_trash_threshold = 0.80
        params.confident_not_trash_threshold = 0.20
        params.high_confidence_threshold = 0.75

    elif difficulty == 4:  # Very Hard
        params.confident_trash_threshold = 0.85
        params.confident_not_trash_threshold = 0.15
        params.high_confidence_threshold = 0.80
        params.investigation_threshold = 0.5

    elif difficulty == 5:  # Chaos - almost everything needs investigation
        params.confident_trash_threshold = 0.90
        params.confident_not_trash_threshold = 0.10
        params.high_confidence_threshold = 0.85
        params.investigation_threshold = 0.6

    return Classifier(params)
