"""
Scoring System - Track robot performance metrics.

Compares robot DECISIONS against GROUND TRUTH to calculate:
- True Positives: Correctly picked up trash
- False Positives: Wrongly picked up non-trash
- True Negatives: Correctly ignored non-trash
- False Negatives: Wrongly ignored trash

Also tracks:
- Precision, Recall, F1 Score
- Accuracy per category
- Investigation rate
- Error patterns
"""
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entities.world_object import WorldObject
from systems.classifier import ClassificationResult, Decision


# =============================================================================
# DECISION OUTCOMES
# =============================================================================

class Outcome(Enum):
    """Possible outcomes of a pickup decision."""
    TRUE_POSITIVE = "TP"   # Picked up actual trash
    FALSE_POSITIVE = "FP"  # Picked up non-trash (error!)
    TRUE_NEGATIVE = "TN"   # Correctly ignored non-trash
    FALSE_NEGATIVE = "FN"  # Ignored actual trash (error!)
    INVESTIGATED = "INV"   # Investigated object (delayed decision)


@dataclass
class DecisionRecord:
    """Record of a single decision."""
    timestamp: float
    object_id: int
    robot_id: int

    # What the robot decided
    decision: Decision
    classification_confidence: float
    trash_probability: float

    # What was actually true
    is_actually_trash: bool
    actual_category: str
    actual_super_category: str

    # Outcome
    outcome: Outcome

    # Context
    distance_at_decision: float
    investigation_count: int = 0  # How many times investigated before deciding


# =============================================================================
# SCORING SYSTEM
# =============================================================================

class ScoringSystem:
    """
    Tracks and analyzes robot decision performance.
    """

    def __init__(self):
        # Core metrics
        self.true_positives: int = 0
        self.false_positives: int = 0
        self.true_negatives: int = 0
        self.false_negatives: int = 0

        # Investigation tracking
        self.investigations: int = 0  # Total investigation events
        self.investigation_resolved_tp: int = 0  # Investigations that led to correct pickup
        self.investigation_resolved_fp: int = 0  # Investigations that led to wrong pickup
        self.investigation_resolved_tn: int = 0  # Investigations that led to correct ignore
        self.investigation_resolved_fn: int = 0  # Investigations that led to wrong ignore

        # Per-category tracking
        self.category_tp: Dict[str, int] = defaultdict(int)
        self.category_fp: Dict[str, int] = defaultdict(int)
        self.category_fn: Dict[str, int] = defaultdict(int)
        self.category_tn: Dict[str, int] = defaultdict(int)

        # Decision history
        self.decision_records: List[DecisionRecord] = []

        # Timing
        self.start_time: float = time.time()

        # Objects tracked
        self.objects_seen: set = set()  # Object IDs that were perceived
        self.objects_decided: set = set()  # Object IDs that got a final decision

    def record_pickup(
        self,
        obj: WorldObject,
        classification: ClassificationResult,
        robot_id: int,
        distance: float,
        investigated: bool = False
    ) -> Outcome:
        """
        Record a pickup decision.

        Args:
            obj: The object being picked up
            classification: The classification that led to this decision
            robot_id: ID of robot making the decision
            distance: Distance at time of decision
            investigated: Whether this was after investigation

        Returns:
            The outcome (TP or FP)
        """
        ground_truth = obj.get_ground_truth()

        if ground_truth.is_actually_trash:
            outcome = Outcome.TRUE_POSITIVE
            self.true_positives += 1
            self.category_tp[ground_truth.super_category] += 1
            if investigated:
                self.investigation_resolved_tp += 1
        else:
            outcome = Outcome.FALSE_POSITIVE
            self.false_positives += 1
            self.category_fp[ground_truth.super_category] += 1
            if investigated:
                self.investigation_resolved_fp += 1

        # Record decision
        record = DecisionRecord(
            timestamp=time.time(),
            object_id=obj.id,
            robot_id=robot_id,
            decision=Decision.TRASH,
            classification_confidence=classification.confidence,
            trash_probability=classification.trash_probability,
            is_actually_trash=ground_truth.is_actually_trash,
            actual_category=ground_truth.category,
            actual_super_category=ground_truth.super_category,
            outcome=outcome,
            distance_at_decision=distance,
            investigation_count=1 if investigated else 0,
        )
        self.decision_records.append(record)
        self.objects_decided.add(obj.id)

        return outcome

    def record_ignore(
        self,
        obj: WorldObject,
        classification: ClassificationResult,
        robot_id: int,
        distance: float,
        investigated: bool = False
    ) -> Outcome:
        """
        Record an ignore decision.

        Args:
            obj: The object being ignored
            classification: The classification that led to this decision
            robot_id: ID of robot making the decision
            distance: Distance at time of decision
            investigated: Whether this was after investigation

        Returns:
            The outcome (TN or FN)
        """
        ground_truth = obj.get_ground_truth()

        if ground_truth.is_actually_trash:
            outcome = Outcome.FALSE_NEGATIVE
            self.false_negatives += 1
            self.category_fn[ground_truth.super_category] += 1
            if investigated:
                self.investigation_resolved_fn += 1
        else:
            outcome = Outcome.TRUE_NEGATIVE
            self.true_negatives += 1
            self.category_tn[ground_truth.super_category] += 1
            if investigated:
                self.investigation_resolved_tn += 1

        # Record decision
        record = DecisionRecord(
            timestamp=time.time(),
            object_id=obj.id,
            robot_id=robot_id,
            decision=Decision.NOT_TRASH,
            classification_confidence=classification.confidence,
            trash_probability=classification.trash_probability,
            is_actually_trash=ground_truth.is_actually_trash,
            actual_category=ground_truth.category,
            actual_super_category=ground_truth.super_category,
            outcome=outcome,
            distance_at_decision=distance,
            investigation_count=1 if investigated else 0,
        )
        self.decision_records.append(record)
        self.objects_decided.add(obj.id)

        return outcome

    def record_investigation(self, obj: WorldObject, robot_id: int):
        """Record that an investigation started (uncertain decision)."""
        self.investigations += 1
        self.objects_seen.add(obj.id)

    def record_seen(self, obj: WorldObject):
        """Record that an object was perceived."""
        self.objects_seen.add(obj.id)

    # =========================================================================
    # METRICS
    # =========================================================================

    def precision(self) -> float:
        """Precision = TP / (TP + FP). Of items picked up, how many were trash?"""
        total = self.true_positives + self.false_positives
        if total == 0:
            return 0.0
        return self.true_positives / total

    def recall(self) -> float:
        """Recall = TP / (TP + FN). Of all trash, how much did we collect?"""
        total = self.true_positives + self.false_negatives
        if total == 0:
            return 0.0
        return self.true_positives / total

    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)."""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / total."""
        total = (self.true_positives + self.true_negatives +
                 self.false_positives + self.false_negatives)
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN). Of non-trash, how many did we wrongly pick up?"""
        total = self.false_positives + self.true_negatives
        if total == 0:
            return 0.0
        return self.false_positives / total

    def false_negative_rate(self) -> float:
        """FNR = FN / (FN + TP). Of trash, how many did we wrongly ignore?"""
        total = self.false_negatives + self.true_positives
        if total == 0:
            return 0.0
        return self.false_negatives / total

    def investigation_rate(self) -> float:
        """What fraction of decisions required investigation?"""
        total_decisions = len(self.decision_records)
        if total_decisions == 0:
            return 0.0
        investigated = sum(1 for r in self.decision_records if r.investigation_count > 0)
        return investigated / total_decisions

    def investigation_success_rate(self) -> float:
        """Of investigations, how many led to correct decisions?"""
        total_investigated = (
            self.investigation_resolved_tp + self.investigation_resolved_fp +
            self.investigation_resolved_tn + self.investigation_resolved_fn
        )
        if total_investigated == 0:
            return 0.0
        correct = self.investigation_resolved_tp + self.investigation_resolved_tn
        return correct / total_investigated

    # =========================================================================
    # CATEGORY ANALYSIS
    # =========================================================================

    def category_precision(self, category: str) -> float:
        """Precision for a specific category."""
        tp = self.category_tp[category]
        fp = self.category_fp[category]
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def category_recall(self, category: str) -> float:
        """Recall for a specific category."""
        tp = self.category_tp[category]
        fn = self.category_fn[category]
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def get_category_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all categories."""
        all_categories = set(
            list(self.category_tp.keys()) +
            list(self.category_fp.keys()) +
            list(self.category_fn.keys()) +
            list(self.category_tn.keys())
        )

        stats = {}
        for cat in all_categories:
            tp = self.category_tp[cat]
            fp = self.category_fp[cat]
            tn = self.category_tn[cat]
            fn = self.category_fn[cat]
            total = tp + fp + tn + fn

            stats[cat] = {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'total': total,
                'precision': self.category_precision(cat),
                'recall': self.category_recall(cat),
            }

        return stats

    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================

    def get_worst_categories(self, by: str = 'errors') -> List[Tuple[str, float]]:
        """Get categories ranked by error rate."""
        stats = self.get_category_stats()

        if by == 'errors':
            # Most total errors (FP + FN)
            errors = [(cat, s['fp'] + s['fn']) for cat, s in stats.items()]
        elif by == 'fp_rate':
            # Highest false positive rate
            errors = [(cat, s['fp'] / max(1, s['total'])) for cat, s in stats.items()]
        elif by == 'fn_rate':
            # Highest false negative rate
            errors = [(cat, s['fn'] / max(1, s['total'])) for cat, s in stats.items()]
        else:
            errors = []

        return sorted(errors, key=lambda x: -x[1])

    def get_confusion_pairs(self) -> Dict[Tuple[str, str], int]:
        """Get pairs of (actual_category, predicted_decision) with counts."""
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)

        for record in self.decision_records:
            decision_str = "pickup" if record.decision == Decision.TRASH else "ignore"
            pairs[(record.actual_super_category, decision_str)] += 1

        return dict(pairs)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def get_summary(self) -> Dict:
        """Get complete summary of performance."""
        elapsed = time.time() - self.start_time
        total_decisions = len(self.decision_records)

        return {
            # Core metrics
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1_score(),
            'accuracy': self.accuracy(),

            # Counts
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'total_decisions': total_decisions,

            # Error rates
            'false_positive_rate': self.false_positive_rate(),
            'false_negative_rate': self.false_negative_rate(),

            # Investigation
            'investigations': self.investigations,
            'investigation_rate': self.investigation_rate(),
            'investigation_success_rate': self.investigation_success_rate(),

            # Object tracking
            'objects_seen': len(self.objects_seen),
            'objects_decided': len(self.objects_decided),

            # Timing
            'elapsed_time': elapsed,
            'decisions_per_minute': (total_decisions / elapsed) * 60 if elapsed > 0 else 0,
        }

    def get_display_stats(self) -> str:
        """Get formatted string for display overlay."""
        s = self.get_summary()
        lines = [
            f"Precision: {s['precision']:.1%}",
            f"Recall: {s['recall']:.1%}",
            f"F1: {s['f1_score']:.2f}",
            f"",
            f"TP: {s['true_positives']}  FP: {s['false_positives']}",
            f"TN: {s['true_negatives']}  FN: {s['false_negatives']}",
            f"",
            f"FP Rate: {s['false_positive_rate']:.1%}",
            f"FN Rate: {s['false_negative_rate']:.1%}",
            f"",
            f"Investigations: {s['investigations']}",
            f"Inv Rate: {s['investigation_rate']:.1%}",
        ]
        return "\n".join(lines)

    def reset(self):
        """Reset all metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.investigations = 0
        self.investigation_resolved_tp = 0
        self.investigation_resolved_fp = 0
        self.investigation_resolved_tn = 0
        self.investigation_resolved_fn = 0
        self.category_tp.clear()
        self.category_fp.clear()
        self.category_fn.clear()
        self.category_tn.clear()
        self.decision_records.clear()
        self.objects_seen.clear()
        self.objects_decided.clear()
        self.start_time = time.time()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_scoring_system: Optional[ScoringSystem] = None


def get_scoring_system() -> ScoringSystem:
    """Get global scoring system instance."""
    global _scoring_system
    if _scoring_system is None:
        _scoring_system = ScoringSystem()
    return _scoring_system


def reset_scoring_system():
    """Reset global scoring system."""
    global _scoring_system
    if _scoring_system is not None:
        _scoring_system.reset()
    else:
        _scoring_system = ScoringSystem()
