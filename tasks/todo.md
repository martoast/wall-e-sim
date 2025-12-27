# Trashly Development Roadmap

> **Mission**: Build autonomous trash-collecting robots, starting with simulation, ending in the real world.

---

## Progress Overview

| Phase | Name | Status | Progress |
|-------|------|--------|----------|
| 1 | Foundation | COMPLETE | ██████████ 100% |
| 2 | Perception Reality | NOT STARTED | ░░░░░░░░░░ 0% |
| 3 | TACO Integration | NOT STARTED | ░░░░░░░░░░ 0% |
| 4 | Physical Realism | NOT STARTED | ░░░░░░░░░░ 0% |
| 5 | Learning Infrastructure | NOT STARTED | ░░░░░░░░░░ 0% |
| 6 | Sim-to-Real Bridge | NOT STARTED | ░░░░░░░░░░ 0% |
| 7 | Real World Pilot | NOT STARTED | ░░░░░░░░░░ 0% |

---

# Phase 1: Foundation (COMPLETE)

What we built:
- [x] Basic simulation environment with Pygame
- [x] Robot entity with articulated arm
- [x] Navigation and obstacle avoidance
- [x] Multi-robot spawning and coordination
- [x] State machine for behavior (PATROL, SEEKING, APPROACHING, PICKING, etc.)
- [x] Basic physics and collision detection
- [x] Swept collision to prevent tunneling
- [x] Nest with ramp for dumping
- [x] Dump queue management
- [x] Debug visualization and telemetry
- [x] Difficulty levels (1-5)
- [x] Click to add/remove objects

---

# Phase 2: Perception Reality (CURRENT PRIORITY)

> **Goal**: Transform from "labeled objects" to "perceived world with uncertainty"

## Why This Matters
The robot currently "cheats" by knowing exactly what is trash. In reality, it must:
1. Observe visual features (not labels)
2. Classify with uncertainty
3. Decide under ambiguity
4. Learn from mistakes

---

## 2.1 World Object System
**Replace Trash class with feature-based WorldObject**

- [ ] Create `entities/world_object.py`
  ```python
  class WorldObject:
      # What the robot CAN perceive
      shape: str          # 'round', 'rectangular', 'irregular', 'crumpled'
      size: float         # radius in pixels
      color: tuple        # RGB
      texture: str        # 'smooth', 'rough', 'shiny', 'matte', 'crinkled'

      # Ground truth (for scoring only - robot cannot see this!)
      is_actually_trash: bool
      category: str       # TACO category
      super_category: str # TACO super-category
      ambiguity: float    # 0.0 = obvious, 1.0 = very ambiguous
  ```

- [ ] Create `entities/object_spawner.py`
  - **Trash types (from TACO taxonomy)**:
    | Super-Category | Types |
    |---------------|-------|
    | Plastic | bottles, bags, wrappers, cups, straws, film |
    | Paper | cardboard, newspaper, receipts, tissues |
    | Metal | cans, foil, bottle caps |
    | Glass | bottles, jars, broken glass |
    | Organic | food waste, fruit peels |
    | Cigarette | butts |
    | Other | styrofoam, batteries |

  - **Non-trash types**:
    | Category | Types |
    |----------|-------|
    | Natural | leaves, rocks, pinecones, sticks, flowers |
    | Valuable | phones, wallets, keys, toys |
    | Ambiguous | worn paper, fabric scraps |

- [ ] Migrate codebase from `trash_group` to `world_objects`
  - [ ] Update simulation.py
  - [ ] Update physics.py
  - [ ] Update behavior.py
  - [ ] Update sensors.py

---

## 2.2 Perception System
**Robots see features, not labels**

- [ ] Create `systems/perception.py`
  ```python
  class PerceptionSystem:
      def perceive(self, obj, robot_pos, robot_angle) -> PerceptionResult:
          """
          Returns what the robot 'sees' - features with noise.
          Does NOT return ground truth!
          """
          distance = calc_distance(robot_pos, obj.position)
          angle_to_obj = calc_angle(robot_pos, obj.position)

          # Add noise based on distance and angle
          perceived_size = obj.size + noise(distance)
          perceived_color = obj.color + color_noise(distance)
          perceived_shape = maybe_misidentify_shape(obj.shape, distance)

          return PerceptionResult(
              position=obj.position + position_noise(distance),
              size=perceived_size,
              color=perceived_color,
              shape=perceived_shape,
              confidence=calc_confidence(distance, angle_to_obj)
          )
  ```

- [ ] Implement sensor noise model
  - Distance: further = noisier features
  - Angle: edge of FOV = worse perception
  - Occlusion: partially hidden = lower confidence
  - Size: small objects harder to classify

---

## 2.3 Classification System
**Probabilistic trash classification**

- [ ] Create `systems/classifier.py`
  ```python
  class Classifier:
      def classify(self, perception: PerceptionResult) -> ClassificationResult:
          """
          Takes perceived features, returns classification + confidence.
          """
          # Feature-based scoring
          trash_score = 0.0

          # Crumpled/irregular shape = more likely trash
          if perception.shape in ['crumpled', 'irregular']:
              trash_score += 0.3

          # Artificial colors = more likely trash
          if is_artificial_color(perception.color):
              trash_score += 0.2

          # Small size = more likely trash
          if perception.size < 30:
              trash_score += 0.2

          # Apply distance-based confidence penalty
          confidence = perception.confidence * base_confidence

          return ClassificationResult(
              is_trash=trash_score > 0.5,
              confidence=confidence,
              trash_probability=trash_score
          )
  ```

- [ ] Implement rule-based classifier v1
- [ ] Add configurable confusion patterns
- [ ] Support category-level classification (not just trash/not-trash)

---

## 2.4 Decision Under Uncertainty
**Replace binary detection with probabilistic decisions**

- [ ] Update behavior.py with new decision logic
  ```python
  # Thresholds
  CONFIDENT_PICKUP = 0.85    # Just grab it
  WORTH_INVESTIGATING = 0.50 # Move closer to check
  PROBABLY_NOT_TRASH = 0.20  # Ignore it

  def decide(self, obj, classification):
      if classification.confidence > CONFIDENT_PICKUP:
          return Action.PICK_UP
      elif classification.confidence > WORTH_INVESTIGATING:
          return Action.INVESTIGATE  # Move closer, re-perceive
      else:
          return Action.IGNORE
  ```

- [ ] Implement INVESTIGATING state
  - Move closer to uncertain object
  - Re-perceive at close range
  - Make final pickup/ignore decision

- [ ] Handle errors gracefully
  - Track false positives (picked up non-trash)
  - Track false negatives (ignored actual trash)

---

## 2.5 Scoring System
**Track real performance metrics**

- [ ] Create `systems/scoring.py`
  ```python
  class ScoringSystem:
      true_positives: int   # Correctly picked up trash
      false_positives: int  # Wrongly picked up non-trash
      true_negatives: int   # Correctly ignored non-trash
      false_negatives: int  # Wrongly ignored trash

      def precision(self):
          return TP / (TP + FP)

      def recall(self):
          return TP / (TP + FN)

      def f1_score(self):
          return 2 * (precision * recall) / (precision + recall)
  ```

- [ ] Log every decision with outcome
- [ ] Display metrics in debug overlay
- [ ] Export metrics for analysis

---

## 2.6 Visual Updates
**Make uncertainty visible**

- [ ] Render objects based on features (not type)
  - Variety in colors, shapes, sizes
  - No more "all trash is brown"

- [ ] Debug mode shows perception state
  - Confidence level above each object
  - Color coding: 🟢 confident trash, 🟡 uncertain, 🔴 confident not-trash
  - Show classification result vs ground truth

---

## 2.7 Testing & Validation

- [ ] Test scenarios
  - Easy: all obvious trash
  - Medium: mixed trash and non-trash
  - Hard: mostly ambiguous objects
  - Edge cases: tiny, distant, occluded

- [ ] Benchmark metrics
  - Precision/recall at different distances
  - Performance per object category
  - Investigation rate vs direct pickup rate

---

## Phase 2 Success Criteria

Phase 2 is COMPLETE when:
- [ ] Robot cannot access `is_actually_trash` directly
- [ ] Perception adds realistic noise based on distance
- [ ] Classification returns confidence scores
- [ ] Robot investigates uncertain objects
- [ ] Scoring tracks TP/FP/TN/FN
- [ ] Debug mode shows confidence levels
- [ ] Both trash and non-trash spawn with variety

---

# Phase 3: TACO Integration

> **Goal**: Connect simulation to real-world data

## 3.1 Dataset Setup
- [ ] Clone TACO: `git clone https://github.com/pedropro/TACO.git`
- [ ] Download images: `python3 download.py`
- [ ] Explore with `demo.ipynb`
- [ ] Analyze category distribution

## 3.2 Train Real Classifier
- [ ] Set up PyTorch environment
- [ ] Implement COCO data loader for TACO
- [ ] Train Mask R-CNN or YOLO on TACO
- [ ] Evaluate: mAP, precision, recall per category
- [ ] Export model for inference

## 3.3 Extract Model Behavior
- [ ] Generate confusion matrix from trained model
- [ ] Extract per-category confidence distributions
- [ ] Document common misclassification patterns
- [ ] Save stats to `data/taco/model_stats.json`

## 3.4 Calibrate Simulation
- [ ] Update sim classifier to match real model behavior
- [ ] Confidence curves match TACO model
- [ ] Confusion patterns match real misclassifications
- [ ] Validate: sim and real classifiers behave similarly

## 3.5 Domain Randomization
- [ ] Use TACO segmentation masks as textures
- [ ] Vary lighting, backgrounds
- [ ] Generate synthetic training data from sim
- [ ] Augment TACO with sim-generated edge cases

---

# Phase 4: Physical Realism

> **Goal**: Make physics match real-world challenges

## 4.1 Grasping Mechanics
- [ ] Grasp success/failure based on object properties
  - Size affects graspability
  - Weight affects carry speed
  - Shape affects grip stability
- [ ] Dropped object handling
- [ ] Retry logic for failed grasps

## 4.2 Terrain System
- [ ] Multiple terrain types
  - Grass (slower movement)
  - Concrete (normal speed)
  - Sand (very slow, objects sink)
  - Slopes (affects navigation)
- [ ] Terrain affects perception (grass occludes small objects)

## 4.3 Environmental Effects
- [ ] Wind (moves light objects, affects small trash)
- [ ] Lighting changes (affects perception confidence)
- [ ] Rain (optional - affects traction, visibility)

## 4.4 Energy Management
- [ ] Battery system
- [ ] Return to charge behavior
- [ ] Energy-efficient path planning

---

# Phase 5: Learning Infrastructure

> **Goal**: Enable ML training in simulation

## 5.1 RL Environment
- [ ] OpenAI Gym compatible interface
- [ ] Observation space (perception output)
- [ ] Action space (move, turn, pickup, etc.)
- [ ] Reward function design

## 5.2 Reward Engineering
- [ ] Reward for true positives
- [ ] Penalty for false positives
- [ ] Penalty for false negatives
- [ ] Efficiency bonuses (coverage, speed)

## 5.3 Training Pipeline
- [ ] Episode management (reset, done conditions)
- [ ] Parallel environment support
- [ ] Curriculum learning (easy → hard)
- [ ] Checkpointing and logging

## 5.4 Data Collection
- [ ] Record all decisions and outcomes
- [ ] Export trajectories for imitation learning
- [ ] Generate labeled perception data

---

# Phase 6: Sim-to-Real Bridge

> **Goal**: Prepare for real hardware deployment

## 6.1 ROS Integration
- [ ] ROS2 node wrapper for simulation
- [ ] Standard message types (sensor_msgs, geometry_msgs)
- [ ] Same control interface as real robot

## 6.2 Sensor Calibration
- [ ] Match sim camera to real camera specs
- [ ] Calibrate noise models to real sensors
- [ ] Depth sensor simulation

## 6.3 Hardware Abstraction
- [ ] Define robot interface (move, turn, pickup)
- [ ] Sim and real implementations of interface
- [ ] Swap between sim/real with config flag

## 6.4 Transfer Validation
- [ ] Test trained policies in increasingly realistic sim
- [ ] Measure sim-to-real gap
- [ ] Domain randomization tuning

---

# Phase 7: Real World Pilot

> **Goal**: Deploy and validate on physical robot

## 7.1 Hardware Setup
- [ ] Select/build robot platform
- [ ] Mount cameras and sensors
- [ ] Integrate compute (Jetson, etc.)
- [ ] Calibrate sensors

## 7.2 Controlled Environment
- [ ] Private campus or test area
- [ ] Known trash placement for testing
- [ ] Safety boundaries
- [ ] Emergency stop system

## 7.3 Progressive Autonomy
- [ ] Stage 1: Human in loop (approve every pickup)
- [ ] Stage 2: Supervised (human monitors, can intervene)
- [ ] Stage 3: Autonomous (human reviews logs after)

## 7.4 Metrics & Iteration
- [ ] Track real-world performance
- [ ] Identify failure modes
- [ ] Update sim to match failures
- [ ] Retrain and redeploy

---

# Quick Reference

## Key Files to Create (Phase 2)

```
entities/
├── world_object.py      # Feature-based objects
└── object_spawner.py    # Diverse object generation

systems/
├── perception.py        # Noisy feature extraction
├── classifier.py        # Probabilistic classification
└── scoring.py           # Performance metrics
```

## Key Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Precision | >90% | N/A |
| Recall | >85% | N/A |
| False Positive Rate | <5% | N/A |
| Investigation Rate | 10-30% | N/A |

---

## Current Status

**Phase**: 2 - Perception Reality
**Task**: 2.1 - Create WorldObject class
**Blockers**: None
**Next Action**: Implement `entities/world_object.py`

---

## Links

- [TACO Dataset](https://github.com/pedropro/TACO)
- [TACO Integration Guide](./taco_integration.md)
- [CEO Vision](./ceo.md)
