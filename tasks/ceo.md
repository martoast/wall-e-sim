# Trashly: CEO Vision Document

## The Mission

**Clean the world, one robot at a time.**

We're building autonomous robots that collect trash from public spaces - parks, beaches, streets, campuses. Not as a gimmick, but as critical infrastructure for a cleaner planet.

---

## Why This Matters

- 8 million tons of plastic enter oceans yearly
- Cities spend billions on manual litter collection
- Human cleanup is dangerous, inefficient, and never-ending
- Climate change demands we stop treating Earth as a landfill

**The opportunity:** Autonomous trash collection that scales.

---

## The North Star

A fleet of WALL-E-style robots that:
1. **See** the world through cameras/sensors (no magic labels)
2. **Understand** what is trash vs. not-trash (with uncertainty)
3. **Navigate** complex real environments (not sanitized simulations)
4. **Coordinate** as a swarm (efficient coverage, no collisions)
5. **Learn** from experience (get better over time)
6. **Operate** safely around humans and animals

---

## Why Simulation First

We're not simulating to avoid reality. We're simulating to **prepare for it**.

### Simulation Advantages:
- **Speed**: 1000x faster iteration than physical prototypes
- **Safety**: Robots can "crash" without real damage or liability
- **Scale**: Test edge cases that would take years to encounter naturally
- **Cost**: $0 per robot-hour vs. $100s for physical testing
- **ML Training**: Generate unlimited synthetic training data

### The Critical Rule:
> **The simulation must be HARDER than reality, not easier.**

If our simulated robots succeed with noisy sensors, uncertain classification, and challenging physics - they'll thrive in the real world.

If they succeed because we gave them perfect information - they'll fail catastrophically when deployed.

---

## The Sim-to-Real Philosophy

### What We MUST Simulate Realistically:

1. **Perception, Not Labels**
   - Robots see pixels/features, not object types
   - Classification has uncertainty and errors
   - Distance, lighting, and occlusion affect recognition
   - Some objects are genuinely ambiguous

2. **Physics That Punishes Cheating**
   - Grasping can fail
   - Objects have different weights and textures
   - Terrain affects movement
   - Wind, slopes, obstacles matter

3. **Sensor Limitations**
   - Limited field of view
   - Noise in depth/distance estimation
   - Occlusion and blind spots
   - Processing latency

4. **Real-World Chaos**
   - Humans walking through the scene
   - Animals, vehicles, unexpected obstacles
   - Weather effects
   - Dynamic environments (trash moves)

### What We CAN Simplify (for now):

- Exact physics engine precision (close enough is fine)
- Photorealistic rendering (features matter, not beauty)
- Full mechanical simulation (abstract the arm/gripper)

---

## The Technical Moat

Our competitive advantage will be:

1. **Perception AI** - The best trash-vs-not-trash classifier
2. **Navigation Intelligence** - Efficient coverage, obstacle handling
3. **Swarm Coordination** - Multi-robot efficiency at scale
4. **Sim-to-Real Transfer** - Models trained in sim that work in reality
5. **Continuous Learning** - Robots that improve from deployment data

---

## Key Resource: TACO Dataset

**TACO** (Trash Annotations in Context) is our foundation for real-world perception.

- **What**: 715+ images of real litter, manually labeled and segmented
- **Where**: Diverse environments - beaches, streets, woods, urban areas
- **Format**: COCO format with 60 categories in 28 super-categories
- **Why it matters**: Object segmentation enables robotic grasping, not just detection

**Repository**: https://github.com/pedropro/TACO

### Strategic Value:
1. **Ground truth taxonomy** - Our simulation uses TACO's real-world categories
2. **Classifier training** - Train production models on real labeled data
3. **Benchmark** - Measure our perception against established metrics
4. **Domain bridge** - Calibrate sim perception to match real classifier behavior

> TACO is not just a dataset. It's our reality anchor - ensuring everything we build connects to the real world.

---

## Development Phases

### Phase 1: Foundation (COMPLETE)
- Basic simulation environment
- Robot navigation and obstacle avoidance
- Multi-robot spawning and coordination
- State machine for behavior
- Basic physics and collision

### Phase 2: Perception Reality (CURRENT PRIORITY)
- Replace labeled trash with feature-based objects
- Implement perception system with uncertainty
- Add object classification with confidence scores
- Introduce ambiguous objects (trash-like non-trash)
- Simulate sensor noise and limitations

### Phase 3: Physical Realism
- Grasping success/failure mechanics
- Terrain variety (grass, concrete, sand, slopes)
- Weather effects (wind, rain, lighting changes)
- Energy/battery management
- Realistic bin capacity and weight

### Phase 4: Learning Infrastructure
- Reinforcement learning hooks
- Reward function design (not just "pick up labeled trash")
- Episode management for training
- Comprehensive data logging
- Behavioral cloning pipeline

### Phase 5: Sim-to-Real Bridge
- ROS/ROS2 integration
- Real sensor model calibration
- Domain randomization for transfer
- Hardware abstraction layer
- Safety system design

### Phase 6: Real World Pilot
- Controlled environment deployment (private campus)
- Remote monitoring and intervention
- Progressive autonomy (human-in-loop -> supervised -> autonomous)
- Data collection for model improvement
- Safety validation

---

## Key Metrics

### Simulation Metrics:
- **Classification Accuracy**: Can robot distinguish trash from non-trash?
- **Coverage Efficiency**: Area cleaned per unit time
- **False Pickup Rate**: How often does it grab non-trash?
- **Miss Rate**: How often does it ignore actual trash?
- **Collision Rate**: Safety in navigation
- **Coordination Efficiency**: Multi-robot utilization

### Real-World Metrics (future):
- Trash collected per hour
- Battery efficiency (trash per charge)
- Human intervention rate
- Safety incident rate
- Public perception/acceptance

---

## Final Thought

> "The goal is not to simulate trash collection. The goal is to build the perception and decision-making systems that will work on real robots. The simulation is the gym where those systems get strong."

Let's build something that matters.
