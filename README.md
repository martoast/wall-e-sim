# WALL-E: Autonomous Garbage Collection Robot Simulation

A sophisticated simulation for developing autonomous trash collection robots that transition from simulation to real-world deployment. The project aims to create WALL-E-style robots capable of cleaning public spaces like parks, beaches, streets, and campuses.

## Project Vision

**Mission:** Build autonomous garbage collection robots using a simulation-first approach, where the simulation is intentionally harder than reality to ensure successful sim-to-real transfer.

**Philosophy:** *"The simulation must be HARDER than reality, not easier"* — robots operate with realistic sensor noise, uncertain classification, and challenging physics to guarantee real-world performance.

## Features

- **Multi-Robot Coordination** — Multiple robots work together with task allocation, patrol zones, and dump queue management
- **Perception-Based Decision Making** — Robots classify objects based on visual features (shape, color, texture) with realistic noise, not ground truth labels
- **Probabilistic Classification** — Confidence-based decisions with investigation states for uncertain objects
- **Comprehensive Scoring** — True Positive/False Positive/True Negative/False Negative tracking with precision, recall, and F1 metrics
- **Realistic Physics** — Collision detection, terrain effects (mud, dirt), and swept collision to prevent tunneling
- **Articulated Arm** — 2-segment arm with claw for trash pickup mechanics
- **Shared Spatial Memory** — Pheromone-like marking of risky areas for emergent coordination

## Project Structure

```
wall_e_sim/
├── config.py              # Centralized configuration parameters
├── main.py                # Entry point with CLI argument parsing
├── simulation.py          # Main simulation loop and orchestration
├── entities/              # Game objects
│   ├── robot.py           # Robot entity with state machine
│   ├── arm.py             # Articulated arm with claw
│   ├── world_object.py    # Feature-based objects (TACO taxonomy)
│   ├── trash.py           # Legacy trash objects
│   ├── nest.py            # Collection bin with ramp
│   ├── obstacle.py        # Rocks and walls
│   └── object_spawner.py  # Diverse object generation
├── systems/               # Autonomous systems
│   ├── behavior.py        # State machine controller
│   ├── perception.py      # Realistic sensor noise simulation
│   ├── classifier.py      # Probabilistic trash classification
│   ├── scoring.py         # TP/FP/TN/FN metrics tracking
│   ├── physics.py         # Collision detection and response
│   ├── navigation.py      # Patrol paths and obstacle avoidance
│   ├── sensors.py         # Vision cone and distance calculations
│   ├── coordinator.py     # Multi-robot task allocation
│   ├── shared_map.py      # Spatial memory system
│   └── telemetry.py       # Event logging and statistics
├── terrain/
│   └── ground.py          # Grid-based terrain with tile types
└── utils/
    └── math_helpers.py    # Vector math utilities
```

## Installation

### Requirements

- Python 3.13+
- Pygame

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd wall-e

# Install dependencies
pip install pygame

# Run the simulation
python main.py
```

## Usage

### Command Line Options

```bash
python main.py [--robots N] [--level L] [--debug]

Options:
  --robots, -r N    Number of robots (default: 1)
  --level, -l L     Difficulty 1-5 (default: 2)
  --debug, -d       Start with debug overlay enabled
```

### Difficulty Levels

| Level | Description | Obstacles | Trash |
|-------|-------------|-----------|-------|
| 1 | Easy | 4 | 10 |
| 2 | Normal | 8 | 20 |
| 3 | Hard | 12 | 30 |
| 4 | Very Hard | 18 | 40 |
| 5 | Chaos | 25 | 50 |

### Controls

| Key | Action |
|-----|--------|
| `F1` / `D` | Toggle debug overlay |
| `SPACE` | Pause/Resume simulation |
| `R` | Reset simulation |
| `T` | Spawn trash at mouse position |
| `Left Click` | Add/Remove trash |
| `Right Click` | Add/Remove obstacle |
| `+` / `=` | Add robot |
| `-` | Remove robot |
| `↑` / `↓` | Speed up / Slow down |
| `ESC` | Quit |

## How It Works

### Robot State Machine

Each robot follows a sophisticated state machine:

```
PATROL → INVESTIGATING → SEEKING → APPROACHING → PICKING → STORING → RETURNING → WAITING → DOCKING → DUMPING → UNDOCKING
```

1. **PATROL** — Wander patrol zone looking for objects
2. **INVESTIGATING** — Move closer to uncertain objects to re-perceive
3. **SEEKING** — Navigate toward detected trash
4. **APPROACHING** — Close enough to attempt pickup
5. **PICKING** — Extend arm and grab trash
6. **STORING** — Retract arm and store in bin
7. **RETURNING** — Return to nest when bin is full
8. **WAITING** — Queue for dump access
9. **DOCKING** — Climb ramp to dump position
10. **DUMPING** — Empty bin into nest

### Perception Pipeline

The robot's decision-making follows a realistic perception pipeline:

```
Perception → Classification → Decision → Scoring
```

1. **Perception Phase**
   - Detect object features: shape, color, texture, size
   - Apply noise based on distance, viewing angle, and occlusion
   - Generate confidence scores for each feature

2. **Classification Phase**
   - Analyze perceived features (not ground truth)
   - Crumpled/irregular shapes → higher trash probability
   - Artificial colors → higher trash probability
   - Output: `trash_probability` (0.0-1.0) and `confidence` (0.0-1.0)

3. **Decision Phase**
   - `trash_probability > 0.60` → **PICK IT UP**
   - `0.35 < trash_probability < 0.60` → **INVESTIGATE**
   - `trash_probability < 0.25` → **IGNORE**

4. **Scoring Phase**
   - Compare decisions against ground truth
   - Track TP, FP, TN, FN metrics
   - Calculate Precision, Recall, F1 Score

### Multi-Robot Coordination

- **Patrol Zones** — Each robot is assigned a coverage area
- **Trash Claims** — Prevents multiple robots targeting the same object
- **Dump Queue** — Orderly access to the collection nest
- **Shared Memory** — Robots mark risky/problematic areas

## Development Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | Complete | Foundation — Basic simulation with robot, navigation, physics |
| 2 | Complete | Perception Reality — Feature-based perception with uncertainty |
| 3 | Planned | TACO Integration — Connect to real-world trash dataset |
| 4 | Planned | Physical Realism — Grasping mechanics, energy systems |
| 5 | Planned | Learning Infrastructure — RL training environment |
| 6 | Planned | Sim-to-Real Bridge — ROS integration |
| 7 | Planned | Real-World Pilot — Hardware deployment |

## Configuration

Key parameters can be adjusted in `config.py`:

| Category | Parameters |
|----------|------------|
| Display | Screen size (1200x800), FPS (30) |
| Robot | Speed, sensor range, bin capacity, vision cone |
| Arm | Segment lengths, extend speed |
| Nest | Position, capacity, ramp angle |
| Terrain | Tile size, mud coverage, speed modifiers |
| Physics | Push force, friction, separation force |
| Debug | Toggle various debug visualizations |

## Debug Overlay

Press `F1` or `D` to enable the debug overlay which shows:

- Robot sensor ranges and vision cones
- Perception results and classification confidence
- TP/FP/TN/FN statistics
- Precision, Recall, F1 scores
- Robot state and current target
- Collision detection zones

## Technologies

- **Python 3.13** — Core language with type hints
- **Pygame** — 2D graphics and game loop
- **Dataclasses** — Structured data for perception results
- **Enums** — Type-safe states and decisions

## Future Integration (Phase 3+)

- **TACO Dataset** — Real-world trash classification data
- **PyTorch** — Train classification models
- **OpenAI Gym** — Reinforcement learning environment
- **ROS2** — Hardware integration for real robots

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please check the `tasks/` directory for:
- `todo.md` — Development roadmap and task breakdown
- `ceo.md` — Vision and strategy documentation
- `taco_integration.md` — Phase 3 planning details
