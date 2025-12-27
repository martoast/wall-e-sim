#!/usr/bin/env python3
"""
WALL-E Garbage Robot Simulation
================================

A Pygame simulation of autonomous garbage collection robots.

Features:
- Autonomous robot with articulated arm
- Patrol behavior with trash detection
- Pickup, store, and dump mechanics
- Central nest with ramp for dumping
- Terrain effects (mud slows robot)
- Debug visualization and telemetry

Controls:
- F1: Toggle debug overlay
- SPACE: Pause/Resume simulation
- R: Reset simulation
- T: Spawn trash at mouse position
- ESC: Quit

Usage:
    python main.py [--robots N] [--level L]

"""
import argparse
import sys

# Ensure we can import from the package
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import Simulation


# Difficulty configurations
DIFFICULTY_LEVELS = {
    1: {'obstacles': 4,  'trash': 10, 'name': 'Easy'},
    2: {'obstacles': 8,  'trash': 20, 'name': 'Normal'},
    3: {'obstacles': 12, 'trash': 30, 'name': 'Hard'},
    4: {'obstacles': 18, 'trash': 40, 'name': 'Very Hard'},
    5: {'obstacles': 25, 'trash': 50, 'name': 'Chaos'},
}


def main():
    parser = argparse.ArgumentParser(
        description="WALL-E Garbage Robot Simulation"
    )
    parser.add_argument(
        '--robots', '-r',
        type=int,
        default=1,
        help='Number of robots to spawn (default: 1)'
    )
    parser.add_argument(
        '--level', '-l',
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help='Difficulty level 1-5 (default: 2)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Start with debug mode enabled'
    )

    args = parser.parse_args()

    difficulty = DIFFICULTY_LEVELS[args.level]

    print("=" * 50)
    print("WALL-E Garbage Robot Simulation")
    print("=" * 50)
    print(f"Robots: {args.robots}")
    print(f"Level:  {args.level} ({difficulty['name']})")
    print(f"  - Obstacles: {difficulty['obstacles']}")
    print(f"  - Trash: {difficulty['trash']}")
    print()
    print("Controls:")
    print("  F1    - Toggle debug overlay")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset simulation")
    print("  Left click  - Add/remove trash")
    print("  Right click - Add/remove obstacle")
    print("  ESC   - Quit")
    print("=" * 50)
    print()

    # Create and run simulation
    sim = Simulation(
        robot_count=args.robots,
        obstacle_count=difficulty['obstacles'],
        initial_trash=difficulty['trash']
    )

    if args.debug:
        sim.debug_mode = True
        sim.telemetry.show_overlay = True

    sim.run()

    print("Simulation ended.")


if __name__ == "__main__":
    main()
