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
    python main.py [--robots N]

"""
import argparse
import sys

# Ensure we can import from the package
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import Simulation


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
        '--debug', '-d',
        action='store_true',
        help='Start with debug mode enabled'
    )

    args = parser.parse_args()

    print("=" * 50)
    print("WALL-E Garbage Robot Simulation")
    print("=" * 50)
    print(f"Spawning {args.robots} robot(s)")
    print()
    print("Controls:")
    print("  F1    - Toggle debug overlay")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset simulation")
    print("  T     - Spawn trash at mouse")
    print("  ESC   - Quit")
    print("=" * 50)
    print()

    # Create and run simulation
    sim = Simulation(robot_count=args.robots)

    if args.debug:
        sim.debug_mode = True
        sim.telemetry.show_overlay = True

    sim.run()

    print("Simulation ended.")


if __name__ == "__main__":
    main()
