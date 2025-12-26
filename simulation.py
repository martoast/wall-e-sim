"""
Main simulation - orchestrates all systems and runs the game loop.
"""
import pygame
import random
from typing import List, Optional

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    TRASH_INITIAL_COUNT, TRASH_SPAWN_INTERVAL,
    OBSTACLE_COUNT, COLOR_BG
)
from entities.robot import Robot
from entities.arm import Arm
from entities.trash import Trash
from entities.nest import Nest
from entities.obstacle import Obstacle
from terrain.ground import Terrain
from systems.sensors import SensorSystem
from systems.navigation import Navigation
from systems.behavior import BehaviorController
from systems.telemetry import Telemetry, EventType
from systems.physics import PhysicsSystem
from systems.coordinator import Coordinator


class Simulation:
    """
    Main simulation class that runs the WALL-E garbage collection simulation.

    Controls:
    - F1: Toggle debug overlay
    - SPACE: Pause/Resume
    - R: Reset simulation
    - T: Spawn trash at mouse position
    - ESC: Quit
    """

    def __init__(self, robot_count: int = 1):
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("WALL-E Garbage Robot Simulation")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        self.running = True
        self.paused = False
        self.debug_mode = False

        # Initialize systems
        self.telemetry = Telemetry()
        self.physics = PhysicsSystem()
        self.coordinator = Coordinator()

        # Create sprite groups
        self.trash_group = pygame.sprite.Group()
        self.obstacle_group = pygame.sprite.Group()

        # Create environment
        self.terrain = Terrain()
        self.nest = Nest()

        # Create obstacles
        self._spawn_obstacles()

        # Create robots
        self.robots: List[Robot] = []
        self.behavior_controllers: List[BehaviorController] = []
        self._spawn_robots(robot_count)

        # Spawn initial trash
        self._spawn_trash(TRASH_INITIAL_COUNT)

        # Assign patrol zones after robots are created
        self.coordinator.assign_patrol_zones(robot_count, SCREEN_WIDTH, SCREEN_HEIGHT)

        # Trash spawn timer
        self.trash_spawn_timer = 0

    def _spawn_obstacles(self):
        """Spawn random obstacles."""
        margin = 100
        nest_avoid = 200

        for _ in range(OBSTACLE_COUNT):
            attempts = 0
            while attempts < 50:
                x = random.randint(margin, SCREEN_WIDTH - margin)
                y = random.randint(margin, SCREEN_HEIGHT - margin)

                # Avoid spawning near nest
                dx = x - self.nest.x
                dy = y - self.nest.y
                if (dx * dx + dy * dy) < nest_avoid * nest_avoid:
                    attempts += 1
                    continue

                obstacle = Obstacle((x, y))
                self.obstacle_group.add(obstacle)
                break

    def _spawn_robots(self, count: int):
        """Spawn robots avoiding obstacles."""
        for i in range(count):
            # Find a valid spawn position
            x, y = self._find_valid_robot_spawn()

            robot = Robot((x, y), robot_id=i)
            arm = Arm(robot)

            sensors = SensorSystem()
            navigation = Navigation()
            behavior = BehaviorController(robot, self.nest, sensors, navigation)

            self.robots.append(robot)
            self.behavior_controllers.append(behavior)

            # Initialize telemetry tracking
            self.telemetry.update_distance(i, (x, y))

    def _find_valid_robot_spawn(self) -> tuple:
        """Find a spawn position that doesn't overlap obstacles or other robots."""
        margin = 100
        robot_size = 50  # Approximate robot size for collision check

        for _ in range(100):  # Max attempts
            x = random.randint(margin, SCREEN_WIDTH // 3)  # Left third of screen
            y = random.randint(margin, SCREEN_HEIGHT - margin)

            # Check obstacles
            valid = True
            test_rect = pygame.Rect(x - robot_size//2, y - robot_size//2, robot_size, robot_size)

            for obstacle in self.obstacle_group:
                if test_rect.colliderect(obstacle.get_rect().inflate(20, 20)):
                    valid = False
                    break

            if not valid:
                continue

            # Check nest
            nest_rect = pygame.Rect(
                self.nest.x - self.nest.width//2 - 50,
                self.nest.y - self.nest.height//2 - 50,
                self.nest.width + 100,
                self.nest.height + 100
            )
            if test_rect.colliderect(nest_rect):
                continue

            # Check other robots
            for robot in self.robots:
                if abs(x - robot.x) < robot_size + 20 and abs(y - robot.y) < robot_size + 20:
                    valid = False
                    break

            if valid:
                return (x, y)

        # Fallback if no valid position found
        return (100, SCREEN_HEIGHT // 2)

    def _spawn_trash(self, count: int):
        """Spawn random trash items."""
        margin = 60
        nest_avoid = 150

        for _ in range(count):
            attempts = 0
            while attempts < 50:
                x = random.randint(margin, SCREEN_WIDTH - margin)
                y = random.randint(margin, SCREEN_HEIGHT - margin)

                # Avoid spawning near nest
                dx = x - self.nest.x
                dy = y - self.nest.y
                if (dx * dx + dy * dy) < nest_avoid * nest_avoid:
                    attempts += 1
                    continue

                # Avoid spawning on obstacles
                too_close = False
                for obstacle in self.obstacle_group:
                    ox, oy = obstacle.position
                    if abs(x - ox) < 50 and abs(y - oy) < 50:
                        too_close = True
                        break

                if not too_close:
                    trash = Trash((x, y))
                    self.trash_group.add(trash)
                    break

                attempts += 1

    def _spawn_trash_at(self, position: tuple):
        """Spawn a single trash item at a specific position."""
        trash = Trash(position)
        self.trash_group.add(trash)

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_F1:
                    self.debug_mode = not self.debug_mode
                    self.telemetry.toggle_overlay()

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.reset()

                elif event.key == pygame.K_t:
                    # Spawn trash at mouse position
                    mouse_pos = pygame.mouse.get_pos()
                    self._spawn_trash_at(mouse_pos)

    def update(self, dt: float):
        """Update simulation state."""
        if self.paused:
            return

        # Update terrain speed modifiers for robots
        for robot in self.robots:
            modifier = self.terrain.get_speed_modifier(robot.x, robot.y)
            robot.set_speed_modifier(modifier)

        # Cleanup stale trash claims
        self.coordinator.cleanup_claims(self.trash_group)

        # Update behavior controllers (sets velocities)
        for i, behavior in enumerate(self.behavior_controllers):
            old_state = behavior.current_state
            behavior.update(dt, self.trash_group, self.obstacle_group, self.robots, self.coordinator, self.telemetry)

            # Log state changes
            if behavior.current_state != old_state:
                self.telemetry.log(
                    EventType.STATE_CHANGE,
                    self.robots[i].id,
                    {'from': old_state.value, 'to': behavior.current_state.value}
                )

        # Apply physics (collision detection and response)
        self.physics.update(
            self.robots,
            self.trash_group,
            self.obstacle_group,
            self.nest,
            dt
        )

        # Update distance tracking after physics
        for i, robot in enumerate(self.robots):
            self.telemetry.update_distance(robot.id, robot.position)

        # Update trash (sprite rects)
        for trash in self.trash_group:
            trash.update()

        # Spawn new trash periodically
        self.trash_spawn_timer += dt
        if self.trash_spawn_timer >= TRASH_SPAWN_INTERVAL * FPS:
            self._spawn_trash(1)
            self.trash_spawn_timer = 0

    def draw(self):
        """Draw everything."""
        # Draw terrain
        self.terrain.draw(self.screen)

        # Draw obstacles
        for obstacle in self.obstacle_group:
            obstacle.draw(self.screen)

        # Draw nest
        self.nest.draw(self.screen)

        # Draw trash
        for trash in self.trash_group:
            trash.draw(self.screen)

        # Draw robots and arms
        for robot in self.robots:
            robot.draw(self.screen)
            if robot.arm:
                robot.arm.draw(self.screen)

        # Draw debug overlays
        if self.debug_mode:
            self.nest.draw_debug(self.screen, self.font)

            for i, robot in enumerate(self.robots):
                self.telemetry.draw_robot_debug(
                    self.screen,
                    robot,
                    self.behavior_controllers[i]
                )

            # Draw physics debug (collision boxes, velocities)
            self.physics.draw_debug(self.screen, self.robots, self.trash_group)

        # Draw telemetry overlay
        fps = self.clock.get_fps()
        self.telemetry.draw_overlay(
            self.screen,
            self.robots,
            self.nest,
            fps
        )

        # Draw pause indicator
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 255, 255))
            rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, 30))
            pygame.draw.rect(
                self.screen, (0, 0, 0),
                (rect.x - 10, rect.y - 5, rect.width + 20, rect.height + 10)
            )
            self.screen.blit(pause_text, rect)

        # Draw controls hint
        if not self.debug_mode:
            hint = self.font.render("F1: Debug | SPACE: Pause | T: Spawn Trash | R: Reset", True, (150, 150, 150))
            self.screen.blit(hint, (10, SCREEN_HEIGHT - 25))

        pygame.display.flip()

    def reset(self):
        """Reset the simulation."""
        # Clear groups
        self.trash_group.empty()
        self.obstacle_group.empty()

        # Regenerate terrain
        self.terrain = Terrain()

        # Reset nest
        self.nest = Nest()

        # Recreate obstacles
        self._spawn_obstacles()

        # Reset robots
        robot_count = len(self.robots)
        self.robots.clear()
        self.behavior_controllers.clear()
        self._spawn_robots(robot_count)

        # Spawn fresh trash
        self._spawn_trash(TRASH_INITIAL_COUNT)

        # Reset systems
        self.telemetry = Telemetry()
        self.physics = PhysicsSystem()
        self.coordinator = Coordinator()
        self.coordinator.assign_patrol_zones(robot_count, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.trash_spawn_timer = 0

    def run(self):
        """Main simulation loop."""
        while self.running:
            dt = 1.0  # Fixed timestep for simplicity

            self.handle_events()
            self.update(dt)
            self.draw()

            self.clock.tick(FPS)

        # Cleanup
        pygame.quit()

        # Export telemetry log
        try:
            self.telemetry.export_log("simulation_log.txt")
            print("Telemetry log exported to simulation_log.txt")
        except Exception as e:
            print(f"Failed to export telemetry: {e}")
