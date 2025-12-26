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
from systems.shared_map import SharedMap


class Simulation:
    """
    Main simulation class that runs the WALL-E garbage collection simulation.

    Controls:
    - Left Click: Spawn trash at mouse position
    - Right Click: Spawn obstacle at mouse position
    - D: Toggle debug overlay
    - SPACE: Pause/Resume
    - R: Reset simulation
    - +/=: Add a robot
    - -: Remove a robot
    - Up Arrow: Speed up simulation
    - Down Arrow: Slow down simulation
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
        self.game_won = False
        self.win_screen_alpha = 0  # For fade-in effect
        self.restart_button_rect = None  # Set when win screen is drawn

        # Simulation speed control
        self.speed_multiplier = 1.0
        self.MIN_SPEED = 0.25
        self.MAX_SPEED = 4.0

        # Initialize systems
        self.telemetry = Telemetry()
        self.physics = PhysicsSystem()
        self.coordinator = Coordinator()
        self.shared_map = SharedMap(SCREEN_WIDTH, SCREEN_HEIGHT)

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

                # Avoid spawning on robots
                if not too_close:
                    for robot in self.robots:
                        if abs(x - robot.x) < 60 and abs(y - robot.y) < 60:
                            too_close = True
                            break

                if not too_close:
                    trash = Trash((x, y), trash_type='general')
                    self.trash_group.add(trash)
                    break

                attempts += 1

    def _spawn_trash_at(self, position: tuple):
        """Spawn a single trash item at a specific position."""
        trash = Trash(position, trash_type='general')
        self.trash_group.add(trash)

    def _add_robot(self):
        """Add a new robot to the simulation."""
        robot_id = len(self.robots)
        x, y = self._find_valid_robot_spawn()

        robot = Robot((x, y), robot_id=robot_id)
        arm = Arm(robot)

        sensors = SensorSystem()
        navigation = Navigation()
        behavior = BehaviorController(robot, self.nest, sensors, navigation)

        self.robots.append(robot)
        self.behavior_controllers.append(behavior)

        # Update telemetry and patrol zones
        self.telemetry.update_distance(robot_id, (x, y))
        self.coordinator.assign_patrol_zones(len(self.robots), SCREEN_WIDTH, SCREEN_HEIGHT)

        # Regenerate patrol paths for all robots with new zones
        for bc in self.behavior_controllers:
            bc._coordinator = self.coordinator
            bc._generate_patrol()

    def _remove_robot(self):
        """Remove the last robot from the simulation."""
        if len(self.robots) <= 1:
            return  # Keep at least one robot

        # Remove last robot
        robot = self.robots.pop()
        behavior = self.behavior_controllers.pop()

        # Release any trash claims
        if behavior.target_trash and self.coordinator:
            self.coordinator.release_claim(behavior.target_trash.id)

        # Leave dump queue if in it
        self.coordinator.leave_queue(robot.id)

        # Reassign patrol zones
        self.coordinator.assign_patrol_zones(len(self.robots), SCREEN_WIDTH, SCREEN_HEIGHT)

        # Regenerate patrol paths for remaining robots
        for bc in self.behavior_controllers:
            bc._coordinator = self.coordinator
            bc._generate_patrol()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.reset()

                elif event.key == pygame.K_t:
                    # Spawn trash at mouse position
                    mouse_pos = pygame.mouse.get_pos()
                    self._spawn_trash_at(mouse_pos)

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    # Add a robot
                    self._add_robot()

                elif event.key == pygame.K_MINUS:
                    # Remove a robot
                    self._remove_robot()

                elif event.key == pygame.K_UP:
                    # Increase simulation speed
                    self.speed_multiplier = min(self.MAX_SPEED, self.speed_multiplier * 1.5)

                elif event.key == pygame.K_DOWN:
                    # Decrease simulation speed
                    self.speed_multiplier = max(self.MIN_SPEED, self.speed_multiplier / 1.5)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if event.button == 1:  # Left click
                    # Check if clicking restart button on win screen
                    if self.game_won and self.restart_button_rect and self.restart_button_rect.collidepoint(mouse_pos):
                        self.reset()
                    elif not self.game_won:
                        self._spawn_trash_at(mouse_pos)
                elif event.button == 3:  # Right click
                    if not self.game_won:
                        self._spawn_obstacle_at(mouse_pos)

    def _spawn_obstacle_at(self, position: tuple):
        """Spawn an obstacle at a specific position."""
        # Check if too close to robots
        for robot in self.robots:
            if abs(position[0] - robot.x) < 60 and abs(position[1] - robot.y) < 60:
                return  # Don't spawn on robots

        # Check if too close to nest
        if abs(position[0] - self.nest.x) < 100 and abs(position[1] - self.nest.y) < 100:
            return  # Don't spawn on nest

        obstacle = Obstacle(position)
        self.obstacle_group.add(obstacle)

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
            behavior.update(dt, self.trash_group, self.obstacle_group, self.robots, self.coordinator, self.telemetry, self.shared_map)

            # Log state changes
            if behavior.current_state != old_state:
                self.telemetry.log(
                    EventType.STATE_CHANGE,
                    self.robots[i].id,
                    {'from': old_state.value, 'to': behavior.current_state.value}
                )

            # CRITICAL: Tell physics which trash this robot is targeting
            # This allows the robot to approach its target trash for pickup
            if behavior.target_trash and not behavior.target_trash.is_picked:
                self.physics.set_robot_target(self.robots[i].id, behavior.target_trash.id)
            else:
                self.physics.set_robot_target(self.robots[i].id, None)

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

        # Check win condition - all trash collected and dumped
        self._check_win_condition()

    def _check_win_condition(self):
        """Check if all trash has been collected and dumped."""
        if self.game_won:
            return

        # Check if there's any trash left on the ground
        if len(self.trash_group) > 0:
            return

        # Check if any robot is still holding trash or has trash in bin
        for robot in self.robots:
            if robot.bin_count > 0:
                return
            if robot.arm and robot.arm.holding:
                return

        # All trash collected and dumped!
        self.game_won = True

    def _draw_win_screen(self):
        """Draw the victory screen overlay."""
        # Fade in effect
        if self.win_screen_alpha < 200:
            self.win_screen_alpha += 5

        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, self.win_screen_alpha))
        self.screen.blit(overlay, (0, 0))

        # Only show text once faded in enough
        if self.win_screen_alpha < 100:
            return

        # Victory text
        title_font = pygame.font.Font(None, 80)
        subtitle_font = pygame.font.Font(None, 36)

        # Main title with green color
        title = title_font.render("ALL TRASH COLLECTED!", True, (100, 255, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60))
        self.screen.blit(title, title_rect)

        # Stats
        total_dumped = self.nest.fill_level
        stats_text = subtitle_font.render(f"Total items recycled: {total_dumped}", True, (200, 200, 200))
        stats_rect = stats_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(stats_text, stats_rect)

        # Restart button
        button_width, button_height = 200, 50
        button_x = SCREEN_WIDTH // 2 - button_width // 2
        button_y = SCREEN_HEIGHT // 2 + 60

        # Store button rect for click detection
        self.restart_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        # Button background
        mouse_pos = pygame.mouse.get_pos()
        if self.restart_button_rect.collidepoint(mouse_pos):
            button_color = (80, 180, 80)  # Hover color
        else:
            button_color = (60, 140, 60)

        pygame.draw.rect(self.screen, button_color, self.restart_button_rect, border_radius=10)
        pygame.draw.rect(self.screen, (100, 255, 100), self.restart_button_rect, 3, border_radius=10)

        # Button text
        button_text = subtitle_font.render("RESTART", True, (255, 255, 255))
        button_text_rect = button_text.get_rect(center=self.restart_button_rect.center)
        self.screen.blit(button_text, button_text_rect)

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

        # Draw ALL debug overlays - controlled by single debug_mode flag
        if self.debug_mode:
            # Draw shared map heatmap (risky areas)
            self.shared_map.draw_debug(self.screen)
            self.nest.draw_debug(self.screen, self.font)

            for i, robot in enumerate(self.robots):
                self.telemetry.draw_robot_debug(
                    self.screen,
                    robot,
                    self.behavior_controllers[i]
                )

            # Draw physics debug (collision boxes, velocities)
            self.physics.draw_debug(self.screen, self.robots, self.trash_group)

            # Draw telemetry stats overlay
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
        if not self.debug_mode and not self.game_won:
            hint = self.font.render("LClick: Trash | RClick: Rock | D: Debug | SPACE: Pause | +/-: Robots | Up/Down: Speed | R: Reset", True, (150, 150, 150))
            self.screen.blit(hint, (10, SCREEN_HEIGHT - 25))

        # Draw speed indicator
        if not self.game_won:
            speed_text = f"Speed: {self.speed_multiplier:.2f}x"
            speed_color = (150, 150, 150) if self.speed_multiplier == 1.0 else (100, 200, 255)
            speed_label = self.font.render(speed_text, True, speed_color)
            self.screen.blit(speed_label, (SCREEN_WIDTH - 100, 10))

        # Draw win screen if game is won
        if self.game_won:
            self._draw_win_screen()

        pygame.display.flip()

    def reset(self):
        """Reset the simulation."""
        # Reset game state
        self.game_won = False
        self.win_screen_alpha = 0
        self.restart_button_rect = None

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
        self.shared_map = SharedMap(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.trash_spawn_timer = 0

    def run(self):
        """Main simulation loop."""
        while self.running:
            dt = self.speed_multiplier  # Adjustable timestep

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
