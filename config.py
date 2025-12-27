# WALL-E Garbage Robot Simulation - Configuration
# All tunable parameters in one place

# =============================================================================
# DISPLAY
# =============================================================================
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 30

# =============================================================================
# ROBOT
# =============================================================================
ROBOT_WIDTH = 50
ROBOT_HEIGHT = 40
ROBOT_SPEED = 2.0
ROBOT_TURN_SPEED = 3.0
ROBOT_BIN_CAPACITY = 10
ROBOT_SENSOR_RANGE = 150  # Detection range - how far the cameras can see
ROBOT_GRAB_RANGE = 30
ROBOT_VISION_CONE = 120  # degrees - realistic camera FOV

# =============================================================================
# ARM
# =============================================================================
ARM_SEGMENT_1_LENGTH = 30
ARM_SEGMENT_2_LENGTH = 25
ARM_EXTEND_SPEED = 2.0
ARM_RETRACT_SPEED = 3.0
CLAW_SIZE = 12

# =============================================================================
# NEST
# =============================================================================
NEST_WIDTH = 100
NEST_HEIGHT = 140
NEST_POSITION = (SCREEN_WIDTH - 150, SCREEN_HEIGHT // 2)
RAMP_WIDTH = 60
RAMP_LENGTH = 80
RAMP_ANGLE = 12
NEST_CAPACITY = 100

# =============================================================================
# TRASH
# =============================================================================
TRASH_INITIAL_COUNT = 20
TRASH_SPAWN_INTERVAL = 3.0  # seconds
TRASH_SIZE_MIN = 8
TRASH_SIZE_MAX = 20

# =============================================================================
# TERRAIN
# =============================================================================
TILE_SIZE = 40
MUD_SPEED_MODIFIER = 0.5
MUD_COVERAGE = 0.15  # 15% of map

# =============================================================================
# OBSTACLES
# =============================================================================
OBSTACLE_COUNT = 8
OBSTACLE_SIZE_MIN = 30
OBSTACLE_SIZE_MAX = 60

# =============================================================================
# COLORS
# =============================================================================
COLOR_BG = (30, 35, 30)
COLOR_ROBOT_BODY = (100, 110, 95)
COLOR_ROBOT_TRACKS = (50, 50, 50)
COLOR_CLAW = (70, 70, 80)
COLOR_CLAW_OPEN = (90, 90, 100)
COLOR_ARM = (80, 85, 90)
COLOR_TRASH = (139, 90, 43)
COLOR_TRASH_CAN = (180, 180, 180)
COLOR_TRASH_BOTTLE = (100, 180, 100)
COLOR_TRASH_PAPER = (200, 180, 150)
COLOR_NEST = (80, 85, 100)
COLOR_NEST_FILL = (60, 65, 80)
COLOR_RAMP = (110, 100, 80)
COLOR_MUD = (60, 45, 30)
COLOR_DIRT = (80, 60, 40)
COLOR_GROUND = (45, 55, 40)
COLOR_OBSTACLE = (90, 90, 90)
COLOR_DEBUG = (255, 255, 0)
COLOR_DEBUG_SENSOR = (0, 255, 0, 50)
COLOR_DEBUG_VISION = (255, 200, 0, 50)

# LED Colors by state
LED_PATROL = (0, 255, 100)
LED_SEEKING = (255, 255, 0)
LED_APPROACHING = (255, 200, 0)
LED_PICKING = (255, 150, 0)
LED_STORING = (255, 100, 50)
LED_RETURNING = (100, 150, 255)
LED_WAITING = (255, 100, 255)  # Magenta - waiting in queue
LED_DOCKING = (150, 150, 255)
LED_DUMPING = (150, 100, 255)
LED_IDLE = (100, 100, 100)

# =============================================================================
# PHYSICS
# =============================================================================
PHYSICS_PUSH_FORCE = 0.8       # How much velocity transfers when pushing
PHYSICS_FRICTION = 0.85        # Friction applied to pushed objects (0-1)
PHYSICS_SEPARATION_FORCE = 2.0 # Force to separate overlapping objects
SCREEN_MARGIN = 20             # Keep objects this far from screen edges

# =============================================================================
# DEBUG
# =============================================================================
DEBUG_SHOW_SENSOR_RANGE = True
DEBUG_SHOW_VISION_CONE = True
DEBUG_SHOW_PATROL_PATH = True
DEBUG_SHOW_STATE_LABEL = True
DEBUG_SHOW_BIN_LEVEL = True
DEBUG_FONT_SIZE = 16
