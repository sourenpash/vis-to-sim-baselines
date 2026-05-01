# Robot API Reference

This page provides detailed API documentation for robot interfaces and configurations in UniMimic.

## Robot Base Classes

### ArticulationCfg

Base configuration for all robot articulations.

```python
from omni.isaac.lab.assets import ArticulationCfg

class ArticulationCfg:
    """Configuration for robot articulation."""
    
    # Asset properties
    prim_path: str = "/World/envs/env_.*/Robot"
    spawn: SpawnCfg = None
    init_state: InitialStateCfg = None
    collision_group: int = 0
    
    # Actuator properties
    actuators: Dict[str, ActuatorCfg] = None
    
    # Soft joint position limits
    soft_joint_pos_limit_factor: float = 1.0
```

### RobotInterface

Base interface for robot control.

```python
from isaaclab_mimic.utils.robots import RobotInterface

class RobotInterface:
    """Base class for robot interfaces."""
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions in radians."""
        
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities in rad/s."""
        
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector position and quaternion."""
        
    def send_joint_command(self, positions: np.ndarray) -> None:
        """Send joint position commands."""
```

## UR5 Robot

### UR5_CFG

```python
from isaaclab_mimic.robots import UR5_CFG

UR5_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={".*": 0.0},
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=3.14,
            effort_limit=150.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
```

### UR5 with Gripper

```python
from isaaclab_mimic.robots import UR5_GRIPPER_CFG

UR5_GRIPPER_CFG = ArticulationCfg(
    # Inherits from UR5_CFG
    spawn=UsdFileCfg(
        usd_path="path/to/ur5_with_gripper.usd",
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_.*", "wrist_.*"],
            # ... arm actuator config
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_.*"],
            velocity_limit=0.2,
            effort_limit=200.0,
            stiffness=1e3,
            damping=1e2,
        ),
    },
)
```

### UR5 Controller

```python
from isaaclab_mimic.utils.robots import UR5Controller

class UR5Controller(RobotInterface):
    """UR5 specific control interface."""
    
    def __init__(self, robot_cfg: ArticulationCfg, device: str = "cuda"):
        """Initialize UR5 controller."""
        
    def inverse_kinematics(
        self, 
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        current_joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute IK solution for target pose."""
        
    def get_jacobian(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Get manipulator Jacobian at given configuration."""
```

## ARX-X7 Robot

### ARX7_CFG

```python
from isaaclab_mimic.robots import ARX7_CFG

ARX7_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="path/to/arx_x7.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -0.785,  # -45 degrees
            "joint_3": 0.0,
            "joint_4": -1.571,  # -90 degrees
            "joint_5": 0.0,
            "joint_6": 0.785,   # 45 degrees
            "joint_7": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            velocity_limit=2.0,
            effort_limit=100.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
)
```

### ARX7 Mobile Base

```python
from isaaclab_mimic.robots import ARX7_MOBILE_CFG

ARX7_MOBILE_CFG = ArticulationCfg(
    # Includes mobile base
    spawn=UsdFileCfg(
        usd_path="path/to/arx_x7_mobile.usd",
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*"],
            velocity_limit=1.0,  # m/s
            effort_limit=100.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            # ... arm config
        ),
    },
)
```

## Dual Arm Configuration

### DualArmRobotCfg

```python
from isaaclab_mimic.robots import DualArmRobotCfg

class DualArmRobotCfg:
    """Configuration for dual-arm robots."""
    
    left_arm: ArticulationCfg = UR5_CFG
    right_arm: ArticulationCfg = UR5_CFG
    
    # Relative positioning
    left_arm_offset: Tuple[float, float, float] = (-0.3, 0.0, 0.0)
    right_arm_offset: Tuple[float, float, float] = (0.3, 0.0, 0.0)
    
    # Coordination settings
    coordination_mode: str = "independent"  # or "coordinated"
```

## IK Solvers

### PyBulletIKSolver

```python
from isaaclab_mimic.utils.robot_ik import PyBulletIKSolver

class PyBulletIKSolver:
    """PyBullet-based IK solver."""
    
    def __init__(
        self,
        urdf_path: str,
        ee_link: str,
        device: str = "cpu"
    ):
        """Initialize solver with URDF."""
        
    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        seed_joints: Optional[np.ndarray] = None,
        tolerance: float = 1e-3,
        max_iter: int = 100
    ) -> Tuple[np.ndarray, bool]:
        """Solve IK for target pose."""
```

### CuRoboIKSolver

```python
from isaaclab_mimic.utils.robot_ik import CuRoboIKSolver

class CuRoboIKSolver:
    """GPU-accelerated IK solver using CuRobo."""
    
    def __init__(
        self,
        robot_cfg: Dict,
        world_cfg: Dict,
        device: str = "cuda"
    ):
        """Initialize CuRobo solver."""
        
    def solve_batch(
        self,
        target_positions: torch.Tensor,  # [B, 3]
        target_quaternions: torch.Tensor,  # [B, 4]
        seed_joints: torch.Tensor,  # [B, N]
        collision_check: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch IK solving on GPU."""
```

### PinkIKSolver

```python
from isaaclab_mimic.utils.robot_ik import PinkIKSolver

class PinkIKSolver:
    """Pink IK solver with task prioritization."""
    
    def __init__(
        self,
        urdf_path: str,
        tasks: List[Task]
    ):
        """Initialize with task hierarchy."""
        
    def solve_with_tasks(
        self,
        current_config: np.ndarray,
        tasks: Dict[str, Any],
        dt: float = 0.01
    ) -> np.ndarray:
        """Solve with multiple task objectives."""
```

## Robot Utilities

### ForwardKinematics

```python
from isaaclab_mimic.utils import ForwardKinematics

class ForwardKinematics:
    """Forward kinematics computation."""
    
    @staticmethod
    def compute_fk(
        joint_positions: torch.Tensor,
        robot_model: RobotModel
    ) -> Dict[str, torch.Tensor]:
        """Compute forward kinematics for all links."""
        
    @staticmethod
    def get_link_transforms(
        joint_positions: torch.Tensor,
        robot_model: RobotModel,
        link_names: List[str]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Get transforms for specific links."""
```

### JacobianComputation

```python
from isaaclab_mimic.utils import JacobianComputation

class JacobianComputation:
    """Jacobian computation utilities."""
    
    @staticmethod
    def geometric_jacobian(
        joint_positions: torch.Tensor,
        robot_model: RobotModel,
        ee_link: str
    ) -> torch.Tensor:
        """Compute geometric Jacobian."""
        
    @staticmethod
    def analytical_jacobian(
        joint_positions: torch.Tensor,
        robot_model: RobotModel,
        ee_link: str,
        reference_frame: str = "world"
    ) -> torch.Tensor:
        """Compute analytical Jacobian."""
```

## Real Robot Interfaces

### ROSRobotsInterface

```python
from isaaclab_mimic.utils.robots import ROSRobotsInterface

class ROSRobotsInterface:
    """ROS-based interface for real robots."""
    
    def __init__(
        self,
        robot_name: str,
        control_topic: str = "/joint_group_position_controller/command",
        state_topic: str = "/joint_states"
    ):
        """Initialize ROS interface."""
        
    def connect(self) -> bool:
        """Establish connection to robot."""
        
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state via ROS."""
        
    def send_trajectory(
        self,
        trajectory: JointTrajectory,
        wait: bool = True
    ) -> bool:
        """Send trajectory to robot."""
```

### URRTDEInterface

```python
from isaaclab_mimic.utils.robots import URRTDEInterface

class URRTDEInterface(RobotInterface):
    """UR robot interface using RTDE."""
    
    def __init__(
        self,
        robot_ip: str,
        control_freq: float = 125.0,
        flags: int = RTDEControlInterface.FLAG_VERBOSE
    ):
        """Initialize RTDE interface."""
        
    def servo_j(
        self,
        joint_positions: np.ndarray,
        time: float = 0.002,
        lookahead_time: float = 0.1,
        gain: int = 300
    ) -> None:
        """Servo to joint positions."""
        
    def move_l(
        self,
        pose: np.ndarray,
        speed: float = 0.25,
        acceleration: float = 1.2
    ) -> None:
        """Linear move in Cartesian space."""
```

## Usage Examples

### Basic Robot Control

```python
# Create robot configuration
robot_cfg = UR5_GRIPPER_CFG.copy()

# Initialize in environment
robot = Articulation(robot_cfg)

# Get robot state
joint_pos = robot.data.joint_pos
ee_pos, ee_quat = robot.data.ee_state

# Send commands
target_joints = compute_target_joints(...)
robot.set_joint_position_target(target_joints)
```

### IK Example

```python
# Initialize IK solver
ik_solver = CuRoboIKSolver(
    robot_cfg=ur5_curobo_cfg,
    world_cfg=world_collision_cfg
)

# Solve IK
target_pos = torch.tensor([[0.5, 0.0, 0.3]])
target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
current_joints = robot.data.joint_pos

solution, success = ik_solver.solve_batch(
    target_pos, 
    target_quat,
    current_joints
)

if success[0]:
    robot.set_joint_position_target(solution[0])
```

### Real Robot Deployment

```python
# Connect to real UR5
real_robot = URRTDEInterface(
    robot_ip="192.168.1.10",
    control_freq=125.0
)

# Control loop
while True:
    # Get state
    state = real_robot.get_robot_state()
    
    # Compute action
    action = policy(state)
    
    # Send command
    real_robot.servo_j(action)
```