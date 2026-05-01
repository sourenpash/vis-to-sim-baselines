# Sim-to-Real Transfer

## Overview

Sim-to-real transfer in UniMimic involves:
- Robot calibration and system identification
- Camera calibration and alignment
- Real-time control interfaces
- Domain adaptation techniques
- Performance validation

## System Setup

### Hardware Requirements

- **Robots**: UR5, ARX-X7, or compatible manipulators
- **Cameras**: Intel RealSense D435/D455, Azure Kinect
- **Compute**: NVIDIA GPU (RTX 3070 or better)
- **Network**: Low-latency connection for real-time control

### Software Stack

```bash
# Install real robot dependencies
pip install pyrealsense2
pip install roboticstoolbox-python
pip install ur-rtde  # For UR robots
```

## Robot Calibration

### 1. Kinematic Calibration

Calibrate robot kinematics for accurate control:

```python
from scripts.calibrate_tools import RobotCalibrator

calibrator = RobotCalibrator(
    robot_type="ur5",
    calibration_poses=50
)

# Collect calibration data
calibration_data = calibrator.collect_data()

# Compute calibration
dh_params, base_transform = calibrator.calibrate(calibration_data)

# Save calibration
calibrator.save_calibration("configs/robot_calib.yaml")
```

### 2. Joint Limits Verification

```python
# Verify joint limits match simulation
real_limits = robot.get_joint_limits()
sim_limits = sim_robot_cfg.joint_limits

for i, (real, sim) in enumerate(zip(real_limits, sim_limits)):
    if not np.allclose(real, sim, rtol=0.01):
        print(f"Joint {i} limit mismatch: real={real}, sim={sim}")
```

## Camera Calibration

### 1. Intrinsic Calibration

```python
from ppt_learning.utils.camera import CameraCalibrator

# Calibrate camera intrinsics
calibrator = CameraCalibrator()
intrinsics = calibrator.calibrate_intrinsics(
    checkerboard_size=(9, 6),
    square_size=0.025  # meters
)
```

### 2. Hand-Eye Calibration

Align camera with robot coordinate frame:

```python
from scripts.calibrate_tools import HandEyeCalibrator

hand_eye = HandEyeCalibrator(
    robot=robot,
    camera=camera,
    calibration_type="eye_to_hand"  # or "eye_in_hand"
)

# Collect calibration poses
poses = hand_eye.collect_calibration_poses(n_poses=20)

# Compute transformation
camera_to_base = hand_eye.calibrate(poses)

# Verify calibration
error = hand_eye.verify_calibration(test_poses=10)
print(f"Calibration error: {error:.3f} mm")
```

### 3. Multi-Camera Setup

```python
from ppt_learning.utils.camera import MultiCameraSystem

# Configure multiple cameras
cameras = MultiCameraSystem([
    {"name": "front", "serial": "123456", "transform": front_transform},
    {"name": "side", "serial": "789012", "transform": side_transform}
])

# Synchronize cameras
cameras.set_sync_mode(master="front")
```

## Real Robot Interface

### 1. UR5 Robot

```python
from isaaclab_mimic.utils.robots import ROSRobotsInterface

# Initialize UR5 interface
robot = ROSRobotsInterface(
    robot_ip="192.168.1.10",
    control_freq=125,  # Hz
    use_rtde=True
)

# Basic control loop
while True:
    # Get current state
    joint_pos = robot.get_joint_positions()
    joint_vel = robot.get_joint_velocities()
    
    # Compute action (from policy)
    action = policy.predict(observation)
    
    # Send command
    robot.send_joint_command(action)
```

### 2. ARX-X7 Robot

```python
from isaaclab_mimic.utils.robots import ARX7Interface

robot = ARX7Interface(
    can_interface="can0",
    control_mode="position"
)

# Enable motors
robot.enable_motors()

# Control with WBC
wbc = WBCController(robot_cfg)
while True:
    joint_cmd = wbc.compute(current_state, task_target)
    robot.send_command(joint_cmd)
```

## Policy Deployment

### 1. Load Trained Model

```python
from ppt_learning.real import RealRobot

# Load checkpoint
model = torch.jit.load("outputs/model.jit")
model.eval()

# Initialize normalizers
obs_normalizer = Normalizer.load("outputs/obs_norm.pkl")
action_normalizer = Normalizer.load("outputs/action_norm.pkl")
```

### 2. Real-Time Inference

```python
class RealTimePolicy:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.action_history = deque(maxlen=10)
        
    @torch.no_grad()
    def predict(self, observation):
        # Preprocess observation
        obs_tensor = self.preprocess(observation)
        
        # Add history if needed
        if self.use_history:
            obs_tensor = self.add_history(obs_tensor)
        
        # Inference
        action = self.model(obs_tensor)
        
        # Post-process
        action = self.postprocess(action)
        
        # Safety checks
        action = self.safety_filter(action)
        
        return action
```

### 3. Observation Processing

```python
def process_real_observation(cameras, robot):
    obs = {}
    
    # Robot proprioception
    obs["joint_pos"] = robot.get_joint_positions()
    obs["joint_vel"] = robot.get_joint_velocities()
    obs["ee_pos"] = robot.get_ee_position()
    obs["ee_quat"] = robot.get_ee_orientation()
    
    # Visual observations
    if "rgb" in required_modalities:
        obs["rgb"] = cameras.get_rgb_images()
        
    if "depth" in required_modalities:
        obs["depth"] = cameras.get_depth_images()
        
    if "pointcloud" in required_modalities:
        obs["pointcloud"] = process_pointcloud(
            cameras.get_pointclouds(),
            voxel_size=0.005
        )
    
    return obs
```

## Domain Adaptation

### 1. System Identification

Match simulation parameters to real robot:

```python
# Measure real robot parameters
friction_coeffs = measure_joint_friction(robot)
damping_coeffs = measure_joint_damping(robot)
payload_mass = measure_payload(robot)

# Update simulation
sim_config.update({
    "joint_friction": friction_coeffs,
    "joint_damping": damping_coeffs,
    "payload_mass": payload_mass
})
```

### 2. Visual Domain Randomization

Apply domain randomization during training:

```yaml
domain_randomization:
  # Lighting
  light_intensity: [0.5, 1.5]
  light_color_temp: [4000, 6500]
  
  # Camera
  camera_noise: 0.01
  camera_position_noise: 0.02
  
  # Materials
  texture_randomization: true
  color_jitter: 0.2
```

### 3. Action Filtering

Smooth actions for real robot:

```python
class ActionFilter:
    def __init__(self, alpha=0.9, max_delta=0.1):
        self.alpha = alpha
        self.max_delta = max_delta
        self.prev_action = None
        
    def filter(self, action):
        if self.prev_action is None:
            self.prev_action = action
            return action
            
        # Exponential smoothing
        filtered = self.alpha * self.prev_action + (1 - self.alpha) * action
        
        # Limit rate of change
        delta = filtered - self.prev_action
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        filtered = self.prev_action + delta
        
        self.prev_action = filtered
        return filtered
```

## Safety Measures

### 1. Joint Limit Protection

```python
def enforce_joint_limits(action, current_pos, limits, margin=0.1):
    """Prevent joint limit violations."""
    next_pos = current_pos + action * dt
    
    for i, (low, high) in enumerate(limits):
        # Add safety margin
        safe_low = low + margin
        safe_high = high - margin
        
        if next_pos[i] < safe_low:
            action[i] = (safe_low - current_pos[i]) / dt
        elif next_pos[i] > safe_high:
            action[i] = (safe_high - current_pos[i]) / dt
            
    return action
```

### 2. Collision Detection

```python
class SafetyMonitor:
    def __init__(self, robot, workspace_limits):
        self.robot = robot
        self.limits = workspace_limits
        self.force_threshold = 50  # Newtons
        
    def check_safety(self):
        # Check workspace limits
        ee_pos = self.robot.get_ee_position()
        if not self.in_workspace(ee_pos):
            return False, "Outside workspace"
            
        # Check force/torque
        ft_reading = self.robot.get_ft_sensor()
        if np.linalg.norm(ft_reading[:3]) > self.force_threshold:
            return False, "Excessive force"
            
        # Check singularity
        jacobian = self.robot.get_jacobian()
        if np.linalg.det(jacobian @ jacobian.T) < 0.001:
            return False, "Near singularity"
            
        return True, "OK"
```

### 3. Emergency Stop

```python
# Hardware e-stop
robot.register_estop_callback(emergency_stop)

def emergency_stop():
    robot.stop()
    robot.disable_motors()
    logger.error("Emergency stop triggered!")
```

## Deployment Script

Complete deployment example:

```python
# scripts/ppt_learning/real/deploy_policy.py

def main():
    # Initialize hardware
    robot = ROSRobotsInterface(args.robot_ip)
    cameras = MultiCameraSystem(args.camera_config)
    
    # Load model
    policy = load_policy(args.checkpoint)
    
    # Safety systems
    safety = SafetyMonitor(robot, workspace_limits)
    action_filter = ActionFilter()
    
    # Control loop
    rate = Rate(args.control_freq)
    
    while True:
        # Safety check
        safe, msg = safety.check_safety()
        if not safe:
            robot.stop()
            logger.warning(f"Safety violation: {msg}")
            continue
            
        # Get observation
        obs = process_real_observation(cameras, robot)
        
        # Predict action
        action = policy.predict(obs)
        
        # Filter and safety checks
        action = action_filter.filter(action)
        action = enforce_joint_limits(action, robot.joint_pos, robot.limits)
        
        # Execute
        robot.send_joint_command(action)
        
        # Maintain frequency
        rate.sleep()
```

## Validation

### 1. Simulation Validation

First validate in realistic simulation:

```python
# Test with realistic parameters
env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**24
env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
env_cfg.sim.dt = 0.001  # 1kHz simulation

# Add realistic noise
env_cfg.observations.add_noise = True
env_cfg.observations.noise_cfg = {
    "joint_pos": 0.001,
    "joint_vel": 0.01,
    "camera": 0.02
}
```

### 2. Real Robot Testing

Progressive testing strategy:

1. **Static Testing**: Verify sensor readings and calibration
2. **Slow Motion**: Run at 10% speed initially
3. **Limited Workspace**: Restrict to safe region
4. **Full Deployment**: Gradually increase speed and workspace

### 3. Performance Metrics

```python
metrics = {
    "success_rate": success_count / total_trials,
    "avg_completion_time": np.mean(completion_times),
    "position_error": np.mean(position_errors),
    "force_exceeded": force_violations / total_trials,
    "inference_time": np.mean(inference_times)
}
```

## Troubleshooting

### Common Issues

1. **Jerky Motion**
   - Increase action filtering
   - Reduce control frequency
   - Check for observation noise

2. **Policy Failure**
   - Verify observation normalization
   - Check camera calibration
   - Validate joint limits

3. **Communication Lag**
   - Use wired connection
   - Optimize observation processing
   - Reduce image resolution

### Debug Tools

```python
# Enable debug visualization
debug_config = {
    "visualize_observations": True,
    "plot_actions": True,
    "save_trajectories": True,
    "record_video": True
}

# Debug logger
logger = DebugLogger(
    save_path="debug_logs/",
    log_freq=10,
    save_on_error=True
)
```

## Best Practices

- Start simple and add complexity
- Test edge cases in simulation first
- Always have hardware e-stops
- Log all data for debugging
- Track model versions and configurations