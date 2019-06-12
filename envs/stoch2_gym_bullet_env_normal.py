"""This file implements the gym environment of stoch.

"""

# import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0,parentdir)

import sys, os
sys.path.append(os.path.realpath('../..'))
import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybRL.envs.bullet_client as bullet_client
import pybRL.envs.stoch as stoch
import os
import pybullet_data
from pkg_resources import parse_version

NUM_SUBSTEPS = 5
NUM_MOTORS = 10
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 360 
RENDER_WIDTH = 480 

class StochBulletEnv(gym.Env):
  """The gym environment for the stoch.

  It simulates the locomotion of a stoch, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the stoch walks in 1000 steps and penalizes the energy
  expenditure.

  """
  metadata = {
      "render.modes": ["human", "rgb_array"],
      "video.frames_per_second": 60
  }

  def __init__(self,
               urdf_root=pybullet_data.getDataPath(),
               action_repeat=1,
               distance_weight=1.0,
               energy_weight=0.2,
               shake_weight=0.0,
               drift_weight=0.0,
               distance_limit=float("inf"),
               observation_noise_stdev=0.0,
               self_collision_enabled=False,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,#not needed to be true if accurate motor model is enabled (has its own better PD)
               leg_model_enabled=True,
               accurate_motor_model_enabled=True,
               motor_kp=1.0,
               motor_kd=0.02,
               torque_control_enabled=False,
               motor_overheat_protection=True,
               hard_reset=False,
               on_rack=False,
               render=False,
               kd_for_pd_controllers=0.3,
               gait='spine',
               velocity_idx=2,
               max_spine_angle = 10.0,
               env_randomizer=None):
    """Initialize the stoch gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
      action_repeat: The number of simulation steps before actions are applied.
      distance_weight: The weight of the distance term in the reward.
      energy_weight: The weight of the energy term in the reward.
      shake_weight: The weight of the vertical shakiness term in the reward.
      drift_weight: The weight of the sideways drift term in the reward.
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      motor_velocity_limit: The velocity limit of each motor.
      pd_control_enabled: Whether to use PD controller for each motor.
      leg_model_enabled: Whether to use a leg motor to reparameterize the action
        space.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in stoch.py for more
        details.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place the stoch back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place the stoch on rack. This is only used to debug
        the walking gait. In this mode, the stoch's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      kd_for_pd_controllers: kd value for the pd controllers of the motors
      env_randomizer: An EnvRandomizer to randomize the physical properties
        during reset().
    """
    self._time_step = 0.01
    self._action_repeat = action_repeat
    self._num_bullet_solver_iterations = 300
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._observation = []
    self._env_step_counter = 0
    self._is_render = render
    self._last_base_position = [0, 0, 0]
    self._distance_weight = distance_weight
    self._energy_weight = energy_weight
    self._drift_weight = drift_weight
    self._shake_weight = shake_weight
    self._distance_limit = distance_limit
    self._observation_noise_stdev = observation_noise_stdev
    self._action_bound = 1
    self._pd_control_enabled = pd_control_enabled
    self._leg_model_enabled = leg_model_enabled
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._motor_kp = motor_kp
    self._motor_kd = motor_kd
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self._cam_dist = 0.6
    self._cam_yaw = 0.0
    self._cam_pitch = -12.0
    self._hard_reset = True
    self._kd_for_pd_controllers = kd_for_pd_controllers
    self._last_frame_time = 0.0
    print("urdf_root=" + self._urdf_root)
    self._env_randomizer = None
    self._info = {}

    # PD control needs smaller time step for stability.
    if pd_control_enabled or accurate_motor_model_enabled:
      self._time_step /= NUM_SUBSTEPS
      self._num_bullet_solver_iterations /= NUM_SUBSTEPS
      self._action_repeat *= NUM_SUBSTEPS

    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bullet_client.BulletClient()

    # Data logging during inference
    self.cg_list = list()
    self.log_buff = False
    self.c = 0

    self.seed()
    self.reset()

    #State space --- Changing to reduce the obs size
    observation_high = self.stoch.GetObservationUpperBound() + OBSERVATION_EPS
    observation_low =  self.stoch.GetObservationLowerBound() - OBSERVATION_EPS
    # observation_high = np.append(observation_high[0:8],observation_high[24:28])
    # observation_low = np.append(observation_low[0:8],observation_low[24:28])
    observation_high = observation_high[24:28]
    observation_low = observation_low[24:28]
    # Gait selection and correspoding attributes

    self.trot, self.bound, self.spine = False, False, False

    if gait == 'trot':
      self.trot = True
    elif gait == 'bound':
      self.bound = True
    elif gait == 'spine':
      self.spine= True

    velocities = [0.5, 1, 1.5, 2, 2.5] # velocities in (m/s)
    sd = [0.1, 0.2, 0.4, 0.55, 0.7] # correspoding gaussian standard deviation
    self.velocity = velocities[velocity_idx]
    self.vel_sd = sd[velocity_idx]

    self.max_spine_angle = max_spine_angle #maximum bend in spine

    # Action space
    action_dim = 5 
    action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
    self.viewer = None
    self._hard_reset = hard_reset  # This assignment need to be after reset()

  def set_env_randomizer(self, env_randomizer):
    self._env_randomizer = env_randomizer

  def configure(self, args):
    self._args = args

  def reset(self):
    """Reset function is called everytime after an episode ends either cause of failure or
    episode timeout. 
    No args required
    Returns: states

    """
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      plane = self._pybullet_client.loadURDF("%s/plane100.urdf" % self._urdf_root)
      self._pybullet_client.changeVisualShape(plane,-1,rgbaColor=[1,1,1,0.9])
      # self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION,0)
      self._pybullet_client.setGravity(0, 0, -10)
      acc_motor = self._accurate_motor_model_enabled
      motor_protect = self._motor_overheat_protection
      self.stoch = (stoch.Stoch(
          pybullet_client=self._pybullet_client,
          urdf_root=self._urdf_root,
          time_step=self._time_step,
          self_collision_enabled=self._self_collision_enabled,
          motor_velocity_limit=self._motor_velocity_limit,
          pd_control_enabled=self._pd_control_enabled,
          accurate_motor_model_enabled=acc_motor,
          motor_kp=self._motor_kp,
          motor_kd=self._motor_kd,
          torque_control_enabled=self._torque_control_enabled,
          motor_overheat_protection=motor_protect,
          on_rack=self._on_rack,
          kd_for_pd_controllers=self._kd_for_pd_controllers))
    else:
      self.stoch.Reset(reload_urdf=False)

    if self._env_randomizer is not None:
      self._env_randomizer.randomize_env(self)
    if  False:
        self._pybullet_client.createConstraint(self.stoch.quadruped, -1, -1, -1,
                                               self._pybullet_client.JOINT_FIXED, [0, 0, 0],
                                               [0, 0, 0], [0, 0, 1])
    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._objectives = []
    self._pybullet_client.resetDebugVisualizerCamera(
        self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])

    return self._noisy_observation()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _transform_action_to_motor_command(self, action):
    """
    Checks bound on the actions
    Run a inverse kinemantics on the polar coordinates received from the agent
    Converts polar coordinates to motor joint angles (radians)

    Returns:
      Motor joint angles to be excuted by the robot. 

    """
    if self._leg_model_enabled:
      for i, action_component in enumerate(action):
        if not (-self._action_bound - ACTION_EPS <= action_component <=
                self._action_bound + ACTION_EPS):
          raise ValueError(
              "{}th action {} out of bounds.".format(i, action_component))
      action, xpos, ypos = self.stoch.ConvertFromLegModel(action)
      self._info['xpos'] = xpos
      self._info['ypos'] = ypos
    return action

  def step(self, action):
    """Step forward the simulation, given the action.

    Args:
      action: A list of desired motor angles for eight motors.

    Returns:
      observations: The angles, velocities and torques of all motors. Orientation of robot body too.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    if self._is_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self._action_repeat * self._time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)
        # time.sleep(1./240.)
        pass
      base_pos = self.stoch.GetBasePosition()
      camInfo = self._pybullet_client.getDebugVisualizerCamera()
      curTargetPos = camInfo[11]
      distance=camInfo[10]
      yaw = camInfo[8]
      pitch=camInfo[9]
      targetPos = [0.95*curTargetPos[0]+0.05*base_pos[0],0.95*curTargetPos[1]+0.05*base_pos[1],curTargetPos[2]]
           
           
      self._pybullet_client.resetDebugVisualizerCamera(
          distance, yaw, pitch, base_pos)

    action_final = np.zeros(10)

    action_half = self._transform_action_to_motor_command(action[:4])

    if self.trot:
      action_final[:4] = action_half

      action_final[4:6] = action_half[2:]
      action_final[6:8] = action_half[:2]

    elif self.bound:
      action_final[:4] = action_half
      action_final[4:8] = action_half
      
    elif self.spine:

      action_final[:4] = action_half
      action_final[4:8] = action_half
      action_final[8] =   self.max_spine_angle * math.pi/180 * (action[4])
      action_final[9] = -self.max_spine_angle * math.pi/180 * (action[4])

    for _ in range(self._action_repeat):
      self.stoch.ApplyAction(action_final)
      self._pybullet_client.stepSimulation()

    self._env_step_counter += 1
    reward = self._reward()
    done = self._termination()
    return np.array(self._noisy_observation()), reward, done, self._info

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.stoch.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_stoch_motor_angles(self):
    """Get the stoch's motor angles.

    Returns:
      A numpy array of motor angles.
    """
    return np.array(
        self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:
                          MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_stoch_motor_velocities(self):
    """Get the stoch's motor velocities.

    Returns:
      A numpy array of motor velocities.
    """
    return np.array(
        self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:
                          MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS])

  def get_stoch_motor_torques(self):
    """Get the stoch's motor torques.

    Returns:
      A numpy array of motor torques.
    """
    return np.array(
        self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:
                          MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_stoch_base_orientation(self):
    """Get the stoch's base orientation, represented by a quaternion.

    Returns:
      A numpy array of stoch's orientation.
    """
    return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

  def is_fallen(self):
    """Decide whether the stoch has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.16 meter), the stoch is considered fallen.

    Returns:
      Boolean value that indicates whether the stoch has fallen.
    """
    orientation = self.stoch.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
    pos = self.stoch.GetBasePosition()

    if self.log_buff:  # sstores position and velocity of COM during testing
      obs = list(pos)
      obs.extend(list(self.stoch.GetBaseVelocity()))
      self.cg_list.append(obs)
      self.c+=1
      print("COM ",self.c)
      if self.c==900:
        np.savetxt("/home/abhik/batch-ppo/logdir/roman/com.txt", self.cg_list)
        print("data saved")


    return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or
            pos[2] < 0.16) 

  def _termination(self):

    position = self.stoch.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    return self.is_fallen() or distance > self._distance_limit

  def gauss(self, x, mean, sd):

    gauss = math.e**(-(float(x-mean)/sd)**2)  
    return gauss

  def _reward(self):
    """
    Calculates reward at every time step based on states and actions

    Reward comprises of a linear combination of robot base velocity, drift, shake and energy penalty

    """
    current_base_position = self.stoch.GetBasePosition()
    base_vel = self.stoch.GetBaseVelocity()

    ## higher the reward higher the speed
    forward_reward = current_base_position[0] - self._last_base_position[0]

    ## gives higher reward as reaching a particular value of velocity
    # forward_reward = self.gauss(base_vel[0], self.velocity, self.vel_sd)

    drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
    self._last_base_position = current_base_position
    energy_reward = np.abs(
        np.dot(self.stoch.GetMotorTorques(),
               self.stoch.GetMotorVelocities())) * self._time_step

    reward = (
        self._distance_weight * forward_reward -
        self._energy_weight * energy_reward + self._drift_weight * drift_reward
        + self._shake_weight * shake_reward)
    self._objectives.append(
        [forward_reward, energy_reward, drift_reward, shake_reward])
    return reward

  def get_objectives(self):
    return self._objectives

  def _get_observation(self):
    self._observation = self.stoch.GetObservation()
    self._info['angles'] = self._observation[0:8]
    self._info['vel'] = self._observation[8:16]
    self._info['torques'] = self._observation[16:24]
    #Only uses orientation of the robot
    return self._observation[24:28]

  def _noisy_observation(self):
    observation = np.array(self._get_observation())
    if self._observation_noise_stdev > 0:
      observation += (np.random.normal(
          scale=self._observation_noise_stdev, size=observation.shape) *
                      self.stoch.GetObservationUpperBound())
    return observation

  # if parse_version(gym.__version__)>=parse_version('0.9.6'):
  #   render = _render
  #   reset = _reset
  #   seed = _seed
  #   step = _step

if(__name__ == "__main__"):
  env = StochBulletEnv(render=True)
  env.reset()
  i = 0
  while(True):
    pass
  pass