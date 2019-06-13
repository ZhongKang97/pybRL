"""This file implements the functionalities of a minitaur using pybullet.

"""
import copy
import math
import numpy as np
from . import motor
import os
import time

INIT_POSITION = [0, 0, 0.267] # 0.29
INIT_ORIENTATION = [0, 0, 0, 1]
# KNEE_CONSTRAINT_POINT_RIGHT = [0.012,0.0,0.043]
# KNEE_CONSTRAINT_POINT_LEFT = [0.,0.0,-0.06]

# KNEE_CONSTRAINT_POINT_RIGHT = [-0.076, 0, 0.014] #hip
# KNEE_CONSTRAINT_POINT_LEFT = [-0.077,0.0,0.0] #knee

KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.076] #hip
KNEE_CONSTRAINT_POINT_LEFT = [0.0,0.0,-0.077] #knee

OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
# LEG_POSITION_STOCH = ["front_left", "back_left", "front_right", "back_right"]
LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]

# MOTOR_NAMES = [
#     "motor_front_leftL_joint", "motor_front_leftR_joint",
#     "motor_back_leftL_joint", "motor_back_leftR_joint",
#     "motor_front_rightL_joint", "motor_front_rightR_joint",
#     "motor_back_rightL_joint", "motor_back_rightR_joint"
# ]

MOTOR_NAMES = [
    # "motor_front_body_spine_joint", "motor_back_body_spine_joint",
    "motor_fl_upper_knee_joint", "motor_fl_upper_hip_joint",
    "motor_bl_upper_knee_joint", "motor_bl_upper_hip_joint",
    "motor_fr_upper_knee_joint", "motor_fr_upper_hip_joint",
    "motor_br_upper_knee_joint", "motor_br_upper_hip_joint",
    "motor_front_body_spine_joint", "motor_back_body_spine_joint"]

    # "motor_front_body_spine_joint", "motor_back_body_spine_joint"
# ]
# LEG_LINK_ID = [2, 3, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 21, 22, 24, 25]
# MOTOR_LINK_ID = [1, 4, 7, 10, 14, 17, 20, 23]
# FOOT_LINK_ID = [3, 6, 9, 12, 16, 19, 22, 25]
# BASE_LINK_ID = -1


class Minitaur(object):
  """The minitaur class that simulates a quadruped robot from Ghost Robotics.

  """

  def __init__(self,
               pybullet_client,
               urdf_root= os.path.join(os.path.dirname(__file__),"../data"),
               time_step=0.01,
               self_collision_enabled=False,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               accurate_motor_model_enabled=False,
               motor_kp=1.0,
               motor_kd=0.02,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               on_rack=False,
               kd_for_pd_controllers=0.3):
    """Constructs a minitaur and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
      urdf_root: The path to the urdf folder.
      time_step: The time step of the simulation.
      self_collision_enabled: Whether to enable self collision.
      motor_velocity_limit: The upper limit of the motor velocity.
      pd_control_enabled: Whether to use PD control for the motors.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_kp: proportional gain for the accurate motor model
      motor_kd: derivative gain for the acurate motor model
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
      kd_for_pd_controllers: kd value for the pd controllers of the motors.
    """
    self.num_motors = 10
    self.num_legs = int(self.num_motors / 2 - 1)
    self._pybullet_client = pybullet_client
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._pd_control_enabled = pd_control_enabled
    self._motor_direction = [1] * self.num_motors
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 3.5
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self.leg = [0.12,0.15015,0.04,0.15501,0.11187,0.04,0.2532,2.803]
    self._width = 960
    self._height = 720
    if self._accurate_motor_model_enabled:
      self._kp = motor_kp
      self._kd = motor_kd
      self._motor_model = motor.MotorModel(
          torque_control_enabled=self._torque_control_enabled,
          kp=self._kp,
          kd=self._kd)
    elif self._pd_control_enabled:
      self._kp = 8
      self._kd = kd_for_pd_controllers
    else:
      self._kp = 1
      self._kd = 1
    self.time_step = time_step
    self.obs_list = list()
    self.act_list = list()
    self.timestamp = list()
    self.log_buff = False
    self.c = 0
    self.Reset()

  def _RecordMassInfoFromURDF(self):
    self._base_mass_urdf = self._pybullet_client.getDynamicsInfo(
        self.quadruped, BASE_LINK_ID)[0]
    self._leg_masses_urdf = []
    self._leg_masses_urdf.append(
        self._pybullet_client.getDynamicsInfo(self.quadruped, LEG_LINK_ID[0])[
            0])
    self._leg_masses_urdf.append(
        self._pybullet_client.getDynamicsInfo(self.quadruped, MOTOR_LINK_ID[0])[
            0])

  def _BuildJointNameToIdDict(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _BuildMotorIdList(self):
    self._motor_id_list = [
        self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES
    ]

  def Reset(self, reload_urdf=True):
    """Reset the minitaur to its initial states.

    Args:
      reload_urdf: Whether to reload the urdf file. If not, Reset() just place
        the minitaur back to its starting position.
    """
    if reload_urdf:
      
      if self._self_collision_enabled:
        self.quadruped = self._pybullet_client.loadURDF(
            "/home/abhik/stoch_two_urdf/urdf/stoch_two_urdf.urdf",
            INIT_POSITION,
            flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
      else:
        self.quadruped = self._pybullet_client.loadURDF(
            "/home/abhik/stoch_two_urdf/urdf/stoch_two_urdf.urdf", INIT_POSITION)#, flags=self._pybullet_client.URDF_USE_MATERIAL_COLORS_FROM_MTL)

      self._BuildJointNameToIdDict()
      # print(self._joint_name_to_id)
      self._BuildMotorIdList()
      # print(self._motor_id_list)
      # self._RecordMassInfoFromURDF()
      self.ResetPose(add_constraint=True)
      if self._on_rack:
        self._pybullet_client.createConstraint(
            self.quadruped, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0], [0, 0, 0.5])
    else:
      self._pybullet_client.resetBasePositionAndOrientation(
          self.quadruped, INIT_POSITION, INIT_ORIENTATION)
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0],
                                              [0, 0, 0])
      self.ResetPose(add_constraint=False)

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors

  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.POSITION_CONTROL,
        targetPosition=desired_angle,
        positionGain=self._kp,
        velocityGain=self._kd,
        force=self._max_force)

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name],
                                   desired_angle)

  def ResetPose(self, add_constraint):
    """Reset the pose of the minitaur.

    Args:
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    for i in range(self.num_legs):
      self._ResetPoseForLeg(i, add_constraint)

    self._ResetPoseForSpine()

  def _ResetPoseForSpine(self):

    knee_friction_force = 0.0
    half_pi = 0.0#math.pi / 2.0
    knee_angle = 0.0#-2.1834

    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_front_body_spine_joint"], # motor
        targetValue = 0, targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_back_body_spine_joint"], # motor
        targetValue = 0, targetVelocity=0)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      # Disable the default motor in pybullet.
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_front_body_spine_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_back_body_spine_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)

    else:
      self._SetDesiredMotorAngleByName("motor_front_body_spine_joint", 0)
      self._SetDesiredMotorAngleByName("motor_back_body_spine_joint", 0)

  def _ResetPoseForLeg(self, leg_id, add_constraint):
    """Reset the initial pose for the leg.

    Args:
      leg_id: It should be 0, 1, 2, or 3, which represents the leg at
        front_left, back_left, front_right and back_right.
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    knee_friction_force = 0.0
    half_pi = 0.0#math.pi / 2.0
    knee_angle = 0.0#-2.1834

    leg_position = LEG_POSITION[leg_id]
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "upper_knee_joint"], # motor
        targetValue = 0, targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id[leg_position + "lower_knee_joint"],
        targetValue = 0, targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "upper_hip_joint"], # motor
        targetValue = 0, targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id[leg_position + "lower_hip_joint"],
        targetValue = 0, targetVelocity=0)

    if add_constraint:
      c = self._pybullet_client.createConstraint(
          self.quadruped, self._joint_name_to_id[leg_position + "lower_hip_joint"],
          self.quadruped, self._joint_name_to_id[leg_position + "lower_knee_joint"],
          self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
          KNEE_CONSTRAINT_POINT_RIGHT, KNEE_CONSTRAINT_POINT_LEFT)

      self._pybullet_client.changeConstraint(c, maxForce=200)


    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      # Disable the default motor in pybullet.
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_"
                                             + leg_position + "upper_knee_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_"
                                             + leg_position + "upper_hip_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)

    else:
      self._SetDesiredMotorAngleByName("motor_" + leg_position + "upper_knee_joint", 0)
      self._SetDesiredMotorAngleByName("motor_" + leg_position + "upper_hip_joint", 0)
                                         

    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id[leg_position + "lower_hip_joint"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id[leg_position + "lower_knee_joint"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)

  def GetBaseVelocity(self):
    """Get the position of minitaur's base.

    Returns:
      The position of minitaur's base.
    """
    vel, _ = (
        self._pybullet_client.getBaseVelocity(self.quadruped))
    return vel

  def GetBasePosition(self):
    """Get the position of minitaur's base.

    Returns:
      The position of minitaur's base.
    """
    position, _ = (
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return position

  def GetBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.

    Returns:
      The orientation of minitaur's base.
    """
    _, orientation = (
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return orientation

  def GetActionDimension(self):
    """Get the length of the action list.

    Returns:
      The length of the action list.
    """
    return self.num_motors

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    upper_bound = np.array([0.0] * self.GetObservationDimension())
    upper_bound[0:self.num_motors] = math.pi  # Joint angle.
    upper_bound[self.num_motors:2 * self.num_motors] = (
        motor.MOTOR_SPEED_LIMIT)  # Joint velocity.
    upper_bound[2 * self.num_motors:3 * self.num_motors] = (
        motor.OBSERVED_TORQUE_LIMIT)  # Joint torque.
    upper_bound[3 * self.num_motors:] = 1.0  # Quaternion of base orientation.
    return upper_bound

  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    return -self.GetObservationUpperBound()

  def GetObservationDimension(self):
    """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
    return len(self.GetObservation())

  def GetObservation(self):
    """Get the observations of minitaur.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
    # observation = []
    # observation.extend(self.GetMotorAngles().tolist())
    # observation.extend(self.GetMotorVelocities().tolist())
    # observation.extend(self.GetMotorTorques().tolist())
    # observation.extend(list(self.GetBaseOrientation()))
    # return observation

    carpos,carorn = self._pybullet_client.getBasePositionAndOrientation(self.quadruped)
    carmat = self._pybullet_client.getMatrixFromQuaternion(carorn)
    # ballpos,ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
    invCarPos,invCarOrn = self._pybullet_client.invertTransform(carpos,carorn)
    # ballPosInCar,ballOrnInCar = self._p.multiplyTransforms(invCarPos,invCarOrn,ballpos,ballorn)
    dist0 = 0.3
    dist1 = 1.
    eyePos = [carpos[0]+dist0*carmat[0],carpos[1]+dist0*carmat[3],carpos[2]+dist0*carmat[6]+0.3]
    targetPos = [carpos[0]+dist1*carmat[0],carpos[1]+dist1*carmat[3],carpos[2]+dist1*carmat[6]+0.3]
    up = [carmat[2],carmat[5],carmat[8]]
    viewMat = self._pybullet_client.computeViewMatrix(eyePos,targetPos,up)
    #viewMat = self._p.computeViewMatrixFromYawPitchRoll(carpos,1,0,0,0,2)
    #print("projectionMatrix:")
    #print(self._p.getDebugVisualizerCamera()[3])
    projMatrix = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    img_arr = self._pybullet_client.getCameraImage(width=self._width,height=self._height,viewMatrix=viewMat,projectionMatrix=projMatrix)
    rgb=img_arr[3]
    np_img_arr = np.reshape(rgb, (self._height, self._width))
    self._observation = np_img_arr
    return self._observation

  def ApplyAction(self, motor_commands):
    """Set the desired motor angles to the motors of the minitaur.

    The desired motor angles are clipped based on the maximum allowed velocity.
    If the pd_control_enabled is True, a torque is calculated according to
    the difference between current and desired joint angle, as well as the joint
    velocity. This torque is exerted to the motor. For more information about
    PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.

    Args:
      motor_commands: The eight desired motor angles.
    """
    if self._motor_velocity_limit < np.inf:
      current_motor_angle = self.GetMotorAngles()
      motor_commands_max = (
          current_motor_angle + self.time_step * self._motor_velocity_limit)
      motor_commands_min = (
          current_motor_angle - self.time_step * self._motor_velocity_limit)
      motor_commands = np.clip(motor_commands, motor_commands_min,
                               motor_commands_max)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      q = self.GetMotorAngles()
      qdot = self.GetMotorVelocities()
      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
            motor_commands, q, qdot)
        if self._motor_overheat_protection:
          for i in range(self.num_motors):
            if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] = 0
            if (self._overheat_counter[i] >
                OVERHEAT_SHUTDOWN_TIME / self.time_step):
              self._motor_enabled_list[i] = False

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque,
                                                 self._motor_direction)

        for motor_id, motor_torque, motor_enabled in zip(
            self._motor_id_list, self._applied_motor_torque,
            self._motor_enabled_list):
          if motor_enabled:
            self._SetMotorTorqueById(motor_id, motor_torque)
          else:
            self._SetMotorTorqueById(motor_id, 0)
      else:
        torque_commands = -self._kp * (q - motor_commands) - self._kd * qdot

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = torque_commands

        # Transform into the motor space when applying the torque.
        self._applied_motor_torques = np.multiply(self._observed_motor_torques,
                                                  self._motor_direction)

        for motor_id, motor_torque in zip(self._motor_id_list,
                                          self._applied_motor_torques):
          self._SetMotorTorqueById(motor_id, motor_torque)
    else:
      motor_commands_with_direction = np.multiply(motor_commands,
                                                  self._motor_direction)
      for motor_id, motor_command_with_direction in zip(
          self._motor_id_list, motor_commands_with_direction):
        self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction)

    if self.log_buff:
      self.act_list.append(motor_commands)
      self.obs_list.append(self.GetObservation())
      self.timestamp.append(time.time())
      self.c+=1
      print("obs ", self.c)
      if self.c==4500:
        np.savetxt("/home/abhik/batch-ppo/logdir/bound/action.txt", self.act_list)
        np.savetxt("/home/abhik/batch-ppo/logdir/bound/observation.txt", self.obs_list)
        np.savetxt("/home/abhik/batch-ppo/logdir/bound/timestamp.txt", self.timestamp)
        print("data saved")


  def GetMotorAngles(self):
    """Get the eight motor angles at the current moment.

    Returns:
      Motor angles.
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
        for motor_id in self._motor_id_list
    ]
    motor_angles = np.multiply(motor_angles, self._motor_direction)
    return motor_angles

  def GetMotorVelocities(self):
    """Get the velocity of all eight motors.

    Returns:
      Velocities of all eight motors.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
        for motor_id in self._motor_id_list
    ]
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorTorques(self):
    """Get the amount of torques the motors are exerting.

    Returns:
      Motor torques of all eight motors.
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
          self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
          for motor_id in self._motor_id_list
      ]
      motor_torques = np.multiply(motor_torques, self._motor_direction)
    return motor_torques

  def ConvertFromLegModel(self, actions):
    """Convert the actions that use leg model to the real motor actions.

    Args:
      actions: The theta, phi of the leg model.
    Returns:
      The eight desired motor angles that can be used in ApplyActions().
    """
    # motor_angle = copy.deepcopy(actions)
    # l1 = 0.12
    # l2 = 0.158
    
    # for i in range(2): 
    #   # r_ac = 0.18 + 0.02*(actions[2*i]+1) #[0.16, 0.22]
    #   # the_ac = actions[2*i+1]*math.pi*40/180 # [-60, 60]

    #   r_ac = 0.19
    #   the_ac = -25.0 * math.pi/180

    #   r_ac = 0.15 + 0.045*(actions[2*i]+1) #[0.16, 0.22]
    #   the_ac = (25.0*actions[2*i+1]-10.0)*math.pi/180.0 # [-60, 60]
      
    #   z = (r_ac**2.0 - l1**2.0 - l2**2.0)/(2.0*l1*l2)
    #   z = np.clip(z, -1.0 , 1.0)
    #   phi = math.acos(z)
    #   alpha = math.atan2(l2*math.sin(phi), l1 + l2*math.cos(phi))
      
    #   theta_hip = alpha
    #   theta_knee = phi

    #   alpha_h = theta_hip - the_ac
    #   alpha_k = theta_knee - alpha_h

    #   motor_angle[2*i+1] = (math.pi/4.0 - alpha_h)
    #   motor_angle[2*i] = -20.0*math.pi/180.0 + alpha_k

    motor_angle = copy.deepcopy(actions)
    
    for i in range(2): 
      # # r_ac = 0.18 + 0.02*(actions[2*i]+1) #[0.16, 0.22]
      # # the_ac = actions[2*i+1]*math.pi*40/180 # [-60, 60]

      r_ac = 0.20
      the_ac = -26* math.pi/180

      r_ac = 0.16 + 0.045*(actions[2*i]+1) #[0.16, 0.045] [0.16, 0.24]
      the_ac = (25.0*actions[2*i+1])*math.pi/180.0 # [-60, 60] [-35, 15][23, -27]  [25]

      x =  r_ac*math.sin(the_ac)
      y = -r_ac*math.cos(the_ac)

      knee, hip = self.inverse_stoch(x,y,self.leg)

      motor_angle[2*i] = knee + 1.2217
      motor_angle[2*i+1] = hip + 2.3562
      

    return motor_angle

  def limiter(self, X):
    if abs(X) >1 :
      X = np.sign(X);
    return X

  def inverse_stoch(self, x,y,Leg):

    l1 =    Leg[0]
    l2 =    Leg[1]
    l4 =    Leg[2]
    l5 =    Leg[3]
    le =    Leg[5]
    tq1 =   Leg[6]
    tq2 =   Leg[7]
    delta = Leg[4]
    xb = [[0,0],[0,0]]
    yb = [[0,0],[0,0]]
    phid = [0,0];psi = [0,0]; theta = [0,0]
    R_base = [[0,0],[0.035,0]]
    xb[0] = R_base[0][0];xb[1] = R_base[1][0]
    yb[0] = R_base[0][1];yb[1] = R_base[1][1]
    l3 = math.sqrt((x-xb[0])**2+(y-yb[0])**2)
    theta[0] = math.atan2((y-yb[0]),(x-xb[0]))
    zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
    zeta = self.limiter(zeta)
    phid[0] = math.acos(zeta)
    psi[0] = math.atan2(l2*math.sin(phid[0]),(l1+l2*math.cos(phid[0])))
    q1 = theta[0] - psi[0]
    q2 = q1 + phid[0]
    xm = l1*math.cos(q1)+l2*math.cos(q2);
    ym = l1*math.sin(q1)+l2*math.sin(q2);
    xi = (xm+xb[0])
    yi = (ym+yb[0])
    #print(xi,yi)

    #left_leg
    #xi = xm + xb[0] - delta*math.cos(q2)
    #yi = ym + yb[0] - delta*math.sin(q2)
    xi = xb[0] + l1*math.cos(q1) + 0.04*math.cos(q2-tq1)
    yi = yb[0] + l1*math.sin(q1) + 0.04*math.sin(q2-tq1)
    R = [xi,yi]; #vector of xi and yi
    l6 = math.sqrt(((xi-xb[1])**2+(yi-yb[1])**2))
    theta[1] = math.atan2((yi-yb[1]),(xi-xb[1]))
    Zeta = (l6**2 - l4**2 - l5**2)/(2*l5*l4)
    leg = 'left'
    Zeta = self.limiter(Zeta);
    phid[1] = math.acos(Zeta);
    psi[1] = math.atan2(l5*math.sin(phid[1]),(l4+l5*math.cos(phid[1])))
    q3 = theta[1]+psi[1]
    q4 = q3-phid[1]
    xm = l4*math.cos(q3)+l5*math.cos(q4)+xb[1]
    ym = l4*math.sin(q3)+l5*math.sin(q4)+yb[1]
    if Zeta == 1:
          [q1, q2] = self.inverse_new(xm,ym,delta,Leg)
   
    # return [q1,q2,q3,q4]
    return [q3, q1]

  def inverse_new(self, xm,ym,delta,Leg):

    l1 = Leg[0]
    l2 = Leg[1]-Leg[4]
    l4 = Leg[2]
    l5 = Leg[3]
    delta = Leg[4]
    xb = [[0,0],[0,0]]
    yb = [[0,0],[0,0]]
    phid = [0,0];psi = [0,0]; theta = [0,0]
    R_base = [[1,0],[-1,0]]
    xb[0] = R_base[0][0];xb[1] = R_base[1][0]
    yb[0] = R_base[0][1];yb[1] = R_base[1][1]
    l3 = math.sqrt((xm-xb[0])**2+(ym-yb[0])**2)
    theta[0] = math.atan2((ym-yb[0]),(xm-xb[0]))
    #print theta[0]
    zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
    zeta = self.limiter(zeta)
    phid[0] = math.acos(zeta)
    psi[0] = math.atan2(l2*math.sin(phid[0]),(l1+l2*math.cos(phid[0])))
    q1 = theta[0] + psi[0]
    q2 = q1 - phid[0]
    xm = l1*math.cos(q1)+l2*math.cos(q2);
    ym = l1*math.sin(q1)+l2*math.sin(q2);
    
    return [q1,q2]
    



  def GetBaseMassFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def SetBaseMass(self, base_mass):
    self._pybullet_client.changeDynamics(
        self.quadruped, BASE_LINK_ID, mass=base_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. All four leg_links have the same mass,
    which is leg_masses[0]. All four motors have the same mass, which is
    leg_mass[1].

    Args:
      leg_masses: The leg masses. leg_masses[0] is the mass of the leg link.
        leg_masses[1] is the mass of the motor.
    """
    for link_id in LEG_LINK_ID:
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, mass=leg_masses[0])
    for link_id in MOTOR_LINK_ID:
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, mass=leg_masses[1])

  def SetFootFriction(self, foot_friction):
    """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in FOOT_LINK_ID:
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, lateralFriction=foot_friction)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)
