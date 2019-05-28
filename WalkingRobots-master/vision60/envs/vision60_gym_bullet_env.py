import numpy as np
import math
import gym
import os
from gym import utils, spaces
import pdb
import envs.walking_controller as walking_controller
import time

import pybullet
import bullet_client
import pybullet_data

INIT_POSITION = [0, 0, 0.5] 
INIT_ORIENTATION = [0, 0, 0, 1]
LEG_POSITION = ["0", "1", "2", "3"]
RENDER_HEIGHT = 720 #360
RENDER_WIDTH = 960 #480 

class Vision60BulletEnv(gym.Env):
    
    def __init__(self,
                 render = False,
                 on_rack = False):
        
        self._is_render = render
        self._on_rack = on_rack
        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        
        self._theta = 0
        self._theta0 = 0
        self._update_action_every = 1.  # update is every 50% of the step i.e., theta goes from 0 to pi/2
        self._frequency = 2.
        self._kp = 50
        self._kd = 5
        self.dt = 0.01
        self._frame_skip = 10
        self._n_steps = 0
        
        self._action_dim = 12
        self._obs_dim = 7
        
        self._last_base_position = [0, 0, 0]
        self._distance_limit = float("inf")

        self._walkcon = walking_controller.WalkingController(gait_type='trot',
                                                             spine_enable = False,
                                                             left_to_right_switch = True,
                                                             planning_space = 'joint_space',
                                                             frequency=self._frequency)
        
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0
    
        ## Gym env related mandatory variables
        observation_high = np.array([10.0] * self._obs_dim)
        observation_low = -observation_high
        self.observation_space = spaces.Box(observation_low, observation_high)
        
        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        
        self.hard_reset()
    
    def hard_reset(self):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
        self._pybullet_client.setTimeStep(self.dt/self._frame_skip)
        
        plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self._pybullet_client.changeVisualShape(plane,-1,rgbaColor=[1,1,1,0.9])
        self._pybullet_client.setGravity(0, 0, -9.8)
        
        cwd = os.getcwd()
        urdf_path = cwd + '/envs/vision60/vision60_v2.5.urdf'
        print(urdf_path)
        self.stoch2 = self._pybullet_client.loadURDF(urdf_path, INIT_POSITION)
        
        self._joint_name_to_id, self._motor_id_list = self.BuildMotorIdList()

        num_legs = 4
        for i in range(num_legs):
            self.ResetLeg(i, add_constraint=True)

        if self._on_rack:
            self._pybullet_client.createConstraint(
                self.stoch2, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, 0.5])
            
        self._pybullet_client.resetBasePositionAndOrientation(self.stoch2, INIT_POSITION, INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])
      
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        
    def reset(self):
        self._pybullet_client.resetBasePositionAndOrientation(self.stoch2, INIT_POSITION, INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])
      
        num_legs = 4
        for i in range(num_legs):
            self.ResetLeg(i, add_constraint=False)

        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._n_steps = 0
              
        return self.GetObservation()
    
    def step(self, action, callback=None):
        energy_spent_per_step = self.do_simulation(action, n_frames = self._frame_skip, callback=self._render)

        ob = self.GetObservation()
        ## calculate reward here
        reward,done,penalty = self._get_reward(energy_spent_per_step)
        
        if done:
            self.reset()

        return ob, reward, done, dict(reward_run=reward, reward_ctrl=-penalty)
    
    def do_simulation(self, action, n_frames, callback=None):
        omega = 2 * math.pi * self._frequency
        self._theta = self._theta0
#         p_index = 0
        energy_spent_per_step = 0
        
        while(self._theta - self._theta0 <= math.pi * self._update_action_every and not self._theta >= 2 * math.pi):

            theta = self._theta
            
            spine_des, leg_m_angle_cmd, d_spine_des, leg_m_vel_cmd = self._walkcon.transform_action_to_motor_joint_command(theta,action)
            self._theta = (omega * self.dt + theta)
            
            
#             if  p_index==0:
#                 print(theta,qpos_act)
                
#             p_index = (p_index + 1)% 5
            
            m_angle_cmd_ext = np.zeros(12)
            m_angle_cmd_ext[[0,3,6,9]] = np.zeros(4)
            m_angle_cmd_ext[[1,4,7,10]] = 0*leg_m_angle_cmd[[0,2,4,6]] + 0
            m_angle_cmd_ext[[2,5,8,11]] = 0*leg_m_angle_cmd[[1,3,5,7]] - 0
            
            m_vel_cmd_ext = np.zeros(12)
            m_vel_cmd_ext[[0,3,6,9]] = np.zeros(4)
            m_vel_cmd_ext[[1,4,7,10]] = 0*leg_m_vel_cmd[[0,2,4,6]]
            m_vel_cmd_ext[[2,5,8,11]] = 0*leg_m_vel_cmd[[1,3,5,7]]
            
            for _ in range(n_frames):
                self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
                self._pybullet_client.stepSimulation()
  
            if callback is not None and self._is_render:
                if callback(mode="rgb_array", close=False) is False:
                    break
                    
            energy_spent = np.abs(np.dot(self.GetMotorTorques(),self.GetMotorVelocities())) * self.dt
            energy_spent_per_step = energy_spent_per_step + energy_spent

        self._theta0 = self._theta % (2* math.pi)
        self._n_steps += 1
        return energy_spent_per_step
  
    def _render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        
        base_pos, _ = self.GetBasePosAndOrientation()
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

    def _termination(self, pos, orientation):
        done = False
        penalty = 0
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]

        # stop episode after ten steps
        if self._n_steps >= 100:
            done = True
            print('%s steps finished. Terminated' % self._n_steps)
            penalty = 0
        else:
            if np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.3:
                print('Oops, Robot about to fall! Terminated')
                done = True
                penalty = penalty + 0.1
            if pos[2] < 0.05:
                print('Robot was too low! Terminated')
                done = True
                penalty = penalty + 1
            if pos[2] > 0.6:
                print('Robot was too high! Terminated')
                done = True
                penalty = penalty + 1

        if done and self._n_steps <= 3:
            penalty = 3
            
        return done, penalty

    def _get_reward(self,energy_spent_per_step):
        current_base_position, current_base_orientation = self.GetBasePosAndOrientation()

        forward_reward = current_base_position[0] - self._last_base_position[0] # added negative reward for staying still
        # forward_reward = np.clip(forward_reward, -0.1, 0.1)

#         walking_velocity_reward = 10 * np.exp(-10*(0.6 - xvel)**2)
        walking_height_reward = 0.5 * np.exp(-10*(-0.04 - current_base_position[2])**2)

        done, penalty = self._termination(current_base_position, current_base_orientation)
        reward = (forward_reward - 0.01 * energy_spent_per_step - penalty) 
        
        return reward, done, penalty

    def _apply_pd_control(self, motor_commands, motor_vel_commands):
        qpos_act = self.GetMotorAngles()
        qvel_act = self.GetMotorVelocities()
        print(qpos_act)
        applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)

        for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
            self.SetMotorTorqueById(motor_id, motor_torque)

        return applied_motor_torque
    
    def GetObservation(self):
        observation = []
        pos, ori = self.GetBasePosAndOrientation()
#         observation.extend(list(pos))
#         observation.extend(self.GetMotorAngles().tolist())
#         observation.extend(self.GetMotorVelocities().tolist())
        return np.concatenate([pos,ori]).ravel()

    def GetMotorAngles(self):
        motor_ang = [self._pybullet_client.getJointState(self.stoch2, motor_id)[0] for motor_id in self._motor_id_list]
        return motor_ang

    def GetMotorVelocities(self):
        motor_vel = [self._pybullet_client.getJointState(self.stoch2, motor_id)[1] for motor_id in self._motor_id_list]
        return motor_vel

    def GetMotorTorques(self):
        motor_torq = [self._pybullet_client.getJointState(self.stoch2, motor_id)[3] for motor_id in self._motor_id_list]
        return motor_torq
    
    def GetBasePosAndOrientation(self):
        position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.stoch2))
        return position, orientation

    def SetMotorTorqueById(self, motor_id, torque):
        self._pybullet_client.setJointMotorControl2(
                  bodyIndex=self.stoch2,
                  jointIndex=motor_id,
                  controlMode=self._pybullet_client.TORQUE_CONTROL,
                  force=torque)

    def BuildMotorIdList(self):
        num_joints = self._pybullet_client.getNumJoints(self.stoch2)
        joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.stoch2, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        
        MOTOR_NAMES = [ "hipRoll0",
                        "hipPitch0", 
                        "knee0",
                        "hipRoll1",
                        "hipPitch1", 
                        "knee1",
                        "hipRoll2",
                        "hipPitch2", 
                        "knee2",
                        "hipRoll3",
                        "hipPitch3", 
                        "knee3"]
        motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]
        return joint_name_to_id, motor_id_list
    
    def ResetLeg(self, leg_id, add_constraint=False):
        leg_position = LEG_POSITION[leg_id]
        self._pybullet_client.resetJointState(
                  self.stoch2,
                  self._joint_name_to_id["hipRoll" + leg_position], # motor
                  targetValue = 0, targetVelocity=0)
        self._pybullet_client.resetJointState(
                  self.stoch2,
                  self._joint_name_to_id["hipPitch" + leg_position],
                  targetValue = 0, targetVelocity=0)
        self._pybullet_client.resetJointState(
                  self.stoch2,
                  self._joint_name_to_id["knee" + leg_position], # motor
                  targetValue = 0, targetVelocity=0)
        
        # set the motors to zero
        self._pybullet_client.setJointMotorControl2(
                              bodyIndex=self.stoch2,
                              jointIndex=(self._joint_name_to_id["hipRoll" + leg_position]),
                              controlMode=self._pybullet_client.VELOCITY_CONTROL,
                              targetVelocity=0,
                              force=0)
        self._pybullet_client.setJointMotorControl2(
                              bodyIndex=self.stoch2,
                              jointIndex=(self._joint_name_to_id["hipPitch" + leg_position]),
                              controlMode=self._pybullet_client.VELOCITY_CONTROL,
                              targetVelocity=0,
                              force=0)
        self._pybullet_client.setJointMotorControl2(
                              bodyIndex=self.stoch2,
                              jointIndex=(self._joint_name_to_id["knee" + leg_position]),
                              controlMode=self._pybullet_client.VELOCITY_CONTROL,
                              targetVelocity=0,
                              force=0)
