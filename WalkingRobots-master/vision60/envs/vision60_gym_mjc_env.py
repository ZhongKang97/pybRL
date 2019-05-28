import numpy as np
import math
import gym
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import pdb
import envs.walking_controller as walking_controller
import time

class Stoch2Env(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        cwd = os.getcwd()
        xml_path = cwd + '/envs/stoch2/stoch_two_urdf/stoch2.xml'

        self._theta = 0
        self._theta0 = 0
        self._update_action_every = 1.  # update is every 50% of the step i.e., theta goes from 0 to pi/2
        self._frequency = 1
        self._action_dim = 10
        self._kp = 8.
        self._kd = 0.2
        self._actuator_indices = [7,8,10,12,14,16,17,19,21,23]

        self._xpos_previous = 0
        self._walkcon = walking_controller.WalkingController(gait_type='trot',
                                                             spine_enable = False,
                                                             left_to_right_switch = False,
                                                             frequency=self._frequency)
        
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path,1) # 2 is the frame_skip here

    def step(self, action, callback=None):

#         xposbefore = self.sim.data.qpos[0]
        # print(xposbefore)
        energy_spent_per_step = self.do_simulation(action, self.frame_skip, callback=callback)
        
#         xposafter = self.sim.data.qpos[0]
#         zposafter = self.sim.data.qpos[2]
        
#         spine_angle = self.sim.data.qpos[[7,16]]
#         fl_leg = self.sim.data.qpos[[8,10]]
#         fr_leg = self.sim.data.qpos[[12,14]]
#         bl_leg = self.sim.data.qpos[[17,19]]
#         br_leg = self.sim.data.qpos[[21,23]]
        
#         spine_vel = self.sim.data.qvel[[7,16]]
#         fl_leg_vel = self.sim.data.qvel[[8,10]]
#         fr_leg_vel = self.sim.data.qvel[[12,14]]
#         bl_leg_vel = self.sim.data.qvel[[17,19]]
#         br_leg_vel = self.sim.data.qvel[[21,23]]
        
#         spine_act = self.sim.data.actuator_force[[0,5]]
#         fl_leg_act = self.sim.data.actuator_force[[1,2]]
#         fr_leg_act = self.sim.data.actuator_force[[3,4]]
#         bl_leg_act = self.sim.data.actuator_force[[6,7]]
#         br_leg_act = self.sim.data.actuator_force[[8,9]]
        
#         print('Spine',spine_angle,spine_vel)
#         print('Front left leg',fl_leg,fl_leg_vel)
#         print('Front right leg',fr_leg,fr_leg_vel)
#         print('Back left leg',bl_leg,bl_leg_vel)
#         print('Back right leg',br_leg,br_leg_vel)
#         print('Spine torque',spine_act)
#         print('Front left torque',fl_leg_act)
#         print('Front right torque',fr_leg_act)
#         print('Back left torque',bl_leg_act)
#         print('Back right torque',br_leg_act)
#         print(self.sim.data.qpos[[7,8,10,12,14,16,17,19,21,23]] - self.sim.data.ctrl)
        
        ob = self._get_obs()
        done, penalty = self.episode_done(ob)
#         reward_run = (xposafter - xposbefore)
#         energy_spent = np.dot(self.sim.data.actuator_force,self.sim.data.qvel[[7,8,10,12,14,16,17,19,21,23]]) * self.dt
        
        ## calculate reward here
#         reward = reward_run - penalty - 2 * energy_spent**2
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        actuator_force = self.sim.data.actuator_force
        reward = self._get_reward(penalty,energy_spent_per_step,qpos,qvel,actuator_force)
        
        if done:
            self.reset_model()

        return ob, reward, done, dict(reward_run=reward, reward_ctrl=-penalty)

    def _get_reward(self,penalty,energy_spent_per_step,qpos,qvel,actuator_force):
        xpos = qpos[0]
        xvel = qvel[0]
        zpos = qpos[2]
        
        distance_travelled = xpos - self._xpos_previous
        self._xpos_previous = xpos

        walking_velocity_reward = 10 * np.exp(-10*(0.6 - xvel)**2)
        walking_height_reward = 0.5 * np.exp(-10*(-0.04 - zpos)**2)
#         print(walking_height_reward)
        
        reward = distance_travelled - penalty - 0.01 * energy_spent_per_step ** 2 + walking_height_reward# + walking_velocity_reward
        
        return reward
    
    def reset_model(self):
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel #+ self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()
   
    def _get_obs(self):
#         return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        return self.sim.data.qpos[0:7]
    
    def episode_done(self,ob):
        done = False
        penalty = 0

        # Convert from Quarternion to Euler
        quat = ob[3:7]
        theta_x0,theta_y0,theta_z0 = self.quat2eul(self.init_qpos[3:7])
        theta_x,theta_y,theta_z = self.quat2eul(quat)

        ang_lim = 0.5 # angle threshold in radians
        ANGLE_COND = (abs(abs(theta_y)-abs(theta_y0))<=ang_lim) and (abs(abs(theta_x)-abs(theta_x0))<=ang_lim) and (abs(abs(theta_z)-abs(theta_z0))<=ang_lim)
        HEIGHT_COND = (ob[2]>=-0.1)
        # print(ob[2])
        # Orientation Penalty
        penalty = penalty + 0 #1*(0.7+(abs(abs(theta_y)-abs(theta_y0)) + abs(abs(theta_x)-abs(theta_x0))+abs(abs(theta_z)-abs(theta_z0))))**3
        #print("Angle Penalty: ",penalty)
        # Action penalty: Prefer lesser energy
        penalty = penalty + 0 #0.01*np.abs( np.dot(self.sim.data.actuator_force[:],self.sim.data.actuator_velocity[:]) )
        #print("Power Penalty: ",np.abs( np.dot(self.sim.data.actuator_force[:],self.sim.data.actuator_velocity[:]) ))

        if not(ANGLE_COND and HEIGHT_COND):
            done = True
#             print("Oops...fail!","Ang condition",ANGLE_COND,"Height condition",HEIGHT_COND)
#             print('Height',ob[2])
            penalty = penalty + 3   #Penalty for the fall or orientation

        return (done,penalty)

    def quat2eul(self,quat):
        theta_z=np.arctan2(2*(quat[0]*quat[3]-quat[1]*quat[2]),
                    np.square(quat[3])-np.square(quat[0]) - \
                        np.square(quat[1])+np.square(quat[2]))
        theta_y=np.arcsin(2*(quat[0]*quat[2]+quat[1]*quat[3]))
        theta_x =np.arctan2(2*(quat[2]*quat[3]-quat[0]*quat[1]),
                    np.square(quat[3])+np.square(quat[0]) - \
                        np.square(quat[1])-np.square(quat[2]))
        return (theta_x,theta_y,theta_z)

    def rotationMatrixToEulerAngles(R) :
        #assert(isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            theta_x = math.atan2(R[2,1] , R[2,2])
            theta_y = math.atan2(-R[2,0], sy)
            theta_z = math.atan2(R[1,0], R[0,0])
        else :
            theta_x = math.atan2(-R[1,2], R[1,1])
            theta_y = math.atan2(-R[2,0], sy)
            theta_z = 0

        return np.array([theta_x, theta_y, theta_z])
    
    def do_simulation(self, action, n_frames, callback=None):
        omega = 2 * math.pi * self._frequency
        self._theta = self._theta0
        p_index = 0
        energy_spent_per_step = 0

        while(self._theta - self._theta0 <= math.pi * self._update_action_every and not self._theta >= 2 * math.pi):

            theta = self._theta
            
            spine_des, leg_m_angle_cmd, d_spine_des, leg_m_vel_cmd = self._walkcon.transform_action_to_motor_joint_command(theta,action)
            self._theta = (omega * self.dt + theta)
            
            qpos_act = self.sim.data.qpos[self._actuator_indices]
            qvel_act = self.sim.data.qvel[self._actuator_indices]
            
#             if  p_index==0:
#                 print(theta,qpos_act)
                
#             p_index = (p_index + 1)% 5
            
            m_angle_cmd_ext = np.zeros(10)
            m_angle_cmd_ext[[1,2,3,4,6,7,8,9]] = leg_m_angle_cmd
            m_angle_cmd_ext[[0,5]] = spine_des

            m_vel_cmd_ext = np.zeros(10)
            m_vel_cmd_ext[[1,2,3,4,6,7,8,9]] = leg_m_vel_cmd
            m_vel_cmd_ext[[0,5]] = d_spine_des

            self.sim.data.ctrl[:] = self._kp * (m_angle_cmd_ext - qpos_act) + self._kd * (m_vel_cmd_ext - qvel_act)
#             self.sim.data.ctrl[:] = m_angle_cmd_ext

            for _ in range(n_frames):
                self.sim.step()

                if callback is not None:
                    if callback(self) is False:
                        break
            
            energy_spent = np.dot(self.sim.data.actuator_force,self.sim.data.qvel[self._actuator_indices]) * self.dt
            energy_spent_per_step = energy_spent_per_step + energy_spent
            
        self._theta0 = self._theta % (2* math.pi)
        return energy_spent_per_step


    def viewer_setup(self):
        #self.viewer.cam.distance = self.model.stat.extent * 0.5
        id = 1
        self.viewer.cam.trackbodyid = id
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[id] = 1.15
        self.viewer.cam.elevation = -10
