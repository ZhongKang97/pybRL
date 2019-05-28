import numpy as np
import math
import gym
import os
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import pdb
import envs.walking_controller as walking_controller
import time

class Stoch2Env(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 render=False,
                 on_rack=False,
                 frame_skip=1):
        
        self._is_render = render
        self._on_rack = on_rack
        
        cwd = os.getcwd()
        model_path = cwd + '/envs/stoch2/stoch_two_urdf/stoch2.xml'

        self.FLx=np.array([-0.0939,-0.0831,-0.0702,-0.0539,-0.0409,-0.0369,-0.0329,-0.0289,-0.0248,-0.0208,
                     -0.0180,-0.0168,-0.0157,-0.0145,-0.0133,-0.0121,-0.0110,-0.0098,-0.0086,-0.0074,
                     -0.0063,-0.0051,-0.0039,-0.0027,-0.0016,-0.0004,0.0008,0.0020,0.0037,0.0055,
                     0.0074,0.0092,0.0111,0.0129,0.0148,0.0168,0.0193,0.0217,0.0241,0.0265,
                     0.0289,0.0305,0.0322,0.0338,0.0354,0.0370,0.0372,0.0370,0.0368,0.0365,
                     0.0363,0.0361,0.0359,0.0357,0.0355,0.0353,0.0352,0.0347,0.0334,0.0321,
                     0.0309,0.0294,0.0269,0.0244,0.0218,0.0195,0.0173,0.0152,0.0130,0.0109,
                     0.0087,0.0066,0.0044,0.0022,0.0001,-0.0021,-0.0042,-0.0064,-0.0085,-0.0107,
                     -0.0128,-0.0155,-0.0181,-0.0208,-0.0234,-0.0261,-0.0287,-0.0314,-0.0363,-0.0418,
                     -0.0473,-0.0528,-0.0583,-0.0641,-0.0719,-0.0802,-0.0912,-0.0949,-0.0949,-0.0939])

        self.FRx=np.array([0.0368,0.0365,0.0363,0.0361,0.0359,0.0357,0.0355,0.0353,0.0352,0.0347,
                      0.0334,0.0321,0.0309,0.0294,0.0269,0.0244,0.0218,0.0195,0.0173,0.0152,
                      0.0130,0.0109,0.0087,0.0066,0.0044,0.0022,0.0001,-0.0021,-0.0042,-0.0064,
                      -0.0085,-0.0107,-0.0128,-0.0155,-0.0181,-0.0208,-0.0234,-0.0261,-0.0287,-0.0314,
                      -0.0363,-0.0418,-0.0473,-0.0528,-0.0583,-0.0641,-0.0719,-0.0802,-0.0912,-0.0949,
                      -0.0949,-0.0939,-0.0939,-0.0831,-0.0702,-0.0539,-0.0409,-0.0369,-0.0329,-0.0289,
                      -0.0248,-0.0208,-0.0180,-0.0168,-0.0157,-0.0145,-0.0133,-0.0121,-0.0110,-0.0098,
                      -0.0086,-0.0074,-0.0063,-0.0051,-0.0039,-0.0027,-0.0016,-0.0004,0.0008,0.0020,
                      0.0037,0.0055,0.0074,0.0092,0.0111,0.0129,0.0148,0.0168,0.0193,0.0217,
                      0.0241,0.0265,0.0289,0.0305,0.0322,0.0338,0.0354,0.0370,0.0372,0.0370])

        self.FLy=np.array([-0.1992,-0.1941,-0.1905,-0.1896,-0.1896,-0.1892,-0.1888,-0.1884,-0.1880,-0.1876,
                      -0.1872,-0.1868,-0.1863,-0.1859,-0.1854,-0.1850,-0.1845,-0.1841,-0.1836,-0.1832,
                      -0.1827,-0.1823,-0.1818,-0.1814,-0.1809,-0.1805,-0.1800,-0.1796,-0.1792,-0.1788,
                      -0.1784,-0.1780,-0.1776,-0.1772,-0.1768,-0.1767,-0.1772,-0.1777,-0.1782,-0.1787,
                      -0.1793,-0.1812,-0.1831,-0.1850,-0.1869,-0.1888,-0.1911,-0.1936,-0.1960,-0.1985,
                      -0.2009,-0.2033,-0.2058,-0.2083,-0.2108,-0.2132,-0.2157,-0.2181,-0.2203,-0.2226,
                      -0.2248,-0.2268,-0.2276,-0.2285,-0.2294,-0.2299,-0.2299,-0.2300,-0.2300,-0.2301,
                      -0.2302,-0.2302,-0.2303,-0.2304,-0.2304,-0.2305,-0.2305,-0.2306,-0.2307,-0.2307,
                      -0.2308,-0.2305,-0.2301,-0.2298,-0.2294,-0.2291,-0.2287,-0.2284,-0.2278,-0.2272,
                      -0.2266,-0.2260,-0.2254,-0.2245,-0.2222,-0.2198,-0.2166,-0.2112,-0.2052,-0.1992])

        self.FRy=np.array([-0.1960,-0.1985,-0.2009,-0.2033,-0.2058,-0.2083,-0.2108,-0.2132,-0.2157,-0.2181,
                      -0.2203,-0.2226,-0.2248,-0.2268,-0.2276,-0.2285,-0.2294,-0.2299,-0.2299,-0.2300,
                      -0.2300,-0.2301,-0.2302,-0.2302,-0.2303,-0.2304,-0.2304,-0.2305,-0.2305,-0.2306,
                      -0.2307,-0.2307,-0.2308,-0.2305,-0.2301,-0.2298,-0.2294,-0.2291,-0.2287,-0.2284,
                      -0.2278,-0.2272,-0.2266,-0.2260,-0.2254,-0.2245,-0.2222,-0.2198,-0.2166,-0.2112,
                      -0.2052,-0.1992,-0.1992,-0.1941,-0.1905,-0.1896,-0.1896,-0.1892,-0.1888,-0.1884,
                      -0.1880,-0.1876,-0.1872,-0.1868,-0.1863,-0.1859,-0.1854,-0.1850,-0.1845,-0.1841,
                      -0.1836,-0.1832,-0.1827,-0.1823,-0.1818,-0.1814,-0.1809,-0.1805,-0.1800,-0.1796,
                      -0.1792,-0.1788,-0.1784,-0.1780,-0.1776,-0.1772,-0.1768,-0.1767,-0.1772,-0.1777,
                      -0.1782,-0.1787,-0.1793,-0.1812,-0.1831,-0.1850,-0.1869,-0.1888,-0.1911,-0.1936])

        self._theta = 0
        self._theta0 = 0
        self._update_action_every = 1.  # update is every step i.e., theta goes from 0 to pi
        self._frame_skip = frame_skip
        self._n_steps = 0

        self._frequency = 2
        self._kp = 30.
        self._kd = 0.4 # DO NOT MODIFY THIS. if you change this, then the xml file needs changing as well
        self._actuator_indices = [7,8,10,12,14,16,17,19,21,23]

        self._xpos_previous = 0
        self._walkcon = walking_controller.WalkingController(gait_type='trot',
                                                             spine_enable = False,
                                                             planning_space = 'polar_task_space',
                                                             left_to_right_switch = True,
                                                             frequency=self._frequency)
        
#         utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 1) # "frame_skip" option here is different. this integrates forward the simulation by "frame_skip=1" steps (for now).
        
        ## Gym env related mandatory variables
        self.obs_dim = 7
        self.action_dim = 10
        self.action = np.zeros(self.action_dim)
        
        observation_high = np.array([5.0] * self.obs_dim)
        observation_low = -observation_high
        self.observation_space = spaces.Box(observation_low, observation_high)
        
        action_high = np.array([1] * self.action_dim)
        action_low  = - action_high
        self.action_space = spaces.Box(action_low, action_high)

    def step(self, action, callback=None):

#         xposbefore = self.sim.data.qpos[0]
        # print(xposbefore)
        energy_spent_per_step, cost_reference = self.do_simulation(action, self._frame_skip, callback=callback)
#         print('energy_spent_per_step',energy_spent_per_step)
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
        
        ## calculate reward here
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        actuator_force = self.sim.data.actuator_force
        reward = self._get_reward(action,penalty,energy_spent_per_step,cost_reference,qpos,qvel,actuator_force)
        
        if done:
            self.reset_model()

        return ob, reward, done, dict(reward_run=reward, reward_ctrl=-penalty)

    def _get_reward(self,action,penalty,energy_spent_per_step,cost_reference,qpos,qvel,actuator_force):
        xpos = qpos[0]
        xvel = qvel[0]
        zpos = qpos[2]
        
        distance_travelled = xpos - self._xpos_previous
        self._xpos_previous = xpos

#         walking_velocity_reward = 10 * np.exp(-10*(0.6 - xvel)**2)
#         walking_height_reward = 2 * np.exp(-2*(0.22 - zpos)**2)
        costreference_reward = np.exp(-2*(0 - cost_reference)**2)
#         print('height',zpos)
#         print(walking_height_reward)
        
#         rt_start, _ = self._walkcon.transform_action_to_rt(0, action)
#         rt_mid, _ = self._walkcon.transform_action_to_rt(math.pi/2, action)
#         print('r and theta start',rt_start,'r and theta mid',rt_mid)
        
#         foot_clearance_reward = 0.5 * np.exp(-10*(0.05 - (rt_mid[0] - rt_mid[2]))**2)
#         foot_clearance_reward = 0.5 * np.exp(-2*(0.23 - rt_mid[0])**2) + 0.5 * np.exp(-2*(0.18 - rt_mid[2])**2)
#         stride_length_reward = 0.5 * np.exp(-10*(1.0 - (rt_start[1] - rt_start[3]))**2) + 0.1 * np.exp(-10*(0.0 - (rt_mid[1] - rt_mid[3]))**2)
        
        reward = distance_travelled - penalty - 0.01 * energy_spent_per_step + 5 * costreference_reward #+ walking_height_reward + foot_clearance_reward + stride_length_reward# + walking_velocity_reward
        
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
        init_quat = self.init_qpos[3:7]
        quat = ob[3:7]
        theta_x0,theta_y0,theta_z0 = self.quat2eul(init_quat)
        theta_x,theta_y,theta_z = self.quat2eul(quat)

        ang_lim = 0.5 # angle threshold in radians
        ANGLE_COND = (abs(abs(theta_y)-abs(theta_y0))<=ang_lim) and (abs(abs(theta_x)-abs(theta_x0))<=ang_lim) and (abs(abs(theta_z)-abs(theta_z0))<=ang_lim)
        HEIGHT_COND = (ob[2]>=0.0)
        # print(ob[2])
        
        if not(ANGLE_COND and HEIGHT_COND):
#             done = True
#             print("Oops...fail!","Ang condition",ANGLE_COND,"Height condition",HEIGHT_COND)
#             print('Height',ob[2])
#             print('Angle',abs(abs(theta_y)-abs(theta_y0)),abs(abs(theta_x)-abs(theta_x0)),abs(abs(theta_z)-abs(theta_z0)))
            penalty = penalty + 0.0   #Penalty for the fall or orientation

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

    def rotationMatrixToEulerAngles(R):
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
        self.action = action
        omega = 2 * math.pi * self._frequency
        self._theta = self._theta0
        p_index = 0
        energy_spent_per_step = 0
        cost_reference = 0

        while(self._theta - self._theta0 <= math.pi * self._update_action_every and not self._theta >= 2 * math.pi):

            theta = self._theta
            
            spine_m_angle_cmd, leg_m_angle_cmd, spine_m_vel_cmd, leg_m_vel_cmd = self._walkcon.transform_action_to_motor_joint_command(theta,action)
            self._theta = (omega * self.dt * n_frames + theta)
            
            qpos_act = self.sim.data.qpos[self._actuator_indices]
            qvel_act = self.sim.data.qvel[self._actuator_indices]
            
#             if  p_index==0:
#                 print(theta,qpos_act)
                
#             p_index = (p_index + 1)% 5
            m_angle_cmd_ext = np.zeros(10)
            m_angle_cmd_ext[[1,2,3,4,6,7,8,9]] = leg_m_angle_cmd
            m_angle_cmd_ext[[0,5]] = spine_m_angle_cmd

            m_vel_cmd_ext = np.zeros(10)
            m_vel_cmd_ext[[1,2,3,4,6,7,8,9]] = leg_m_vel_cmd
            m_vel_cmd_ext[[0,5]] = spine_m_vel_cmd

            self.sim.data.ctrl[:] = self._kp * (m_angle_cmd_ext - qpos_act) + self._kd * (m_vel_cmd_ext )
#             self.sim.data.ctrl[:] = m_angle_cmd_ext

            for _ in range(n_frames):
                self.sim.step()
                
                joint_power = np.multiply(self.sim.data.actuator_force, self.sim.data.qvel[self._actuator_indices]) # Power output of individual actuators
                joint_power[ joint_power < 0.0] = 0.0 # Zero all the negative power terms
                energy_spent = np.sum(joint_power) * self.dt # not divided by n_frames, self.dt itself is integration step
                energy_spent_per_step += energy_spent
                
            if callback is not None:
                if callback(self) is False:
                    break
                    
            cost_reference += self.CostReferenceGait(theta,qpos_act)
            
        self._theta0 = self._theta % (2* math.pi)
        return energy_spent_per_step, cost_reference

    def viewer_setup(self):
        id = 1
        self.viewer.cam.trackbodyid = id
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[id] = 1.15
        self.viewer.cam.elevation = -10
        
    def GetMotorAngles(self):
        return self.sim.data.qpos[self._actuator_indices]
    
    def GetDesiredMotorAngles(self):
        _, leg_m_angle_cmd, _, _ = self._walkcon.transform_action_to_motor_joint_command(self._theta,self.action)
        
        return leg_m_angle_cmd

    def GetMotorVelocities(self):
        return self.sim.data.qvel[self._actuator_indices]

    def GetMotorTorques(self):
        return self.sim.data.actuator_force
    
    def GetXYTrajectory(self,action):
        rt = np.zeros((4,100))
        rtvel = np.zeros((4,100))
        xy = np.zeros((4,100))
        xyvel = np.zeros((4,100))
        
        for i in range(100):
            theta = 2*math.pi/100*i
            rt[:,i], rtvel[:,i] = self._walkcon.transform_action_to_rt(theta, action)
            
            r_ac1 = rt[0,i]
            the_ac1 = rt[1,i]
            r_ac2 = rt[2,i]
            the_ac2 = rt[3,i]
            
            xy[0,i] =  r_ac1*math.sin(the_ac1)
            xy[1,i] = -r_ac1*math.cos(the_ac1)
            xy[2,i] =  r_ac2*math.sin(the_ac2)
            xy[3,i] = -r_ac2*math.cos(the_ac2)
            
#             if xy[1,i] < -0.233:
#                 xy[1,i] = -0.233
            
#             if xy[3,i]< -0.233:
#                 xy[3,i] = -0.233
            
        return xy

    def CostReferenceGait(self,theta,q):
        i = int(theta/2/math.pi*100)
        xy = self._walkcon.forwardkinematics(q)
        ls_error = (xy[0] - self.FRx[i])**2 + (xy[1] - self.FRy[i])**2 + (xy[2] - self.FLx[i])**2 + (xy[3] - self.FLy[i])**2
        return ls_error
 