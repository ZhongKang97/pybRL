# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for realizing walking controllers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
from scipy.linalg import solve

PI = math.pi

class WalkingController():
    
    def __init__(self,
                 gait_type='trot',
                 leg = [0.12,0.15015,0.04,0.15501,0.11187,0.04,0.2532,2.803],
                 spine_enable = False,
                 motor_offset = [-0.,0.],
                 frequency=2,
                 planning_space = 'joint_space',
                 left_to_right_switch = float('nan')
                 ):
        
        ## These are empirical parameters configured to get the right controller, these were obtained from previous training iterations
        
        self._trot = False
        self._bound = False
        self._canter = False
        self._step_in_place = False
        self._planning_space = planning_space
        
        if math.isnan(left_to_right_switch):
            if (planning_space == 'cartesian_task_space' or planning_space == 'polar_task_space'):
                self._left_to_right_switch = True
            else:
                self._left_to_right_switch = False
        else:
            self._left_to_right_switch = left_to_right_switch
            
        self._action_leg_indices = [1,2,3,4,6,7,8,9]
        self._action_spine_indices = [0,5]
        self._action_rt_indices = [10,11]
        self._frequency = frequency
        # print(self._planning_space)
        # print(gait_type)
        print('#########################################################')
        print('This training is for',gait_type,'in',self._planning_space)
        print('#########################################################')
        self._MOTOR_OFFSET = motor_offset
        if gait_type == 'trot':
            self._trot = True
             
            self._action_ref = np.array([ -0.68873949,  -2.7171507,    0.64782447,  -2.78440302,
                                           0.87747347,   1.1122558,   -5.73509876,  -0.57981057,
                                          -2.78440302, -17.35424259,  -1.41528624,  -0.68873949,
                                          -0.57981057,  2.25623534,   4.15258502,   0.87747347])

        elif gait_type == 'bound':
            self._bound = True
            
            self._action_ref = np.array([ -1.61179252,  -9.92289586,   0.4231481,   -2.56823015, 
                                            0.48072314,  -3.61462618,  -4.51084818,   3.20596271,
                                           -2.56823015, -15.83252898,  0.25214076,  -1.61179252,
                                           3.20596271,  23.74662442,   6.49896502,   0.48072314])
        elif gait_type == 'canter':
            self._canter = True
            
            self._action_ref = np.array([ 0.31126051, -0.54898837, -0.35217553, -1.78440302, 
                              -0.12252653, -3.69983287, -4.73509876, -1.57981057,
                              -1.78440302,  0.2640965,  -0.41528624,  0.31126051,
                              -1.57981057,  1.98315648,  3.15258502, -0.12252653])
        elif gait_type == 'step_in_place':
            self._step_in_place = True

            self._action_ref = np.array([ -1.61179252,  -9.92289586,   0.4231481,   -2.56823015, 
                                            0.48072314,  -3.61462618,  -4.51084818,   3.20596271,
                                           -2.56823015, -15.83252898,  0.25214076,  -1.61179252,
                                           3.20596271,  23.74662442,   6.49896502,   0.48072314])

        if (self._planning_space == 'joint_space'):
            self._RT_OFFSET = [0.0,-0.0]
            self._RT_SCALINGFACTOR = np.array([1/3,1/3])
            self._action_space_to_command = self._action_joint_space 

        elif(self._planning_space == 'cartesian_task_space'):
            self._RT_OFFSET = [0.20,-0.0]
            self._RT_SCALINGFACTOR = np.array([0.045/4,0.045/3])
            self._action_space_to_command = self._action_cartesian_task_space

        elif(self._planning_space == 'polar_task_space'):
            self._RT_OFFSET = [0.23,-0.0]
            self._RT_SCALINGFACTOR = np.array([0.045/1.5,25./1.25*PI/180.0]) # gait 16 was 0.045/1.5 and 25./1.5
            self._action_space_to_command = self._action_polar_task_space
        
        self._spine_enable = spine_enable

        if self._spine_enable:
            self._SPINE_SCALINGFACTOR = np.array([0.05/3])
            self._action_spine_ref = np.array([ 1.30790679, 0., 0., -0.45147199, -1.30790679, 0., 0., 0.45147199])
        
    #_action_planning_space method for different task space
    def _action_joint_space(self,tau, stance_leg, action_ref): #
        j_ang, j_vel = self._transform_action_to_joint_angle_via_bezier_polynomials(tau, stance_leg, action_ref)
        leg_motor_angle,leg_motor_vel = j_ang, j_vel
        return leg_motor_angle,leg_motor_vel
    
    def _action_polar_task_space(self,tau, stance_leg, action_ref):
        r_theta, dr_dtheta = self._transform_action_to_r_and_theta_via_bezier_polynomials(tau, stance_leg, action_ref)
        leg_motor_angle, leg_motor_vel = self._transform_r_and_theta_to_hip_knee_angles(r_theta, dr_dtheta)
        return leg_motor_angle,leg_motor_vel
    
    def _action_cartesian_task_space(self,tau, stance_leg, action_ref):
        xy, dxdy = self._transform_action_to_xy_via_bezier_polynomials(tau, stance_leg, action_ref)
        leg_motor_angle, leg_motor_vel = self._transform_xy_to_hip_knee_angles(xy, dxdy) 
        return leg_motor_angle,leg_motor_vel
    
    def transform_action_to_motor_joint_command(self, theta, action):
        if theta > PI:
            tau = (theta - PI)/PI  # as theta varies from pi to 2 pi, tau varies from 0 to 1    
            stance_leg = 1 # for 0 to pi, stance leg is left leg. (note robot behavior sometimes is erratic, so stance leg is not explicitly seen)
        else:
            tau = theta / PI  # as theta varies from 0 to pi, tau varies from 0 to 1    
            stance_leg = 0 # for pi to 2 pi stance leg is right leg.
        action_ref = self._extend_leg_action_space_for_hzd(tau,action)

        leg_motor_angle,leg_motor_vel  = self._action_space_to_command(tau, stance_leg, action_ref)#selects between planning_spaces
        
        if self._spine_enable:
            tau_spine = theta/2/PI
            action_spine = self._extend_spine_action_space_for_hzd(tau,action)
            spine_m_angle_cmd, spine_m_vel_cmd = self._transform_action_to_spine_actuation_via_bezier_polynomials(tau_spine, stance_leg, action_spine)
        else:
            spine_m_angle_cmd = np.zeros(2)
            spine_m_vel_cmd = np.zeros(2)
    
        #leg_motor_angle,leg_motor_vel = j_ang, j_vel
        leg_m_angle_cmd = self._spread_motor_commands(leg_motor_angle)
        leg_m_vel_cmd = self._spread_motor_commands(leg_motor_vel)

        return spine_m_angle_cmd, leg_m_angle_cmd, spine_m_vel_cmd, leg_m_vel_cmd

    def _Bezier_polynomial(self,tau,nTraj):
        Phi = np.zeros(4*nTraj)
        for i in range(nTraj):
            TAU = (tau)
            Phi[4*i + 0] = (1 - TAU) ** 3
            Phi[4*i + 1] = TAU ** 1 * (1 - TAU) ** 2
            Phi[4*i + 2] = TAU ** 2 * (1 - TAU) ** 1
            Phi[4*i + 3] = TAU ** 3

        return Phi

    def _Bezier_polynomial_derivative(self,tau,nTraj):
        Phi = np.zeros(4*nTraj)
        for i in range(nTraj):
            TAU = (tau)
            Phi[4*i + 0] = - 3 * (1 - TAU) ** 2
            Phi[4*i + 1] = (1 - TAU) ** 2 - 2 * TAU * (1 - TAU)
            Phi[4*i + 2] = - TAU ** 2 + 2 * TAU * (1 - TAU)
            Phi[4*i + 3] = 3 * TAU ** 2

        return Phi
    
    def _extend_leg_action_space_for_hzd(self,tau,action):
        action_ref = np.zeros(16)
        action_ref[[2,3,6,7,10,11,14,15]] = self._action_ref[[2,3,6,7,10,11,14,15]] + action[self._action_leg_indices]
        
        if self._left_to_right_switch:
            # hybrid zero dynamics based coefficients are defined here, some of the action values are reassigned
            action_ref[0] = action_ref[11] # r of left leg is matched with r of right leg
            action_ref[4] = action_ref[15] # theta of left leg is matched with theta of right leg
            action_ref[8] = action_ref[3] # r of left leg is matched with r of right leg
            action_ref[12]= action_ref[7] # theta of left leg is matched with theta of right leg

            action_ref[1] = 6 * action_ref[11] - action_ref[10] # rdot of one leg is matched with rdot of opposite leg
            action_ref[5] = 6 * action_ref[15] - action_ref[14] # thetadot of one leg is matched with thetadot of opposite leg
            action_ref[9] = 6 * action_ref[3] - action_ref[2] # rdot of one leg is matched with rdot of opposite leg
            action_ref[13]= 6 * action_ref[7] - action_ref[6] # thetadot of one leg is matched with thetadot of opposite leg
            
        else:
            action_ref[0] = action_ref[3] # r at tau=0 is matched with tau=1
            action_ref[4] = action_ref[7] # theta at tau=0 is matched with tau=1

            action_ref[1] = action_ref[10] # 
            action_ref[5] = action_ref[11] # 

            action_ref[8:] = action_ref[:8]
#         print('action_ref',action_ref)    
        return action_ref
        
    def _extend_spine_action_space_for_hzd(self,tau,action):
        action_spine = np.zeros(8)
        action_spine[[0,3]] = self._action_spine_ref[[0,3]] + action[self._action_spine_indices]
        action_spine[[4,7]] = self._action_spine_ref[[4,7]] - action[self._action_spine_indices]
        
        return action_spine
        
    def _transform_action_to_joint_angle_via_bezier_polynomials(self, tau, stance_leg, action):

        joint_ang = np.zeros(4)
        joint_vel = np.zeros(4)
        Weight_ac = np.zeros(16)
        
        if self._left_to_right_switch:
            if stance_leg == 0:
                Weight_ac = action  # the first half of the action the values are for stance leg
            else:
                Weight_ac[0:8] = action[8:16]
                Weight_ac[8:16] = action[0:8]
        else:
            Weight_ac = action  # the first half of the action the values are for stance leg

        omega = 2 * PI * self._frequency # omega frequency used here
        Phi = self._Bezier_polynomial(tau,4)
        dPhidt = self._Bezier_polynomial_derivative(tau,4)
        
        if not self._left_to_right_switch:
            tau_aux = (tau + 0.5) % 1
            Phi_aux = self._Bezier_polynomial(tau_aux,4)
            dPhidt_aux = self._Bezier_polynomial_derivative(tau_aux,4)
            
            for i in range(2):
                Phi[4*(i+2):4*(i+2)+4] = Phi_aux[4*i:4*i+4]
                dPhidt[4*(i+2):4*(i+2)+4] = dPhidt_aux[4*i:4*i+4]
                
        for i in range(4):
            joint_ang[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] + self._MOTOR_OFFSET[i % 2]
            joint_vel[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] 

        return joint_ang, joint_vel
    
    def _transform_action_to_r_and_theta_via_bezier_polynomials(self, tau, stance_leg, action):

        rt = np.zeros(4)
        drdt = np.zeros(4)
        Weight_ac = np.zeros(16)
        
        if self._left_to_right_switch:
            if stance_leg == 0:
                Weight_ac = action  # the first half of the action the values are for stance leg
            else:
                Weight_ac[0:8] = action[8:16]
                Weight_ac[8:16] = action[0:8]
        else:
            Weight_ac = action  # the first half of the action the values are for stance leg

        omega = 2 * PI * self._frequency # omega frequency used here
        Phi = self._Bezier_polynomial(tau,4)
        dPhidt = self._Bezier_polynomial_derivative(tau,4)
        
        if not self._left_to_right_switch:
            tau_aux = (tau + 0.5) % 1
            Phi_aux = self._Bezier_polynomial(tau_aux,4)
            dPhidt_aux = self._Bezier_polynomial_derivative(tau_aux,4)
            
            for i in range(2):
                Phi[4*(i+2):4*(i+2)+4] = Phi_aux[4*i:4*i+4]
                dPhidt[4*(i+2):4*(i+2)+4] = dPhidt_aux[4*i:4*i+4]
                
        for i in range(4):
            rt[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] + self._RT_OFFSET[i % 2] 
            drdt[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] 

        return rt, drdt

    def _transform_action_to_xy_via_bezier_polynomials(self, tau, stance_leg, action):

        xy = np.zeros(4)
        yx = np.zeros(4)
        dxdy = np.zeros(4)
        dydx = np.zeros(4)
        Weight_ac = np.zeros(16)
        
        if self._left_to_right_switch:
            if stance_leg == 0:
                Weight_ac = action  # the first half of the action the values are for stance leg
            else:
                Weight_ac[0:8] = action[8:16]
                Weight_ac[8:16] = action[0:8]
        else:
            Weight_ac = action  # the first half of the action the values are for stance leg

        omega = 2 * PI * self._frequency # omega frequency used here
        Phi = self._Bezier_polynomial(tau,4)
        dPhidt = self._Bezier_polynomial_derivative(tau,4)

        if not self._left_to_right_switch:
            tau_aux = (tau + 0.5) % 1
            Phi_aux = self._Bezier_polynomial(tau_aux,4)
            dPhidt_aux = self._Bezier_polynomial_derivative(tau_aux,4)
            
            for i in range(2):
                Phi[4*(i+2):4*(i+2)+4] = Phi_aux[4*i:4*i+4]
                dPhidt[4*(i+2):4*(i+2)+4] = dPhidt_aux[4*i:4*i+4]
                
        for i in range(4):
            yx[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] + self._RT_OFFSET[i % 2] 
            dydx[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] 

        # this negates the y direction because the legs are pointed downwards
        yx[[0,2]] = - yx[[0,2]]
        dydx[[0,2]] = - dydx[[0,2]]
        
        xy = yx[[1,0,3,2]]
        dxdy = dydx[[1,0,3,2]]
        
        return xy, dxdy
    
    def _transform_action_to_spine_actuation_via_bezier_polynomials(self, tau_spine, stance_leg, action):
        spine_des = np.zeros(2)
        d_spine_des = np.zeros(2)
        Weight_ac = np.zeros(8)
        Weight_ac = action
        
        omega = 2 * PI * self._frequency # omega frequency used here
        Phi = self._Bezier_polynomial(tau_spine,2)
        dPhidt = self._Bezier_polynomial_derivative(tau_spine,2)

        for i in range(2):
            spine_des[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._SPINE_SCALINGFACTOR
            d_spine_des[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._SPINE_SCALINGFACTOR

        return spine_des, d_spine_des
    
    def _transform_r_and_theta_to_hip_knee_angles(self, r_and_theta, dr_and_dtheta):
        motor_angle = self._ConvertRThetatoHipKneeJointMotorAngle(r_and_theta)
        motor_vel = self._ConvertRThetatoHipKneeJointMotorVel(r_and_theta, dr_and_dtheta, motor_angle)

        return motor_angle, motor_vel

    def _transform_xy_to_hip_knee_angles(self, xy, dxdy):
        motor_angle = self._ConvertXYtoHipKneeJointMotorAngle(xy)
        motor_vel = self._ConvertXYtoHipKneeJointMotorVel(xy, dxdy, motor_angle)

        return motor_angle, motor_vel
    
    def _ConvertRThetatoHipKneeJointMotorAngle(self, r_and_theta):
        """Convert the r and theta values that use leg model to the real motor actions.

        Args:
          r_and_theta: The theta, phi of the leg model.
        Returns:
          The eight desired motor angles that can be used in ApplyActions().
        """
        motor_angle = np.zeros(4)
        xy = np.zeros(4)

        for i in range(2):

            r_ac = r_and_theta[2*i] # first value is r and second value is theta
            the_ac = r_and_theta[2*i+1] # already converted to radians in stoch2_gym_env
            # print('r',r_ac)
            # print('theta',the_ac)

            xy[2*i] =  r_ac*math.sin(the_ac)
            xy[2*i+1] = -r_ac*math.cos(the_ac) # negative y direction for using the IK solver
            
        motor_angle = self._ConvertXYtoHipKneeJointMotorAngle(xy)
        
        return motor_angle
    
    def _ConvertXYtoHipKneeJointMotorAngle(self, xy):
        """Convert the r and theta values that use leg model to the real motor actions.

        Args:
          xy: The theta, phi of the leg model.
        Returns:
          The eight desired motor angles that can be used in ApplyActions().
        """
        motor_angle = np.zeros(4)

        for i in range(2):
            x =  xy[2*i]
            y =  xy[2*i+1]
            
#             if (y > -0.145) or (y<-0.235):
#                 print('error y',y)
#             elif (x>(-1*(y+0.01276)/(1.9737))) or (x<((y+0.01276)/(1.9737))):
#                 print('error x',x)
            
            knee, hip, _, _ = self._inverse_stoch2(x,y,self._leg)

            motor_angle[2*i] = hip + self.MOTOROFFSETS[0]
            motor_angle[2*i+1] = knee + self.MOTOROFFSETS[1]
        
        return motor_angle

    def _ConvertRThetatoHipKneeJointMotorVel(self, r_and_theta, dr_and_dtheta, motor_angle):
        """Convert the r and theta values that use leg model to the real motor actions.
        Args:
          r_and_theta: r and theta of the legs.
          dr_and_dtheta: r and theta dot of the legs
          angle: 
        Returns:
          The eight desired motor angles that can be used in ApplyActions().
        """
#         print(r_and_theta,dr_and_dtheta)

        joint_velocity = np.zeros(4)
        #dalpha = np.zeros(2)
    
        return joint_velocity

    def _ConvertXYtoHipKneeJointMotorVel(self, xy, dxdy, motor_angle):
        """Convert the r and theta values that use leg model to the real motor actions.

        Args:
          xy: x and y pos of legs
          dxdy: xdot and ydot of legs
          angle: motor angle
        Returns:
          The eight desired motor angles that can be used in ApplyActions().
        """
        # print(r_and_theta,dr_and_dtheta)

        joint_velocity = np.zeros(4)
  
        return joint_velocity

    def _limiter(self, X):
        if abs(X) >1 :
            X = np.sign(X);
        return X

    def _spread_motor_commands(self, angles):
        """
        This function distributes the four angle commands obtained from basis functions
        to eight individual joint angles. This distribution depends the gait.
        """
        motor_angles = np.zeros(8)

        if self._canter:
            motor_angles[:4] = angles
            motor_angles[4:] = angles

        if self._trot or self._step_in_place:
            motor_angles[:4] = angles
            motor_angles[4:6] = angles[2:]
            motor_angles[6:] = angles[:2]

        if self._bound:
            motor_angles[:2] = angles[:2]
            motor_angles[2:4] = angles[:2]

            motor_angles[4:6] = angles[2:]
            motor_angles[6:] = angles[2:]

        return motor_angles

    def transform_action_to_rt(self, theta, action):
        if theta > PI:
            tau = (theta - PI)/PI  # as theta varies from pi to 2 pi, tau varies from 0 to 1    
            stance_leg = 1 # for 0 to pi, stance leg is left leg. (note robot behavior sometimes is erratic, so stance leg is not explicitly seen)
        else:
            tau = theta / PI  # as theta varies from 0 to pi, tau varies from 0 to 1    
            stance_leg = 0 # for pi to 2 pi stance leg is right leg.
            
        if self._planning_space == 'polar_task_space':
            action_ref = self._extend_leg_action_space_for_hzd(tau,action)
            r_theta, rdot_thetadot = self._transform_action_to_r_and_theta_via_bezier_polynomials(tau, stance_leg, action_ref)
        else:
            r_theta = np.zeros(2)
            raise Exception('Error: r, theta are evaluated only for polar task space. Change the task space')

        return r_theta, rdot_thetadot
    
    def forwardkinematics(self,q):
        q_fl = q[1:3]
        q_fr = q[3:5]
        q_bl = q[6:8]
        q_br = q[8:10]
        
        xy = np.zeros(4)
        _, xy[0:2] = self.ik_leg.forwardKinematics(q_fl-self.MOTOROFFSETS)
        _, xy[2:4] = self.ik_leg.forwardKinematics(q_fr-self.MOTOROFFSETS)
        
        return xy