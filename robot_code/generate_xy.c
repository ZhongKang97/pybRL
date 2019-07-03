#include<stdio.h>
#include<stdlib.h>
#define PI 3.14159265358979323846
int main()
{
    double theta =0;double theta0 = 0;
    double tau = 0;
    int stance_leg = 0;
    // For action ref
    double action_ref_trot[16]= { -0.68873949,-2.7171507,    0.64782447,  -2.78440302,
                            0.87747347,1.1122558,   -5.73509876,  -0.57981057,
                            -2.78440302,-17.35424259,  -1.41528624,  -0.68873949,
                            -0.57981057,2.25623534,   4.15258502,   0.87747347};
    int action_ref_list[8] = {2,3,6,7,10,11,14,15};
    int action_leg_indices[8] = {1,2,3,4,6,7,8,9};
    double action_ref_final[16];
    double action[10];

    //For action_to_xy
    double xy[4] = {0};
    double x_leg1_coords[1000];
    double x_leg2_coords[1000];
    double y_leg1_coords[1000];
    double y_leg2_coords[1000];
    double yx[4] = {0};
    int xy_list[4] = {1,0,3,2};
    double weight_ac[16] = {0};
    double phi[16] = {0};
    double RT_OFFSET[2] = {0.20, 0.0};
    double RT_SCALINGFACTOR[2] = {0.045/4, 0.045/3};
    const double omega = 2*PI*2;

    int count  =0;
    for (theta = 0; theta < 2*PI; theta = theta + 0.001*2*PI)
    {

        if(theta < PI)
        {
            tau = theta/PI;
            stance_leg = 0;
        }
        else
        {
            tau = (theta - PI)/ PI;
            stance_leg = 1;
        }
        // Action Ref function
        for(int i =0; i< 8; i ++)
        {
            action_ref_final[action_ref_list[i]] = action_ref_trot[action_ref_list[i]] + action[action_leg_indices[i]];
        }
        action_ref_final[0] = action_ref_final[11]; // r of left leg is matched with r of right leg
        action_ref_final[4] = action_ref_final[15]; // theta of left leg is matched with theta of right leg
        action_ref_final[8] = action_ref_final[3] ;// r of left leg is matched with r of right leg
        action_ref_final[12]= action_ref_final[7]; //theta of left leg is matched with theta of right leg

        action_ref_final[1] = 6 * action_ref_final[11] - action_ref_final[10]; // rdot of one leg is matched with rdot of opposite leg
        action_ref_final[5] = 6 * action_ref_final[15] - action_ref_final[14]; // thetadot of one leg is matched with thetadot of opposite leg
        action_ref_final[9] = 6 * action_ref_final[3] - action_ref_final[2]; // rdot of one leg is matched with rdot of opposite leg
        action_ref_final[13]= 6 * action_ref_final[7] - action_ref_final[6]; // thetadot of one leg is matched with thetadot of opposite leg
        // Action Ref Function Done

        //Action_to_xy_via_Bezier function
        for(int i=0; i<16; i++)
        {
            weight_ac[i] = action_ref_final[i];
        }
        // Bezier Polynomial function start
        for(int i=0; i<16; i++)
        {
            

            phi[4*i + 0] = (1-tau)*(1-tau)*(1-tau);
            phi[4*i + 1] = tau*(1-tau)*(1-tau);
            phi[4*i + 2] = tau*tau*(1-tau);
            phi[4*i + 3] = tau*tau*tau;

        }
        //Bezier polynomial function end

        for(int i = 0; i<4; i++)
        {
            for(int j =0; j<4; j++)
            {
                yx[i] += weight_ac[4*i + j]*phi[4*i+j];
            }
            yx[i] = yx[i]*RT_SCALINGFACTOR[i %2 ] + RT_OFFSET[i%2];
            
        }
        yx[0] = -1*yx[0];
        yx[2] = -1*yx[2];
        for (int i =0; i<4; i++)
        {
            xy[i] = yx[xy_list[i]];
        }
        x_leg1_coords[count] = xy[0];
        x_leg2_coords[count] = xy[2];

        y_leg1_coords[count] = xy[1];
        y_leg2_coords[count] = xy[3];
        count++;
    }
    for(int i=0; i<1000; i++)
    {
        printf("%f,%f,%f,%f\n",x_leg1_coords[i],y_leg1_coords[i],x_leg2_coords[i],y_leg2_coords[i] );
    }

}