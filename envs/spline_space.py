import matplotlib.pyplot as plt 
import numpy as np 
from dataclasses import dataclass
PI = np.pi
@dataclass
class pt:
    x : float = 0.0
    y: float = 0.0

def _generate_spline_ref(size, limit_radius, limit_thetas):
    spline_ref = np.zeros(size)
    x= []
    y= []
    for i in range(spline_ref.size):
        theta = i*(2*PI/size)
        if(theta > PI):
            theta = theta - 2*PI
        idx = np.abs(theta - limit_thetas).argmin()
        # print('diff: ', np.abs(theta - limit_thetas).min())
        spline_ref[i] = limit_radius[idx]
        x.append(limit_radius[idx]*np.cos(theta))
        y.append(limit_radius[idx]*np.sin(theta))
    return spline_ref, [x,y]

def get_spline(action, center):
    cubic_spline = lambda coeffts, t: coeffts[0] + t*coeffts[1] + t*t*coeffts[2] + t*t*t*coeffts[3]
    spline_fit = lambda y0, y1, d0, d1: np.array([y0, d0, 3*(y1-y0) -2*d0 - d1, 2*(y0 - y1) + d0 + d1 ])
    theta = 0
    x= []
    y =[]
    r1= []
    n = action.size -1
    while(theta < 2*PI):

        idx = int((theta - 1e-4)*n/(2*PI))
        tau = (theta - 2*PI*idx/n) /(2*PI/n)
        y0 = action[idx]
        y1 = action[idx+1]
        if idx == 0 :
            d0 = 0 # Slope at start-point is zero
        else:
            d0 = (action[idx+1] - action[idx-1])/2 # Central difference
        if idx == n-1:
            d1 = 0 # Slope at end-point is zero
        else:
            d1 = (action[idx+2] - action[idx])/2 # Central difference


        coeffts = spline_fit(y0, y1, d0, d1)
        r = cubic_spline(coeffts, tau)
        r1.append(r)
        x.append(-r * np.cos(theta)+ center[0])
        y.append(r * np.sin(theta) + center[1])
        theta = theta + 2*PI/1000
    
    return np.array(x), np.array(y), np.array(r1)

y = np.arange(-0.145, -0.245, -0.001)

x_max = np.zeros(y.size)
x_min = np.zeros(y.size)

trap_pts = []
count = 0 
for pt in y:
    x_max[count] = (pt+0.01276)/1.9737
    x_min[count] = -1*(pt+0.01276)/1.9737
    count = count + 1

center = [0, -0.195]
radius = 0.042
thetas = np.arange(0, 2*np.pi, 0.001)
x_circ = np.zeros(thetas.size)
y_circ = np.zeros(thetas.size)
count = 0
# for every theta there is a max r, if I find that, then I can search the entire space
#Need to  check and see if this works
# for val in  x_max:

for theta in thetas:
    x_circ[count] = radius*np.cos(theta) + center[0]
    y_circ[count] = radius*np.sin(theta) + center[1]
    count = count + 1

x_bottom = np.arange(x_min[-1], x_max[-1], -0.001)
x_top = np.arange(x_min[0], x_max[0], -0.001)


final_x = np.concatenate([x_max, np.flip(x_bottom), x_min, x_top])
final_y = np.concatenate([y, np.ones(x_bottom.size)*y[-1], y, np.ones(x_top.size)*y[0]])
final_thetas = np.arctan2(final_y - center[1], final_x - center[0])
final_radius = np.sqrt(np.square(final_x - center[0]) + np.square(final_y - center[1])) - 0.005
check_x = np.multiply(final_radius, np.cos(final_thetas)) + center[0]
check_y = np.multiply(final_radius, np.sin(final_thetas)) + center[1]
np.save("stoch2/ik_check_thetas", final_thetas)
np.save("stoch2/ik_check_radius", final_radius)

action = np.ones(100)
# action = np.array([ 1.18621837,-1.24374914,-0.2546842,1.368942,1.30855425,-0.23257767,-0.76869941,-1.10599863,0.1056882,1.24348629])
# action = np.clip(action, -1, 1)
# mul_ref = np.array([0.08233419, 0.07341638, 0.04249794, 0.04249729, 0.07341638, 0.08183298,0.07368498, 0.04149645, 0.04159619, 0.07313576])
# action = np.multiply(action, mul_ref) * 0.5
# action_spline_ref = np.multiply(np.ones(action.size),mul_ref) * 0.5
# action = action + action_spline_ref
mul_ref, pts = _generate_spline_ref(action.size, final_radius, final_thetas)
action = np.multiply(action, mul_ref)
# print(mul_ref)
action = np.append(action, action[0])
# print(action)

action = np.array([0.0409,0.0134,0.0266,0.0234,0.0185,0.0117,0.0528,0.0731,0.078,0.0651,0.024,0.0057,0.0017,0.0,0.0,0.0335,0.0711,0.1153,0.0409]) * 0.85
# action = np.array([0.0466,0.0347,0.0357,0.0251,0.0252,0.0253,0.033,0.0552,0.0601,0.0539,0.02,0.012,0.0138,0.015,0.019,0.0293,0.0706,0.1153,0.0466])
action = np.array([0.048,0.0391,0.035,0.025,0.0238,0.0242,0.0326,0.0522,0.0561,0.0525,0.0252,0.0113,0.0154,0.0159,0.0187,0.0275,0.0624,0.1153,0.048])
action = np.array([0.0476,0.0372,0.0367,0.0254,0.0234,0.0238,0.0308,0.0485,0.0518,0.0514,0.0261,0.0166,0.0164,0.0182,0.0189,0.0269,0.0546,0.1153,0.0476])
action = np.array([0.0469,0.038,0.035,0.0251,0.0237,0.0231,0.0311,0.0453,0.0472,0.0471,0.0329,0.0195,0.0194,0.0193,0.0202,0.027,0.0477,0.1113,0.0469])
action = np.array([0.0463,0.0368,0.0343,0.026,0.0238,0.0238,0.0305,0.0429,0.0457,0.0458,0.0344,0.0218,0.0218,0.0202,0.0211,0.027,0.0456,0.0965,0.0463])
action = np.array([0.0461,0.0367,0.0358,0.027,0.0237,0.0236,0.0292,0.0403,0.0429,0.0446,0.037,0.0257,0.0235,0.0217,0.0214,0.0267,0.042,0.0853,0.0461])
final_str = '{'
for x in action:
    final_str = final_str + str(round(x,4)) + ','
final_str = final_str + '};'
print(final_str)
x_spline, y_spline, r_spline = get_spline(action, center)

# print(x_top)
plt.figure(1)

plt.plot(final_x,final_y,'r', label = 'robot workspace')
plt.plot(x_circ, y_circ, 'g', label = 'circle search space')
# plt.plot(check_x, check_y, 'b')
plt.plot(x_spline, y_spline,'y', label = 'spline search space')
plt.plot(np.array(pts[0])+center[0], np.array(pts[1])+center[1], 'purple', label ='spline interpol pts')
plt.legend()
# plt.figure(2)
# dict1 = {}
# for i in range(final_thetas.size):
#     dict1[final_thetas[i]] = final_radius[i]
# th = []
# r = []
# for i in sorted(dict1):
#     th.append(i)
#     r.append(dict1[i])
# plt.plot(th, r)
plt.show()


# import pandas as pd 
# df = pd.read_csv('data.txt')

# for i, row in df.iterrows():
#     leg1_x = row['leg1_x']
#     leg1_y = row['leg1_y']
#     leg2_x = row['leg2_x']
#     leg2_y = row['leg2_y']
#     if(leg1_y > -0.145 or leg1_y < -0.245):
#         print('Invalid point (Y): ',leg1_x,leg1_y)
#     else:
#         if(leg1_x > -1*(leg1_y+0.01276)/1.9737 or leg1_x < 1*(leg1_y+0.01276)/1.9737):
#             print('Invalid point (X): ',leg1_x,leg1_y)
#     if(leg2_y > -0.145 or leg2_y < -0.245):
#         print('Invalid point (Y): ',leg2_x,leg2_y)
#     else:
#         if(leg2_x > -1*(leg2_y+0.01276)/1.9737 or leg2_x < 1*(leg2_y+0.01276)/1.9737):
#             print('Invalid point (X): ',leg2_x,leg2_y)
# print("all valid")