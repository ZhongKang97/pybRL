import numpy as np
import math
import matplotlib.pyplot as plt



def spline_fit(y0, y1, d0, d1):
    a = y0
    b = d0
    c = 3*(y1-y0) -2*d0 - d1
    d = 2*(y0 - y1) + d0 + d1
    return np.array([a, b, c, d])


def cubic_spline(coeffts, t):
    a = coeffts[0] + t*coeffts[1] + t*t*coeffts[2] + t*t*t*coeffts[3]
    return a



size=100

r  = np.zeros(size)
th = np.zeros(size)

x  = np.zeros(size)
y  = np.zeros(size)


# Number of segments
n=10

for k in range(5):
	pts = 10*np.ones(n+1) + 4*np.random.rand(n+1)	
        #pts = 10*np.ones(n+1)
	pts[n] = pts[0]  # C0 continuity	
	print(pts)
        for i in range(size):
            idx = i/n
            tau = float(i - n*idx)/n

            y0 = pts[idx]
            y1 = pts[idx+1]
            if idx == 0 :
                d0 = 0 # Slope at start-point is zero
            else:
                d0 = (pts[idx+1] - pts[idx-1])/2 # Central difference
            if idx == n-1:
                d1 = 0 # Slope at end-point is zero
            else:
                d1 = (pts[idx+2] - pts[idx])/2 # Central difference

            coeffts = spline_fit(y0, y1, d0, d1)

            r[i]  = cubic_spline(coeffts, tau)
            th[i] = i*(2*3.14159)/size
	    x[i]  = r[i] * math.cos(th[i])
	    y[i]  = r[i] * math.sin(th[i])
        #plt.plot(th, r)
	plt.plot(x, y)

plt.show()







