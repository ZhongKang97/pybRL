import numpy as np
def cg_solve(f_Ax, b, x_0=None, cg_iters=10, residual_tol=1e-10):
    """ Works well only with sparse matrices. Implements conjugate gradients. Takes in a function that returns Ax, and a constant matrix B. 
    It returns x, which is a close solution """
    x = np.zeros_like(b) #if x_0 is None else x_0
    r = b.copy() #if x_0 is None else b-f_Ax(x_0)
    p = r.copy()
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x

if(__name__ == "__main__"):
        A = np.array([[1.,2.],[3.,4.]])
        def f_ax(val):
                return A @ val
        B = np.array([1.1,1.6])
        x = cg_solve(f_ax,B)
        print('Ax: ',A@x,'B: ',B)