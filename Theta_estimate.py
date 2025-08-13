
import numpy as np
import scipy.optimize as spo

#%%----------------Estimation for the parameter Theta------------------------------------
def Theta_est(Y, A, Z, f_X, g_X, eta):
    ZC0 = np.vstack((A,A*(Z>eta)))
    p = ZC0.shape[0]
    def TF(*args):   
        ZC = np.vstack((A,A*(Z>eta)))
        loss_total = Y-ZC.T @ args[0] - f_X - g_X*(Z>eta)
        Loss_F = (0.5*(loss_total)**2).mean()
        return Loss_F
    result = spo.minimize(TF,np.zeros(p),method='SLSQP') #nonconvex optimaization
    return result['x']

