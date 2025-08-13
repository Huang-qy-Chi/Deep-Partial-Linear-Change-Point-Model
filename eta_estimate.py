import numpy as np

        

#%% ---------------------------change point est by grid search--------------------------
def eta_est(Y, A, Z, f_X, g_X, Theta, seq = 0.01):
    #establish the grid of zeta
    Z_min = np.min(Z)
    Z_max = np.max(Z)
    #num = np.floor((Z_max-Z_min)/seq)
    #np.arange(0, 3, 0.1)
    zeta_grid = np.arange(Z_min, Z_max, seq)  #the search grid

    #define the log-likelihood loss
    def BZ2(*args):
        ZC = np.vstack((A,A*(Z>args[0])))
        loss_total = Y-ZC.T @ Theta - f_X - g_X*(Z>args[0])
        Loss_F = (0.5*(loss_total)**2).mean()
        return Loss_F
    
    zeta_loss = []
    for zeta in zeta_grid:
        zeta_loss.append(BZ2(zeta))
    loss_min = min(zeta_loss)
    loc = zeta_loss.index(loss_min)
    zeta_est = zeta_grid[loc]
    zeta_est = zeta_est.astype(np.float32)
    return zeta_est









