#%%
import numpy as np
import torch 
from dnn_estimate import dnn_est
from Theta_estimate import Theta_est
from eta_estimate import eta_est

#%%-------------------------Estimation by iteration-----------------------------
def CPDPLM(train_data,val_data,test_data,Theta0, eta0,\
            n_layer, n_node, n_lr, n_epoch, patiences,show_val=False,maxloop=100,seq=0.01):
    Y_train = train_data['Y']
    X_train = train_data['X']
    A_train = train_data['A']
    Z_train = train_data['Z']
    # index for whether it converges
    C_index = 0

    # iterate estimation
    for i in range(maxloop):
        # DNN estimation
        dnn_res = dnn_est(train_data, val_data, test_data,Theta0,eta0,\
                           n_layer, n_node, n_lr, n_epoch, patiences,show_val=show_val)
        f_train = dnn_res['f_train']
        g_train = dnn_res['g_train']
        f_C_train = dnn_res['f_C_train']
        f_val = dnn_res['f_val']
        g_val = dnn_res['g_val']
        f_C_val = dnn_res['f_C_val']
        f_test = dnn_res['f_test']
        g_test = dnn_res['g_test']
        f_C_test = dnn_res['f_C_test']
        val_loss = dnn_res['Val_loss']
        maxepoch = dnn_res['Early_stop']

        # Parameter estimation
        Theta = Theta_est(Y_train, A_train, Z_train, f_train, g_train, eta0)

        #Change point estimation
        eta = eta_est(Y_train, A_train, Z_train, f_train, g_train, Theta, seq = seq)

        # whether stop the loop
        if (np.max(abs(Theta0-Theta)) <= 0.01):
            C_index = 1
            break
        Theta0 = Theta
        eta0 = eta
    
    return{
        'C_index': C_index, 
        'Theta': Theta, 
        'eta': eta,
        'f_test': f_test,
        'g_test': g_test,
        'f_C_test': f_C_test,
        'f_train': f_train,
        'g_train': g_train,
        'f_C_train': f_C_train,
        'f_val': f_val,
        'g_val': g_val,
        'f_C_val': f_C_val,
        'val_loss': val_loss,
        'val_loss': val_loss,
        'maxepoch': maxepoch
    }



























































