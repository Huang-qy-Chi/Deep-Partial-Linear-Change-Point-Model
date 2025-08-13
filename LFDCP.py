
import torch
from torch import nn
import numpy as np

# for the score and information

# simultaneously read the a_b
def LFDCP(A1_train, A1_val, train_data,val_data,Theta,eta,f_C_train, f_C_val,\
          n_layer,n_node,n_lr,n_epoch,patiences=10,show_val = True):
    """
    This function outputs the least favorite direction to construct 
    the information matrix and calculate the SE of beta and gamma.
    """
    if show_val == True:
        print('DNN_iteration')
    
    # Convert training and test data to tensors
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    Y_train = torch.tensor(train_data['Y'], dtype=torch.float32)
    Z_train = torch.tensor(train_data['Z'], dtype=torch.float32)
    A1_train = torch.tensor(A1_train, dtype=torch.float32)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    A_val = torch.tensor(val_data['A'], dtype=torch.float32)
    Y_val = torch.tensor(val_data['Y'], dtype=torch.float32)
    Z_val = torch.tensor(val_data['Z'], dtype=torch.float32)
    A1_val = torch.tensor(A1_val, dtype=torch.float32)

    Theta = torch.tensor(Theta, dtype=torch.float32)
    eta = torch.tensor(eta, dtype=torch.float32)
    f_C_train = torch.tensor(f_C_train, dtype=torch.float32)
    f_C_val = torch.tensor(f_C_val, dtype=torch.float32)
    d_X = X_train.size()[1]

    # Define the DNN model : outfeature=1, input the X to find the LFD of f(X) and g(X) respectively by input A1 and A2
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X, out_features=1, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0):
            super(DNNModel, self).__init__()
            layers = []
            # Input layer
            layers.append(torch.nn.Linear(in_features, hidden_nodes))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(drop_rate))
            # Hidden layers
            for _ in range(hidden_layers):
                layers.append(torch.nn.Linear(hidden_nodes, hidden_nodes))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(drop_rate))
            # Output layer
            layers.append(torch.nn.Linear(hidden_nodes, out_features))
            self.linear_relu_stack = torch.nn.Sequential(*layers)
        
        def forward(self, x):
            return self.linear_relu_stack(x)
        

    # Initialize model and optimizer
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)

    # Custom loss function for binary decision: via MSEloss
    # LS loss for LFD for reading a with dim n*1
    def my_loss(A1,Z,A,Theta,eta,Y,f_C_X,a):  
        #least square
        ind = Z>eta
        ind = ind.unsqueeze(1)
        A2 = A.unsqueeze(1)
        A2 = torch.cat((A2,A2*ind),dim = 1) 
        epsilon = (Y-A2@Theta-f_C_X)   # n*1  
        loss_fun = (epsilon**2)*((A1-a)**2)
        return loss_fun.mean()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    #Loop
    for epoch in range(n_epoch):
        # Training loop
        model.train()  # Set model to training mode
        a_train = model(X_train).squeeze()
        loss = my_loss(A1_train,Z_train, A_train, Theta, eta, Y_train, f_C_train,a_train)
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            a_val = model(X_val).squeeze()
            val_loss = my_loss(A1_val,Z_val, A_val, Theta, eta, Y_val, f_C_val,a_val)
            if show_val == True:
                print('epoch=', epoch, 'val_loss=', val_loss.detach().numpy())
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            patience_counter += 1
            if show_val == True:
                print('patience_counter =', patience_counter)
            
        if patience_counter >= patiences:
            if show_val == True:
                print(f'Early stopping at epoch {epoch + 1}, ', 'validation—loss=', val_loss.detach().numpy())
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        # prediction of mu0(x) for baseline, mu1(x) for untreat and mu2(x) for treated
        model.eval()
        a_train = model(X_train).squeeze()

    return a_train.detach().numpy()
    








