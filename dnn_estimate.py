import torch
import numpy as np

#%%--------DNN estimation with given Theta and eta------------
def dnn_est(train_data, val_data, test_data, Theta, eta, n_layer, n_node, n_lr, n_epoch, patiences,show_val=True):
    if show_val == True:
        print('DNN_iteration')
    # Convert training and test data to tensors
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    Y_train = torch.tensor(train_data['Y'], dtype=torch.float32)
    Z_train = torch.tensor(train_data['Z'], dtype=torch.float32)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    A_val = torch.tensor(val_data['A'], dtype=torch.float32)
    Y_val = torch.tensor(val_data['Y'], dtype=torch.float32)
    Z_val = torch.tensor(val_data['Z'], dtype=torch.float32)

    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    A_test = torch.tensor(test_data['A'], dtype=torch.float32)
    Y_test = torch.tensor(test_data['Y'], dtype=torch.float32)
    Z_test = torch.tensor(test_data['Z'], dtype=torch.float32)

    Theta = torch.tensor(Theta, dtype=torch.float32)
    eta = torch.tensor(eta, dtype=torch.float32)
    d_X = X_train.size()[1]

    # Define the DNN model : outfeature=2, f(x) untreat and g(x) treated respectively
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X, out_features=2, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0):
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
    # my_loss = torch.nn.MSELoss(reduction='mean')
    def my_loss(Z,A,X,Theta,eta,Y,fg_X):  #LS loss
        #least square
        ind = Z>eta
        ind = ind.unsqueeze(1)
        A1 = torch.ones(A.shape[0])
        A1 = A1.unsqueeze(1)
        A = A.unsqueeze(1)
        A = torch.cat((A,A*ind),dim = 1) 
        A2 = torch.cat((A1,ind),dim=1)
        mu_TX1 = torch.sum(fg_X*A2, dim = 1)
        loss_fun = 0.5*(Y-A@Theta-mu_TX1)**2  
        return loss_fun.mean()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    #Loop
    for epoch in range(n_epoch):
        # Training loop
        model.train()  # Set model to training mode
        fg_X_train= model(X_train).squeeze()
        f_X_train = fg_X_train[:,0]
        g_X_train = fg_X_train[:,1]
        f_X_C_train = f_X_train + g_X_train*(Z_train>eta)
        loss = my_loss(Z_train, A_train, X_train, Theta, eta, Y_train, fg_X_train)
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            fg_X_val = model(X_val).squeeze()
            f_X_val = fg_X_val[:,0]
            g_X_val = fg_X_val[:,1]
            f_X_C_val = f_X_val + g_X_val*(Z_val>eta)
            val_loss = my_loss(Z_val, A_val, X_val, Theta, eta, Y_val, fg_X_val)
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
        fg_X_test = model(X_test).squeeze()
        f_X_test = fg_X_test[:,0]
        g_X_test = fg_X_test[:,1]
        f_X_C_test = f_X_test+g_X_test*(Z_test>eta)




    return{
        'Early_stop': epoch,
        'Val_loss': val_loss.detach().numpy(),
        'f_train': f_X_train.detach().numpy(),
        'g_train': g_X_train.detach().numpy(),
        'f_C_train': f_X_C_train.detach().numpy(),
        'f_val': f_X_val.detach().numpy(),
        'g_val': g_X_val.detach().numpy(),
        'f_C_val': f_X_C_val.detach().numpy(),
        'f_test': f_X_test.detach().numpy(),
        'g_test': g_X_test.detach().numpy(),
        'f_C_test': f_X_C_test.detach().numpy(),
        'model': model
    }











































































