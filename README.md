# Deep-Partial-Linear-Change-Point-Model
This code mainly focuses on estimating a partial linear model with a change point:
$$Y=A^\top\beta+f(X)+(A^\top\gamma+g(X))I(Z>\eta)+\epsilon,$$
where $Y\in R$ represents the response, $A\in R^p$ means the covariate (treatment) with linear effects, $X\in R^r$ denotes
other covariates that may have complex behavior, $I(\cdot)$ means the indicator function, and $Z\in R$ represents the change-point covariate. 
An additional impact $A_i^\top\gamma+g(X_i)$ will occur if the $i$ th observation satisfies $Z_i>\eta$, where $\eta\in R$ 
is an unknown threshold.
The function inputs data $(Y_i,A_i,X_i,Z_i)$ and output the estimation of the parameter $(\beta,\gamma)$, the change-point
$\eta$ and the estimation of two functions $(f,g)$. 

Readers can attach to the file $\textsf{main0.ipynb}$, which automatically imports functions in $\textbf{\textsf{.py}}$, and it reports the bias, SSE, ESE and CP (close to 0.95) of $(\beta,\gamma)$ (with semiparametric efficiency and asymptotic normality), the bias of $\eta$ and the relative error (RE) of $(f,g)$ on test data with 10 replications. 
To avoid overfitting, the hyperparameters n_lr (learning rate), n_node (width), n_layer (depth), n_epoch (max epoch)
and patiences (when to early stop) need to be adjusted, and the grid for $\eta$ grid search can be adjusted 
by seq (initial=0.01).

The code requires Python version>=3.13.2, Numpy >=2.2.4, Pytorch>=2.7.0 and Scipy>=1.15.2. 

Copyright © 2025 Q. Huang. All rights reserved. 

13/08/2025, HungHom, Kowloon, HongKong, China.





