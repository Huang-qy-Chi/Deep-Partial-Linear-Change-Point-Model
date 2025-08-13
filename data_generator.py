
import numpy as np
import numpy.random as ndm

#%%————————————————————————————————————————————————————————————————————————————————————
def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_Deep(n, corr, Theta, eta):
    """
    This function generates data.
    Input:
        n: sample size,
        corr: correlation between X,
        Theta: true value of parameter (beta, gamma),
        eta: true value of change point;
    Output: 
        Y: response, 
        A: treatment,
        X: nuisance covariate,
        Z: change-point covariate,
        f_X: true value of f(X),
        g_X: true value of g(X),
        f_X_C: true value of f(X)_g(X)I(Z>eta).
    """    
    #parametric A
    A = ndm.binomial(1, 0.5, n) 
    #change point Z
    Z = ndm.normal(loc=2, scale=1, size=n) 
    Z = np.clip(Z, 1.5, 2.5)
    AC = np.vstack((A, A*(Z>eta)))
    AC = AC.T

    #nonparametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    def multivariatet(mu,Sigma,N,M):
        d = len(Sigma)
        g = np.tile(np.random.gamma(N/2,1/2,M),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
        return mu + Z/np.sqrt(g/N)
    X = multivariatet(mean,cov,5,n)
    X = np.clip(X, 0, 2) #t-distributed with [0,2]

    #DNN function f and g
    f_X = np.sqrt(X[:,0]*X[:,1])/5 + X[:,2]**2*X[:,3]/4 \
        + np.log(X[:,3]+1)/3 + np.exp(X[:,4])/2 - 1.91
    g_X = (0.2*(np.sqrt(X[:,0]*X[:,1])/5 + X[:,2]**2*X[:,3]/4 \
        + np.log(X[:,3]+1)/3 + np.exp(X[:,4])/2)**2 -1.36)
    Ind = (Z>eta)
    f_X_C = f_X+g_X*Ind

    #error  epsilon: N(0,1)
    epsilon = ndm.normal(loc=0,scale=1,size=n)
    # truncation
    # epsilon = np.clip(epsilon,-1,1)

    #output Y
    Y = AC@Theta + f_X_C +epsilon

    return {
        'Y': np.array(Y, dtype='float32'),
        'A': np.array(A, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'f_X': np.array(f_X, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'f_X_C': np.array(f_X_C, dtype='float32'),
        'Z': np.array(Z, dtype = 'float32')
    }




