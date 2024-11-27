import autograd.numpy as np
from autograd import hessian, value_and_grad
from scipy.optimize import minimize



squared_exponential = lambda tau_squared, kappa, scale: kappa**2*np.exp(-0.5*tau_squared/scale**2)
sigmoid = lambda x: 1./(1+np.exp(-x))

def create_se_kernel(X1, X2, theta, jitter=1e-6):
    """ returns the NxM kernel matrix between the two sets of input X1 and X2 
    
    arguments:
    X1     -- NxD matrix
    X2     -- MxD matrix
    theta  -- vector with 3 element: [kappa, scale, eta]
    jitter -- scalar
    
    returns NxM matrix    
    """

    kappa = theta[0]
    scale = theta[1]

    # compute all the pairwise squared distances efficiently
    dists_squared = np.sum((np.expand_dims(X1, 1) - np.expand_dims(X2, 0))**2, axis=-1)
    
    # squared exponential covariance function
    K = squared_exponential(dists_squared, kappa, scale)
    
    # add jitter for numerical stability
    if len(X1) == len(X2) and np.allclose(X1, X2):
        K = K + jitter*np.identity(len(X1))
    
    return K


def log_lik_bernoulli(y, t): 
    """ implement log p(t=1|y) using the sigmoid inverse link function """
    p = sigmoid(y)
    return t.ravel()*np.log(p) + (1-t.ravel())*np.log(1-p)

#######################################################################################
# Gaussian process model with non-Gaussian likelihoods
########################################################################################

class GaussianProcessModel(object):
    
    def __init__(self, Phi, t, theta, log_lik_fun):
        
        # data
        self.Phi = Phi          # N x D
        self.t = t              # N x 1
        self.N = len(self.Phi)
                
        # hyperparameters and log likelihood function
        self.theta = theta
        self.log_lik_fun = log_lik_fun
        
        # prepare kernel for training data
        self.K = create_se_kernel(self.Phi, self.Phi, self.theta)
        self.L = np.linalg.cholesky(self.K)

        # posterior approximation
        self.compute_laplace_approximation()
        
    def log_likelihood(self, y):
        """ evaluate log likelihood for data t and latent function values y """
        return np.sum(self.log_lik_fun(y, self.t))
        
    def compute_laplace_approximation(self):
        """ computes posterior mean and covariance using Laplace approx. """
        
        # prepare objective function
        def obj(a):
            
            # mean using m = K@a
            m = self.K@a

            # log prior and likelihood contributions
            log_prior =  - 0.5*self.N*np.log(2*np.pi)  -0.5*np.sum(np.log(np.diag(self.L)**2)) -0.5*m.T@a
            log_lik = self.log_likelihood(m)
            
            # return negative log joint p(y, t)
            return - log_prior - log_lik

        # find mode/MAP of posterior using gradients
        result = minimize(value_and_grad(obj), np.zeros(len(self.t)), jac=True)

        # store result
        self.a = result.x
        self.m = self.K@self.a
        self.W = -hessian(self.log_likelihood)(self.m)
        self.A = self.W + np.linalg.inv(self.K)
        self.S = np.linalg.inv(self.K@self.W + np.identity(self.N))@self.K         
        return self
    
    def compute_posterior_y(self, Phi_pred, pointwise=True):
        """ computes the mean and covariance of  distribuion of p(y^*|t, x^*) """
        
        # compute kernel matrix for the new predictions
        Kp = create_se_kernel(Phi_pred, Phi_pred, self.theta)
        
        # compute kernel matrix between the new predictions and training set
        k = create_se_kernel(Phi_pred, self.Phi, self.theta)
        
        # compute mean
        mean = k@np.linalg.solve(self.K, self.m)
        
        # comptue variance
        h = np.linalg.solve(self.K, k.T)
        
        if pointwise:
            var = np.diag(Kp) - np.diag(h.T@(self.K-self.S)@h)
        else:
            var = Kp - h.T@(self.K-self.S)@h

        return mean[:, None], var


#######################################################################################
# Helper function for sampling multivariate Gaussians
########################################################################################


def generate_samples(mean, K, M, jitter=1e-8):
    """ returns M samples from a zero-mean Gaussian process with kernel matrix K
    
    arguments:
    K      -- NxN kernel matrix
    M      -- number of samples (scalar)
    jitter -- scalar
    returns NxM matrix
    """
    
    L = np.linalg.cholesky(K + jitter*np.identity(len(K)))
    zs = np.random.normal(0, 1, size=(len(K), M))
    fs = mean + np.dot(L, zs)
    return fs

### Predictions

def compute_predictive_prob_MC(mu_y, Sigma_y, sample_size=2000):
    """
        The function computes p(t^* = 1|t, x^*) using Monte Carlo sampling  as in eq. (2).
        The function also returns the samples generated in the process for plotting purposes

        arguments:
        mu_y             -- N x 1 vector
        Sigma_y          -- N x N matrix
        sample_size      -- positive integer

        returns:
        p                -- N   vector
        y_samples        -- sample_size x N matrix
        sigma_samples    -- sample_size x N matrix

    """

    # generate samples from y ~ N(mu, Sigma)
    y_samples = generate_samples(mu_y, Sigma_y, sample_size).T 

    # apply inverse link function (elementwise)
    sigma_samples = sigmoid(y_samples)

    # return MC estimate, samples of y and sigma(y)
    return np.mean(sigma_samples, axis=0), y_samples, sigma_samples

def predictive(X, gp):
    mu_y, var_y = gp.compute_posterior_y(X, pointwise=True)
    p, _, _ = compute_predictive_prob_MC(mu_y, np.diag(var_y))
    return p
