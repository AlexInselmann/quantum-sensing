from scipy.stats import norm
import numpy as np

def phi(x,sig=1):
    return  norm.pdf(x, 0, sig) #1.0 / (np.power(2.0 * np.pi*sig**2,0.25)) * np.exp(-np.power((x) / 2*sig, 2.0) / 4)  

def p(x,g,a,b): #probability of measurement outcome x
    if np.isscalar(a):
        return  (a**2) * (phi(x-g)**2) + (b**2) * (phi(x+g)**2)
    elif len(x) != len(a):
        # shape (N_sim, len(x))
        return (a**2)[:,None] * (phi(x-g)[None,:])**2  + (b**2)[:,None] * (phi(x + g)[None,:])**2 
    else: # case N_sim = len(x)
        return (a**2) * (phi(x-g))**2  + (b**2) * (phi(x + g))**2

def cplus_VN(x,g, a,b):#coeficent of plus state post single measurement
    return a * phi(x - g) / np.sqrt(p(x, g, a, b))
    

def cminus_VN(x,g,a,b): #coeficent of minus state post single measurement
    return b * phi(x + g) / np.sqrt(p(x, g, a, b))



def Xeuler_sim(N_sim, N, g, a0=1/np.sqrt(2) ,b0=None,r=0, delta_t=1, seed = 42):#
    '''
    Random walk with fixed step size, in a neaumann system. Can run multiple simulations at once.

    Input:
    N_sim: number of simulations
    N: number of measurements per simulation
    g: interaction strength
    a0: initial state coefficient plus
    b0: initial state coefficient minus
    r: rate for bit flip 

    Output:
    X: array with the particle position measurement at different time steps - measurement record.
    a: array with the particle state coefficient plus at different time steps.
    b: array with the particle state coefficient minus at different time steps.

    '''
    np.random.seed(seed) #seed for reproducibility

    if b0 is None:
        b0 = np.sqrt(1 - a0**2)
        
    assert a0**2 + b0**2 == 1, 'Initial state coefficients do not sum to 1'

    if N_sim == N: # dimensions course problems in vectorization if these are equal
        print('N_sim and N are equal, changing N_sim to N+1')
        N_sim += 1
        
    
    X_span = np.linspace(-10,10,1000) if N_sim != 1000 else np.linspace(-10,10,1001)
    
    X = np.zeros((N_sim, N)) #spot to plug in new position, one less measurement than states (starts and ends with states
    a = np.zeros((N_sim, N+1)) #coeficitent in plus state
    b = np.zeros((N_sim, N+1)) #coeficitent in minus state

    #Initial conditions
    a[:,0] = np.repeat(a0,N_sim)
    b[:,0] = np.repeat(b0,N_sim)

#    X[0] = #np.random.choice(X_span,p=p(X_span,gstrong,a[0],b[0])/p(X_span,gstrong,a[0],b[0]).sum()) #First measurement
    for i in range(N):#creating N_sim number of quantum trajectory in parallel
        P = p(X_span,g,a[:,i],b[:,i]) 
        P = P/P.sum(axis=1)[:,None]
        X[:, i] = np.array([np.random.choice(X_span,p=P[i,:]) for i in range(N_sim)])#Collect position measurement from the previus state, do not depend directly on the prevues position measurement!
        k = np.random.uniform(0,1) 
        if 0<k<r*delta_t:#bitflip!
            a[:,i+1] =  cminus_VN(X[:,i],g,a[:,i],b[:,i])
            b[:,i+1] = cplus_VN(X[:,i],g, a[:,i],b[:,i]) #Depends on initial state but next. Can it be complex from time evolution)?
        else:
            a[:,i+1] = cplus_VN(X[:,i],g, a[:,i],b[:,i]) #Depends on initial state but next. Can it be complex from time evolution)?
            b[:,i+1] = cminus_VN(X[:,i],g,a[:,i],b[:,i])
    return X,a,b


