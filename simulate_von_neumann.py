from scipy.stats import norm
import numpy as np


def phi(x,sig=1):
    return  norm.pdf(x, 0, sig) #1.0 / (np.power(2.0 * np.pi*sig**2,0.25)) * np.exp(-np.power((x) / 2*sig, 2.0) / 4)  

def p(x,g,a,b): #probability of measurement outcome x at state a,b
    if np.isscalar(a):
        return  (abs(a)**2) * (phi(x-g)**2) + (abs(b)**2) * (phi(x+g)**2)#=
    elif len(x) != len(a):
        # shape (N_sim, len(x))
        return abs(a[:,None])**2 * (phi(x-g)[None,:])**2  + abs(b[:,None])**2 * (phi(x + g)[None,:])**2 
    else: # case N_sim = len(x)
        return abs(a)**2 * (phi(x-g))**2  + abs(b)**2 * (phi(x + g))**2

#Perhaps better to just write up the hamilton???
def cplus_H(a,H):
    return H@np.array([1,0])*a

def cminus_H(b,H):
    return  H@np.array([0,1])*b

def cplus_VN(x,g, a,b):#coeficent of plus state post single measurement
    return a * phi(x - g) / np.sqrt(p(x, g, a, b))


def cminus_VN(x,g,a,b): #coeficent of minus state post single measurement
    return b * phi(x + g) / np.sqrt(p(x, g, a, b))


def Xeuler_sim(N_sim, N, g, U_s, a0=1/np.sqrt(2) ,b0=None,r=0, delta_t=0.05, seed = None):#
    '''
    Random walk with fixed step size, in a neaumann system. Can run multiple simulations at once.

    Input:
    N_sim: number of simulations
    N: number of measurements per simulation
    g: interaction strength
    a0: initial state coefficient plus
    b0: initial state coefficient minus
    r: successrate for a measurement to happen.

    Output:
    X: array with the particle position measurement at different time steps - measurement record.
    a: array with the particle state coefficient plus at different time steps.
    b: array with the particle state coefficient minus at different time steps.

    '''
    if seed is not None: #seed for reproducibility
        np.random.seed(seed) 

    if b0 is None:
        b0 = np.sqrt(1 - abs(a0)**2) # this can include a complex phase too!
        
    assert abs(a0)**2 + abs(b0)**2 == 1, 'Initial state coefficients do not sum to 1'

    if N_sim == N: # dimensions cause problems in vectorization if these are equal
        print('N_sim and N are equal, changing N_sim to N+1')
        N_sim += 1
    
    X_span = np.linspace(-10,10,1000) if N_sim != 1000 else np.linspace(-10,10,1001)
    t_m = []

    X = np.zeros((N_sim, N)) #spot to plug in new position, one less measurement than states (starts and ends with states
    a = np.zeros((N_sim, N+1),dtype=np.complex64) #coeficitent in plus state
    b = np.zeros((N_sim, N+1),dtype=np.complex64) #coeficitent in minus state
    m_error = np.zeros((N_sim, N+1)) #Keep track when there is not performed a measurement

    #Initial conditions
    a[:,0] = np.repeat(a0,N_sim)
    b[:,0] = np.repeat(b0,N_sim)
    m_error[:,0] = np.repeat(0,N_sim) #Times for trajectory n to not do a measurement. 
    

#    X[0] = #np.random.choice(X_span,p=p(X_span,gstrong,a[0],b[0])/p(X_span,gstrong,a[0],b[0]).sum()) #First measurement
    for i in range(N):#creating N_sim number of quantum trajectory in parallel 
        k = np.random.uniform(0,1, size=N_sim) 
        #Include sucess rate measurement or not! Remove measurement vs remove measurement record
        if k<r*delta_t:#Measurement is done
            #Measure meter at time t_m
            P = p(X_span,g,a[:,i],b[:,i]) 
            P = P/P.sum(axis=1)[:,None]
            X[:, i] = np.array([np.random.choice(X_span,p=P[i,:]) for i in range(N_sim)])#Collect position measurement from the previus state, do not depend directly on the prevues position measurement!
            t_m.append(i*delta_t)
            #Collapse system acording to meter
            a[:,i+1] =  cplus_VN(X[:,i],g, a[:,i],b[:,i]) 
            b[:,i+1] =  cminus_VN(X[:,i],g,a[:,i],b[:,i])#Depends on initial state but next. Can it be complex from time evolution)?
            
        else: #No measurement -> no partial collapse of the system in this timestep.
            a[:,i+1] = a[:,i] #Depends on initial state but next. Can it be complex from time evolution)?
            b[:,i+1] = b[:,i]
            m_error[:,i+1] += 1

        #Evovle system thorught system hamilton                                                                                                                                                                                                                                                                                                     
        a[:,i+1] = (U_s@np.array([1,0]))[0]*a[:,i]
        b[:,i+1] = (U_s@np.array([0,1]))[1]*b[:,i]
    return X,t_m,a,b,m_error

#Compare epsilon with delta t (rabi oscilation and measurement) Plot t_rabi>>t_95, t_rabi approx t_95 and t_rabi<<t_95

#how long does it take before 95% of the states
#Find average a of a's bigger and 1/2 and smaller than 1/2

def add_wnoise(x,sigma_wn):
    x += np.random.normal(0,sigma_wn,len(x))
    return x


'''
def Xeuler_event(tstop=tstop,N0=N0): #Let's start with constant rate!
    T=[0]
    N = [N0]
    while sum(T)<=tstop:
        a = np.random.uniform(0,1)
        if 0<a<r(N[-1])/K(N[-1]):
            tau = np.random.exponential(1/r(N[-1])) #Sample of time intervals for constant rate
            N.append(N[-1]-1)
        else:
            tau = np.random.exponential(1/g(N[-1]))
            N.append(N[-1]+1)
        T.append(tau)
        #if len(T)==100:
        #    break
    return T, N
'''

if __name__ == '__main__':
    print('hej')
    print(Xeuler_sim(10, 5, 0.3, np.array([[1,0],[0,-1]]), a0=1/np.sqrt(2)))