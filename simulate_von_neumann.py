from scipy.stats import norm
import numpy as np


def phi(x,sig=1):
    return  norm.pdf(x, 0, sig) #1.0 / (np.power(2.0 * np.pi*sig**2,0.25)) * np.exp(-np.power((x) / 2*sig, 2.0) / 4)  

def p(x,g,a,b): #probability of measurement outcome x at state a,b
    if np.isscalar(a):
        return  (abs(a)**2) * (phi(x-g)**2) + (abs(b)**2) * (phi(x+g)**2)
    elif len(x) != len(a):
        # shape (N_sim, len(x))
        return abs(a[:,None])**2 * (phi(x-g)[None,:])**2  + abs(b[:,None])**2 * (phi(x + g)[None,:])**2 
    else: # case N_sim = len(x)
        return abs(a)**2 * (phi(x-g))**2  + abs(b)**2 * (phi(x + g))**2

#Both unitaries are written in the 0/1 basis (just like the rest)
def U_z(epsilon, t):
    return np.array([[np.exp(-1j*epsilon*t),0],[0,np.exp(1j*epsilon*t)]])

def U_x(epsilon, t):
    return np.array([[np.cos(epsilon*t),-1j*np.sin(epsilon*t)],
                     [-1j*np.sin(epsilon*t),np.cos(epsilon*t)]])

def cplus_VN(x,g, a,b):#coeficent of plus state post single measurement
    return a * phi(x - g) / np.sqrt(p(x, g, a, b))

def cminus_VN(x,g,a,b): #coeficent of minus state post single measurement
    return b * phi(x + g) / np.sqrt(p(x, g, a, b))


def Xeuler_sim(N_sim, N, g, epsilon, a0=1/np.sqrt(2) ,b0=None,rm=None,rd=None, delta_t=1, seed = None,verbose=False):#
    '''
    Random walk with fixed step size, in a neaumann system. Can run multiple simulations at once.

    Input:
    N_sim: number of simulations
    N: number of measurements per simulation
    g: interaction strength
    a0: initial state coefficient plus
    b0: initial state coefficient minus
    rm: errorrate for a measurement to happen.
    rd: errorrate for a detection to happen.

    Output:
    X: array with the particle position measurement at different time steps - measurement record.
    a: array with the particle state coefficient plus at different time steps.
    b: array with the particle state coefficient minus at different time steps.

    '''
    if seed is not None: #seed for reproducibility
        np.random.seed(seed) 

    if b0 is None:
        b0 = np.sqrt(1 - abs(a0)**2) # this can include a complex phase too!
    
    if rm is None:
        rm = 0

    if rd is None:
        rd = 0
        
    assert abs(a0)**2 + abs(b0)**2 == 1, 'Initial state coefficients do not sum to 1'

    if N_sim == N: # dimensions cause problems in vectorization if these are equal
        print('N_sim and N are equal, changing N_sim to N+1')
        N_sim += 1
    
    X_span = np.linspace(-10,10,1000) if N_sim != 1000 else np.linspace(-10,10,1001)
    # create dictionaries in list to store the measurements
    measurements = [{'X':[],'t':[]} for _ in range(N_sim)]

    
    #X = np.zeros((N_sim, N)) #spot to plug in new position, one less measurement than states (starts and ends with states
    a = np.zeros((N_sim, N+1),dtype=np.complex128) #coeficitent in plus state
    b = np.zeros((N_sim, N+1),dtype=np.complex128) #coeficitent in minus state
    #m_error = np.zeros((N_sim, N+1)) #Keep track when there is not performed a measurement

    #Initial conditions
    a[:,0] = np.repeat(a0,N_sim)
    b[:,0] = np.repeat(b0,N_sim)
    #m_error[:,0] = np.repeat(0,N_sim) #Times for trajectory n to not do a measurement. 
    
    p_merror = rm*delta_t
    p_derror = rd*delta_t
    ptot_error = p_derror + p_merror
    
    if verbose: 
        print('measurement error:', p_merror)
        print('pd_succes:', p_derror)
        print('ptot_succes:', ptot_error)
        
    
    assert ptot_error >= 0 or ptot_error < 1, 'Error probability exceeds 1 or is negative, not a probability'

#    X[0] = #np.random.choice(X_span,p=p(X_span,gstrong,a[0],b[0])/p(X_span,gstrong,a[0],b[0]).sum()) #First measurement
    for i in range(N):#creating N_sim number of quantum trajectory in parallel 
        k = np.random.uniform(0,1, size=N_sim) 
        #Include sucess rate measurement or not! Remove measurement vs remove measurement record
        
        #if any(k < ptot_error): # does any error occur?
            
        ## Normal measurements
        idx_m = np.where(ptot_error < k)[0] # Index of measurements

        #Measure meter at time t_m
        P = p(X_span,g,a[:,i],b[:,i]) 
        P = P/P.sum(axis=1)[:,None]

        
        # make and save measurement
        Xmeasure = np.array([np.random.choice(X_span,p=P[idx,:]) for idx in range(N_sim)]) #Collect position measurement from the previus state, do not depend directly on the prevues position measurement!
        for j, idx in enumerate(idx_m):
            measurements[idx]['X'].append(Xmeasure[j])
            measurements[idx]['t'].append(i*delta_t)

        #Collapse system acording to meter for those that did a measurement
        a[idx_m,i+1] =  cplus_VN(Xmeasure[idx_m],g, a[idx_m,i],b[idx_m,i]) 
        b[idx_m,i+1] =  cminus_VN(Xmeasure[idx_m],g,a[idx_m,i],b[idx_m,i]) 

        if  any(k < p_merror):   
            ## Measurement errors -> no measurement and the state is the same as before
            idx_merror = np.where(k < p_merror)[0]   # Index of measurements errors

            for idx in idx_merror:
                a[idx,i+1] = a[idx,i]
                b[idx,i+1] = b[idx,i]

            a[idx_merror,i+1] = a[idx_merror,i] 
            b[idx_merror,i+1] = b[idx_merror,i]

        if any(np.logical_and(p_merror < k,k < ptot_error)):
            ## Detection errors  -> do not save measurement, but state changes
            idx_derror = np.where(np.logical_and(p_merror < k,k < ptot_error))[0]  # Index of detection errors    
            
            # No detection error
            a[idx_derror,i+1] =  cplus_VN(Xmeasure[idx_derror],g, a[idx_derror,i],b[idx_derror,i]) 
            b[idx_derror,i+1] =  cminus_VN(Xmeasure[idx_derror],g,a[idx_derror,i],b[idx_derror,i]) 
        
       
            
        #Evovle system thorught system hamilton                                                                                                                                                                                                                                                                                                     
        a[:,i+1] = (U_x(epsilon, delta_t)@np.array([a[:,i+1],b[:,i+1]]))[0]
        b[:,i+1] = (U_x(epsilon, delta_t)@np.array([a[:,i+1],b[:,i+1]]))[1]
    
    # order measurement to a np.array
    for i in range(N_sim):
        measurements[i]['X'] = np.array(measurements[i]['X'])
        measurements[i]['t'] = np.array(measurements[i]['t'])
    

    return measurements, a, b

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
    measurements, a, b = Xeuler_sim(10, 5, 0.3, np.array([[1,0],[0,-1]]), a0=1/np.sqrt(2), b0=None, r=15., delta_t=0.05, seed = 1)
    
    print(f"Measurements lenght: {len(measurements)}")
    print(f"First measurements: {measurements[0]}")
    print(f"a shape: {a.shape}")
    print(f"b shape: {b.shape}")
    