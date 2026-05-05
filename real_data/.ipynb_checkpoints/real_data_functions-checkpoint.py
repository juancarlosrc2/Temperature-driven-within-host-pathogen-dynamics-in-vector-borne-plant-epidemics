import numba as nb
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

#temperature as function of time

@nb.jit(nopython=True)
def T(t,a,b):
    return (a+b)/2+(b-a)*(-np.cos(t*2*np.pi))/2





#f(T(t)) function
@nb.jit(nopython=True)

def f(temperatures):

    #parameters
    Tg_base,Tg_1,Tg_opt,Tg_2,Tg_max=12,18,28,32,35
    
    m_2,m_3,m_4=1,-1.25,-3
    
    b_1,b_2,b_3,b_4=8,-14,35+14,105

    m_1=(4+b_1)/Tg_1
    

    
    n_data = len(temperatures)
    f_T = np.zeros(n_data)

    for T_index in range(n_data):
        
        T=temperatures[T_index]
        
        if T<Tg_base:
            
            f_T[T_index] = 0
    
        if T>=Tg_base and T<Tg_1:
            
            f_T[T_index] = m_1*T-b_1
    
        if T>=Tg_1 and T<Tg_opt:
            f_T[T_index] = m_2*T+b_2
    
        if T>=Tg_opt and T<Tg_2:
            f_T[T_index] = m_3*T+b_3
    
        if T>=Tg_2 and T<Tg_max:
            f_T[T_index] = m_4*T+b_4
    
        if T>=Tg_max:
            f_T[T_index] = 0
            
    return f_T




#\gamma(t) function
@nb.jit(nopython=True)
def gammas(f_T,n,MGDD_R):
    
    return n/MGDD_R*f_T



#CDD function
@nb.jit(nopython=True)
def g(temperatures):
    T_0 = 6

    n_data = len(temperatures)
    g_T = np.zeros(n_data)

    for T_index in range(n_data):
        T = temperatures[T_index]
        
        if T>T_0:
            g_T[T_index] = 0
    
        else:
            g_T[T_index] = 6-T
            
    return g_T




#\chi(t) function
@nb.jit(nopython=True)
def chis(g_T,n,MGDD_R):
    
    return n/MGDD_R*g_T





#\mathcal{F} function
@nb.jit(nopython=True)
def F(MGDD):
    return 1/(1+np.exp(-0.012*(MGDD-975)))





#function to get the daily mean array

#this function get a temperature array and take the daily mean, the output is a vector with a temperature value (the mean) for
#each day 

@nb.jit(nopython=True)

def daily_mean_arrays(array,day_fraction):

    l_array = len(array)

    l_daily_mean_array = ceil(l_array/day_fraction)
    
    daily_mean_array = np.zeros(l_array)

    for k in range(l_daily_mean_array):
        
        daily_mean_array[k*day_fraction:(k+1)*day_fraction] = np.mean(array[k*day_fraction:(k+1)*day_fraction])
        
    
    return daily_mean_array










#function to get the derivatives of each compartment

#this function takes an array "X" with the values of the population in each compartment, an array "alphas" with the values of
# \alpha_i for all i, the value of \gamma and the value \chi, and return an array with the derivatives of each compartment

#this fucntion is used in the Vector_borne function

@nb.jit(nopython=True)

def dX_ms(X,alphas,gamma,chi,n,beta,mu,Gamma,delta,N_H,N_v):
    
    #the array "derivatives" store the values of the derivative of each compartment
    #the last five elements of contain the values of I_{n+1}, S_H, R_H, S_v and I_v. The rest contain the values I_{i} 
    derivatives=np.zeros(n+5,float)
    I_n1,S_H,R_H,S_v,I_v=X[-5],X[-4],X[-3],X[-2],X[-1]

    
    #
    sum = np.dot(X[:n+1],alphas)
    
    #derivatives of I_i
    
    derivatives[0]=beta*S_H*I_v/N_H-gamma*X[0] +chi*(X[1]-X[0]) 
    derivatives[1:n]=np.array([gamma*(X[x-1]-X[x])+chi*(X[x+1]-X[x]) for x in range(1,n)])
    derivatives[n-1] = gamma*(X[n-2]-X[n-1])+chi*(-X[n-1])
    
    
    #derivatives of the rest of compartments
    derivatives[-5]=gamma*X[-6]-Gamma*X[-5]
    derivatives[-4]=-beta*S_H*I_v/N_H + chi*X[0]                                          
    derivatives[-3]=Gamma*X[-5]                                                   
    derivatives[-2]=delta*N_v-S_v*sum/N_H - mu*S_v                                    
    derivatives[-1]=S_v*sum/N_H -mu*I_v
    
    
    return derivatives


#-----------------------------------------------------------------------------------

def plot_compartments_vs_t(X,times,show=True):

    n = np.shape(X)[0] -5

    fig, ax = plt.subplots(2,3,figsize=(15,10))


    ax[0,0].plot(times,X[-1],"r"),ax[0,1].plot(times,X[-2],"m"),ax[0,2].plot(times,X[-3],"b")




    ax[0,0].set_xlabel("$t$",fontsize=14) ,ax[0,1].set_xlabel("$t$",fontsize=14), ax[0,2].set_xlabel("$t$",fontsize=14)
    ax[0,0].set_ylabel("$I_v$",fontsize=14) ,ax[0,1].set_ylabel("$S_v$",fontsize=14), ax[0,2].set_ylabel("$R_H$",fontsize=14)



    ax[1,0].plot(times,np.sum(X[:n+1,:],axis=0),"r"),ax[1,1].plot(times,X[-4],"m")



    ax[1,2].plot(times,X[-4],"m"), ax[1,2].plot(times,np.sum(X[:n+1,:],axis=0),"r"), ax[1,2].plot(times,X[-3],"b")


    ax[1,0].set_xlabel("$t$",fontsize=14) ,ax[1,1].set_xlabel("$t$",fontsize=14), ax[1,2].set_xlabel("$t$",fontsize=14)
    ax[1,0].set_ylabel("$I_H$",fontsize=14) ,ax[1,1].set_ylabel("$S_H$",fontsize=14) 

    ax[1,2].legend(["$S_H$","$I_H$","$R_H$","$S_v$","$I_v$"])

    plt.tight_layout()



    if show:
        plt.show()

    plt.close(fig)
    
    return fig


#Vector borne mass action-------------------------------------------


#function to get the derivatives of each compartment

#this function takes an array "X" with the values of the population in each compartment, an array "alphas" with the values of
# \alpha_i for all i, the value of \gamma and the value \chi, and return an array with the derivatives of each compartment

#this fucntion is used in the Vector_borne function


@nb.jit(nopython=True)

def Vector_borne_ms(parameters,initial_conditions,T_t,n,day_fraction,N_H,N_v):


    #parameters and initial conditions
    beta,Alpha,MGDD_R,Gamma,mu,delta=parameters
    
    n = int(n)
    I_H_0,I_v_0 = initial_conditions
    

    
    #
    Delta_MGDD=MGDD_R/n
    
    MGDD = np.array([x*Delta_MGDD for x in range(n+1)])
    
    alphas = Alpha*F(MGDD)


    #temperature function and rates functions
    
    f_t = f(T_t)
    g_t = g(T_t)

    """
    f_t = daily_mean_arrays(f_t,day_fraction)
    g_t = daily_mean_arrays(g_t,day_fraction)
    """

    # number of full days
    n_days = len(f_t) // 24



    f_t = f_t[:n_days * 24].reshape(n_days, 24)

    daily_f = np.empty(n_days)

    for i in range(n_days):
        s = 0.0
        for j in range(24):
            s += f_t[i, j]
        daily_f[i] = s / 24.0
    f_t = daily_f


    g_t = g_t[:n_days * 24].reshape(n_days, 24)

    daily_g = np.empty(n_days)

    for i in range(n_days):
        s = 0.0
        for j in range(24):
            s += g_t[i, j]
        daily_g[i] = s / 24.0
    g_t = daily_g



    f_t = np.array([x for x in f_t for _ in range(day_fraction)])
    g_t = np.array([x for x in g_t for _ in range(day_fraction)])

    factor = 365


    """
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if a<6:
        t_6 = 1 - np.arccos(6*2/(a-b)-(a+b)/(a-b))/(2*np.pi)
        index_i = np.argmin(np.abs(times-t_6))
        g_t[:index_i] = 0
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    
    gamma_t = gammas(f_t,n,MGDD_R)*factor
    chi_t = chis(g_t,n,MGDD_R)*factor

    number_of_days = len(T_t)
    years = number_of_days/365

    
    #integration time, time step, and time array
    dt = 1/(day_fraction*365)
    N = len(f_t)#*day_fraction#int(np.ceil(years/dt))
    times=np.array(list(range(N)))*dt
    
    
    #array to store the values of the population in each compartment at each time
    X=np.zeros((n+5,N),float)
    
    

    #array to store the values of the population in each compartment at time dt(k-1)
    X_0=np.zeros(n+5,float)


    #X_0 = np.array([0. for x in range(n+5)])
    
    X_0[0]=N_H*I_H_0
    X_0[n+1:]=np.array([N_H*(1-I_H_0),0,N_v*(1-I_v_0),N_v*I_v_0])
    
    

    #array to store the values of the population in each compartment at time en kdt
    X_t=np.copy(X_0)
    


    #the column 0 of the array "X" store the initial conditions
    X[:,0]=X_t

    
    
    #----------------RK4--------------------
    
    
    for k in range(1,N):
        
        gamma_1, gamma_2, gamma_3  = gamma_t[k-1], gamma_t[k-1], gamma_t[k]
        chi_1, chi_2, chi_3  = chi_t[k-1], chi_t[k-1], chi_t[k]
          
    
        
        k1 = dX_ms(X_t,alphas,gamma_1,chi_1,n,beta,mu,Gamma,delta,N_H,N_v)
    
        k2 = dX_ms(X_t + k1*dt/2,alphas,gamma_2,chi_2,n,beta,mu,Gamma,delta,N_H,N_v)
        k3 = dX_ms(X_t + k2*dt/2,alphas,gamma_2,chi_2,n,beta,mu,Gamma,delta,N_H,N_v)
        k4 = dX_ms(X_t + k3*dt,alphas,gamma_3,chi_3,n,beta,mu,Gamma,delta,N_H,N_v)
    
        X_t = X_0 + 1/6*(k1+2*k2+2*k3+k4)*dt
    
        X[:,k] = X_t
        X_0 = X_t
        
    #---------------------------------------------------
    
    return X,times





@nb.jit(nopython=True)

def Vector_borne_ms_Rinf(parameters,initial_conditions,T_t,n,day_fraction,N_H,N_v):


    #parameters and initial conditions
    beta,Alpha,MGDD_R,Gamma,mu,delta=parameters
    
    n = int(n)
    I_H_0,I_v_0 = initial_conditions
    

    
    #
    Delta_MGDD=MGDD_R/n
    
    MGDD = np.array([x*Delta_MGDD for x in range(n+1)])
    
    alphas = Alpha*F(MGDD)


    #temperature function and rates functions
    
    f_t = f(T_t)
    g_t = g(T_t)

    f_t = np.array([x for x in f_t for _ in range(day_fraction)])
    g_t = np.array([x for x in g_t for _ in range(day_fraction)])

    factor = 365
    
    gamma_t = gammas(f_t,n,MGDD_R)*factor
    chi_t = chis(g_t,n,MGDD_R)*factor

    number_of_days = len(T_t)
    years = number_of_days/365

    
    #integration time, time step, and time array
    dt = 1/(day_fraction*365)
    N = len(f_t)#*day_fraction#int(np.ceil(years/dt))

    N_store = int(np.ceil(N/day_fraction))

    times=np.array(list(range(N)))*dt
    
    
    #---------------array to store the values of the population in each compartment at each time-----------------
    X=np.zeros((n+5,N_store),float)
    
    

    #array to store the values of the population in each compartment at time dt(k-1)
    X_0=np.zeros(n+5,float)


    #X_0 = np.array([0. for x in range(n+5)])
    
    X_0[0]=N_H*I_H_0
    X_0[n+1:]=np.array([N_H*(1-I_H_0),0,N_v*(1-I_v_0),N_v*I_v_0])
    
    

    #array to store the values of the population in each compartment at time en kdt
    X_t=np.copy(X_0)
    


    #the column 0 of the array "X" store the initial conditions
    X[:,0]=X_t

    
    
    #----------------RK4--------------------
    
    
    for k in range(1,N):
        
        gamma_1, gamma_2, gamma_3  = gamma_t[k-1], gamma_t[k-1], gamma_t[k]
        chi_1, chi_2, chi_3  = chi_t[k-1], chi_t[k-1], chi_t[k]
          
    
        
        k1 = dX_ms(X_t,alphas,gamma_1,chi_1,n,beta,mu,Gamma,delta,N_H,N_v)
    
        k2 = dX_ms(X_t + k1*dt/2,alphas,gamma_2,chi_2,n,beta,mu,Gamma,delta,N_H,N_v)
        k3 = dX_ms(X_t + k2*dt/2,alphas,gamma_2,chi_2,n,beta,mu,Gamma,delta,N_H,N_v)
        k4 = dX_ms(X_t + k3*dt,alphas,gamma_3,chi_3,n,beta,mu,Gamma,delta,N_H,N_v)
    
        X_t = X_0 + 1/6*(k1+2*k2+2*k3+k4)*dt


        if k%day_fraction == 0:
            k_store = k//day_fraction
            X[:,k_store] = X_t

        X_0 = X_t


        years_check = 365*150*day_fraction

        if k > years_check and k%(years_check) == 0:

            k_store = k//day_fraction
            
            R_H_check = np.copy(X[-3][:k_store][-10*365::2*365])
            R_H_check_diff = np.diff(R_H_check,n=1)


            if np.all(R_H_check_diff<1e-6) & np.all(R_H_check>I_H_0):
        
                #print(R_H_check[-1],k*dt)
                break
            
            if np.max(X[80:n+1])<1e-9:
                break
        
    #---------------------------------------------------
    
    return X,times[::day_fraction]



#-------------------------------------------------e5-------------------------------------------
@nb.jit(nopython=True)
def get_daily_mean_numba(arr):
    n = 24
    length = len(arr) // n * n
    daily_count = length // n
    result = np.empty(daily_count)
    
    for i in range(daily_count):
        s = 0.0
        for j in range(n):
            s += arr[i*n + j]
        result[i] = s / n
    return result



@nb.jit(nopython=True)

def Vector_borne_ms_e5(parameters,initial_conditions,T_t,n,day_fraction,N_H,N_v):


    #parameters and initial conditions
    beta,Alpha,MGDD_R,Gamma,mu,delta=parameters
    
    n = int(n)
    I_H_0,I_v_0 = initial_conditions
    

    
    #
    Delta_MGDD=MGDD_R/n
    
    MGDD = np.array([x*Delta_MGDD for x in range(n+1)])
    
    alphas = Alpha*F(MGDD)


    #temperature function and rates functions
    
    f_t = f(T_t)
    g_t = g(T_t)
    
    f_t = get_daily_mean_numba(f_t)
    g_t = get_daily_mean_numba(g_t)
    
    """
    f_t = daily_mean_arrays(f_t,24)
    g_t = daily_mean_arrays(g_t,24)
    """

    f_t = np.array([x for x in f_t for _ in range(day_fraction)])
    g_t = np.array([x for x in g_t for _ in range(day_fraction)])

    factor = 365


    """
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if a<6:
        t_6 = 1 - np.arccos(6*2/(a-b)-(a+b)/(a-b))/(2*np.pi)
        index_i = np.argmin(np.abs(times-t_6))
        g_t[:index_i] = 0
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    
    gamma_t = gammas(f_t,n,MGDD_R)*factor
    chi_t = chis(g_t,n,MGDD_R)*factor

    number_of_days = len(T_t)
    years = number_of_days/365

    
    #integration time, time step, and time array
    dt = 1/(day_fraction*365)
    N = len(f_t)#*day_fraction#int(np.ceil(years/dt))
    times=np.array(list(range(N)))*dt
    
    
    #array to store the values of the population in each compartment at each time
    X=np.zeros((n+5,N),float)
    
    

    #array to store the values of the population in each compartment at time dt(k-1)
    X_0=np.zeros(n+5,float)


    #X_0 = np.array([0. for x in range(n+5)])
    
    X_0[0]=N_H*I_H_0
    X_0[n+1:]=np.array([N_H*(1-I_H_0),0,N_v*(1-I_v_0),N_v*I_v_0])
    
    

    #array to store the values of the population in each compartment at time en kdt
    X_t=np.copy(X_0)
    


    #the column 0 of the array "X" store the initial conditions
    X[:,0]=X_t

    
    
    #----------------RK4--------------------
    
    
    for k in range(1,N):
        
        gamma_1, gamma_2, gamma_3  = gamma_t[k-1], gamma_t[k-1], gamma_t[k]
        chi_1, chi_2, chi_3  = chi_t[k-1], chi_t[k-1], chi_t[k]
          
    
        
        k1 = dX_ms(X_t,alphas,gamma_1,chi_1,n,beta,mu,Gamma,delta,N_H,N_v)
    
        k2 = dX_ms(X_t + k1*dt/2,alphas,gamma_2,chi_2,n,beta,mu,Gamma,delta,N_H,N_v)
        k3 = dX_ms(X_t + k2*dt/2,alphas,gamma_2,chi_2,n,beta,mu,Gamma,delta,N_H,N_v)
        k4 = dX_ms(X_t + k3*dt,alphas,gamma_3,chi_3,n,beta,mu,Gamma,delta,N_H,N_v)
    
        X_t = X_0 + 1/6*(k1+2*k2+2*k3+k4)*dt
    
        X[:,k] = X_t
        X_0 = X_t
        
    #---------------------------------------------------
    
    return X,times



@nb.jit(nopython=True)

def Vector_borne_ms_Rinf_e5(parameters,initial_conditions,T_t,n,day_fraction,N_H,N_v):


    #parameters and initial conditions
    beta,Alpha,MGDD_R,Gamma,mu,delta=parameters
    
    n = int(n)
    I_H_0,I_v_0 = initial_conditions
    

    
    #
    Delta_MGDD=MGDD_R/n
    
    MGDD = np.array([x*Delta_MGDD for x in range(n+1)])
    
    alphas = Alpha*F(MGDD)


    #temperature function and rates functions
    
    f_t = f(T_t)
    g_t = g(T_t)

    
    f_t = get_daily_mean_numba(f_t)
    g_t = get_daily_mean_numba(g_t)
    

    """
    f_t = daily_mean_arrays(f_t,24)
    g_t = daily_mean_arrays(g_t,24)
    """


    f_t = np.array([x for x in f_t for _ in range(day_fraction)])
    g_t = np.array([x for x in g_t for _ in range(day_fraction)])

    factor = 365
    
    gamma_t = gammas(f_t,n,MGDD_R)*factor
    chi_t = chis(g_t,n,MGDD_R)*factor

    number_of_days = len(f_t)
    years = number_of_days/365

    
    #integration time, time step, and time array
    dt = 1/(day_fraction*365)
    N = len(f_t)#*day_fraction#int(np.ceil(years/dt))

    N_store = int(np.ceil(N/day_fraction))

    times=np.array(list(range(N)))*dt
    
    
    #---------------array to store the values of the population in each compartment at each time-----------------
    X=np.zeros((n+5,N_store),float)
    
    

    #array to store the values of the population in each compartment at time dt(k-1)
    X_0=np.zeros(n+5,float)


    #X_0 = np.array([0. for x in range(n+5)])
    
    X_0[0]=N_H*I_H_0
    X_0[n+1:]=np.array([N_H*(1-I_H_0),0,N_v*(1-I_v_0),N_v*I_v_0])
    
    

    #array to store the values of the population in each compartment at time en kdt
    X_t=np.copy(X_0)
    


    #the column 0 of the array "X" store the initial conditions
    X[:,0]=X_t

    
    
    #----------------RK4--------------------
    
    
    for k in range(1,N):
        
        gamma_1, gamma_2, gamma_3  = gamma_t[k-1], gamma_t[k-1], gamma_t[k]
        chi_1, chi_2, chi_3  = chi_t[k-1], chi_t[k-1], chi_t[k]
          
    
        
        k1 = dX_ms(X_t,alphas,gamma_1,chi_1,n,beta,mu,Gamma,delta,N_H,N_v)
    
        k2 = dX_ms(X_t + k1*dt/2,alphas,gamma_2,chi_2,n,beta,mu,Gamma,delta,N_H,N_v)
        k3 = dX_ms(X_t + k2*dt/2,alphas,gamma_2,chi_2,n,beta,mu,Gamma,delta,N_H,N_v)
        k4 = dX_ms(X_t + k3*dt,alphas,gamma_3,chi_3,n,beta,mu,Gamma,delta,N_H,N_v)
    
        X_t = X_0 + 1/6*(k1+2*k2+2*k3+k4)*dt


        if k%day_fraction == 0:
            k_store = k//day_fraction
            X[:,k_store] = X_t

        X_0 = X_t


        years_check = 365*150*day_fraction

        if k > years_check and k%(years_check) == 0:

            k_store = k//day_fraction
            
            R_H_check = np.copy(X[-3][:k_store][-10*365::2*365])
            R_H_check_diff = np.diff(R_H_check,n=1)


            if np.all(R_H_check_diff<1e-6) & np.all(R_H_check>I_H_0):
        
                #print(R_H_check[-1],k*dt)
                break
            
            if np.max(X[80:n+1])<1e-9:
                break
        
    #---------------------------------------------------
    
    return X,times[::day_fraction]