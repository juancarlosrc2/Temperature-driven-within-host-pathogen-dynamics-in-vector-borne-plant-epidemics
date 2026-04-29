import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(".."))
import vector_borne_functions as vbf

T = vbf.T
f = vbf.f
gammas = vbf.gammas
g = vbf.g
chis = vbf.chis
F = vbf.F
daily_mean_arrays = vbf.daily_mean_arrays
dX = vbf.dX
Vector_borne = vbf.Vector_borne



dX_Rinf = vbf.dX_Rinf
Vector_borne_Rinf = vbf.Vector_borne_Rinf

def sum_alpha_i(MGDD_R):
    

    c_1 = 0.012
    c_2 = 975
    
    sum_alpha = MGDD_R + 1/c_1*np.log( (1 + np.exp(-c_1 * (MGDD_R-c_2) ) ) /(1 + np.exp(c_1 * c_2) ) ) 

    return sum_alpha


def get_rinf_tpeak_r0(parameter_list_,temperatures):

    dimension = len(temperatures)
    n_parameters = len(parameter_list_)



    R_inf_t_values= np.zeros((n_parameters,dimension))
    R_0_t_values = np.zeros((n_parameters,dimension))
    peak_time_values = np.zeros((n_parameters,dimension))

    years_R = 2000
    MGDD_R = 1500.
    n = 200
    day_fraction = 3
    factor = 365


    I_H_0 = 0.
    I_v_0 = 0.001

    sum_alpha = sum_alpha_i(MGDD_R)

    for parameter_combination_index in range(n_parameters):
        
        
        Alpha, beta, Gamma, mu = parameter_list_[parameter_combination_index]
        delta = mu
        parameters,initial_conditions=[beta,Alpha,MGDD_R,Gamma,mu,delta],[I_H_0,I_v_0]

        
        print('parameter combination index:',parameter_combination_index)

        
        for temperature_index in range(dimension):

            a = temperatures[temperature_index]
            b = a
            

            print(temperature_index)
            
            X,times=Vector_borne_Rinf(parameters,initial_conditions,years_R,a,b,n,day_fraction)
            max_R_inf = np.max(X[-3])  
            R_inf_t_values[parameter_combination_index,temperature_index] = max_R_inf

            if max_R_inf > 0.1:

                I_H = np.sum(X[:n+1,:],axis=0)
                peak_time_values[parameter_combination_index,temperature_index] = np.argmax(I_H)/(365*day_fraction)

            

            f_t_constant = f(np.array([a,a]))[0]*factor
            #f_t_constant = f([a])*factor
            
            R_0 = (sum_alpha/f_t_constant*Gamma + 1)*(beta*Alpha)/(mu*Gamma)

            R_0_t_values[parameter_combination_index,temperature_index] = R_0

    return R_inf_t_values, R_0_t_values, peak_time_values


with open("parameter_list_.pkl", "rb") as efe:
    parameter_list_ = pickle.load(efe)

temperatures = np.loadtxt('temperatures.txt')


R_inf_t_values, R_0_t_values, peak_time_values = get_rinf_tpeak_r0(parameter_list_,temperatures)


np.savetxt('r_infty_constant.txt', R_inf_t_values)
np.savetxt('r_0_constant.txt', R_0_t_values)
np.savetxt('peak_time_constant.txt', peak_time_values)
