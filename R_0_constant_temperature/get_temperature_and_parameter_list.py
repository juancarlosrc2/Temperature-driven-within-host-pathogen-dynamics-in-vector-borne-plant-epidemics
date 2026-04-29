import numpy as np
import pickle


# number of temperature values between (12.1 and 34.9)
dimension = 100

#R_0^* values
r_star_values = np.linspace(0.1,1.3,dimension)

# list of parameter for which calculation will be done. All parameters 1 except one that give the value of the corresponding R_0^*
parameter_list_ = []

# array with temperature and f(T) values
temperatures = np.linspace(12.1,34.9,dimension,float)

for i in range(dimension):

    j = np.random.randint(0,4)
    list_i = [1. for x in range(4)]
    if j<2:
        list_i[j] = r_star_values[i]
    else:
        list_i[j] = 1/r_star_values[i]

    
    parameter_list_.append(list_i)

print(parameter_list_)


with open("parameter_list_.pkl", "wb") as f:
    pickle.dump(parameter_list_, f)

np.savetxt('temperatures.txt',temperatures)
np.savetxt('r_star_values.txt',r_star_values)

