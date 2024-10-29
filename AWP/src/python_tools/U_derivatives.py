import numpy as np

# Define a multivariable function
def U(state):
    mu = 0.012277471
    r_1 = np.sqrt((state[0] + mu)**2 + state[1]**2 + state[2]**2)
    r_2 = np.sqrt((state[0] - 1 + mu)**2 + state[1]**2 + state[2]**2)
    return 0.5 * (state[0]**2 + state[1]**2) + (1 - mu)/r_1 + mu/r_2

def H_r(state):
    pos_diff = state[6:9] - state[0:3]
    return np.linalg.norm(pos_diff)

def H_rr(state):
    pos_diff = state[6:9] - state[0:3]
    vel_diff = state[9:12] - state[3:6]
    return (pos_diff @ vel_diff)/np.linalg.norm(pos_diff)

# Numerical derivative (first derivative)
def partial_derivative(var_index, point, func,  h=1e-7):
    point = np.array(point)
    point_plus_h = point.copy()
    point_minus_h = point.copy()
    
    point_plus_h[var_index] += h
    point_minus_h[var_index] -= h
    
    return (func(point_plus_h) - func(point_minus_h)) / (2 * h)

# Numerical second derivative
def second_derivative(var_index1, var_index2, point, func=U, h=1e-7):
    point = np.array(point)
    
    # Compute f(x + h, y + h), f(x + h, y - h), f(x - h, y + h), f(x - h, y - h)
    point_plus_plus = point.copy()
    point_plus_minus = point.copy()
    point_minus_plus = point.copy()
    point_minus_minus = point.copy()
    
    point_plus_plus[var_index1] += h
    point_plus_plus[var_index2] += h
    
    point_plus_minus[var_index1] += h
    point_plus_minus[var_index2] -= h
    
    point_minus_plus[var_index1] -= h
    point_minus_plus[var_index2] += h
    
    point_minus_minus[var_index1] -= h
    point_minus_minus[var_index2] -= h

    return (func(point_plus_plus) - func(point_plus_minus) - func(point_minus_plus) + func(point_minus_minus)) / (4 * h * h)