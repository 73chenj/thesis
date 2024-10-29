# The LiAISON EKF using the Capstone mission and another made up ELFO mission
import numpy as np
from pymatreader import read_mat
import spiceypy as sp
from CR3BP import CR3BP
import rotations
import U_derivatives
import matplotlib.pyplot as plt

mat_capstone = read_mat('MATLAB/HALO/output/ORBdataCapstone.mat')
mat_ELFO = read_mat('MATLAB/HALO/output/ORBdataClementine.mat')

seq_capstone = mat_capstone['orb']['seq']['a']['XJ2000']
seq_ELFO = mat_ELFO['orb']['seq']['a']['XJ2000']
time_seq_capstone = mat_capstone['orb']['seq']['a']['t'] - mat_capstone['orb']['seq']['a']['t'][0]
time_seq_ELFO = mat_ELFO['orb']['seq']['a']['t'] - mat_ELFO['orb']['seq']['a']['t'][0]

# cr3bt model as ground truth
d = 384400
cr3bp = CR3BP('earth-moon')
capstone_prev = capstone_init = [0.98772 - seq_capstone[0][0]/d, seq_capstone[0][1]/d, seq_capstone[0][2]/d, seq_capstone[0][3], seq_capstone[0][4], seq_capstone[0][5]]
ELFO_prev = ELFO_init = [0.98772 - seq_ELFO[0][0]/d, seq_ELFO[0][1]/d, seq_ELFO[0][2]/d, seq_ELFO[0][3], seq_ELFO[0][4], seq_ELFO[0][5]]
test_et_ELFO, test_states_ELFO = cr3bp.propagate_orbit(ELFO_init, tspan=0.921)

# To make sure the time step is constant
timestep = 30 # seconds
cr3bp_states_capstone = [capstone_init]
cr3bp_states_ELFO = [ELFO_init]
for i in range(int(1/(timestep*2.66381e-6))):
    #print(i)
    cr3bp_et_capstone, cr3bp_state_capstone = cr3bp.propagate_orbit(capstone_prev, tspan=timestep*2.66381e-6)
    cr3bp_et_ELFO, cr3bp_state_ELFO = cr3bp.propagate_orbit(ELFO_prev, tspan=timestep*2.66381e-6)
    cr3bp_states_capstone.append(cr3bp_state_capstone[-1])
    cr3bp_states_ELFO.append(cr3bp_state_ELFO[-1])
    capstone_prev = cr3bp_state_capstone[-1]
    ELFO_prev = cr3bp_state_ELFO[-1]

# Function for converting state vectors to measurements
def state_to_measurements(satellite1, satellite2):
    pos_vector_difference = np.array([satellite1[0] - satellite2[0], satellite1[1] - satellite2[1], satellite1[2] - satellite2[2]])
    vel_vector_difference = np.array([satellite1[3] - satellite2[3], satellite1[4] - satellite2[4], satellite1[5] - satellite2[5]])
    range = np.linalg.norm(pos_vector_difference)
    range_rate = np.linalg.norm(vel_vector_difference)
    return range, range_rate

# Range and range rate measurements (Every 30 seconds)
ranges = []
range_rates = []
for i in range(len(cr3bp_states_capstone)):
    range_i, range_rate_i = state_to_measurements(cr3bp_states_capstone[i], cr3bp_states_ELFO[i])
    range_i += np.random.normal(0, 1/d) # Error is represented with a Gaussian variable with mean 0m and STD 1m
    range_rate_i += np.random.normal(0, 0.00001) # Error is represented with a Gaussian variable with mean 0mm/s and STD 1mm/s
    ranges.append(range_i)
    range_rates.append(range_rate_i)
measurements = np.column_stack((ranges, range_rates))

a_priori_capstone = seq_capstone[0]
a_priori_ELFO = seq_ELFO[0]

def cr3bp_state_transition(x, mu):
    # Unpack the state vector
    x1, x2, x3, vx1, vx2, vx3 = x
    
    # Distances to primary and secondary bodies
    r1 = np.sqrt((x1 + mu)**2 + x2**2 + x3**2)
    r2 = np.sqrt((x1 - 1 + mu)**2 + x2**2 + x3**2)
    
    # Equations of motion
    ddx1 = 2 * vx2 + x1 - (1 - mu) * (x1 + mu) / r1**3 - mu * (x1 - 1 + mu) / r2**3
    ddx2 = -2 * vx1 + x2 - (1 - mu) * x2 / r1**3 - mu * x2 / r2**3
    ddx3 = -(1 - mu) * x3 / r1**3 - mu * x3 / r2**3
    
    # State transition function
    return np.array([vx1, vx2, vx3, ddx1, ddx2, ddx3])

def calculate_F_jacobian(states, f, mu, epsilon = 10e-7):
    individual_jacobians = []
    # for i in range(len(states)):
    #     top_left = U_derivatives.second_derivative(0, 0, states[i][:3]) + 1
    #     top_middle = U_derivatives.second_derivative(0, 1, states[i][:3])
    #     top_right = U_derivatives.second_derivative(0, 2, states[i][:3])
    #     middle_left = U_derivatives.second_derivative(1, 0, states[i][:3])
    #     middle_middle = U_derivatives.second_derivative(1, 1, states[i][:3]) + 1
    #     middle_right = U_derivatives.second_derivative(1, 2, states[i][:3])
    #     bottom_left = U_derivatives.second_derivative(2, 0, states[i][:3])
    #     bottom_middle = U_derivatives.second_derivative(2, 1, states[i][:3])
    #     bottom_right = U_derivatives.second_derivative(2, 2, states[i][:3])
        
    #     cr3bp_jacobian_stm = np.array([[1, 0, 0, 1, 0, 0],
    #                                 [0, 1, 0, 0, 1, 0],
    #                                 [0, 0, 1, 0, 0, 1],
    #                                 [top_left, top_middle, top_right, 0, 2, 0],
    #                                 [middle_left, middle_middle, middle_right, -2, 0, 0],
    #                                 [bottom_left, bottom_middle, bottom_right, 0, 0, 0]])
        
    #     individual_jacobians.append(cr3bp_jacobian_stm)
    for i in range(len(states)):
        F = np.zeros((6, 6))
        fx0 = f(states[i], mu)  # Evaluate function at the nominal state
        
        # Perturb each state variable
        for j in range(6):
            x_perturbed = np.copy(states[i])
            x_perturbed[j] += epsilon
            f_perturbed = f(x_perturbed, mu)
            F[:, j] = (f_perturbed - fx0) / epsilon  # Approximate the derivative
        individual_jacobians.append(F)
    
    combined_jacobian = np.block([[individual_jacobians[0], np.zeros(individual_jacobians[0].shape)], 
                                  [np.zeros(individual_jacobians[0].shape), individual_jacobians[1]]])
    
    return combined_jacobian

def state_to_measurements_2(state):
    pos_diff = state[6:9] - state[0:3]
    vel_diff = state[9:12] - state[3:6]
    return np.array([np.linalg.norm(pos_diff), (pos_diff @ vel_diff)/np.linalg.norm(pos_diff)])

def calculate_H_jacobian(state, f, epsilon=1e-7):
    H = np.zeros((2, 12))
    f0 = f(state)  # Evaluate the measurement function at the nominal state
    
    # Perturb each state variable
    for i in range(12):
        state_perturbed = np.copy(state)
        state_perturbed[i] += epsilon
        f_perturbed = f(state_perturbed)
        H[:, i] = (f_perturbed - f0) / epsilon  # Approximate the derivative
    
    return H

measurement_covariance = np.diag([1/d, 0.00001])
prev_state_covariance_matrix = np.diag([0.00001, 0.000001, 0.000001, 0.0001, 0.0001, 0.0001, 0.000001, 0.000001, 0.000001, 0.0001, 0.0001, 0.0001])
prediction_covariance = np.diag([0.000001, 0.000001, 0.000001, 0.0001, 0.0001, 0.0001, 0.000001, 0.000001, 0.000001, 0.0001, 0.0001, 0.0001])/1000000

# The actual Kalman Filter
prev_state_capstone_EKF = capstone_init
prev_state_ELFO_EKF = ELFO_init
bad_measurements = []
capstone_x2 = []
for i in range(len(cr3bp_states_capstone)):
    #print(i)
    cr3bp_et_capstone_EKF, cr3bp_states_capstone_EKF = cr3bp.propagate_orbit(prev_state_capstone_EKF, tspan=timestep*2.66381e-6)
    cr3bp_et_ELFO_EKF, cr3bp_states_ELFO_EKF = cr3bp.propagate_orbit(prev_state_ELFO_EKF, tspan=timestep*2.66381e-6)
    cr3bp_states_capstone_EKF[-1] += np.array([np.random.normal(0, 0.000001), np.random.normal(0, 0.000001), np.random.normal(0, 0.000001), np.random.normal(0, 0.0001), np.random.normal(0, 0.0001), np.random.normal(0, 0.0001)])/1000000
    cr3bp_states_ELFO_EKF[-1] += np.array([np.random.normal(0, 0.000001), np.random.normal(0, 0.000001), np.random.normal(0, 0.000001), np.random.normal(0, 0.0001), np.random.normal(0, 0.0001), np.random.normal(0, 0.0001)])/1000000

    # Calculate Kalman Gain
    H_jacobian = calculate_H_jacobian(np.concatenate([prev_state_capstone_EKF, prev_state_ELFO_EKF]), state_to_measurements_2)
    F_jacobian = calculate_F_jacobian([prev_state_capstone_EKF, prev_state_ELFO_EKF], cr3bp_state_transition, 0.012150585609624)
    state_covariance_matrix = F_jacobian @ prev_state_covariance_matrix @ F_jacobian.T + prediction_covariance
    kalman_gain = state_covariance_matrix @ H_jacobian.T @ np.linalg.inv(H_jacobian @ state_covariance_matrix @ H_jacobian.T + measurement_covariance)
    # if i > 669:
    #     print(a_posteriori)
    #     print(F_jacobian)
    #     print(prev_state_covariance_matrix)
    #     print(H_jacobian)
    #     print(kalman_gain)
    #     break

    # Calculate a posteriori
    full_prediction_vector = np.concatenate((cr3bp_states_capstone_EKF[-1], cr3bp_states_ELFO_EKF[-1]))
    pred_range, pred_range_rate = state_to_measurements(full_prediction_vector[0:6], full_prediction_vector[6:12])
    pred_measurements = np.array([pred_range, pred_range_rate])
    bad_measurements.append(pred_measurements)
    # print(full_prediction_vector)
    # print(np.concatenate((seq_capstone[i], seq_ELFO[i])))
    a_posteriori = full_prediction_vector + kalman_gain @ (measurements[i] - pred_measurements)
    capstone_x2.append(a_posteriori[2])
    # print(full_prediction_vector)
    # print(kalman_gain @ (measurements[i] - pred_measurements))
    # exit()
    #print(a_posteriori) # Not even fucking close
    

    # Updating for next step
    prev_state_capstone_EKF = a_posteriori[0:6]
    prev_state_ELFO_EKF = a_posteriori[6:12]
    prev_state_covariance_matrix = (np.identity(12) - kalman_gain @ H_jacobian) @ state_covariance_matrix
    # print(state_covariance_matrix)

test_et_capstone, test_states_capstone = cr3bp.propagate_orbit(capstone_init, tspan=0.921)
cr3bp.plot_3d()
test_et_ELFO, test_states_ELFO = cr3bp.propagate_orbit(ELFO_init, tspan=0.921)

print(np.concatenate((test_states_capstone[-1], test_states_ELFO[-1])))
print(np.concatenate((cr3bp_states_capstone[-1], cr3bp_states_ELFO[-1])))
print(a_posteriori)
print(state_to_measurements(a_posteriori[0:6], a_posteriori[6:12]))
print(np.linalg.norm(np.concatenate((cr3bp_states_capstone[-1], cr3bp_states_capstone[-1])) - a_posteriori))

# Plotting the range and range rates
# Extract x and y values
x_values = time_seq_capstone
y_values = [arr[0] for arr in bad_measurements][0:len(x_values)]

# Create the plot
plt.plot(x_values, y_values, marker='o')

# Add labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Range (Earth to Moon distances)')
plt.title('Range vs Time')

# Show the plot
plt.show()
