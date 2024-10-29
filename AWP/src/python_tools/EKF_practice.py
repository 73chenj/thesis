import numpy as np

c = 0.69
state_init = np.array([2, 3])

def func(state, timestep):
    return np.array([state[0] + state[1] * timestep, state[1] - c * timestep * state[1]**3])

def calculate_F_jacobian(state, timestep):
    return np.array([[1, timestep], [0, 1 - 3 * timestep * state[1]**2]])

def calculate_H_jacobian(state):
    return np.array([[state[0]/abs(state[0]), 0], [0, 2 * state[1]]])

num_steps = 40
ground_truth = []
prev_state = state_init
step = 0.01
# Ground Truth
for i in range(num_steps):
    ground_truth.append(func(prev_state, step))
    prev_state = func(prev_state, step)
#print(np.array(ground_truth))
measurements = []
for i in range(len(ground_truth)):
    x_measurement = abs(ground_truth[i][0]) + np.random.normal(0, 0.0001)
    v_measurement = ground_truth[i][1]**2 + np.random.normal(0, 0.0001)
    measurements.append(np.array([x_measurement, v_measurement]))

filter_results = []
prev_state = state_init
prev_cov = np.zeros((2, 2))
prediction_cov = np.zeros((2, 2))
prev_cov = [[0.001, 0], [0, 0.001]]
prediction_cov = [[0.001, 0], [0, 0.001]]
measurement_cov = [[0.0001, 0], [0, 0.0001]]
for i in range(num_steps):
    F_jacobian = calculate_F_jacobian(prev_state, step)
    H_jacobian = calculate_H_jacobian(prev_state)

    a_priori = func(prev_state, step)
    a_priori[0] += np.random.normal(0, 0.001)
    a_priori[1] += np.random.normal(0, 0.001)
    state_cov = F_jacobian @ prev_cov @ F_jacobian.T + prediction_cov
    kalman_gain = state_cov @ H_jacobian.T @ np.linalg.inv(H_jacobian @ state_cov @ H_jacobian.T + measurement_cov)
    a_posteriori = a_priori + kalman_gain @ (measurements[i] - H_jacobian @ a_priori)
    filter_results.append(a_posteriori)
    prev_state = a_posteriori
    prev_cov = (np.identity(2) - kalman_gain @ H_jacobian) @ state_cov
#print(np.array(filter_results))
print(np.array(ground_truth) - np.array(filter_results))