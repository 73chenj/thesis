"""
The LiAISON EKF using the Capstone and Clementine orbits

CR3BP model as ground truth
The unit of distance is normalised such that the distance between earth and moon is 1
Frame of reference is rotational centred at the barycentre of earth and moon
The unit of time is normalised such that the orbital period of the moon around the Earth is 2pi

state vector is [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
"""

import numpy as np
from pymatreader import read_mat
import spiceypy as sp
from CR3BP import CR3BP
import matplotlib.pyplot as plt

mat_capstone = read_mat("MATLAB/HALO/output/ORBdataCapstone.mat")
mat_ELFO = read_mat("MATLAB/HALO/output/ORBdataClementine.mat")

seq_capstone = mat_capstone["orb"]["seq"]["a"]["XJ2000"]
seq_ELFO = mat_ELFO["orb"]["seq"]["a"]["XJ2000"]
time_seq_capstone = (
    mat_capstone["orb"]["seq"]["a"]["t"] - mat_capstone["orb"]["seq"]["a"]["t"][0]
)
time_seq_ELFO = mat_ELFO["orb"]["seq"]["a"]["t"] - mat_ELFO["orb"]["seq"]["a"]["t"][0]

d = 384400
cr3bp = CR3BP("earth-moon")
# Using the same starting states of Capstone and Clementine but transformed into CR3BP propagator coordinates
capstone_prev = capstone_init = [
    0.98772 + seq_capstone[0][0] / d,
    seq_capstone[0][1] / d,
    seq_capstone[0][2] / d,
    seq_capstone[0][3],
    seq_capstone[0][4],
    seq_capstone[0][5],
]
ELFO_prev = ELFO_init = [
    0.98772 + seq_ELFO[0][0] / d,
    seq_ELFO[0][1] / d,
    seq_ELFO[0][2] / d,
    seq_ELFO[0][3],
    seq_ELFO[0][4],
    seq_ELFO[0][5],
]
test_et_capstone, test_states_capstone = cr3bp.propagate_orbit(
    capstone_init, tspan=0.921
)  # 0.921 is 4 days in the normalised time units

# np.random.seed(seed=19)
# np.random.seed(seed=1701)
# np.random.seed(seed=501)
# np.random.seed(seed=212)
# np.random.seed(seed=66)
# To make sure the time step is constant. Basically propagating every 30 seconds without any random noise until the end of the 4 days and using those values as the ground truth.
# Need to do this because the CR3BP propagator does not have constant timesteps.
timestep = 30  # seconds
cr3bp_states_capstone = [capstone_init]
cr3bp_states_ELFO = [ELFO_init]
for i in range(int(0.921 / (timestep * 2.663811e-6))):
    # print(i)
    cr3bp_et_capstone, cr3bp_state_capstone = cr3bp.propagate_orbit(
        capstone_prev, tspan=timestep * 2.663811e-6
    )  # 2.663811e-6 is one second in the normalised time units
    cr3bp_et_ELFO, cr3bp_state_ELFO = cr3bp.propagate_orbit(
        ELFO_prev, tspan=timestep * 2.663811e-6
    )
    cr3bp_states_capstone.append(cr3bp_state_capstone[-1])
    cr3bp_states_ELFO.append(cr3bp_state_ELFO[-1])
    capstone_prev = cr3bp_state_capstone[-1]
    ELFO_prev = cr3bp_state_ELFO[-1]

# # Assuming cr3bp_states_capstone and cr3bp_states_ELFO are lists of numpy arrays
# cr3bp_states_capstone = np.array(cr3bp_states_capstone)
# cr3bp_states_ELFO = np.array(cr3bp_states_ELFO)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# Function for converting state vectors to measurements
def state_to_measurements(state):
    pos_diff = state[6:9] - state[0:3]
    vel_diff = state[9:12] - state[3:6]
    return np.array(
        [np.linalg.norm(pos_diff), (pos_diff @ vel_diff) / np.linalg.norm(pos_diff)]
    )


# Correct range and range rate measurements (Every 30 seconds)
ranges = []
range_rates = []
for i in range(len(cr3bp_states_capstone)):
    range_i, range_rate_i = state_to_measurements(
        np.concatenate((cr3bp_states_capstone[i], cr3bp_states_ELFO[i]))
    )
    range_i += np.random.normal(
        0, 1 / d
    )  # Error is represented with a Gaussian variable with mean 0m and STD 1km
    range_rate_i += np.random.normal(
        0, 0.00001
    )  # Error is represented with a Gaussian variable with mean 0 and STD 1cm/s
    ranges.append(range_i)
    range_rates.append(range_rate_i)
measurements = np.column_stack((ranges, range_rates))

# Next 2 functions are used to calculate state transition Jacobian
def cr3bp_state_transition(x):
    mu = 0.012150585609624  # Mass ratio between Earth and Moon
    # Unpack the state vector
    x1, x2, x3, vx1, vx2, vx3 = x

    # Distances to primary and secondary bodies
    r1 = np.sqrt((x1 + mu) ** 2 + x2**2 + x3**2)
    r2 = np.sqrt((x1 - 1 + mu) ** 2 + x2**2 + x3**2)

    # Equations of motion
    ddx1 = 2 * vx2 + x1 - (1 - mu) * (x1 + mu) / r1**3 - mu * (x1 - 1 + mu) / r2**3
    ddx2 = -2 * vx1 + x2 - (1 - mu) * x2 / r1**3 - mu * x2 / r2**3
    ddx3 = -(1 - mu) * x3 / r1**3 - mu * x3 / r2**3

    # State transition function
    return np.array([vx1, vx2, vx3, ddx1, ddx2, ddx3])


def calculate_F_jacobian(states, f, epsilon=10e-7):
    individual_jacobians = []
    for i in range(len(states)):
        F = np.zeros((6, 6))
        fx0 = f(states[i])  # Evaluate function at the nominal state

        # Perturb each state variable
        for j in range(6):
            x_perturbed = np.copy(states[i])
            x_perturbed[j] += epsilon
            f_perturbed = f(x_perturbed)
            F[:, j] = (f_perturbed - fx0) / epsilon  # Approximate the derivative
        individual_jacobians.append(F)

    combined_jacobian = np.block(
        [
            [individual_jacobians[0], np.zeros(individual_jacobians[0].shape)],
            [np.zeros(individual_jacobians[0].shape), individual_jacobians[1]],
        ]
    )

    return combined_jacobian


# Calculating Jacobian for relating states to measurements
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


measurement_covariance = np.diag([1 / d, 0.00001])
prev_state_covariance_matrix = np.diag(  # Initial covariance
    [
        1 / d,
        1 / d,
        1 / d,
        0.0001,
        0.0001,
        0.0001,
        1 / d,
        1 / d,
        1 / d,
        0.0001,
        0.0001,
        0.0001,
    ]
)  
prediction_covariance = np.diag(
    [
        1 / d,
        1 / d,
        1 / d,
        0.0001,
        0.0001,
        0.0001,
        1 / d,
        1 / d,
        1 / d,
        0.0001,
        0.0001,
        0.0001,
    ]
)

# The actual Kalman Filter
prev_state_capstone_EKF = capstone_init
prev_state_ELFO_EKF = ELFO_init
bad_measurements = []
all_a_posterioris = []
for i in range(len(cr3bp_states_capstone)):
    # print(i)
    cr3bp_et_capstone_EKF, cr3bp_states_capstone_EKF = cr3bp.propagate_orbit(
        prev_state_capstone_EKF, tspan=timestep * 2.66381e-6
    )
    cr3bp_et_ELFO_EKF, cr3bp_states_ELFO_EKF = cr3bp.propagate_orbit(
        prev_state_ELFO_EKF, tspan=timestep * 2.66381e-6
    )
    cr3bp_states_capstone_EKF[-1] += np.array(
        [
            np.random.normal(0, 1 / d),
            np.random.normal(0, 1 / d),
            np.random.normal(0, 1 / d),
            np.random.normal(0, 0.0001),
            np.random.normal(0, 0.0001),
            np.random.normal(0, 0.0001),
        ]
    )
    cr3bp_states_ELFO_EKF[-1] += np.array(
        [
            np.random.normal(0, 1 / d),
            np.random.normal(0, 1 / d),
            np.random.normal(0, 1 / d),
            np.random.normal(0, 0.0001),
            np.random.normal(0, 0.0001),
            np.random.normal(0, 0.0001),
        ]
    )

    # Calculate Kalman Gain
    H_jacobian = calculate_H_jacobian(
        np.concatenate([prev_state_capstone_EKF, prev_state_ELFO_EKF]),
        state_to_measurements,
    )
    F_jacobian = calculate_F_jacobian(
        [prev_state_capstone_EKF, prev_state_ELFO_EKF], cr3bp_state_transition
    )
    state_covariance_matrix = (
        F_jacobian @ prev_state_covariance_matrix @ F_jacobian.T + prediction_covariance
    )
    kalman_gain = (
        state_covariance_matrix
        @ H_jacobian.T
        @ np.linalg.inv(
            H_jacobian @ state_covariance_matrix @ H_jacobian.T + measurement_covariance
        )
    )

    # Calculate a posteriori
    full_prediction_vector = np.concatenate(
        (cr3bp_states_capstone_EKF[-1], cr3bp_states_ELFO_EKF[-1])
    )
    pred_range, pred_range_rate = state_to_measurements(full_prediction_vector)
    pred_measurements = np.array([pred_range, pred_range_rate])
    bad_measurements.append(pred_measurements)
    a_posteriori = full_prediction_vector + kalman_gain @ (
        measurements[i] - pred_measurements
    )
    all_a_posterioris.append(a_posteriori)

    # Updating for next step
    prev_state_capstone_EKF = a_posteriori[0:6]
    prev_state_ELFO_EKF = a_posteriori[6:12]
    prev_state_covariance_matrix = (
        np.identity(12) - kalman_gain @ H_jacobian
    ) @ state_covariance_matrix
    # print(state_covariance_matrix)

test_et_capstone, test_states_capstone = cr3bp.propagate_orbit(
    capstone_init, tspan=0.921
)
test_et_ELFO, test_states_ELFO = cr3bp.propagate_orbit(ELFO_init, tspan=0.921)
# cr3bp.plot_3d()

print(np.concatenate((test_states_capstone[-1], test_states_ELFO[-1])))
print(np.concatenate((cr3bp_states_capstone[-1], cr3bp_states_ELFO[-1])))
print(a_posteriori)
print(state_to_measurements(a_posteriori))
print(
    np.linalg.norm(
        np.concatenate((cr3bp_states_capstone[-1], cr3bp_states_capstone[-1]))
        - a_posteriori
    )
)

'''
Plotting
'''
# Create 3D plot of the predicted trajectories
EKF_states_capstone = np.array([arr[0:6] for arr in all_a_posterioris])
EKF_states_ELFO = np.array([arr[6:12] for arr in all_a_posterioris])
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the trajectory for cr3bp_states_capstone
ax.plot(
    EKF_states_capstone[:, 0],
    EKF_states_capstone[:, 1],
    EKF_states_capstone[:, 2],
    label="CAPSTONE",
)

# Plot the trajectory for cr3bp_states_ELFO
ax.plot(
    EKF_states_ELFO[:, 0],
    EKF_states_ELFO[:, 1],
    EKF_states_ELFO[:, 2],
    label="ELFO",
)

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory of CAPSTONE and ELFO")

# Add legend
ax.legend()
plt.show()
fig.savefig("AWP/output/EKF_Trajectory_of_CAPSTONE_and_ELFO.png")

# Plot range and range rate measurements
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(time_seq_capstone, [arr[0] for arr in bad_measurements][0:len(time_seq_capstone)], "o")
ax1.set_title("Ranges")

ax2.plot(time_seq_capstone, [arr[1] for arr in bad_measurements][0:len(time_seq_capstone)], "o")
ax2.set_title("Range Rates")

plt.tight_layout()
plt.show()
fig.savefig("AWP/output/EKF_Ranges_and_Range_Rates.png")


# Plot each state vector value
# Create a 3x4 grid of subplots
fig, axs = plt.subplots(3, 4, figsize=(15, 10))
# Plotting on each subplot
axs[0, 0].plot(time_seq_capstone, [arr[0] for arr in all_a_posterioris][0:len(time_seq_capstone)],'r')
axs[0, 0].plot(time_seq_capstone, [arr[0] for arr in cr3bp_states_capstone][0:len(time_seq_capstone)], 'g')
axs[0, 0].set_title('Capstone X')

axs[0, 1].plot(time_seq_capstone, [arr[1] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[0, 1].plot(time_seq_capstone, [arr[1] for arr in cr3bp_states_capstone][0:len(time_seq_capstone)], 'g')
axs[0, 1].set_title('Capstone Y')

axs[0, 2].plot(time_seq_capstone, [arr[2] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[0, 2].plot(time_seq_capstone, [arr[2] for arr in cr3bp_states_capstone][0:len(time_seq_capstone)], 'g')
axs[0, 2].set_title('Capstone Z')

axs[0, 3].plot(time_seq_capstone, [arr[3] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[0, 3].plot(time_seq_capstone, [arr[3] for arr in cr3bp_states_capstone][0:len(time_seq_capstone)], 'g')
axs[0, 3].set_title('Capstone vX')

axs[1, 0].plot(time_seq_capstone, [arr[4] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[1, 0].plot(time_seq_capstone, [arr[4] for arr in cr3bp_states_capstone][0:len(time_seq_capstone)], 'g')
axs[1, 0].set_title('Capstone vY')

axs[1, 1].plot(time_seq_capstone, [arr[5] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[1, 1].plot(time_seq_capstone, [arr[5] for arr in cr3bp_states_capstone][0:len(time_seq_capstone)], 'g')
axs[1, 1].set_title('Capstone vZ')

axs[1, 2].plot(time_seq_capstone, [arr[6] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[1, 2].plot(time_seq_capstone, [arr[0] for arr in cr3bp_states_ELFO][0:len(time_seq_capstone)], 'g')
axs[1, 2].set_title('ELFO X')

axs[1, 3].plot(time_seq_capstone, [arr[7] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[1, 3].plot(time_seq_capstone, [arr[1] for arr in cr3bp_states_ELFO][0:len(time_seq_capstone)], 'g')
axs[1, 3].set_title('ELFO Y')

axs[2, 0].plot(time_seq_capstone, [arr[8] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[2, 0].plot(time_seq_capstone, [arr[2] for arr in cr3bp_states_ELFO][0:len(time_seq_capstone)], 'g')
axs[2, 0].set_title('ELFO Z')

axs[2, 1].plot(time_seq_capstone, [arr[9] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[2, 1].plot(time_seq_capstone, [arr[3] for arr in cr3bp_states_ELFO][0:len(time_seq_capstone)], 'g')
axs[2, 1].set_title('ELFO vX')

axs[2, 2].plot(time_seq_capstone, [arr[10] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[2, 2].plot(time_seq_capstone, [arr[4] for arr in cr3bp_states_ELFO][0:len(time_seq_capstone)], 'g')
axs[2, 2].set_title('ELFO vY')

axs[2, 3].plot(time_seq_capstone, [arr[11] for arr in all_a_posterioris][0:len(time_seq_capstone)], 'r')
axs[2, 3].plot(time_seq_capstone, [arr[5] for arr in cr3bp_states_ELFO][0:len(time_seq_capstone)], 'g')
axs[2, 3].set_title('ELFO vZ')

plt.tight_layout()
plt.show()
fig.savefig("AWP/output/EKF_and_benchmark_state_vectors.png")
