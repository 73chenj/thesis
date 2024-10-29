import numpy as np

# Constants
omega = 2.6617e-6  # Approximate angular velocity of Earth-Moon system (rad/s)

# CR3BP propagator uses rotational frame and Quentin's propagator used intertial frame. Need a way to convert.
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0, 0, 1]
    ])
# Function to transform position from rotating frame to inertial frame
def rotating_to_inertial(pos_rotating, time):
    theta = omega * time  # Rotation angle at time t
    R = rotation_matrix(theta)
    pos_inertial = np.dot(R, pos_rotating)
    return pos_inertial
# Function to transform position from inertial frame to rotating frame
def inertial_to_rotating(pos_inertial, time):
    theta = omega * time  # Rotation angle at time t
    R = rotation_matrix(-theta)
    pos_rotating = np.dot(R, pos_inertial)
    return pos_rotating
# Function to transform velocity from rotating frame to inertial frame
def rotating_to_inertial_velocity(pos_rotating, vel_rotating, time):
    theta = omega * time
    R = rotation_matrix(theta)
    # Transform position and velocity to inertial frame
    pos_inertial = np.dot(R, pos_rotating)
    vel_inertial = np.dot(R, vel_rotating) + np.cross([0, 0, omega], pos_rotating)
    return np.concatenate((pos_inertial, vel_inertial))
# Function to transform velocity from inertial frame to rotating frame
def inertial_to_rotating_velocity(pos_inertial, vel_inertial, time):
    theta = omega * time
    R = rotation_matrix(-theta)
    # Transform position and velocity to rotating frame
    pos_rotating = np.dot(R, pos_inertial)
    vel_rotating = np.dot(R, vel_inertial) - np.cross([0, 0, omega], pos_inertial)
    return np.concatenate((pos_rotating, vel_rotating))