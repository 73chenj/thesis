# The LiAISON EKF using the Capstone mission and another made up ELFO mission
import numpy as np
from pymatreader import read_mat
import spiceypy as sp
from CR3BP import CR3BP
import rotations
import U_derivatives



mat_capstone = read_mat('MATLAB/HALO/output/ORBdataCapstone.mat')
mat_ELFO = read_mat('MATLAB/HALO/output/ORBdataClementine.mat')

seq_capstone = mat_capstone['orb']['seq']['a']['XJ2000']
time_seq_capstone = mat_capstone['orb']['seq']['a']['t'] - mat_capstone['orb']['seq']['a']['t'][0]
seq_ELFO = mat_ELFO['orb']['seq']['a']['XJ2000']
print(seq_capstone[0])


def convert(state_j2000):
    d = 384400  # Distance between the two bodies (e.g., Earth-Moon distance in km)
    T = 2*np.pi  # Orbital period (Moon's orbital period in seconds)

    # Positions of the primary bodies in J2000 coordinates (example values)
    # Earth position
    earth_position = np.array([0, 0, 0])  # Earth at origin
    # Moon position (example, in km)
    moon_position = np.array([d, 0, 0])  # Moon at (d, 0, 0)

    # Calculate center of mass position
    m1 = 1.0  # Mass of Earth (normalized)
    m2 = 0.012277471  # Mass of Moon (normalized)
    CM_position = (m2 / (m1 + m2)) * moon_position/d

    # Convert position to CR3BP coordinates
    x_cr3bp = (state_j2000[0] / d) - CM_position[0]
    y_cr3bp = (state_j2000[1] / d) - CM_position[1]
    z_cr3bp = (state_j2000[2] / d) - CM_position[2]
    print(state_j2000[0]/d)

    # Convert velocity to CR3BP coordinates
    vx_cr3bp = state_j2000[3] / T
    vy_cr3bp = state_j2000[4] / T
    vz_cr3bp = state_j2000[5] / T

    # New state in CR3BP coordinates
    return np.array([x_cr3bp + 1, y_cr3bp, z_cr3bp, vx_cr3bp, vy_cr3bp, vz_cr3bp])

state_cr3bp = convert(seq_capstone[0])

# cr3bt model as ground truth
earth_to_moon = 384400
cr3bp = CR3BP('earth-moon')
capstone_prev = capstone_init = [state_cr3bp]
# To make sure the time step is constant
cr3bp_states_capstone = [capstone_init]
cr3bp_et_capstone, cr3bp_state_capstone = cr3bp.propagate_orbit(state_cr3bp, tspan=0.921)

cr3bp.plot_3d()
