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
seq_ELFO = mat_ELFO['orb']['seq']['a']['XJ2000']
time_seq_capstone = mat_capstone['orb']['seq']['a']['t'] - mat_capstone['orb']['seq']['a']['t'][0]
time_seq_ELFO = mat_ELFO['orb']['seq']['a']['t'] - mat_ELFO['orb']['seq']['a']['t'][0]
print(seq_capstone[0])

d = 384400
cr3bp = CR3BP('earth-moon')
# Using the same starting states of Capstone and Clementine but transformed into CR3BP propagator coordinates
capstone_prev = capstone_init = [0.98772 + seq_ELFO[0][0]/d, seq_ELFO[0][2]/d, seq_ELFO[0][1]/d, seq_ELFO[0][3], seq_ELFO[0][5], seq_ELFO[0][4]]
ELFO_prev = ELFO_init = [0.98772 + seq_ELFO[0][0]/d, seq_ELFO[0][1]/d, seq_ELFO[0][2]/d, seq_ELFO[0][3], seq_ELFO[0][4], seq_ELFO[0][5]]
test_et_capstone, test_states_capstone = cr3bp.propagate_orbit(capstone_init, tspan=1.8) # 0.921 is 4 days in the normalised time units
cr3bp.plot_3d()
