It's Quentin's propagator repo except the AWP folder contains the CR3BP propagator.

AWP/src/python_tools/LiAISON.py is where I implemented my EKF

AWP/src/python_tools/CR3BP.py is the implementation for the cr3bp propagator

I ran Quentin's propagator for the Capstone and Clemtine missions for 4 days so those outputs are in the output folder in Matlab which LiAISON.py uses. It only used the initial state really, I used the CR3BP model to propagate from there.
