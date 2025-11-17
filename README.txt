Dual Pulse Nucleoside Labeling experiment simulations

py file roles:

 - gECC.py is gillespie simulation for dual labeling
 - gillespie.py is base gillespie simulation class
 - analytics.py includes all analytic & numeric tools used to produce information in the SI
 - plotting.py produced the graphs in the work

Gillespie Simulations and numerics model the following process for measuring cell proliferation via the cell cycle:

 - The system is initialised with a number of cells divided between G1, S and G2M phases according to steady-state proportions.
 - At time t=0, all S phase cells are labeled blue / EdU+
 - From then until the prespecified waiting time is up, both labeled and unlabeled cells cycle as normal
 - At the BrdU labeling time, all cells in S phase become BrdU+, making them either EdU-BrdU+ (red) or EdU+BrdU+ (purple).
 - For a further 0.5 hours, all four gatings of cell continue cycling at unmodified rates
 - After this time, the number of cells of each label type is counted
 - Gillespie repeats this process to get an estimate of the mean and standard deviation in cell counts resulting from the process

Numerical integration is performed on a first order ODE obtained from solving the linear Master equation which the Gillespie simulations follow.
Plotting of the figures included in our work was carried out with the tools included in plotting.py with our large simulation dataset, not included in the repository.

Code transferred during November 2025 from original project code by Alastar Phelan.