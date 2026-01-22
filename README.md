### Dual Pulse Nucleoside Labeling experiment simulations ###

# py file roles: #

 - gillespie.py is the base gillespie simulation class, detailing the iteration method and basic parameter inputs
 - gECC.py uses base classes from gillespie.py to implement dual pulse nucleoside labeling experiment batches
 - analytics.py; analytic_snr.py and analytic_snr_erlang.py include all analytic & numeric tools used to produce information in the SI and generate lookup tables
 - plotting.py produced all graphs in the work and is best suited to running in interactive mode
 - dpnl_optimizer_tool.py contains functions for recommending an optimal waiting time between pulses across a range of scenarios, with several example inputs.
 - main.py manages use of the dpnl_optimizer_tool.py code with the user-friendly site https://atphelan.github.io/optimising-dpnl-scheduling-ui/

#Gillespie Simulations and numerics model the following process for measuring cell proliferation via the cell cycle: #

 - The system is initialised with a number of cells divided between G1, S and G2M phases according to steady-state proportions.
 - At time t=0, all S phase cells are labeled blue / EdU+
 - From then until the pre-specified waiting time is up, both labeled and unlabeled cells cycle as normal
 - At the BrdU labeling time, all cells in S phase become BrdU+, making them either EdU-BrdU+ (red) or EdU+BrdU+ (purple).
 - For a further 0.5 hours, all four gatings of cell continue cycling at unmodified rates
 - After this time, the number of cells of each label type is counted
 - Gillespie repeats this process to get an estimate of the mean and standard deviation in cell counts resulting from the process

Numerical integration is performed on a first order ODE obtained from solving the linear Master equation which the Gillespie simulations follow.
Plotting of the figures included in our work was carried out with the tools included in plotting.py with our large simulation dataset, included in the repository.

To test the robustness of results, Erlang-distributed cell cycle phase times and noisy initial conditions were simulated - code for these can be found in gECC.py.

Alastar Phelan, January 2026.
