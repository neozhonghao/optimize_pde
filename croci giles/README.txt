Code and data used to generate the figures in the paper Effects of round-to-nearest and stochastic rounding in the numerical solution of the heat equation in low precision by M. Croci and M. B. Giles
========================================================================================================================================================================================================

* To generate the data, it is sufficient to run all the commands in ./run_paper.sh one by one (see the script for further details). However, the generated data is already in ./results/
* The main script used for the simulations is pde.py
* The data used to generate Figure 1 (right) is in ./results/vtk_data/zeroBC/. This figure was generated using paraview.

Dependencies:
-------------

* Libchopping (https://bitbucket.org/croci/libchopping/src/master/), and its dependencies.
* Python3, numpy, scipy, sympy, matplotlib, mpi4py, pybind11, pyevtk (optional, only for vtk export).
* Intel MKL library, OpenMP, Intel C++ compiler icpc (recommended, but optional).
