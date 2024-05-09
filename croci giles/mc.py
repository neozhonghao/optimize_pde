#########################################################################################################################################
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#########################################################################################################################################

import numpy as np
from numpy import sqrt
import sys
import os

if os.environ.get("OMPI_COMM_WORLD_SIZE") is not None:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpiRank = comm.Get_rank()
    mpiSize = comm.Get_size()
else:
    comm = None
    mpiRank = 0
    mpiSize = 1

def mc(mc_fn, N, logfile = None):
    """
    Standard Monte Carlo test routine.

    mlmc_fn: the user low-level routine. Its interface is
      sums = mc_fn(N)
    with inputs
      N = number of paths
    and a list of numpy arrays of outputs
      sums[i][0] = sum(P)
      sums[i][1] = sum(P**2)
      sums[i][2] = sum(P**3)
      sums[i][3] = sum(P**4)

    N: number of samples for convergence tests
    """

    # First, convergence tests

    write(logfile, "\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "***     Convergence, Expected Value, Variance, Cost    ***\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "\n  ave(P)      var(P)      kurt(P)     cost(P) \n")
    write(logfile, "--------------------------------------------------\n")

    avg = []
    var = []
    kurt = []
    cost = []

    sums, costl = mc_fn(N)

    l = len(sums)

    for k in range(l):
        cost.append(costl)
        for i in range(len(sums[k])):
            sums[k][i] = sums[k][i]/N

        avg.append(sums[k][0])

        var.append(max(abs(sums[k][1]-sums[k][0]**2), 1.0e-16)) 
        kurt.append((sums[k][3] - 4*sums[k][0]*sums[k][2] + 6*sums[k][0]**2*sums[k][1] - 3*sums[k][0]**4)/var[-1]**2.)

        write(logfile, "%8.4e  %8.4e  %8.4e  %8.4e \n" % (avg[-1], var[-1], kurt[-1], cost[-1]))

        write(logfile, "\n")

    return avg

def next_divisible_n(n):
    return int(np.ceil(float(n)/mpiSize))*mpiSize

def mc_solve(mc_fn, N0 = 32, tol = 0.05, logfile=None):

    N = 0
    N_new = next_divisible_n(N0)
    err = np.inf

    it = 0

    while err > tol and N_new > 1: # 5% error on the mean
        sums_new,_ = mc_fn(N_new)
        N += N_new
        it += 1
        if np.isinf(err):
            l = len(sums_new)
            avg = [0.0]*l
            var = [0.0]*l
            sums = []
            for k in range(l):
                sums.append([0.0]*len(sums_new[k]))

        for k in range(l):
            for i in range(len(sums[k])):
                sums[k][i] = sums[k][i] + sums_new[k][i]
                sums_new[k][i] = sums[k][i]/N

            avg[k] = sums_new[k][0]
            var[k] = max(abs(sums_new[k][1]-sums_new[k][0]**2), 1.0e-16)

        err = 0.0
        errs = np.zeros((l,))
        for k in range(l):
            if abs(avg[k]) > 1.0e-10:
                errs[k] = sqrt(var[k]/N)/abs(avg[k])
            else:
                errs[k] = 0.0

            err = max(err, errs[k])

        N_new = next_divisible_n(min(max(int(np.ceil(N*(err/tol)**2)) - N, 0), 1000))
        write(logfile, "Iteration %d, N samples: %d" % (it,N) + "; max error: %f" % err + "; new samples needed: %d\n" % N_new)

    return avg, var


def write(logfile, msg):
    """
    Write to both sys.stdout and to a logfile.
    """
    if mpiRank == 0:
        if logfile is not None:
            logfile.write(msg)
            logfile.flush()
        sys.stdout.write(msg)
        sys.stdout.flush()


