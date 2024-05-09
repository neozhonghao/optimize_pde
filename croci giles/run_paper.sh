#!/bin/bash

#########################################################################################################################################
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#########################################################################################################################################

echo 'While this script contains all the commands required to generate the paper data we advise against running this script directly, especially on a laptop since it requires too many resourced. Please run each one of the commands in this script individually on a computing node.'

exit

nprocs=8
nthreads=1

./run_all.sh 1
./run_all.sh 2
./run_all.sh 3

./MC_run.sh FE 1 $nprocs $nthreads
./MC_run.sh BE 1 $nprocs $nthreads
./MC_run.sh FE 2 $nprocs $nthreads
./MC_run.sh BE 2 $nprocs $nthreads
./MC_run.sh FE 3 $nprocs $nthreads

python3 pde.py BE -d 1 -s --noprint -ll &
OMP_NUM_THREADS=$nthreads OMP_PROC_BIND=false mpiexec -n $nprocs python3 pde.py BE -d 1 -s -MC -ll --noprint &

python3 IC_plot.py
