#!/bin/bash

#########################################################################################################################################
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#########################################################################################################################################

method=$1
dim=$2
nprocs=$3
nthreads=$4

OMP_NUM_THREADS=$nthreads OMP_PROC_BIND=false mpiexec -n $nprocs python3 pde.py $method -d $dim -s -MC --noprint &
OMP_NUM_THREADS=$nthreads OMP_PROC_BIND=false mpiexec -n $nprocs python3 pde.py $method -d $dim -s -MC --nodelta --noprint &
OMP_NUM_THREADS=$nthreads OMP_PROC_BIND=false mpiexec -n $nprocs python3 pde.py $method -d $dim -s -MC --nosmart --noprint &
