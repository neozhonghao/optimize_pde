#!/bin/bash

#########################################################################################################################################
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#########################################################################################################################################

dim=$1
if [[ $dim == 1 ]]
then
    export OLD_OMP_NUM_THREADS=$OMP_NUM_THREADS
    export OMP_NUM_THREADS=2
fi

python3 pde.py FE -d $dim -s --noprint &
python3 pde.py FE -d $dim -s --noprint --nodelta &
python3 pde.py FE -d $dim -s --noprint --nosmart &

if [[ $dim -lt 3 ]]
then 
    python3 pde.py BE -d $dim -s --noprint &
    python3 pde.py BE -d $dim -s --noprint --nodelta &
    python3 pde.py BE -d $dim -s --noprint --nosmart &
fi

if [[ $dim == 1 ]]
then
    export OMP_NUM_THREADS=$OLD_OMP_NUM_THREADS
fi
