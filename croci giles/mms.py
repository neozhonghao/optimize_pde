#########################################################################################################################################
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#########################################################################################################################################

from numpy import ndarray
from sympy import diff
from sympy.abc import x,y,z,t
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr

def mms(solution_string):

    u = parse_expr(solution_string)
    #dim = len(u.free_symbols)
    dim = sum([int(item in solution_string) for item in "xyz"])
    symbols_list = [t,x,y,z][:(dim+1)]

    source = lambdify(symbols_list, diff(u,t) - laplace(u))
    exact_sol = lambdify(symbols_list, u)

    return exact_sol, source

def laplace(u):
    dim = sum([int(item in u.free_symbols) for item in [x,y,z]])
    symbols_list = [x,y,z][:dim]
    out = 0
    for w in symbols_list:
        out += diff(u,w,2)

    return out
