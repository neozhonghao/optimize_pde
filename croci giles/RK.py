#########################################################################################################################################
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#########################################################################################################################################

from numpy import array
import sympy as sy
from sympy import diff, Poly, N, sympify
from sympy.abc import x,z,e
from sympy.functions.special.polynomials import chebyshevu, chebyshevt
from sympy.solvers.solveset import linsolve

def to_float(l):
    return [float(item) for item in l]

def dchebt(j,x,n=1):
    return diff(chebyshevt(j,x),x,n)

def get_RKC1_coeffs(s, e=1/sympify(20)):
    assert s > 0
    #s = 4
    #e = 0

    e = sympify(e)

    def get_matrix(Qs,z):
        s = len(Qs);
        A = [Qs[0].all_coeffs()]
        for i in range(1,s):
            A.append([sympify(0)]*i + Qs[i].all_coeffs())
        return sy.Matrix(A).T

    w0 = 1 + e/s**2
    w1 = chebyshevt(s,w0)/dchebt(s,x).subs(x,w0)

    b = lambda j : 1/chebyshevt(j,w0)
    c = lambda j : w1*dchebt(j,x).subs(x,w0)/chebyshevt(j,w0)
    P = lambda j : b(j)*chebyshevt(j,w0+w1*z)
    Q = lambda j,k : b(j)/b(k)*chebyshevu(j-k, w0+w1*z)

    cj = to_float([c(j) for j in range(1,s+1)])

    Ps = Poly(sy.expand(P(s)))
    Qs = []
    for i in range(1,s+1):
        Qs.append(Poly(Q(s,i),z))

    St = Poly(sy.simplify((Ps-1)/z),z)

    A = get_matrix(Qs,z)
    rhs = sy.Matrix(St.all_coeffs())

    weights = linsolve((A,rhs)).args[0]
    Qscoeffs = [to_float((Q*w).all_coeffs()) for Q,w in zip(Qs,weights)]

    Stcoeffs = to_float(St.all_coeffs())
    Scoeffs = to_float(Ps.all_coeffs())

    return Scoeffs, Stcoeffs, Qscoeffs, cj

def get_RKC2_coeffs(s, e=sympify(2)/sympify(13)):
    assert s > 1
    #s = 4
    #e = 0

    e = sympify(e)

    def get_matrix(Qs,z):
        s = len(Qs);
        A = [Qs[0].all_coeffs()]
        for i in range(1,s):
            A.append([sympify(0)]*i + Qs[i].all_coeffs())
        return sy.Matrix(A).T

    w0 = 1 + e/s**2
    w1 = dchebt(s,x).subs(x,w0)/dchebt(s,x,2).subs(x,w0)

    def b(j):
        if j < 2: j=2
        return dchebt(j,x,2).subs(x,w0)/(dchebt(j,x).subs(x,w0)**2)

    def a(j):
        if j == 0: return 1-b(0);
        if j == 1: return 1-b(1)*w0
        return 1-b(j)*chebyshevt(j,w0)

    def c(j):
        if j == 1:
            return w1*dchebt(2,x,2).subs(x,w0)/(dchebt(2,x).subs(x,w0)**2)
        else:
            return w1*dchebt(j,x,2).subs(x,w0)/dchebt(j,x).subs(x,w0)


    P = lambda j : a(j) + b(j)*chebyshevt(j,w0+w1*z)
    Q = lambda j,k : b(j)/b(k)*chebyshevu(j-k, w0+w1*z)

    cj = to_float([c(j) for j in range(1,s+1)])

    Ps = Poly(sy.expand(P(s)))
    Qs = []
    for i in range(1,s+1):
        Qs.append(Poly(Q(s,i),z))

    St = Poly(sy.simplify((Ps-1)/z),z)

    A = get_matrix(Qs,z)
    rhs = sy.Matrix(St.all_coeffs())

    weights = linsolve((A,rhs)).args[0]
    Qscoeffs = [to_float((Q*w).all_coeffs()) for Q,w in zip(Qs,weights)]

    Stcoeffs = to_float(St.all_coeffs())
    Scoeffs = to_float(Ps.all_coeffs())

    return Scoeffs, Stcoeffs, Qscoeffs, cj

def get_RK_coefficients(method, stages = 4):
    if method == "RK4":
        #S = lambda z : z**0 + z + z**2/2 + z**3/6 + z**4/24; Stilde = lambda z : z**0 + z/2 + z**2/6 + z**3/24;
        #Q1 = lambda z : (z**0 + z + z**2/2 + z**3/4)/6; Q2 = lambda z : (2*z**0 + z + z**2/2)/6; Q3 = lambda z : (2*z**0 + z)/6; Q4 = lambda z : z**0/6
        cj       = [0, 1/2, 1/2, 1] # evaluation points
        Scoeffs  = [1/24, 1/6, 1/2, 1, 1]
        Stcoeffs = [1/24, 1/6, 1/2, 1]
        Qscoeffs = [[1/24, 1/12, 1/6, 1/6], [1/12, 1/3 ,2/3], [1/12, 1/3, 2/3], [1/6]]
    elif method == "RKC1" or method == "RKC":
        Scoeffs, Stcoeffs, Qscoeffs, cj = get_RKC1_coeffs(stages)
    elif method == "RKC2":
        stages += 1
        Scoeffs, Stcoeffs, Qscoeffs, cj = get_RKC2_coeffs(stages)
    else:
        Scoeffs, Stcoeffs, Qscoeffs, cj = None, None, None, None

    return {"S" : Scoeffs, "St" : Stcoeffs, "Qs" : Qscoeffs, "cj" : cj}
