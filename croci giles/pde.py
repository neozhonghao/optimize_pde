#########################################################################################################################################
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#########################################################################################################################################

import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from RK import get_RK_coefficients
from chopping import *
import time
import warnings
import os
import sys
if os.environ.get("OMPI_COMM_WORLD_SIZE") is not None:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpiRank = comm.Get_rank()
    mpiSize = comm.Get_size()
else:
    comm = None
    mpiRank = 0
    mpiSize = 1

######################################################

fmt = 'b'

options = Option(True)
options.set_format(fmt)
options.set_round(1)

options_sr = Option(True)
options_sr.set_format(fmt)
options_sr.set_round(5)
options_sr.set_seed(mpiRank*100)

if options.get_format() == "h":
    u = 2**-11
else:
    u = 2**-8

######################################################

def csr_data(A):
    return A.data, A.indices, A.indptr

def vtk_export(coords, data, name="figure"):
    from pyevtk.hl import pointsToVTK, gridToVTK
    x,y,z = coords
    gridToVTK("./results/vtk_data/" + name, x,y,z, pointData=data)


class LPProblem(object):
    def __init__(self, method, params, source, exact_sol = None, bc = 0):
        self.method = method
        if params["exact_linalg"] == True:
            params["delta_form"] = True

        self.dim = params["dim"]
        
        self.params = params

        self.source = source
        self.exact_sol = exact_sol
        self.bc = bc

        from inspect import getsourcelines
        self.timedep = getsourcelines(source)[0][1].split(";")[0].count("t") > 1 # logical, returns True if the source is time-dependent

        self.mats = None
        self.vecs = None

        s = 4
        if "RKC" in method:
            s = 16
            try: p = int(method[-1])
            except ValueError: p = 1
            self.setup_RKC(p, s)

        self.RK_coeffs = get_RK_coefficients(method, s)

    def MGsolve(self, b, tol = 1.0e-7, max_it = 50, smoother = "jacobi"):
        dim = self.dim
        lmbda = self.lmbda
        if self.method == "IM":
            lmbda /= 2

        B = self.mats[1]

        if smoother == "jacobi":
            nsmooth = 4
        else:
            nsmooth = 4

        bb = add_zero_boundary(dim, b)
        x = 0.0*bb

        err = np.inf
        rate = 0.0
        k = 0

        while(err > tol and rate < 0.6 and k < max_it):
            y = mgv(dim, bb, lmbda, smoother, nsmooth, x)
            if isinstance(bb, np.ndarray):
                temp = norm(residual(dim, bb, lmbda, y), np.inf)
            else:
                # no point in computing the residual in low precision
                temp = norm(residual(dim, bb.array(), lmbda, y.array()), np.inf)

            rate = temp/err
            k += 1
            #print(rate, k)
            if rate < 1:
                err = temp
                x = y

        if k >= 50:
            print("WARNING! MG did not converge. Residual: ", err)

        out = remove_zero_boundary(dim, x)

        return out

    def assemble_matrices(self, K, lmbda, dim, method):
        e = np.ones((K,))
        L = sp.spdiags([-e,2*e,-e],[-1,0,1], K, K, format="csr")
        I = sp.eye(L.shape[0])
        if   dim == 1: A = L
        elif dim == 2: A = sp.kron(I,L) + sp.kron(L,I)
        elif dim == 3: A = sp.kron(sp.kron(I,I),L) + sp.kron(L,sp.kron(I,I)) + sp.kron(sp.kron(I,L),I)

        A = -lmbda*A.tocsr()
        B = None

        if   method == "BE": B = sp.eye(K**dim) - A
        elif method == "IM": B = sp.eye(K**dim) - A/2

        return A,B

    def cut_borders(self, x, tensorize = False):
        islpv = not isinstance(x, np.ndarray) 
        if tensorize and islpv: raise NotImplementedError

        if islpv: op = x.option; x = x.array()

        if self.dim == 1: return x[1:-1]

        l = len(x.shape)
        if l == 1:
            m = int(len(x)**(1/self.dim))
            out = x.reshape(tuple([m]*self.dim))
        else:
            out = x

        if self.dim == 2: out = out[1:-1,1:-1]
        if self.dim == 3: out = out[1:-1,1:-1,1:-1]
        if not tensorize: out = out.flatten()
        if islpv: out = LPV(out, op) 
        return out

    def tensorize(self,x):
        m = int(np.rint(len(x)**(1/self.dim)))
        if self.dim == 1: return x
        if self.dim == 2: return x.reshape((m,m))
        if self.dim == 3: return x.reshape((m,m,m))

    def LPV_polymatvec(self, coeffs, A, x):
        p = len(coeffs)
        if p == 0:
            raise ValueError("polymatvec: coefficient vector cannot have zero length")

        out = coeffs[0]*x
        for i in range(1,p):
            out = matvec_csr(*csr_data(A), out) + coeffs[i]*x

        return out

    def polymatvec(self, coeffs, A, x):
        p = len(coeffs)
        if p == 0:
            raise ValueError("polymatvec: coefficient vector cannot have zero length")

        out = coeffs[0]*x
        for i in range(1,p):
            out = A@out + coeffs[i]*x

        return out

    def zero_bc(self, U):
        islpv = not isinstance(U, np.ndarray) 
        if islpv: op = U.option; U = U.array()

        if self.dim == 1: U[0],U[-1] = 0.0,0.0
        else:
            U = self.tensorize(U)
            if self.dim == 2:
                U[0,:]  = 0.0 
                U[-1,:] = 0.0 
                U[:,0]  = 0.0 
                U[:,-1] = 0.0 
            if self.dim == 3:
                U[0,:,:]  = 0.0 
                U[-1,:,:] = 0.0 
                U[:,0,:]  = 0.0 
                U[:,-1,:] = 0.0 
                U[:,:,0]  = 0.0 
                U[:,:,-1] = 0.0 

            U = U.flatten()

        if islpv: U = LPV(U, op)

        return U

    def fix_bc(self, fx):
        islpv = not isinstance(fx, np.ndarray) 
        if islpv: op = fx.option; fx = fx.array()

        bc = self.bc
        if self.dim == 1:
            fx[0] += bc*self.lmbda; fx[-1] += bc*self.lmbda 
        else:
            fx = self.tensorize(fx)
            if self.dim == 2:
                fx[0,:]  += bc*self.lmbda 
                fx[-1,:] += bc*self.lmbda 
                fx[:,0]  += bc*self.lmbda 
                fx[:,-1] += bc*self.lmbda 
            if self.dim == 3:
                fx[0,:,:]  += bc*self.lmbda 
                fx[-1,:,:] += bc*self.lmbda 
                fx[:,0,:]  += bc*self.lmbda 
                fx[:,-1,:] += bc*self.lmbda 
                fx[:,:,0]  += bc*self.lmbda 
                fx[:,:,-1] += bc*self.lmbda 

            fx = fx.flatten()

        if islpv: fx = LPV(fx, op)

        return fx

    def setup_RKC(self, p, s):
        from scipy.special import eval_chebyt, eval_chebyu

        if p > 2: raise NotImplementedError

        T   = lambda s,x : eval_chebyt(s,x)
        dT  = lambda s,x : s*eval_chebyu(s-1,x) # T_s'
        d2T = lambda s,x : s/(x**2-1)*((s+1)*eval_chebyt(s,x) - eval_chebyu(s,x)) # T''
        if p == 1:
            gm = None
            e = 0.05
            w0 = 1 + e/s**2
            w1 = T(s,w0)/dT(s,w0)
            mu1 = w1/w0
            b = [1./T(j,w0) for j in range(s+1)]
            # NOTE: using MATLAB-like indexing here for consistency with papers
            mu = [0.0, mu1] + [2*w1*b[j]/b[j-1] for j in range(2,s+1)]
            nu = [0.0, 0.0] + [2*w0*b[j]/b[j-1] for j in range(2,s+1)]
            kp = [0.0, 0.0] + [-b[j]/b[j-2] for j in range(2,s+1)]
            cj = [0.0] + [w1*dT(j,w0)/T(j,w0) for j in range(1,s)] + [1.0]

        elif p == 2:
            e = 2./13
            w0 = 1 + e/s**2
            w1 = dT(s,w0)/d2T(s,w0)
            b2 = d2T(2,w0)/(dT(2,w0)**2)
            b1 = b2;
            b0 = b2;
            a0 = 1 - b0; a1 = 1 - b1*w0
            b = [b0,b1,b2] + [d2T(j,w0)/(dT(j,w0)**2) for j in range(3,s+1)]
            a = [a0, a1] + [1 - b[j]*T(j,w0) for j in range(2,s+1)]

            mu1 = b[1]*w1
            mu = [0.0, mu1]   + [2*w1*b[j]/b[j-1]      for j in range(2,s+1)]
            nu = [0.0, 0.0]   + [2*w0*b[j]/b[j-1]      for j in range(2,s+1)]
            kp = [0.0, 0.0]   + [-b[j]/b[j-2]          for j in range(2,s+1)]
            gm = [0.0, 0.0]   + [-a[j-1]*mu[j]         for j in range(2,s+1)]
            cj = [0.0, w1*b2] + [w1*d2T(j,w0)/dT(j,w0) for j in range(2,s  )] + [1.0]

        self.RKC_params = {"p": p, "s" : s, "mu" : mu, "nu": nu, "kp" : kp, "gm" : gm, "cj" : cj}

    def get_sources(self, t0, t1):
        method = self.method
        source = self.source
        dt     = self.dt
        x      = self.x

        if   method == "FE": fx = source(t0,*x)
        elif method == "BE": fx = source(t1,*x)
        elif method == "IM": fx = source((t0+t1)/2,*x)
        elif "RK" in method and self.timedep:
            cj = self.RK_coeffs["cj"]
            A,B = self.mats
            fs    = [self.cut_borders(source(t0+c*dt,*x)) for c in cj]
            fs_c  = [LPV(f, options) for f in fs]
            fs_sr = [LPV(f, options_sr) for f in fs]

            Qscoeffs = self.RK_coeffs["Qs"]
            fx    = dt*sum([self.polymatvec(Qcoeff, A, f) for Qcoeff,f in zip(Qscoeffs, fs)])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fx_c  = dt*sum([self.LPV_polymatvec(Qcoeff, A, f) for Qcoeff,f in zip(Qscoeffs, fs_c)])
                fx_sr = dt*sum([self.LPV_polymatvec(Qcoeff, A, f) for Qcoeff,f in zip(Qscoeffs, fs_sr)])
        elif "RK" in method and not self.timedep:
            A,B = self.mats
            fx    = self.cut_borders(source(t0,*x))
            fx_c  = LPV(fx, options)
            fx_sr = LPV(fx, options_sr)

            Stcoeffs = self.RK_coeffs["St"]
            fx = dt*self.polymatvec(Stcoeffs, A, fx)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fx_c  = dt*self.LPV_polymatvec(Stcoeffs, A, fx_c)
                fx_sr = dt*self.LPV_polymatvec(Stcoeffs, A, fx_sr)

        if "RK" not in method:
            fx = dt*self.cut_borders(fx); #NOTE: if the initial condition does not satisfy the BCs, this should be different.
            if not self.params["smart_differencing"] or not self.params["delta_form"]:
                fx = self.fix_bc(fx)
            
            return (fx, LPV(fx, options), LPV(fx, options_sr))
        else:
            #fx = dt*self.cut_borders(fx); #NOTE: if the initial condition does not satisfy the BCs, this should be different.
            if not self.params["smart_differencing"] or not self.params["delta_form"]:
                raise NotImplementedError("We have not fixed the BCs yet for RK")
                fx = self.fix_bc(fx)
            if self.timedep:
                raise NotImplementedError("We have not fixed the BCs yet for RK")
            return (fx, fx_c, fx_sr)

    def dUex(self, U, f):
        method = self.method
        if "RK" in method: Stcoeffs = self.RK_coeffs["St"]

        A,B = self.mats

        if self.params["smart_differencing"] and self.params["delta_form"]:
            f = self.fix_bc(f)

        if   method == "FE":       return A@U + f
        elif method == "RK4":      return self.update_rk4(U)
        elif "RKC" in method:      return self.update_rkc(U)
        elif "RK" in method:       return self.polymatvec(Stcoeffs, A, A@U) + f
        elif method in ["BE","IM"] and self.params["use_multigrid"]: return self.MGsolve(A@U + f)
        elif method in ["BE","IM"] and not self.params["use_multigrid"]: return spl.spsolve(B, A@U + f)

    def update_rk4(self, U):
        if self.timedep: raise NotImplementedError
        source = self.source
        fx = self.dt*self.cut_borders(source(0,*self.x))
        islpv = not isinstance(U, np.ndarray)
        if islpv:
            fx = LPV(fx, U.option)
            def B(U):
                return lmbda*diffND(U, self.dim, self.bc)
        else:
            fx = self.fix_bc(fx)
            A,_ = self.mats
            def B(U):
                return A@U

        k1 = B(U)        + fx
        k2 = B(U + k1/2) + fx
        k3 = B(U + k2/2) + fx
        k4 = B(U + k3)   + fx
        out  = (k1 + 2*(k2 + k3) + k4)/6

        return out

    def update_rkc(self, U):
        method = self.method

        if self.timedep: raise NotImplementedError
        source = self.source
        fx = self.dt*self.cut_borders(source(0,*self.x))
        islpv = not isinstance(U, np.ndarray)
        if islpv:
            fx = LPV(fx, U.option)
            def B(U):
                return lmbda*diffND(U, self.dim, self.bc)
        else:
            A,_ = self.mats
            def B(U):
                return self.fix_bc(A@U)

        if "RKC1" in method:
            s, mu, nu, kp = (self.RKC_params[it] for it in ["s", "mu", "nu", "kp"])

            d0 = 0
            z1 = B(U)
            d1 = mu[1]*(B(U) + fx)
            d = [d0, d1]
            c0 = 0; c1 = mu[1]
            c = [c0, c1]
            dj = d1
            cj = c1
            for j in range(2,s+1):
                dj = nu[j]*d[-1] + kp[j]*d[-2] + mu[j]*(B(d[-1] + U) + fx)
                cj = nu[j]*c[-1] + kp[j]*c[-2] + mu[j]
                d[-2] = d[-1]; d[-1] = dj; c[-2] = c[-1]; c[-1] = cj

            return dj

        elif "RKC2" in method:
            s, mu, nu, kp, gm = (self.RKC_params[it] for it in ["s", "mu", "nu", "kp", "gm"])

            d0 = 0
            z1 = B(U);
            f0 = fx
            d1 = mu[1]*(z1 + f0)
            d = [d0, d1]
            c0 = 0; c1 = mu[1]
            c = [c0, c1]
            dj = d1
            cj = c1
            for j in range(2,s+1):
                dj = nu[j]*d[-1] + kp[j]*d[-2] + mu[j]*B(d[-1] + U) + (mu[j] + gm[j])*f0 + gm[j]*z1
                cj = nu[j]*c[-1] + kp[j]*c[-2] + mu[j] + gm[j]
                d[-2] = d[-1]; d[-1] = dj; c[-2] = c[-1]; c[-1] = cj

            return dj

    def dU(self, U, f):
        if self.params["exact_linalg"]: return LPV(self.dUex(U.array(), f.array()), U.option)

        method = self.method
        lmbda  = self.lmbda
        A,B    = self.mats

        if self.params["smart_differencing"]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ldiff = lmbda*diffND(U, self.dim, self.bc)
        else:
            ldiff = matvec_csr(*self.csr_mats[0], U)

        if method == "FE":
            return ldiff + f
        elif "RK4" in method: return self.update_rk4(U)
        elif "RKC" in method: return self.update_rkc(U) 
        elif "RK" in method:
            Stcoeffs = self.RK_coeffs["St"]
            return self.LPV_polymatvec(Stcoeffs, A, ldiff) + f
        elif method in ["BE","IM"]:
            nbands = self.K**(self.dim-1)
            rhs = ldiff + f
            if self.params["use_multigrid"]:
                return self.MGsolve(rhs)
            else:
                return solve_symmetric(B.toarray(), nbands, rhs)

    def nondelta(self, U, f):
        method = self.method
        lmbda  = self.lmbda
        A,B    = self.mats

        if method == "FE":
            return matvec_csr(*self.csr_mats[1], U) + f
        elif method == "BE":
            nbands = self.K**(self.dim-1)
            if self.params["use_multigrid"]:
                return self.MGsolve(U+f)
            else:
                return solve_symmetric(B.toarray(), nbands, U+f)
        elif method == "IM":
            nbands = self.K**(self.dim-1)
            rhs = matvec_csr(*self.csr_mats[1], U) + f
            if self.params["use_multigrid"]:
                return self.MGsolve(rhs)
            else:
                return solve_symmetric(B.toarray(), nbands, rhs)
        elif "RK" in method:
            Scoeffs = self.RK_coeffs["S"]
            return self.LPV_polymatvec(Scoeffs, A, U) + f

    def update(self, U, f):
        try: mode = U.option.get_round()
        except AttributeError: return U + self.dUex(U,f)

        if mode == 1:
            if self.params["delta_form"]:
                check = U + self.dU(U, f)
            else:
                check = self.nondelta(U, f)

        elif mode == 5:
            if self.params["delta_form"]:
                if not self.params["std_linalg"]:
                    check = U + self.dU(U, f)
                else:
                    U.option = options; f.option = options;
                    temp = self.dU(U,f)
                    temp.option = options_sr; U.option = options_sr
                    check = U + temp
            else:
                check = self.nondelta(U, f)

        else:
            raise NotImplementedError("Not implemented for other rounding modes")

        if self.params["lambda_loop"]:
            try: assert all(np.isfinite(check.array()))
            except AssertionError:
                check = LPV(np.ones_like(check.array()), check.option)
        else:
            assert all(np.isfinite(check.array()))

        return check

    def get_local_errors(self, U_c, U_sr, fx_c, fx_sr):
        tot_err_c = 0.0

        if self.params["delta_form"]:
            if not self.params["MC"]:
                Ut_c  = self.dUex(U_c.array(), fx_c.array())
                UUt_c = self.update(U_c.array(), fx_c.array())
            Ut_sr = self.dUex(U_sr.array(), fx_sr.array())

            if not self.params["MC"]:
                UU_c = self.update(U_c, fx_c).array()
                U_c  = self.dU(U_c,fx_c).array()

            if not self.params["std_linalg"]:
                U_sr = self.dU(U_sr, fx_sr).array()
            else:
                U_sr.option = options; fx_sr.option = options;
                U_sr = self.dU(U_sr,fx_sr).array()

        else:
            if not self.params["MC"]:
                Ut_c  = self.update(U_c.array(), fx_c.array())
            Ut_sr = self.update(U_sr.array(), fx_sr.array())

            if not self.params["MC"]:
                U_c  = self.update(U_c,fx_c).array()
            U_sr = self.update(U_sr, fx_sr).array()

        if not self.params["MC"]:
            err_c  = norm(Ut_c-U_c,np.inf)/u
            if self.params["delta_form"]:
                tot_err_c = norm(UUt_c - UU_c, np.inf)/u
        else:
            err_c = 0.0

        err_sr = norm(Ut_sr-U_sr,np.inf)/u

        return np.array([err_c, err_sr, tot_err_c])


    def get_diff_error(self, U_c, U_sr): 

        lmbda = self.lmbda
        A,B   = self.mats
        dt = self.dt
        h = self.h
        mod_csr_data = list(csr_data(A))
        mod_csr_data[0] = mod_csr_data[0]/dt

        if self.params["smart_differencing"]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if not self.params["MC"]:
                    AU_c  = (diffND(U_c,self.dim,self.bc)/h**2).array()

                AU_sr = (diffND(U_sr,self.dim,self.bc)/h**2).array()
        else:
            if not self.params["MC"]:
                AU_c  = matvec_csr(*mod_csr_data, U_c).array()
            AU_sr = matvec_csr(*mod_csr_data, U_sr).array()


        Uex_sr = A@(U_sr.array())
        if self.params["smart_differencing"]:
            Uex_sr = self.fix_bc(Uex_sr)
        Uex_sr = Uex_sr/dt

        if not self.params["MC"]:
            Uex_c  = A@(U_c.array())
            if self.params["smart_differencing"]:
                Uex_c = self.fix_bc(Uex_c)
            Uex_c = Uex_c/dt

        if not self.params["MC"]:
            nrm_c  = norm(Uex_c,np.inf)
        else:
            nrm_c = 0

        nrm_sr = norm(Uex_sr,np.inf)

        errs = np.zeros((2,))
        # NOTE: unnormalized now since we moved the BCs
        if nrm_c > 0:
            errs[0] = norm(Uex_c-AU_c,np.inf)/u
        if nrm_sr > 0:
            errs[1] = norm(Uex_sr-AU_sr,np.inf)/u

        return errs

    def solve(self, dt, lmbda, T=1):

        method    = self.method
        params    = self.params
        source    = self.source
        exact_sol = self.exact_sol
        dim       = self.dim

        if dim == 1 and self.timedep:
            h = np.sqrt(dt/lmbda);
            n = int(np.round(1/h))
            h = 1./n;
            dt = lmbda*h**2
        else:
            h = np.sqrt(dt/lmbda);
            n = 2**int(np.round(np.log2(1/h)))
            h = 1./n;
            dt = lmbda*h**2


        K = n-1
        N = K**dim

        self.dt    = dt
        self.h     = h
        self.lmbda = lmbda
        self.K     = K
        self.N     = N
        self.T     = T

        if mpiRank == 0:
            print("method = ", method, ", dt = ", dt, ", h = ", h, ", lambda = ", lmbda)
            print("params = ", params)

        if n-1 < 2: raise ValueError("Mesh size too small!")

        x = [np.linspace(0,1,n+1)]*dim
        if dim > 1:
            x = np.meshgrid(*x)
        self.x = x

        #################################################################################

        self.mats = self.assemble_matrices(K, lmbda, dim, method)
        A = self.mats[0]
        C = None
        if method == "FE":
            C = csr_data(sp.eye(A.shape[0]) + A)
        elif method == "IM":
            C = csr_data(sp.eye(A.shape[0]) + A/2)
        self.csr_mats = (csr_data(A), C)

        if exact_sol is not None:
            U_ex = self.cut_borders(exact_sol(T,*x))
        else:
            U_ex = self.bc*np.ones((N,))

        U    = self.bc*np.ones((N,))
        self.ICstrings = (str(self.bc), str(self.bc))

        #Ufunc = lambda x : 1 + np.sin(8*np.pi*x[0])
        #self.ICstrings = ("sin", "1 + sin(8*pi*x)")
        #U = self.cut_borders(Ufunc(self.x))

        #self.ICstrings = ("neg", "negativeSSsol + 2")
        #U = -U_ex + 2

        #self.ICstrings = ("random", "randn + 1")
        #U = 1 + np.random.randn(N)

        #self.ICstrings = ("hat", "3/2 - |x-1/2|")
        #U = 3/2 - abs(x[0]-1/2)
        #U = self.cut_borders(U)

        U_c  = LPV(U,options)
        U_sr = LPV(U,options_sr)

        #################################################################################

        t = 0
        local_errs = np.zeros((3,))
        diff_errs  = np.zeros((2,))
        while t <= T:
            t += dt

            fx,fx_c,fx_sr = self.get_sources(t-dt,t)

            if params["compute_local_errors"]:
                local_errs = np.maximum(self.get_local_errors(U_c, U_sr, fx_c, fx_sr), local_errs)
                if method == "FE":
                    diff_errs  = np.maximum(self.get_diff_error(U_c, U_sr), diff_errs)

            U = self.update(U,fx)
            assert all(np.isfinite(U))
            if not params["MC"]:
                U_c  = self.update(U_c,fx_c)
            U_sr = self.update(U_sr, fx_sr)

        #################################################################################

        if params["compute_local_errors"]:
            if not params["MC"]:
                local_errs[0] = local_errs[0]/norm(U_c.array(), np.inf)
            local_errs[1] = local_errs[1]/norm(U_sr.array(), np.inf)


        if params["compute_local_errors"] and mpiRank == 0:
            print("Local errors: ", local_errs[0], local_errs[1], local_errs[2], "; Differencing errors: ", diff_errs[0], diff_errs[1])

        if norm(U_ex) > 0 and mpiRank == 0:
            print("Errors (inf norm): ", norm(U_ex-U,np.inf)/norm(U_ex,np.inf), norm(U_ex-U_c.array(),np.inf)/norm(U_ex,np.inf), norm(U_ex-U_sr.array(),np.inf)/norm(U_ex,np.inf))
            print("Errors (2   norm): ", norm(U_ex-U,2)/norm(U_ex,2), norm(U_ex-U_c.array(),2)/norm(U_ex,2), norm(U_ex-U_sr.array(),2)/norm(U_ex,2))

        if not params["MC"]:
            nrm_c  = norm(U_c.array(), np.inf);  nrm2_c  = norm(U_c.array(), 2);
        else:
            nrm_c = 1; nrm2_c = 1
        nrm_sr = norm(U_sr.array(), np.inf); nrm2_sr = norm(U_sr.array(), 2);
        global_errs = [norm(U-U_c.array(),np.inf)/nrm_c/u, norm(U-U_sr.array(), np.inf)/nrm_sr/u, norm(U-U_c.array(),2)/nrm2_c/u, norm(U-U_sr.array(), 2)/nrm2_sr/u]
        if mpiRank == 0:
            print("Global errors: ", global_errs)

        if params["vtk_export"]:
            self.export_to_vtk(U_ex, U, U_c, U_sr)

        if params["plot_results"]:
            self.plot_results(U_ex, U, U_c, U_sr)

        return np.array(global_errs), local_errs, diff_errs

    def export_to_vtk(self, U_ex, U, U_c, U_sr):
        K = self.K
        y = np.linspace(0,1,K+2); y = y[1:-1]
        yy = np.linspace(0,1,max(100,K+2))

        if not self.params["delta_form"]:       string = "nodelta"
        elif self.params["smart_differencing"]: string = "delta_smart"
        else:                                   string = "delta_nosmart"
        name = "./results/vtk_data/" + self.method + "_" + string + "_%dD" % self.dim + "_IC-" + self.ICstrings[0]

        if self.dim == 1:
            print("Warning: 1D exporting to vtk is not supported. Saving to npz file...")
            coords = np.linspace(0,1,K+2)
            sol_c  = np.concatenate([[self.bc],U_c.array(),[self.bc]])
            sol_sr = np.concatenate([[self.bc],U_sr.array(),[self.bc]])
            sol_ex = np.concatenate([[self.bc],U_ex,[self.bc]])
            np.savez(name + "_" + str(self.dt) + ".npz", x=coords, RtN=sol_c, SR=sol_sr, exact=sol_ex, info=self.ICstrings[1])
            return
        elif self.dim == 2:
            z = np.array([0.0]); zz = z
            coords = [item for item in np.meshgrid(y,y,z)]
            coords2 = [item for item in np.meshgrid(yy,yy,zz)]
        elif self.dim == 3:
            coords = [item for item in np.meshgrid(y,y,y)]
            coords2 = [item for item in np.meshgrid(yy,yy,yy)]

        if norm(U_ex) > 0:
            U_ex = exact_sol(self.T,*coords2[:self.dim])
            vtk_export(coords2, {"exact" : U_ex}, name = name + "_exact")

        if self.dim == 2:
            data = {"double" : self.tensorize(U)[...,np.newaxis], "RtN" : self.tensorize(U_c.array())[...,np.newaxis], "SR" : self.tensorize(U_sr.array())[...,np.newaxis]}
        else:
            data = {"double" : self.tensorize(U), "RtN" : self.tensorize(U_c.array()), "SR" : self.tensorize(U_sr.array())}

        vtk_export(coords, data, name=name + "_" + str(self.dt))

    def plot_results(self, U_ex, U, U_c, U_sr):
        from mpl_toolkits.mplot3d import Axes3D  
        # Axes3D import has side effects, it enables using projection='3d' in add_subplot
        import matplotlib.pyplot as plt
        if self.dim == 3:
            raise NotImplementedError("3D plotting not supported")

        method = self.method
        h = self.h
        x = self.x

        fontsize=35
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{bm}\usepackage{amsmath}')
        plt.rc('font', family='serif', size=fontsize)
        plt.rc('xtick', labelsize=fontsize)     
        plt.rc('ytick', labelsize=fontsize)

        fig = plt.figure(1,figsize=[15,15])
        if self.dim == 2:
            ax = fig.add_subplot(111, projection='3d')

        if norm(U_ex) > 0:
            xx = self.dim*[np.linspace(0+h,1-h,max(100,self.K))]; xx = np.meshgrid(*xx); U_ex = exact_sol(self.T,*xx)
            if self.dim == 1:
                plt.plot(xx[0], U_ex, "k-",linewidth=3, label="exact")
                plt.plot(x[0][1:-1], U, 'g-', linewidth=3, label="double") 
            elif self.dim == 2:
                surf = ax.plot_surface(xx[0],xx[1],U_ex, label="exact")
                surf._facecolors2d=surf._facecolors3d
                surf._edgecolors2d=surf._edgecolors3d
                surf = ax.plot_surface(self.cut_borders(x[0],True),self.cut_borders(x[1],True),self.tensorize(U), label="double")
                surf._facecolors2d=surf._facecolors3d
                surf._edgecolors2d=surf._edgecolors3d
            else:
                raise NotImplementedError("3D plotting not supported")
        else:
            if self.dim == 1:
                plt.plot(x[0][1:-1], U, 'g-', linewidth=3, label="double") 
            elif self.dim == 2:
                surf = ax.plot_surface(self.cut_borders(x[0],True),self.cut_borders(x[1],True),self.tensorize(U), label="double")
                surf._facecolors2d=surf._facecolors3d
                surf._edgecolors2d=surf._edgecolors3d
            else:
                raise NotImplementedError("3D plotting not supported")

        if self.dim == 1:
            plt.plot(x[0][1:-1], U_c.array() , 'c-', linewidth=3, label="std rounding") 
            plt.plot(x[0][1:-1], U_sr.array(), 'r-',linewidth=3, label="stoch rounding") 
        elif self.dim == 2:
            surf = ax.plot_surface(self.cut_borders(x[0], True),self.cut_borders(x[1],True),self.tensorize(U_c.array()), label="std rounding")
            surf._facecolors2d=surf._facecolors3d
            surf._edgecolors2d=surf._edgecolors3d
            surf = ax.plot_surface(self.cut_borders(x[0], True),self.cut_borders(x[1],True),self.tensorize(U_sr.array()), label="stoch rounding")
            surf._facecolors2d=surf._facecolors3d
            surf._edgecolors2d=surf._edgecolors3d
        else:
            raise NotImplementedError("3D plotting not supported")

        if self.dim == 1:
            plt.legend(loc = 1, fontsize=25)
            plt.xlabel("$x$")
            plt.ylabel("$C^\infty(x)$")
        else:
            ax.legend(fontsize=25)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$C^\infty(x)$")

        if   method == "FE": methodstr = "Forward Euler"
        elif method == "BE": methodstr = "Backward Euler"
        elif method == "IM": methodstr = "Implicit midpoint"
        else: methodstr = method
        plt.title("%s, $\lambda=%.2f$, $\Delta t = %.2e$" % (methodstr, self.lmbda, dt))
        plt.show()

    def solve_MC(self, dt, lmbda, T=1):
        from mc import mc_solve
        from contextlib import redirect_stdout

        self.max_local_err = 0.0

        def mc_fn(N):
            nprocs  = min(mpiSize,max(N,1))
            NN      = [int(N/nprocs)]*nprocs 
            NN[0]  += N%nprocs
            NN     += [0 for i in range(mpiSize - nprocs)]

            n_outputs = 2

            sums = [[0.0]*4 for i in range(n_outputs)]
            for n in range(NN[mpiRank]):
                with redirect_stdout(open("/dev/null","w")):
                    global_errs, local_errs, diff_errs = self.solve(dt, lmbda, T)

                self.max_local_err = max(self.max_local_err, local_errs[1])

                for i in range(4):
                    sums[0][i] += global_errs[1]**(i+1)
                    sums[1][i] += (global_errs[3]**2)**(i+1)

            if mpiSize > 1:
                for k in range(n_outputs):
                    for i in range(4):
                        sums[k][i] = comm.allreduce(sums[k][i], op = MPI.SUM)

            return sums, T/dt*(dt/lmbda)**(-self.dim/2)*N

        avg, var = mc_solve(mc_fn, N0 = 32, tol = 0.05)

        if mpiSize > 1:
            self.max_local_err = comm.allreduce(self.max_local_err, op = MPI.MAX)

        ge = avg[:2]
        ge[1] = np.sqrt(ge[1])
        le = self.max_local_err
        return ge, le, 0.0

if __name__ == "__main__":
    from mms import mms
    import argparse
    parser = argparse.ArgumentParser()

    default_params = {"exact_linalg"          : False,
                      "std_linalg"            : False,
                      "use_multigrid"         : True,
                      "plot_results"          : True,
                      "vtk_export"            : False,
                      "delta_form"            : True, 
                      "smart_differencing"    : True, 
                      "compute_local_errors"  : False, 
                      "save_results"          : False, 
                      "loop"                  : False, 
                      "lambda_loop"           : False,
                      "noprint"               : False,
                      "MC"                    : False,
                      "dim"                   : 1,
                      "time_dependent_source" : False,
                      }

    # Non-optional argument
    parser.add_argument('method', action="store", type=str, help="Specifies numerical method to use")

    # Optional arguments
    parser.add_argument("-d", "--dim", dest="dim", action="store", type=int, help="Specifies dimension to use") 
    parser.add_argument("-MC", "--MC", dest="MC", action="store_true", help="Run MC simulation on stochastic rounding only") 
    parser.add_argument("-nomg", "--nomg", dest="use_multigrid", action="store_false", help="Uses Gaussian elimination rather than multigrid") 
    parser.add_argument("--exact_linalg", dest="exact_linalg", action="store_true", help="Computes update in double precision, then truncate. Applies to both rounding modes") 
    parser.add_argument("-ll", "--lambda-loop", dest="lambda_loop", action="store_true", help="Performs a loop with increasing lambda") 
    parser.add_argument("-td", "--time-dependent-source", dest="time_dependent_source", action="store_true", help="Uses time-dependent source") 
    parser.add_argument("--std_linalg", dest="std_linalg", action="store_true", help="Computes update using round-to-nearest in half precision. Only applies to stochastic rounding in delta form") 
    parser.add_argument("-np", "--noplot", dest="plot_results", action="store_false", help="Deactivates plotting of the results") 
    parser.add_argument("-vtk", "--vtk_export", dest="vtk_export", action="store_true", help="Exports the solution data to vtk format") 
    parser.add_argument("-nd", "--nodelta", dest="delta_form", action="store_false", help="Use the non-delta form implementation of the numerical scheme") 
    parser.add_argument("-nsd", "--nosmart", dest="smart_differencing", action="store_false", help="Deactivates the use of accurate finite differencing when computing A*U") 
    parser.add_argument("-loc", "--local", dest="compute_local_errors", action="store_true", help="Computes the local rounding errors") 
    parser.add_argument("-s", "--save", dest="save_results", action="store_true", help="Saves global and local errors to file. This also activates local error computation and deactivates plotting") 
    parser.add_argument("-l", "--loop", dest="loop", action="store_true", help="Computes results with a hierarchy of discretisations rather than a single one") 
    parser.add_argument("--noprint", dest="noprint", action="store_true", help="Redirects print stdout to /dev/null") 
    [parser.set_defaults(key=val) for key,val in default_params.items()]

    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")

    method = args.method
    params = vars(args)

    dim = params["dim"]

    tailstring = "_%dD" % dim + "_" + options.get_format()

    if method not in ["FE", "BE", "IM", "RK4", "RKC", "RKC1", "RKC2"]: raise NotImplementedError("Specified method (%s) not implemented" % method)
    if params["lambda_loop"]: params["loop"] = True; tailstring += "_LL"
    if params["save_results"]: params["plot_results"] = False; params["compute_local_errors"] = True; params["loop"] = True
    if not params["loop"]: params["save_results"] = False
    if params["noprint"]: sys.stdout = open("/dev/null", "w")
    if params["plot_results"] and dim == 3: params["plot_results"] = False
    if dim == 1: params["use_multigrid"] = False
    if mpiSize > 1 and not params["MC"]: raise RuntimeError("No need for MPI unless doing Monte Carlo since the solver already uses OpenMP for parallelisation")
    if params["MC"]: params["plot_results"] = False; params["vtk_export"] = False; params["compute_local_errors"] = True; tailstring += "_MC"
    if params["time_dependent_source"]: tailstring += "_timedep"

    bc = 1.0
    T  = 1
    
    lmbda = (0.5-2**-4)/dim
    h = 2**-5
    dt = lmbda*h**2

    # if h is picked first, this matches what is actually run by the algorithm
    # since h must be in a power of 2
    L = 6 - int(dim > 1) + int(params["time_dependent_source"])
    lmbdas = (0.5-2**-4)/dim*np.ones(L)
    hs = 2.**(-2-np.arange(L))
    dts = lmbdas*hs**2

    if params["lambda_loop"]:
        L += 1
        hs = (2**-7)*np.ones((L+2,))
        lmbdas = 2.**np.arange(5,L+2+5)+5
        dts = lmbdas*hs**2

    if dim == 1:
        eigmode = 1
        ex_sol = "16*(x*(x-1))**2+" + str(bc)
        if params["time_dependent_source"]: ex_sol = "1/5*(1+4*cos(2*pi*t))*" + ex_sol
        exact_sol,source = mms(ex_sol)
    elif dim == 2:
        ex_sol = "(16*x*y*(1-x)*(1-y))**2+" + str(bc)
        if params["time_dependent_source"]: ex_sol = "1/5*(1+4*cos(2*pi*t))*" + ex_sol
        exact_sol,source = mms(ex_sol)
    else:
        ex_sol = "(64*x*y*z*(1-x)*(1-y)*(1-z))**2+" + str(bc)
        if params["time_dependent_source"]: ex_sol = "1/5*(1+4*cos(2*pi*t))*" + ex_sol
        exact_sol,source = mms(ex_sol)

    problem = LPProblem(method, params, source, exact_sol, bc=bc)
    if not params["loop"]:
        if not params["MC"]:
            errs = problem.solve(dt, lmbda = lmbda,T=T)
        else:
            errs = problem.solve_MC(dt, lmbda = lmbda, T=T)
            if mpiRank == 0: print(errs)
    else:
        if not params["MC"]:
            errs = [problem.solve(dt, lmbda = lmbda,T=T) for dt,lmbda in zip(dts,lmbdas)]
        else:
            errs = [problem.solve_MC(dt, lmbda = lmbda, T=T) for dt,lmbda in zip(dts,lmbdas)]
            print(errs)

        global_errors = np.array([err[0] for err in errs])
        local_errors  = np.array([err[1] for err in errs])
        diff_errors   = np.array([err[2] for err in errs])

        if mpiRank == 0:
            print("Global errors:\n\n", global_errors)

            if params["compute_local_errors"]:
                print("Local errors:\n\n", local_errors)
                print("Differencing errors:\n\n", diff_errors)

            if params["save_results"]:
                if not params["delta_form"]:       string = "nodelta"
                elif params["smart_differencing"]: string = "delta_smart"
                else:                              string = "delta_nosmart"
                outdict = {"global" : global_errors, "local" : local_errors, "diff" : diff_errors, "dt" : dts, "lmbda" : lmbdas}
                np.savez("./results/" + method + "_" + string + tailstring + ".npz", **outdict)
