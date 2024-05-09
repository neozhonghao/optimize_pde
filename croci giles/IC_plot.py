#########################################################################################################################################
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#########################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import subprocess
from glob import glob
import os
import sys

savefig = False
BW = True

################################################# PYPLOT SETUP ###############################################

# change the round factor if you like
r = 1

screens = [l.split()[-3:] for l in subprocess.check_output(
    ["xrandr"]).decode("utf-8").strip().splitlines() if " connected" in l]

sizes = []
for s in screens:
    w = float(s[0].replace("mm", "")); h = float(s[2].replace("mm", "")); d = ((w**2)+(h**2))**(0.5)
    sizes.append([2*round(n/25.4, r) for n in [w, h, d]])

gnfntsz = 30
fntsz = 25
ttlfntsz = 35

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{amsmath}\usepackage{bm}\usepackage{relsize}\DeclareMathOperator{\uu}{\mathfrak{u}}\DeclareMathOperator{\EPS}{\mathlarger{\mathlarger{\mathlarger{\varepsilon}}}}')
plt.rc('font', family='serif', size=gnfntsz)
plt.rc('xtick', labelsize=gnfntsz)     
plt.rc('ytick', labelsize=gnfntsz)

###################################### LOAD DATA FILES ########################################

basepath = "./results/vtk_data/"

methods = ["FE"]

strings = ["delta_smart"]
nicerstrings = [r"delta form"]

files = glob(basepath + "*1D*npz")
files.sort()

data = []
[data.append(dict(np.load(f))) for f in files]

x = data[0]['x']
U_ex = data[0]['exact']
U_sr = data[0]['SR'] # NOTE: they are all visually the same, it turns out, so no need to plot all of them or the plot will get messy

ICs = [r"$\uu_0 = 1$", r"$\uu_0 = 3/2 - |x-1/2|$", r"$\uu_0 = 1 + \text{noise}$", r"$\uu_0 = 1 + \sin(8\pi x)$"]
legend = ["double (same as exact)", "SR, all initial conditions"] + ["RtN, " + item for item in ICs]

################################################ PLOT RESULTS #########################################################

plt.close("all")
colors = ["steelblue", "tab:green", "tab:orange", "tab:cyan", "gold", "slategrey", "tab:brown"]
markers = "*^+xo*sD"
linestyles = ["-", (0,(1,1)), (0,(5,5)), (0,(3,1,1,1)), (0,(1,3))]
linestyles_glob = ["", "-."]
colormap = plt.cm.Paired
cmap = [colormap((2*i+1)/12) for i in range(9)]

def newfig(fign,small=False):
    figsize=sizes[0][:-1]
    if small: figsize[0] /= 2
    fig = plt.figure(fign, figsize=figsize)
    fig.patch.set_facecolor("white")
    return fig

fig = newfig(1,small=True)
if BW:
    colors = ["dimgrey"]
    plt.plot(x, U_ex, '-', color = "darkgrey", linewidth=8)
    plt.plot(x, U_sr, '-', color = "black", linewidth=3)
else:
    plt.plot(x, U_ex, 'k-', linewidth=4)
    plt.plot(x, U_sr, 'r-', linewidth=4)
for i,dat in enumerate(data):
    plt.plot(x,dat["RtN"], color=colors[0], linestyle=linestyles[i], linewidth=4)

plt.xlabel("$x$")
plt.ylabel("$\hat{U}^{\infty}$")
plt.xlim([0.,1.])
plt.legend(tuple(legend), fontsize=fntsz, ncol=1)

fig.tight_layout()

if savefig: fig.savefig("ICplot1D.eps", format='eps', dpi=600)
else: plt.show()
