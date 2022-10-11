import numpy as np
import sys, time

from Params import *
from layers import *
from menu import *


"""
Below is the default parameters used in generating test sequence,
if any parameters need to be changed, change it here.
"""
KWARGS = {
    'fcen' : 1.25, 'df' : 2.0, 'nfreqs' : 81,
    'period' : 0.95, 'gap' : 0.05, 'length':0.72, 'ngeom' : 21,
    'method' : 'rcwa',
}


class Geom(Structure):
    l1, gap, l2 = 0.40, 0.05, 0.27
    h = 0.46  # thickness of the main device
    w = 0.20  # width of the main device
    # nl, nh = (1.45+0j)/1.33, (3.45+0j)/1.33  # refractive index of low RI and high RI medium
    nl, nh = 1.45+0j, 3.45+0j  # refractive index of low RI and high RI medium
    Lx, Ly = 0.95, 0.60  # periodicity
    hsub = 0.4  # thickness of substrate
    hcap = 1 - h - hsub  # thickness of cap
    # nrf = 1.+0j #1.33
    # ntm = 1.+0j #1.33
    nrf = 1.33+0j #1.33
    ntm = 1.33+0j #1.33

    def init(self, params:RCWAParams):
        self.period = (self.Lx, self.Ly)
        self.Nx, self.Ny = int(1/params.dx+1), int(1/params.dy+1)
        self.x, self.y = np.mgrid[0:self.Lx:1j*self.Nx, 0:self.Ly:1j*self.Ny]
        mask = ((self.x>=0)*(self.x<=self.l1) + (self.x>=self.l1+self.gap)*(self.x<=self.l1+self.gap+self.l2)) * ((self.y>=self.w)*(self.y<=2*self.w))
        self.ur = [1.+0j, 1.+0j, 1.+0j]
        self.er = [self.nl**2]
        self.hs = [self.hsub]

        eps = np.where(mask, self.nh**2, self.nl**2)
        self.er.append(eps)
        self.hs.append(self.h)

        self.er.append(self.nl**2)
        self.hs.append(self.hcap)

        self.errf, self.urrf = self.nrf**2, 1.+0j
        self.ertm, self.urtm = self.ntm**2, 1.+0j


def main_loop(params, source, geom):
    sim = Layers(params, source, geom)
    sim.solve()
    R = sim.Rtot
    T = sim.Ttot
    F = sim.F
    return R, T, F


"""
Calling from script, first argument is option, 
and the other two are the index in group and number of groups,
such that the job can be seperated into several parts to be finished by different BC4 nodes.
If no multiple nodes speed up is needed, use 0 and 1 as grp and ngrps,
i.e., "python main-fdtd.py opt 0 1"
"""

option, grp, ngrps = map(int, sys.argv[1:])
option_func = FUNCTION_LIST[option]
param_list = option_func(**KWARGS)[grp::ngrps]

if option == CONVERGE_TEST:
    Nmodes = []
    ref = []
    trm = []
    forces = []
    time_usage = []
    for n, nmodes in enumerate(param_list):
        Nmodes.append(nmodes)
        params = RCWAParams()
        params.dx, params.dy = 1e-3, 1e-3
        params.dtype = np.complex64
        params.Nmx, params.Nmy = nmodes, nmodes
        params.init()
        source = Source()
        source.wl = 1.064
        source.init(params)
        geom = Geom()
        geom.init(params)

        start = time.time()
        R, T, force = main_loop(params, source, geom)
        end = time.time()
        ref.append(R)
        trm.append(T)
        forces.append(force)
        time_usage.append(end-start)
    np.savez(
        "rcwa_converge_grp{}_{}".format(grp, ngrps),
        nmodes = Nmodes,
        R = ref,
        T = trm,
        force = forces,
        time = time_usage,
    )
elif option == SPECTRUM_TEST:
    wls = param_list
    ref = []
    trm = []
    forces = []
    time_usage = []
    for n, wl in enumerate(param_list):
        params = RCWAParams()
        params.dx, params.dy = 1e-3, 1e-3
        params.dtype = np.complex64
        params.Nmx, params.Nmy = 41, 41
        params.init()
        source = Source()
        source.wl = wl
        source.init(params)
        geom = Geom()
        geom.init(params)

        start = time.time()
        R, T, force = main_loop(params, source, geom)
        end = time.time()
        ref.append(R)
        trm.append(T)
        forces.append(force)
        time_usage.append(end-start)
    np.savez(
        "rcwa_spectrum_grp{}_{}".format(grp, ngrps),
        wl = wls,
        R = ref,
        T = trm,
        force = forces,
        time = time_usage,
    )
elif option == GEOM_SWEEP:
    poses = []
    ref = []
    trm = []
    forces = []
    time_usage = []
    for pos, l1, l2 in param_list:
        poses.append(pos)
        params = RCWAParams()
        params.dx, params.dy = 1e-3, 1e-3
        params.dtype = np.complex64
        params.Nmx, params.Nmy = 41, 41
        params.init()
        source = Source()
        source.wl = 1.064/1.33
        source.init(params)
        geom = Geom()
        geom.l1 = l1
        geom.l2 = l2
        geom.gap = KWARGS['gap']
        geom.init(params)

        start = time.time()
        R, T, force = main_loop(params, source, geom)
        end = time.time()
        ref.append(R)
        trm.append(T)
        forces.append(force)
        time_usage.append(end-start)
    np.savez(
        "rcwa_geomsweep_grp{}_{}".format(grp, ngrps),
        pos = poses,
        R = ref,
        T = trm,
        force = forces,
        time = time_usage,
    )
else:
    raise ValueError("Unknown option")



