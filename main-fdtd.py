import meep as mp
import numpy as np
import sys, time

from Sim import *
from menu import *



"""
Below is the default parameters used in generating test sequence,
if any parameters need to be changed, change it here.
"""
KWARGS = {
    'fcen' : 1.25, 'df' : 2.0, 'nfreqs' : 81,
    'period' : 0.95, 'gap' : 0.05, 'length':0.72, 'ngeom' : 21,
    'method' : 'fdtd',
}


class Params:
    res = 100 # 200
    sx, sy = 0.95, 0.60
    l1, gap, l2 = 0.40, 0.05, 0.27
    w, h, dsub, dcap = 0.20, 0.46, 0.40, 0.60
    dpad, dpml = 0.5, 1#40/res, 100/res
    n0, nl, nh = 1.33/1.33, 1.45/1.33, 3.45/1.33  # refractive index of background, low and high materials
    wl, df, nfreqs = 1.064/1.33, 0.4, 21
    pol = mp.Ex

    def __init__(self):
        pass

    def init(self):
        self.fcen = 1./self.wl
        self.thick = self.dsub + self.dcap
        self.sz = 2*(self.dpml + self.dpad) + self.thick
        self.mon_pt = mp.Vector3(z=self.thick/2 + self.dpad/5)


class Geometry:
    def __init__(self, params):
        self.cell_size = mp.Vector3(params.sx, params.sy, params.sz)
        self.empty_geom, self.geom = self._get_geoms(params)
        self.pml = [mp.PML(thickness=params.dpml, direction=mp.Z)]

    def _get_geoms(self, params):
        empty_geom = [
            mp.Block(
                size=self.cell_size,
                center=mp.Vector3(),
                material=mp.Medium(index=params.n0),
            ),  # background
        ]
        # geom_z = -params.sz/2 + params.dpml + params.dpad + params.dsub + params.h/2
        geom = [
            mp.Block(
                size=mp.Vector3(params.sx, params.sy, params.thick),
                center=mp.Vector3(),
                material=mp.Medium(index=params.nl),
            ),  # substrate

            mp.Block(
                size=mp.Vector3(params.l1, params.w, params.h),
                center=mp.Vector3(
                    x=params.l1/2 - params.sx/2,
                    z=params.dsub+params.h/2 - params.thick/2,
                ),
                material=mp.Medium(index=params.nh),
            ),  # block 1
            mp.Block(
                size=mp.Vector3(params.l2, params.w, params.h),
                center=mp.Vector3(
                    x=params.l1+params.gap+params.l2/2 - params.sx/2,
                    z=params.dsub + params.h / 2 - params.thick/2,
                ),
                material=mp.Medium(index=params.nh),
            ),  # block 2
        ]
        return empty_geom, empty_geom+geom


class Source:
    def __init__(self, params, **kwargs):
        self.src = mp.GaussianSource(
            frequency=params.fcen,
            fwidth=params.df,
            is_integrated=True,
            **kwargs,
        )
        self.sources = self._get_sources(params)

    def _get_sources(self, params):
        source = mp.Source(
            self.src,
            component=params.pol,
            size=mp.Vector3(params.sx, params.sy),
            center=mp.Vector3(z=params.dpml+params.dpad/2 - params.sz/2)
        )
        return [source]


def main_loop(params, is_spectrum=False):
    geom = Geometry(params)
    src = Source(params)

    sim = MVSim(params, geom, src)
    if is_spectrum:
        sim.set_monitors(nperiods=10, df=params.df, nfreqs=params.nfreqs)
    else:
        sim.set_monitors(nperiods=10)
    sim.run(tol=1e-5)
    flux = sim.get_flux()  # [incident_flux, reflect_flux, transmitted_flux]
    fx = mp.get_forces(sim.force_fields[0])
    fy = mp.get_forces(sim.force_fields[1])
    fz = mp.get_forces(sim.force_fields[2])
    forces = np.array([fx, fy, fz])
    return flux, forces


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
    resolutions = []
    fluxes = []
    forces = []
    time_usage = []
    for n, res in enumerate(param_list):
        resolutions.append(res)
        params = Params()
        params.res = int(res)
        params.init()
        start = time.time()
        flux, force = main_loop(params)
        end = time.time()
        fluxes.append(flux)
        forces.append(force)
        time_usage.append(end-start)
    if mp.am_master():
        np.savez(
            "fdtd_converge_grp{}_{}".format(grp, ngrps),
            res = resolutions,
            flux = fluxes,
            force = forces,
            time = time_usage,
        )
elif option == SPECTRUM_TEST:
    params = Params()
    wls = param_list
    params.wl = 1/KWARGS['fcen']
    params.df = KWARGS['df']
    params.nfreqs = KWARGS['nfreqs']
    params.init()
    start = time.time()
    flux, force = main_loop(params, is_spectrum=True)
    end = time.time()
    if mp.am_master():
        np.savez(
            "fdtd_spectrum",
            wl = wls,
            flux = flux,
            force = force,
            time = end-start,
        )
elif option == GEOM_SWEEP:
    poses = []
    fluxes = []
    forces = []
    time_usage = []
    for pos, l1, l2 in param_list:
        poses.append(pos)
        params = Params()
        params.l1 = l1
        params.l2 = l2
        params.gap = KWARGS['gap']
        params.init()
        start = time.time()
        flux, force = main_loop(params)
        end = time.time()
        fluxes.append(flux)
        forces.append(force)
        time_usage.append(end-start)
    if mp.am_master():
        np.savez(
            "fdtd_geomsweep_grp{}_{}".format(grp, ngrps),
            pos = poses,
            flux = fluxes,
            force = forces,
            time = time_usage,
        )
else:
    raise ValueError("Unknown option")



