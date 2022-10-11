import meep as mp
import numpy as np
import matplotlib.pyplot as plt

from Sim import *


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


fluxes = []
reses = []
forces = []
mst = []
for res in range(50, 300, 50):
    if mp.am_master(): print(f"resolution {res} started")
    params = Params()
    params.res = res
    params.n0 = 1.
    params.init()
    geom = Geometry(params)
    src = Source(params)

    sim = MVSim(params, geom, src)
    sim.set_monitors(nperiods=10)
    sim.run(tol=1e-5)
    flux = sim.get_flux()
    fluxes.append(flux)
    res_ref, res_trm, f_eig = sim.get_eigenmode_force(nbands=9, flux=fluxes[0])
    forces.append(f_eig)
    reses.append(res)
    fx = mp.get_forces(sim.force_fields[0])[0]
    fy = mp.get_forces(sim.force_fields[1])[0]
    fz = mp.get_forces(sim.force_fields[2])[0]
    mst.append([fx, fy, fz])
    if mp.am_master(): print(f"\n {res}, {flux[0]}, {flux[1]}, {flux[2]}, {f_eig[0]}, {f_eig[1]}, {f_eig[2]}, {fx}, {fy}, {fz} \n")
    if mp.am_master(): print(f"resolution {res} completed!")


with open("res_conv.csv", 'w') as f:
    f.write("Res, Inc, Ref, Tran, Fx(md), Fy(md), Fz(md), Fx(mst), Fy(mst), Fz(mst)\n")
    for res, flux, f_eig, f_mst in zip(reses, fluxes, forces, mst):
        f.write(f"{res}, {flux[0]}, {flux[1]}, {flux[2]}, {f_eig[0]}, {f_eig[1]}, {f_eig[2]}, {f_mst[0]}, {f_mst[1]}, {f_mst[2]} \n")