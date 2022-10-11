import meep as mp
import numpy as np

class MVSim:
    def __init__(self, params, geom, src):
        self.params = params
        self.geom = geom
        self.src = src

        self.empty = mp.Simulation(
            resolution=params.res,
            cell_size=geom.cell_size,
            boundary_layers=geom.pml,
            geometry=geom.empty_geom,
            sources=src.sources,
            k_point=mp.Vector3(),
            force_complex_fields=True,
        )
        self.sim = mp.Simulation(
            resolution=params.res,
            cell_size=geom.cell_size,
            boundary_layers=geom.pml,
            geometry=geom.geom,
            sources=src.sources,
            k_point=mp.Vector3(),
            force_complex_fields=True,
        )

    def set_monitors(self, nperiods=100, df=0, nfreqs=1):
        mon_pt = self.params.mon_pt
        mon_size = mp.Vector3(self.params.sx, self.params.sy)
        comps = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
        directions = [mp.X, mp.Y, mp.Z]
        empty = self.empty
        sim = self.sim
        fcen = self.params.fcen

        nf_inc = empty.add_dft_fields(comps, fcen, df, nfreqs, center=-1 * mon_pt, size=mon_size)
        nf_ref = sim.add_dft_fields(comps, fcen, df, nfreqs, center=-1 * mon_pt, size=mon_size)
        nf_trm = sim.add_dft_fields(comps, fcen, df, nfreqs, center=mon_pt, size=mon_size)

        fr_plus = mp.FluxRegion(center=mon_pt, size=mon_size)
        fr_minus = mp.FluxRegion(center=-1 * mon_pt, size=mon_size)
        flux_inc = empty.add_flux(fcen, df, nfreqs, fr_minus)
        flux_rec = sim.add_flux(fcen, df, nfreqs, fr_minus)
        flux_trm = sim.add_flux(fcen, df, nfreqs, fr_plus)

        n2f_inc = empty.add_near2far(fcen, df, nfreqs, fr_minus, nperiods=nperiods)
        n2f_ref = sim.add_near2far(fcen, df, nfreqs, fr_minus, nperiods=nperiods)
        n2f_trm = sim.add_near2far(fcen, df, nfreqs, fr_plus, nperiods=nperiods)

        force_field_x = sim.add_force(fcen, df, nfreqs,
                                      *[mp.ForceRegion(w * mon_pt, size=mon_size, direction=mp.X, weight=w) for w in
                                        [+1, -1]])
        force_field_y = sim.add_force(fcen, df, nfreqs,
                                      *[mp.ForceRegion(w * mon_pt, size=mon_size, direction=mp.Y, weight=w) for w in
                                        [+1, -1]])
        force_field_z = sim.add_force(fcen, df, nfreqs,
                                      *[mp.ForceRegion(w * mon_pt, size=mon_size, direction=mp.Z, weight=w) for w in
                                        [+1, -1]])
        force_fields = [force_field_x, force_field_y, force_field_z]
        # force_fields = []
        # for d in directions:
        #     force_regions = []
        #     for w in [+1, -1]:
        #         region = mp.ForceRegion(center=mon_pt, size=mon_size, direction=d, weight=w)
        #         force_regions.append(region)
        #     field = sim.add_force(fcen, 0, 1, *force_regions)
        #     force_fields.append(field)

        self.near_fields = [nf_inc, nf_ref, nf_trm]
        self.flux_fields = [flux_inc, flux_rec, flux_trm]
        self.n2f_fields = [n2f_inc, n2f_ref, n2f_trm]
        self.force_fields = force_fields

    def run(self, tol=1e-6):
        if isinstance(self.src.src, mp.GaussianSource):
            self._run_peak(tol=tol)
        elif isinstance(self.src.src, mp.ContinuousSource):
            self._run_pw()
        else:
            raise TypeError("Unknown source type!")

    def _run_peak(self, tol=1e-6):
        self.empty.run(until_after_sources=mp.stop_when_fields_decayed(.5, mp.Hy, self.params.mon_pt, tol))
        self._load_minus()
        self.sim.run(until_after_sources=mp.stop_when_fields_decayed(.5, mp.Hy, self.params.mon_pt, tol))

    def _run_pw(self):
        pass

    def _load_minus(self):
        flux_data = self.empty.get_flux_data(self.flux_fields[0])
        n2f_data = self.empty.get_near2far_data(self.n2f_fields[0])
        self.sim.load_minus_flux_data(self.flux_fields[1], flux_data)
        self.sim.load_minus_near2far_data(self.n2f_fields[1], n2f_data)

    def get_eigenmode_force(self, nbands=5, flux=1.):
        try:
            ref = self.flux_fields[1]
            trm = self.flux_fields[2]
            res_ref = self.sim.get_eigenmode_coefficients(ref, range(1, nbands+1))
            res_trm = self.sim.get_eigenmode_coefficients(trm, range(1, nbands+1))
            alpha_ref = res_ref.alpha[:,:, 1]
            knorm_ref = np.asarray([np.asarray(kdom)/self.params.fcen/self.params.n0 for kdom in res_ref.kdom])
            alpha_trm = res_trm.alpha[:,:, 0]
            knorm_trm = np.asarray([np.asarray(kdom)/self.params.fcen/self.params.n0 for kdom in res_trm.kdom])
            f_ref = np.sum(knorm_ref * np.abs(alpha_ref)**2 * np.array([-1,-1,2]), axis=0)
            f_trm = np.sum(knorm_trm * np.abs(alpha_trm)**2 * np.array([1,1,0]), axis=0)
            return res_ref, res_trm, (f_ref + f_trm)/flux
        except Exception:
            print(res_ref.kdom)
            print(res_trm.kdom)

    def get_flux(self):
        fluxes = [mp.get_fluxes(flux) for flux in self.flux_fields]
        return np.array(fluxes)

