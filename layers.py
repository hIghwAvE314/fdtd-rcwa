import numpy as np
import scipy as sp
import abc
from scipy import linalg as LA
from scipy import sparse as spa
from scipy.sparse import linalg as spLA
import matplotlib.pyplot as plt


from Params import *
from matrices import *



class Layer:
    def __init__(self, er, ur, h=0.):
        self.is_homo = not (isinstance(er, np.ndarray) or isinstance(ur, np.ndarray))
        self.nr = np.sqrt(er*ur)
        self.er = er
        self.ur = ur
        self.h = h
        self.er_fft = None
        self.ur_fft = None

    @log("Solving Layer")
    def init(self, params:RCWAParams, wvm:WaveVectorMatrix):
        if self.is_homo:
            self.W, self.Lam, self.V = wvm.homo_decompose(self.er, self.ur)
            W, V = self.W, self.V
        else:
            self.er_fft, er = self._init_convol_mat(self.er, params, buffer=self.er_fft)
            self.ur_fft, ur = self._init_convol_mat(self.ur, params, buffer=self.ur_fft)
            er = numpy2torch(er.astype(params.dtype), device=params.device) if isinstance(er, np.ndarray) else er
            ur = numpy2torch(ur.astype(params.dtype), device=params.device) if isinstance(ur, np.ndarray) else ur
            W, self.Lam, V = wvm.general_decompose(er, ur)
            self.W = totorch(W, device='cpu')
            self.V = totorch(V, device='cpu')
        return self.W, self.Lam, self.V

    def _init_convol_mat(self, component, params, buffer=None):
        if isinstance(component, np.ndarray):
            if buffer is None: buffer = fft2(component)
            comp_mn = roll(buffer, params.Mx, params.My)
            conv = convol_matrix(comp_mn, params.Mx, params.My)
        else:
            conv = component * spa.identity(params.Nmodes, dtype=params.dtype, format='csc')
        return buffer, conv

    @log("Calculating Layer S-Matrix")
    def get_Smat(self, params, K, W0, V0, k0):
        W, Lam, V = self.init(params, K)
        if self.is_homo:
            Smat = get_homo_Smatrix(params.Nmodes, Lam, V, V0, k0, self.h, device=params.device)
        else:
            Smat = get_Smatrix(W, Lam, V, W0, V0, k0, self.h, device=params.device)
        return Smat




class Layers(list):
    def __init__(self, params:RCWAParams, src:Source, geom:Structure):
        self.params = params
        self.src = src
        self.geom = geom
        self._init(params, src, geom)

    @log("Initialising layers simulation")
    def _init(self, params, src, geom):
        self.K = WaveVectorMatrix(src, geom, params)
        self.e_src = src.e_src

        self.ref_layer = Layer(geom.errf, geom.urrf)
        self.trm_layer = Layer(geom.ertm, geom.urtm)
        self.gap_layer = Layer(1+0j, 1+0j)

        Vrf = self.ref_layer.init(params, self.K)[-1]
        Vtm = self.trm_layer.init(params, self.K)[-1]
        V0 = self.gap_layer.init(params, self.K)[-1]

        self.ref_Smat = get_refSmatrix(params.Nmodes, Vrf, V0, device=params.device)
        self.trm_Smat = get_trmSmatrix(params.Nmodes, Vtm, V0, device=params.device)

        self.nlayers = len(geom.hs)
        self.layers = [Layer(geom.er[i], geom.ur[i], geom.hs[i]) for i in range(self.nlayers)]
        super().__init__(self.layers)
        self.dev_Smat = None
    
    @log("Solving the problems")
    def solve(self):
        n = 0
        Smats = [self.ref_Smat]
        W0 = self.gap_layer.W
        V0 = self.gap_layer.V
        k0 = self.src.k0
        Nmodes = self.params.Nmodes
        for layer in self.layers:
            print(f"Solving for layer {n}...")
            n += 1
            Smat = layer.get_Smat(self.params, self.K, W0, V0, k0)
            Smats.append(Smat)
        Smats.append(self.trm_Smat)
        self.Smat = get_total_Smat(*Smats)
        print("Solving for the fields and diffraction efficiency...")
        self.get_DE()
        self.is_conserve = self._power_conserve()
        self.get_force()

    def get_DE(self):
        # self.e_ref = self.ref_layer.W @ self.Smat.S11 @ inv(self.ref_layer.W) @ self.e_src
        # self.e_trm = self.trm_layer.W @ self.Smat.S21 @ inv(self.ref_layer.W) @ self.e_src
        self.e_ref = torch2numpy(self.Smat.S11) @ self.e_src
        self.e_trm = torch2numpy(self.Smat.S21) @ self.e_src
        rx = self.e_ref.T[:self.params.Nmodes]
        ry = self.e_ref.T[self.params.Nmodes:]
        rz = - inv(self.K.Kz_rf) @ (self.K.Kx@rx + self.K.Ky@ry)
        tx = self.e_trm.T[:self.params.Nmodes]
        ty = self.e_trm.T[self.params.Nmodes:]
        tz = - inv(self.K.Kz_tm) @ (self.K.Kx@tx + self.K.Ky@ty)
        self.rcoeff = np.array([rx, ry, rz])
        self.tcoeff = np.array([tx, ty, tz])
        r2 = np.sum(np.abs(self.rcoeff)**2, axis=0)
        t2 = np.sum(np.abs(self.tcoeff)**2, axis=0)
        ck = np.real(self.K.k_inc[2]/self.ref_layer.nr)
        crf = np.real(self.K.Kz_rf/self.ref_layer.nr) / ck
        ctm = np.real(self.K.Kz_tm/self.trm_layer.nr) / ck
        self.Ref = (crf @ r2).reshape(self.params.Nmx, self.params.Nmy)
        self.Trm = (ctm @ t2).reshape(self.params.Nmx, self.params.Nmy)
        self.Rtot = np.sum(self.Ref)
        self.Ttot = np.sum(self.Trm)

    def get_force(self, file=''):
        Nmx, Nmy = self.params.Nmx, self.params.Nmy
        Kx = np.real(self.K.Kx.diag.reshape(Nmx, Nmy))
        Ky = np.real(self.K.Ky.diag.reshape(Nmx, Nmy))
        Kz_rf = self.K.Kz_rf.diag.reshape(Nmx, Nmy)
        Kz_tm = self.K.Kz_tm.diag.reshape(Nmx, Nmy)
        mask_rf = np.nonzero(np.real(Kz_rf))
        mask_tm = np.nonzero(np.real(Kz_tm))
        crf = np.sqrt(1 - np.real(Kz_rf[mask_rf]/self.ref_layer.nr)**2)
        ctm = np.sqrt(1 - np.real(Kz_tm[mask_tm]/self.trm_layer.nr)**2)
        phi_rf = np.arctan2(Ky[mask_rf], Kx[mask_rf])
        phi_tm = np.arctan2(Ky[mask_tm], Kx[mask_tm])
        Fx = np.sum(self.Ref[mask_rf]*crf*np.cos(phi_rf) + self.Trm[mask_tm]*ctm*np.cos(phi_tm))
        Fy = np.sum(self.Ref[mask_rf]*crf*np.sin(phi_rf) + self.Trm[mask_tm]*ctm*np.sin(phi_tm))
        Fz = 1 + np.sum(self.Ref[mask_rf]*np.real(Kz_rf[mask_rf]/self.ref_layer.nr) - self.Trm[mask_tm]*np.real(Kz_tm[mask_tm]/self.trm_layer.nr))
        self.F = np.array([Fx, Fy, Fz])
        if file:
            np.savez(
                file,
                Kx = np.real(self.K.Kx.diag).reshape(self.params.Nmx, self.params.Nmy),
                Ky = np.real(self.K.Ky.diag).reshape(self.params.Nmx, self.params.Nmy),
                Kz_rf = self.K.Kz_rf.diag.reshape(Nmx, Nmy),
                Kz_tm = self.K.Kz_tm.diag.reshape(Nmx, Nmy),
                Kz_0 = self.K.Kz_0.diag.reshape(Nmx, Nmy),
                Ref = self.Ref,
                Trm = self.Trm,
                F = self.F,
            )
        return self.F

    def _power_conserve(self):
        return np.isclose(self.Rtot + self.Ttot, 1)

    def change_nmodes(self, Nmx, Nmy):
        print(f"Changing total modes number to {Nmx},{Nmy}")
        self.params.Nmx = Nmx
        self.params.Nmy = Nmy
        self.params.init()
        self.src.init(self.params)
        self.geom.init(self.params)
        self._init(self.params, self.src, self.geom)
        self.solve()

    def converge_test(self, Max, step=2, comp='xy', atol=1e-4):
        if not hasattr(self, 'Rtot'):
            self.solve()
        R = [self.Rtot]
        T = [self.Ttot]
        F = [self.F]
        Mx = self.params.Nmx
        My = self.params.Nmy
        Nx = (Max - Mx) / step 
        Ny = (Max - My) / step 
        N = max(Nx, 1) if 'x' in comp else 1
        N = max(N, Ny) if 'y' in comp else N
        if not hasattr(self, 'is_converge'): self.is_converge=False
        for n in range(int(N)):
            Mx = Mx + step if 'x' in comp else Mx
            My = My + step if 'y' in comp else My
            print(Mx, My)
            self.change_nmodes(Mx, My)
            if np.isclose(self.F[0], F[-1][0], atol=atol) or np.isclose(self.Ttot, T[-1], atol=atol):
                print(f"Convergence is reached at mode ({Mx},{My}), Reflectance {np.real(self.Rtot)}, Transmittance {np.real(self.Ttot)}, Force {self.F}")
                self.is_converge = True
            else:
                self.is_converge = False
            R.append(self.Rtot)
            T.append(self.Ttot)
            F.append(self.F)
        return R, T, F




