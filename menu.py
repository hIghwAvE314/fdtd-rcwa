import numpy as np

CONVERGE_TEST = 0
SPECTRUM_TEST = 1
GEOM_SWEEP = 2

def _gap_position(pos=-0.05, period=0.95, gap=0.05, length=0.72):
    """
    -period/2 + gap/2 <= pos <= -period/2 + length - gap/2
    if period == 0.95 and gap == 0.05 and length == (0.40+0.05+0.27=0.72):
    -0.45 (-0.95/2+0.05/2) <= pos <= 0.22 (-0.95/2+0.40+0.05+0.27-0.05/2), default -0.05
    returns: l1, l2
    """
    xmin = -period/2
    l1 = pos - xmin - gap/2
    l2 = length - l1 - gap
    return l1, l2

def converge_test(method='fdtd', **kwargs):
    """
    convergence test for 'fdtd' and 'rcwa' method
    returns : a list of resolution/nmodes
    """
    if method == 'fdtd':
        res = np.array([10*n for n in range(1,10)] + list(range(100, 300, 20)))
        return res
    elif method == 'rcwa':
        nmodes = np.arange(11, 71, 2)
        return nmodes


def spectrum(fcen=1.25, df=2.0, nfreqs=41, **kwargs):
    """
    fcen = 1.25 <==> wl = 1.064/1.33=0.8
    df = 2.0 so that the spectrum span the only 0 mode case and +-2 mode case
    returns : a list of wavelength used
    """
    freqs = np.linspace(fcen-0.5*df, fcen+0.5*df, nfreqs)
    return 1/freqs

def geom_sweep(period=0.95, gap=0.05, length=0.72, ngeom=21, **kwargs):
    """
    sweep gap center from one end to the other
    returns : list of l1 and l2 used
    """
    pos = np.linspace(-period/2 + gap/2, -period/2 + length - gap/2, ngeom)
    l1, l2 = _gap_position(pos=pos, period=period, gap=gap, length=length)
    return list(zip(pos, l1, l2))


FUNCTION_LIST = [converge_test, spectrum, geom_sweep]