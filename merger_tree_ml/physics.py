
import numpy as np
import astropy
from astropy.cosmology import Planck13

# Set default cosmology
def default_cosmo():
    """ Return default cosmology  Planck 2013 """
    cosmo = Planck13
    cosmo.sigma8 = 0.82
    cosmo.ns = 0.96
    #cosmo.Lambda = 0.169
    cosmo.Lambda = cosmo.sigma8 * cosmo.Om0   # shape param Lambda = Omega_m * sigma_8
    return cosmo

# Analytical functions
def calc_omega(z):
    """ Compute omega(z), the natural time variable in the EPS theorem
    Following the approximation from Neistein &  Dekel (2008):

    omega(z) = 1.260 [1 + z + 0.09 (1+z)^{-1} + 0.24 e^{-1.16z}]

    """
    return 1.260 * (1 + z + 0.09 / (1 + z) + 0.24 * np.exp(-1.16 * z))

def calc_u(x):
    """ u(x) = 64.087 * [ 1 + 1.074 x^0.3 - 1.581 x^0.4 + 0.954 x^0.5 - 0.185 x^0.6]^{-10} """
    return 64.087 * (1 + 1.074*x**0.3 - 1.581*x**0.4 + 0.954*x**0.5 - 0.185*x**0.6)**(-10)


def calc_S(M, cosmo=None):
    """ Compute S(M), the variance of the density field smoothed with
    a spherical top-hat window function of a radius that on average encompasses a mass M in real space

    S(M) = u^2(c0 * Lambda * M^{1/3}/ Omega_m^{1/3}) * sigma8^2 / u^2(32 * Lambda)

    where Lambda=0.169 is power spectrum shape parameter and c0=3.804 x 10^{-4}
    """

    if cosmo is None:
        cosmo = default_cosmo()
    c0 = 3.804e-4
    x1 = c0 * cosmo.Lambda * M**(1/3) / cosmo.Om0**(1/3)
    x2 = 32 * cosmo.Lambda
    return calc_u(x1)**2  * cosmo.sigma8**2 / calc_u(x2)**2

