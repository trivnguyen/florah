
import numpy as np
import astropy
import astropy.units as u

import nbodykit
from nbodykit import cosmology


DEFAULT_COSMO = cosmology.Planck13
DEFAULT_Pk = cosmology.power.LinearPower(DEFAULT_COSMO, 0, transfer='CLASS') # P(k, 0)

def Pk(z, cosmo=DEFAULT_COSMO):
    Pk = cosmology.power.LinearPower(cosmo, z, transfer='CLASS')
    return Pk


def calc_omega(z, cosmo=DEFAULT_COSMO):
    """ Calculate omega w(z), the natural time variable in EPS. Define as

        w(z) = delta_{c, 0} / D(z)

    where delta_{c, 0}=1.6865 is the density hold at z=0, and D is the linear growth function
    """
    bg = cosmology.background.MatterDominated(
        cosmo.Omega0_m, cosmo.Omega0_lambda, cosmo.Omega0_k)
    a = 1 / (1 + z)
    return 1.6865 / bg.D1(a)

def m2r(m, z, cosmo=DEFAULT_COSMO):
    """ Convert mass to radius using the top hat kernel

    R^3(M, z) = 3 * M / 4 pi rho_bar(z)

    where rho_bar is the average matter density at redshift z,
    defined as rho_bar = Omega_m(z) * rho_c(z)

    Assume that m is in solar mass unit

    """
    u_density = 10**10 * (u.Msun / cosmo.h) / (u.Mpc / cosmo.h)**3
    rho_bar = cosmo.rho_m(z) * u_density
    r = np.power(3 * m * u.Msun / (4 * np.pi * rho_bar), 1/3)
    r = r.to_value(u.Mpc / cosmo.h)
    return r

def calc_Svar(m, z, cosmo=DEFAULT_COSMO, P=None, kmin=1e-8, kmax=1e3):
    """ Calculate the mass variance S(M) """
    if P is None:
        # assume P(k, 0)
        P = cosmology.power.LinearPower(cosmo, 0, transfer='CLASS')
    r = m2r(m, z, cosmo)
    sigma_r = P.sigma_r(r, kmin=kmin, kmax=kmax)
    S = sigma_r**2
    return S


### old code
# Set default cosmology
# def default_cosmo():
#     """ Return default cosmology  Planck 2013 """
#     cosmo = Planck13
#     cosmo.sigma8 = 0.82
#     cosmo.ns = 0.96
#     cosmo.Lambda = cosmo.sigma8 * cosmo.Om0   # shape param Lambda = Omega_m * sigma_8
#     # shape parameter
#     cosmo.Lambda = cosmo.Om0 * cosmo.h * np.exp(
#         -cosmo.Ob0 * (1 + np.sqrt(2 * cosmo.h) / cosmo.Om0))
#     return cosmo

# # Analytical functions
# def calc_omega(z):
#     """ Compute omega(z), the natural time variable in the EPS theorem
#     Following the approximation from Neistein &  Dekel (2008):

#     omega(z) = 1.260 [1 + z + 0.09 (1+z)^{-1} + 0.24 e^{-1.16z}]

#     """
#     return 1.260 * (1 + z + 0.09 / (1 + z) + 0.24 * np.exp(-1.16 * z))

# def calc_u(x):
#     """ u(x) = 64.087 * [ 1 + 1.074 x^0.3 - 1.581 x^0.4 + 0.954 x^0.5 - 0.185 x^0.6]^{-10} """
#     return 64.087 * (1 + 1.074*x**0.3 - 1.581*x**0.4 + 0.954*x**0.5 - 0.185*x**0.6)**(-10)


# def calc_S(M, cosmo=None):
#     """ Compute S(M), the variance of the density field smoothed with
#     a spherical top-hat window function of a radius that on average encompasses a mass M in real space

#     S(M) = u^2(c0 * Lambda * M^{1/3}/ Omega_m^{1/3}) * sigma8^2 / u^2(32 * Lambda)

#     where Lambda=0.169 is power spectrum shape parameter and c0=3.804 x 10^{-4}
#     """

#     if cosmo is None:
#         cosmo = default_cosmo()
#     c0 = 3.804e-4
#     x1 = c0 * cosmo.Lambda * M**(1/3) / cosmo.Om0**(1/3)
#     x2 = 32 * cosmo.Lambda
#     return calc_u(x1)**2  * cosmo.sigma8**2 / calc_u(x2)**2

