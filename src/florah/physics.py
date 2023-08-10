
# import astropy.units as u
# import numpy as np
# import scipy.interpolate as interpolate

# try:
#     from nbodykit import cosmology
# except:
#     cosmology = None

# DEFAULT_COSMO = cosmology.Planck13
# DEFAULT_Pk = cosmology.power.LinearPower(DEFAULT_COSMO, 0, transfer='CLASS') # P(k, 0)

# def Pk(z, cosmo=DEFAULT_COSMO):
#     Pk = cosmology.power.LinearPower(cosmo, z, transfer='CLASS')
#     return Pk

# def calc_omega(z, cosmo=DEFAULT_COSMO):
#     """ Calculate omega w(z), the natural time variable in EPS. Define as

#         w(z) = delta_{c, 0} / D(z)

#     where delta_{c, 0}=1.6865 is the density hold at z=0, and D is the linear growth function
#     """
#     bg = cosmology.background.MatterDominated(
#         cosmo.Omega0_m, cosmo.Omega0_lambda, cosmo.Omega0_k)
#     a = 1 / (1 + z)
#     return 1.6865 / bg.D1(a)

# def calc_redshift(omega, cosmo=DEFAULT_COSMO):
#     """ Calculate reshift z from omega w(z) by interpolation """
#     z_arr = np.linspace(0, 1000, 10000)
#     omega_arr = calc_omega(z_arr, cosmo=cosmo)
#     return interpolate.interp1d(omega_arr, z_arr)(omega)


# def m2r(m, z, cosmo=DEFAULT_COSMO):
#     """ Convert mass to radius using the top hat kernel

#     R^3(M, z) = 3 * M / 4 pi rho_bar(z)

#     where rho_bar is the average matter density at redshift z,
#     defined as rho_bar = Omega_m(z) * rho_c(z)

#     Assume that m is in solar mass unit

#     """
#     u_density = 10**10 * (u.Msun / cosmo.h) / (u.Mpc / cosmo.h)**3
#     rho_bar = cosmo.rho_m(z) * u_density
#     r = np.power(3 * m * u.Msun / (4 * np.pi * rho_bar), 1/3)
#     r = r.to_value(u.Mpc / cosmo.h)
#     return r

# def calc_Svar(m, z, cosmo=DEFAULT_COSMO, P=None, kmin=1e-8, kmax=1e3):
#     """ Calculate the mass variance S(M) """
#     if P is None:
#         # assume P(k, 0)
#         P = cosmology.power.LinearPower(cosmo, 0, transfer='CLASS')
#     r = m2r(m, z, cosmo)
#     sigma_r = P.sigma_r(r, kmin=kmin, kmax=kmax)
#     S = sigma_r**2
#     return S
