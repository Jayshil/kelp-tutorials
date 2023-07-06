import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

from batman import TransitModel

import astropy.units as u
from astropy.constants import R_jup, R_sun
from astropy.stats import sigma_clip, mad_std

import arviz

import numpyro
# Set the number of cores on your machine for parallelism:
cpu_cores = 1
numpyro.set_host_device_count(cpu_cores)

from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
from jax import numpy as jnp
from jax.random import PRNGKey, split

from kelp import Planet
from kelp.jax import reflected_phase_curve_inhomogeneous
import pickle
import corner

# Fitting data to inhomogeneous pc

## Reading the data
binphase, binflux, binerror = np.loadtxt('Data/binned_data.dat', usecols=(0,1,2), unpack=True)

## Planetary parameters
t0 = 2454967.27687  # Esteves et al. 2015
period = 4.8854892  # Esteves et al. 2015
T_s = 5933  # NASA Exoplanet Archive
rp = 1.622 * R_jup  # Esteves et al. 2015
rstar = 1.966 * R_sun  # ±0.013 (NASA Exoplanet Archive)
duration = 5.1313 / 24  # Morton et al. 2016
a = 0.06067 * u.AU  # Esteves et al. 2015
b = 0.5599  # Esteves et al. 2015 +0.0045-0.0046
rho_star = 0.238 * u.g / u.cm ** 3  # Southworth et al. 2012 ±0.010
a_rs = float(a / rstar)
a_rp = float(a / rp)
rp_rstar = float(rp / rstar)
eclipse_half_dur = duration / period / 2

planet = Planet(
    per=period,
    t0=t0,
    inc=np.degrees(np.arccos(b/a_rs)),
    rp=rp_rstar,
    ecc=0,
    w=90,
    a=a_rs,
    u=[0, 0],
    fp=1e-6,
    t_secondary=t0 + period/2,
    T_s=T_s,
    rp_a=rp_rstar/a_rs,
    name='Kepler-7 b'
)

# compute a static eclipse model:
bintime = binphase * period + t0
eclipse_kepler = TransitModel(
    planet, bintime,
    transittype='secondary',
    supersample_factor=100,
    exp_time=bintime[1] - bintime[0]
).light_curve(planet)

# renormalize to ppm:
eclipse_kepler = 1e6 * (eclipse_kepler - 1)

def model():
    # Define reflected light phase curve model according to
    # Heng, Morris & Kitzmann ("HMK", 2021)

    # We reparameterize the omega_0 and omega_prime with the
    # following parameters with uniform priors and limits from [0, 1]:
    omega_a = numpyro.sample('omega_a', dist.Uniform(low=0, high=1))
    omega_b = numpyro.sample('omega_b', dist.Uniform(low=0, high=1))

    # and we derive the "native" parameters for the HMK model from these
    # re-cast parameters:
    omega_0 = numpyro.deterministic('omega_0', jnp.sqrt(omega_a) * omega_b)
    omega_prime = numpyro.deterministic('omega_prime', jnp.sqrt(omega_a) * (1 - omega_b))

    # We sample for the start/stop longitudes of the dark central region:
    x1 = numpyro.sample('x1', dist.Uniform(low=-np.pi/2, high=0.4))  # [rad]
    x2 = numpyro.sample('x2', dist.Uniform(low=0.4, high=np.pi/2))  # [rad]

    # Sample for the geometric albedo:
    A_g = numpyro.sample('A_g', dist.Uniform(low=0, high=1))

    # construct an inhomogeneous reflected light phase curve model
    flux_ratio_ppm, g, q = reflected_phase_curve_inhomogeneous(
        binphase, omega_0, omega_prime, x1, x2, A_g, a_rp
    )

    offset = numpyro.sample('flux_offset', dist.Uniform(low=-20, high=20))
    # Construct a composite phase curve model
    flux_model = eclipse_kepler * flux_ratio_ppm + offset

    # Keep track of the q and g values at each step in the chains
    numpyro.deterministic('q', q)
    numpyro.deterministic('g', g)

    # Construct our likelihood
    numpyro.sample('obs',
        dist.Normal(
            loc=flux_model, scale=binerror
        ), obs=binflux
    )

# Random numbers in jax are generated like this:
rng_seed = 42
rng_keys = split(
    PRNGKey(rng_seed),
    cpu_cores
)

# Define a sampler, using here the No U-Turn Sampler (NUTS)
# with a dense mass matrix:
sampler = NUTS(
    model,
    dense_mass=True
)

# Monte Carlo sampling for a number of steps and parallel chains:
mcmc = MCMC(
    sampler,
    num_warmup=1_000,
    num_samples=5_000,
    num_chains=cpu_cores
)

# Run the MCMC
mcmc.run(rng_keys)

# arviz converts a numpyro MCMC object to an `InferenceData` object based on xarray:
result = arviz.from_numpyro(mcmc)

plt.figure()
plt.errorbar(binphase, binflux, binerror, fmt='o', color='k', ecolor='silver')

n_models_to_plot = 50
keys = ['omega_0', 'omega_prime', 'x1', 'x2', 'A_g', 'flux_offset']

for i in range(n_models_to_plot):
    sample_index = (
        np.random.randint(0, high=mcmc.num_chains),
        np.random.randint(0, high=mcmc.num_samples)
    )
    omega_0, omega_prime, x1, x2, A_g, offset = np.array([
        result.posterior[k][sample_index][0] for k in keys
    ])
    flux_ratio_ppm, g, q = reflected_phase_curve_inhomogeneous(
        binphase, omega_0, omega_prime, x1, x2, A_g, a_rp
    )
    flux_model = flux_ratio_ppm * eclipse_kepler + offset
    plt.plot(binphase, flux_model, alpha=0.1, color='DodgerBlue')
plt.gca().set(
    xlabel='Phase',
    ylabel='$F_p/F_\mathrm{star}$ [ppm]',
    title='Kepler-7 b'
)

plt.savefig('Data/fig1.png', dpi=500)

# make a corner plot
corner.corner(
    result,
    quiet=True,
);
plt.savefig('Data/corner.png', dpi=500)

# Dumping a pickle
pickle.dump(result,open('Data/posteriors.pkl','wb'))