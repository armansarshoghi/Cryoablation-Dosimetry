"""Hierarchical dose-response models relating minimum temperature to acute
cell death, and the dose metrics derived from them.

Every condition in the study shares one death metric. Per radial bin,

    n_expected = max(n_baseline, n_post_total)
    death      = (n_expected - n_live) / n_expected

so n_live is a count out of n_expected and is bounded by it in every bin. The
observation model is a binomial thinning of the expected population,

    n_live ~ BetaBinomial(n_expected, p_survive = 1 - f(x)/100, phi)

with a three-parameter logistic mean and Gaussian replicate effects. Three
observation models are provided: 'gaussian' (per-bin death percentage, Normal
likelihood), 'binomial' (counts, no overdispersion) and 'betabinomial' (counts
with a concentration parameter). Only the two count models share an observation
space, so only those two can be compared by leave-one-out cross-validation; the
Gaussian is compared through posterior predictive checks instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUTS_BACKEND = None
for _name in ('nutpie', 'numpyro'):
    try:
        __import__(_name)
        NUTS_BACKEND = _name
        break
    except Exception:
        continue


# ==================================================================
#  Dose metrics
# ==================================================================
# The fitted plateau L is below 100 % in every condition, so a threshold can be
# read either relative to that plateau or on an absolute percentage scale. The
# two coincide only when L = 100. `dose_metric` makes the choice explicit.

def dose_metric(x0, k, L, q, mode: str):
    """Return the x at which the logistic reaches the q-threshold.

    mode='relative' : solve f(x) = (q/100) * L  ->  x0 + log(100/q - 1)/k
                      (q = 50 returns exactly x0, for any L)
    mode='absolute' : solve f(x) = q            ->  x0 + log(L/q - 1)/k
                      (undefined whenever L <= q)
    """
    x0 = np.asarray(x0, dtype=float)
    k = np.asarray(k, dtype=float)
    L = np.asarray(L, dtype=float)
    q = np.asarray(q, dtype=float)
    if mode == 'relative':
        with np.errstate(divide='ignore', invalid='ignore'):
            out = x0 + np.log(100.0 / q - 1.0) / k
        return np.where(q > 0, out, np.nan)
    if mode == 'absolute':
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = L / q - 1.0
            out = x0 + np.log(ratio) / k
        return np.where(ratio > 0, out, np.nan)
    raise ValueError(f"mode must be 'relative' or 'absolute', got {mode!r}")


def logistic(x, L, k, x0):
    return L / (1.0 + np.exp(k * (np.asarray(x, dtype=float) - x0)))


# ==================================================================
#  Data assembly
# ==================================================================

def build_rows(df, *, x_col, group_col, replicate_col, bin_size,
               live_col='n_live', exp_col='n_expected', groups=None):
    """Collapse to one row per (group, replicate, bin).

    Counts are summed within the bin; the death fraction is recomputed from the
    summed counts rather than averaged, which is what makes the count
    likelihood coherent.
    """
    d = df.copy()
    d['bin'] = (d[x_col] / bin_size).round() * bin_size
    g = (d.groupby([group_col, replicate_col, 'bin'], observed=True)
           .agg(n_live=(live_col, 'sum'), n_expected=(exp_col, 'sum'))
           .reset_index())
    g = g[g['n_expected'] > 0].copy()
    g['death_pct'] = 100.0 * (g['n_expected'] - g['n_live']) / g['n_expected']
    if groups is not None:
        g = g[g[group_col].isin(groups)]
    g = g.rename(columns={group_col: 'group', replicate_col: 'replicate'})
    codes = {v: i for i, v in enumerate(sorted(g['group'].unique()))}
    g['g_idx'] = g['group'].map(codes)
    rep_key = g[['group', 'replicate']].drop_duplicates().reset_index(drop=True)
    rep_key['r_idx'] = np.arange(len(rep_key))
    g = g.merge(rep_key, on=['group', 'replicate'])
    rep_key['g_idx'] = rep_key['group'].map(codes)
    return g.reset_index(drop=True), list(codes), rep_key['g_idx'].values


# ==================================================================
#  Model
# ==================================================================

def build_model(rows, n_groups, rep2group, likelihood, *, domain='temperature'):
    """Hierarchical three-parameter logistic with the chosen observation model.

    The plateau is capped at 100 %: a death percentage above 100 is not
    interpretable, and the count likelihoods require the mean to be a
    probability.
    """
    import pymc as pm

    x = rows['bin'].values.astype(float)
    g_idx = rows['g_idx'].values.astype(int)
    r_idx = rows['r_idx'].values.astype(int)
    n_rep = int(r_idx.max()) + 1

    # Distance curves are steep and positive, temperature curves shallow and
    # centred well below zero.
    if domain == 'temperature':
        k_sigma, x0_mu, x0_sigma, tau_sigma = 1.0, float(np.median(x)), 20.0, 5.0
    else:
        k_sigma, x0_mu, x0_sigma, tau_sigma = 2.0, float(np.median(x)), 10.0, 2.0

    with pm.Model() as model:
        L = pm.TruncatedNormal('L', mu=90.0, sigma=15.0, lower=50.0, upper=100.0,
                               shape=n_groups)
        k = pm.HalfNormal('k', sigma=k_sigma, shape=n_groups)
        x0 = pm.Normal('x0', mu=x0_mu, sigma=x0_sigma, shape=n_groups)
        tau = pm.HalfNormal('tau', sigma=tau_sigma, shape=n_groups)
        z = pm.Normal('z', mu=0.0, sigma=1.0, shape=n_rep)
        x0_r = pm.Deterministic('x0_r', x0[rep2group] + tau[rep2group] * z)

        death_pct = L[g_idx] / (1.0 + pm.math.exp(k[g_idx] * (x - x0_r[r_idx])))

        if likelihood == 'gaussian':
            sigma_y = pm.HalfNormal('sigma_y', sigma=10.0)
            pm.Normal('obs', mu=death_pct, sigma=sigma_y,
                      observed=rows['death_pct'].values)
        else:
            p_surv = pm.math.clip(1.0 - death_pct / 100.0, 1e-6, 1.0 - 1e-6)
            n_exp = rows['n_expected'].values.astype(int)
            n_live = rows['n_live'].values.astype(int)
            if likelihood == 'binomial':
                pm.Binomial('obs', n=n_exp, p=p_surv, observed=n_live)
            elif likelihood == 'betabinomial':
                # phi is the beta-binomial concentration: large phi -> binomial.
                phi = pm.Gamma('phi', alpha=2.0, beta=0.05)
                pm.BetaBinomial('obs', n=n_exp, alpha=p_surv * phi,
                                beta=(1.0 - p_surv) * phi, observed=n_live)
            else:
                raise ValueError(likelihood)
    return model


def sample(model, *, seed=2026, draws=1500, tune=1500, chains=4,
           target_accept=0.95):
    """Sample, then attach the pointwise log-likelihood.

    nutpie ignores pymc's `idata_kwargs`, so requesting the log-likelihood at
    sample time is silently dropped and LOO becomes impossible. Computing it
    afterwards works with any backend.
    """
    import pymc as pm
    kw = dict(draws=draws, tune=tune, chains=chains, cores=1,
              target_accept=target_accept, random_seed=seed, progressbar=False)
    if NUTS_BACKEND:
        kw['nuts_sampler'] = NUTS_BACKEND
    with model:
        idata = pm.sample(**kw)
        if 'log_likelihood' not in idata.groups():
            pm.compute_log_likelihood(idata, progressbar=False)
    return idata


# ==================================================================
#  Diagnostics and derived quantities
# ==================================================================

def diagnostics(idata, var_names=('L', 'k', 'x0', 'tau')) -> pd.DataFrame:
    """Per-parameter mean, 95 % HDI, R-hat, bulk and tail ESS, and MCSE."""
    import arviz as az
    names = [v for v in var_names if v in idata.posterior]
    for extra in ('sigma_y', 'phi'):
        if extra in idata.posterior:
            names.append(extra)
    s = az.summary(idata, var_names=names, hdi_prob=0.95, round_to=None,
                   stat_focus='mean')
    return s.reset_index().rename(columns={'index': 'parameter'})


def posterior_dose(idata, group_index, q, mode) -> np.ndarray:
    """Posterior draws of the dose metric, drawing (x0, k, L) jointly."""
    post = idata.posterior
    x0 = post['x0'].stack(s=('chain', 'draw')).values[group_index]
    k = post['k'].stack(s=('chain', 'draw')).values[group_index]
    L = post['L'].stack(s=('chain', 'draw')).values[group_index]
    return np.asarray(dose_metric(x0, k, L, q, mode), dtype=float)


def hdi_of(draws, prob=0.95):
    import arviz as az
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return (np.nan, np.nan)
    lo, hi = az.hdi(d, hdi_prob=prob)
    return float(lo), float(hi)


# ==================================================================
#  Confocal prolate-ellipsoid geometry
# ==================================================================

L_ACTIVE = 22.0           # mm, IceSphere 1.5 CX active freezing length


def tip_to_ellipsoid(r_tip, L_active: float = L_ACTIVE) -> dict:
    """Confocal prolate ellipsoid through (z = c, r = r_tip), c = L_active/2.

    Solves a^2 - a*r_tip - c^2 = 0 for the semi-major axis, then b^2 = a^2 - c^2.
    """
    r_tip = float(r_tip)
    c = float(L_active) / 2.0
    if not np.isfinite(r_tip) or r_tip <= 0:
        return dict(r_tip=r_tip, length=np.nan, width_max=np.nan, volume=np.nan)
    a = 0.5 * (r_tip + np.sqrt(r_tip * r_tip + 4.0 * c * c))
    b = np.sqrt(a * a - c * c)
    return dict(r_tip=r_tip, length=2.0 * a, width_max=2.0 * b,
                volume=(4.0 / 3.0) * np.pi * a * b * b)


def find_isotherm_dist(tmin_curve, dist_grid, target) -> float:
    """Largest distance at which T_min(d) <= target, linearly interpolated."""
    tmin_curve = np.asarray(tmin_curve, dtype=float)
    dist_grid = np.asarray(dist_grid, dtype=float)
    below = np.where(tmin_curve <= target)[0]
    if len(below) == 0:
        return np.nan
    i = below[-1]
    if i + 1 >= len(dist_grid):
        return float(dist_grid[i])
    d1, d2 = dist_grid[i], dist_grid[i + 1]
    T1, T2 = tmin_curve[i], tmin_curve[i + 1]
    if T2 == T1:
        return float(d1)
    return float(d1 + (target - T1) / (T2 - T1) * (d2 - d1))


def stefan_number(T_probe, T_medium, T_fusion, c_p_solid, c_p_liquid, L_f) -> float:
    """St* = c_p,s (T_F - T_P) / [L_f + c_p,l (T_M - T_F)].

    Temperatures in degC, heat capacities in J/(kg K), latent heat in J/kg.
    """
    return (c_p_solid * (T_fusion - T_probe)) / (L_f + c_p_liquid * (T_medium - T_fusion))
