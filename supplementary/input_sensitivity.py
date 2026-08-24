"""Input uncertainty of the lung-adapted ice-ball prediction.

Propagates the tissue-property ranges and the probe nadir temperature through
the full three-cycle solver, and ranks the inputs by partial rank correlation
with the predicted radial extent.

A direct Monte Carlo would need thousands of 26-minute chained solves, so the
design is a Latin hypercube over the seven uncertain inputs; a quadratic
response surface in the radial extent is fitted to it and sampled densely for
the predictive interval. The two predicted axes are deterministic functions of
that single output, so their predictive correlation is near one and only
marginal intervals per axis are meaningful.

Runtime roughly 40 minutes on one core; produces the values reported in the
supplementary sensitivity figure. Run from the repository root:

    python supplementary/input_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc, rankdata, truncnorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lung_chain  # noqa: E402
import dose_response as dr  # noqa: E402

L_WATER_JKG = 334e3
N_DESIGN = 220
N_SAMPLE = 40_000
SEED = 11

# name -> (distribution, mean, sd, lower, upper). Solver units are W/(mm K)
# and kg/mm^3; the table below is in the units the properties are quoted in.
INPUTS = {
    # deflated lung conductivity, range 0.28-0.478 W/m/K
    'k_def':   ('truncnorm', 0.387, 0.090, 0.28, 0.478),
    # deflated density, no spread reported: +/-10 % assumed
    'rho_def': ('truncnorm', 1050.0, 105.0, 800.0, 1300.0),
    # deflated water content 80.3 %, SD 2.96
    'lwd_def': ('truncnorm', 80.3, 2.96, 70.0, 90.0),
    # inflated conductivity, single value with range 0.161-0.198
    'k_inf':   ('truncnorm', 0.179, 0.0093, 0.161, 0.198),
    # inflated density 394, SD 159
    'rho_inf': ('truncnorm', 394.0, 159.0, 255.0, 604.0),
    # inflated water content 27.2 %, SD 4.3
    'lwd_inf': ('truncnorm', 27.2, 4.3, 15.0, 45.0),
    # probe boundary nadir
    'T_min':   ('norm', -93.0, 15.0, -np.inf, np.inf),
}
NAMES = list(INPUTS)
LABELS = {'T_min': 'Probe nadir temperature',
          'k_def': 'Deflated lung conductivity',
          'rho_def': 'Deflated lung density',
          'lwd_def': 'Deflated lung water content',
          'rho_inf': 'Inflated lung density',
          'k_inf': 'Inflated lung conductivity',
          'lwd_inf': 'Inflated lung water content'}


def _ppf(name, u):
    kind, mu, sd, lo, hi = INPUTS[name]
    if kind == 'norm':
        return norm.ppf(u, loc=mu, scale=sd)
    a, b = (lo - mu) / sd, (hi - mu) / sd
    return truncnorm.ppf(u, a, b, loc=mu, scale=sd)


def solver_props(draw):
    """Convert a draw in quoted units to the solver's units."""
    return dict(
        k_def=draw['k_def'] * 1e-3,
        rho_def=draw['rho_def'] * 1e-9,
        L_def=(draw['lwd_def'] / 100.0) * L_WATER_JKG,
        k_inf=draw['k_inf'] * 1e-3,
        rho_inf=draw['rho_inf'] * 1e-9,
        L_inf=(draw['lwd_inf'] / 100.0) * L_WATER_JKG)


def run_design(n=N_DESIGN, seed=SEED):
    rng = np.random.default_rng(seed)
    U = qmc.LatinHypercube(d=len(NAMES), seed=seed).random(n)
    reps = rng.choice(['N1', 'N2', 'N3'], size=n)
    out = []
    for i in range(n):
        draw = {nm: float(_ppf(nm, U[i, j])) for j, nm in enumerate(NAMES)}
        try:
            res = lung_chain.chain_protocol(reps[i], 'VC3', draw['T_min'],
                                            props=solver_props(draw))
        except Exception as exc:
            print(f'  design {i} failed: {exc}')
            continue
        if res is None or not np.isfinite(res['rt']):
            continue
        out.append({**draw, 'rep': reps[i], 'r_tip': res['rt'],
                    'long_mm': res['length'], 'short_mm': res['width']})
        if (i + 1) % 25 == 0:
            print(f'  design {i + 1}/{n}', flush=True)
    return pd.DataFrame(out)


def prcc(df, col, target='r_tip'):
    """Partial rank correlation of one input with the target, controlling for
    the others."""
    others = [c for c in NAMES if c != col]
    Z = np.column_stack([np.ones(len(df))] + [rankdata(df[c].values) for c in others])
    x, y = rankdata(df[col].values), rankdata(df[target].values)
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def response_surface(df):
    """Quadratic surface in the standardised inputs, fitted to the radial extent."""
    X = df[NAMES].values
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    design = [np.ones(len(Z))] + [Z[:, i] for i in range(Z.shape[1])]
    design += [Z[:, i] * Z[:, j] for i in range(Z.shape[1])
               for j in range(i, Z.shape[1])]
    A = np.column_stack(design)
    beta, *_ = np.linalg.lstsq(A, df['r_tip'].values, rcond=None)
    fit = A @ beta
    resid = df['r_tip'].values - fit
    r2 = 1 - resid.var() / df['r_tip'].values.var()
    return (mu, sd, beta), float(r2), float(resid.std(ddof=1))


def evaluate_surface(surface, Z):
    mu, sd, beta = surface
    design = [np.ones(len(Z))] + [Z[:, i] for i in range(Z.shape[1])]
    design += [Z[:, i] * Z[:, j] for i in range(Z.shape[1])
               for j in range(i, Z.shape[1])]
    return np.column_stack(design) @ beta


def main():
    print(f'Latin hypercube, {N_DESIGN} chained three-cycle solves')
    design = run_design()
    print(f'{len(design)} completed\n')

    ranking = pd.DataFrame([{'input': LABELS[c], 'prcc': prcc(design, c)}
                            for c in NAMES])
    ranking['abs_prcc'] = ranking['prcc'].abs()
    print('Partial rank correlation with the predicted radial extent')
    print(ranking.sort_values('abs_prcc', ascending=False)[['input', 'prcc']]
          .to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    surface, r2, resid_sd = response_surface(design)
    print(f'\nResponse surface R2 = {r2:.4f}, residual SD = {resid_sd:.3f} mm')

    rng = np.random.default_rng(SEED)
    U = rng.random((N_SAMPLE, len(NAMES)))
    X = np.column_stack([_ppf(nm, U[:, j]) for j, nm in enumerate(NAMES)])
    mu, sd, _ = surface
    r_tip = evaluate_surface(surface, (X - mu) / sd)
    r_tip += rng.normal(0.0, resid_sd, size=len(r_tip))

    axes = np.array([[g['length'], g['width_max']] for g in
                     (dr.tip_to_ellipsoid(r) for r in r_tip)])
    print('\nPredictive distribution')
    for i, axis in enumerate(('long', 'short')):
        lo, hi = np.percentile(axes[:, i], [2.5, 97.5])
        print(f'  {axis:5s} axis  mean {axes[:, i].mean():6.2f} mm   '
              f'95% interval {lo:.2f} - {hi:.2f} mm')
    print(f'  predicted axis correlation {np.corrcoef(axes.T)[0, 1]:.4f} '
          '(both axes follow from one solver output)')


if __name__ == '__main__':
    main()
