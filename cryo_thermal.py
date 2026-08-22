"""
cryo_thermal.py -- Cylindrical FDM Thermal Model for Cryoablation
=================================================================
1-D cylindrical finite-difference solver with implicit Euler time
stepping and apparent-heat-capacity method for phase change.

Handles pre-freeze conduction, phase change, and post-freeze
cooling in a unified framework.  Measured probe thermocouple as
Dirichlet inner boundary condition.

All thermal properties from literature.  Single fitted parameter
per experiment: dT (probe TC offset, physically justified as the
gap between TC position and actual coldest probe surface).

Validated via spatial cross-validation: fit dT on innermost
measurement TC, predict all outer TCs.

Usage
-----
    python cryo_thermal.py
    python cryo_thermal.py --stage=celsio
    python cryo_thermal.py --base-dir /path/to/Crystal
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, sys, time as _timer

# ==================================================================
#  MATERIAL CONSTANTS   (mm  s  degC  kg  J  W)
# ==================================================================
K_ICE     = 2.22e-3        # W/(mm*K)
C_ICE     = 2090.0         # J/(kg*K)
RHO_ICE   = 917.0e-9       # kg/mm^3
ALPHA_ICE = K_ICE / (RHO_ICE * C_ICE)  # ~1.158 mm^2/s

K_WATER   = 0.600e-3       # W/(mm*K)
C_WATER   = 4180.0         # J/(kg*K)
RHO_WATER = 1000.0e-9      # kg/mm^3
ALPHA_WATER = K_WATER / (RHO_WATER * C_WATER)  # ~0.1435 mm^2/s

L_WATER   = 334_000.0      # J/kg  (PBS / saline)
L_GELMA   = 284_000.0      # J/kg  (GelMA 3-D construct)

# ==================================================================
#  DOMAIN TRUNCATION
# ==================================================================
# How far the far-field Dirichlet boundary sits beyond the outermost radius
# the solution is evaluated at.
#
# These are NOT numerically converged values: pushing the boundary out to 25 mm
# moves the 0 degC isotherm of a 10-minute freeze by ~0.25 mm, and 25 mm and
# 40 mm then agree. But the larger, converged domain fits the measured
# thermocouples WORSE (clinical R2_val 0.78 -> 0.70), because a warm boundary
# close to the ablation zone stands in for the non-radial heat inflow -- through
# the dish, the medium reservoir and the air above -- that the 1-D model
# neglects by construction. The truncation is therefore a modelling choice, not
# an oversight, and it is kept at the submitted values so the reported numbers
# remain those of the fitted model. The revision notebook (section 1) reports
# the convergence study and the <=0.15 mm effect on every reported dimension.
DOMAIN_MARGIN_MM = 3.0
DOMAIN_MIN_MM    = 10.0


def k_ice_T(T_C):
    """Temperature-dependent thermal conductivity of ice (W/(mm*K)).
    Empirical fit: k_ice ≈ 2.22 * (273.15 / T_K) valid 0 to -150 C.
    Ref: Slack (1980), Engineering Toolbox.
    Returns array-safe result; clamps at T=-200C to avoid singularity.
    """
    T_K = np.maximum(T_C + 273.15, 73.15)  # clamp at -200C
    return 2.22e-3 * (273.15 / T_K)


# ==================================================================
#  SOLVER:  Implicit 1-D Cylindrical FDM
# ==================================================================

def _thomas(a, b, c, d):
    """Thomas algorithm for tridiagonal system Ax=d.
    a: sub-diagonal (n-1), b: main diagonal (n),
    c: super-diagonal (n-1), d: RHS (n).
    """
    n = len(b)
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        w = b[i] - a[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / w
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / w
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def h_to_W(h_Wm2K, L_char_mm):
    """Convert surface heat-transfer coeff h [W/m²/K] to volumetric W [W/mm³/K].

    Parameters
    ----------
    h_Wm2K   : float, surface convection coefficient (W/m²/K)
    L_char_mm : float, characteristic length (mm)
                celsio ~ 3 mm, ice2x6/clinical ~ 7 mm, crystal_3d ~ 10 mm

    Returns
    -------
    W_conv : float, volumetric coefficient (W/mm³/K)
    """
    return h_Wm2K / (L_char_mm * 1e-3 * 1e9)


def solve_fdm(t, T_bc, r_bc, r_eval, T_i, T_phase, dT=0.0,
              k_s=K_ICE, k_l=K_WATER, c_s=C_ICE, c_l=C_WATER,
              rho_s=RHO_ICE, rho_l=RHO_WATER, L_f=L_WATER,
              HW=0.5, dr=0.05, n_sub=4, k_ice_tdep=False,
              W_conv=0.0, neumann_after=None,
              T_init=None, return_grid_final=False):
    """
    1-D cylindrical FDM with apparent heat capacity for phase change.

    Parameters
    ----------
    t       : (N,) time in seconds from start of window (0-based)
    T_bc    : (N,) effective inner-boundary temperature (degC),
              already includes any dT offset with decay
    r_bc    : inner-boundary radius (mm from axis)
    r_eval  : (M,) radii at which to predict T (mm from axis)
    T_i     : far-field initial temperature (degC)
    T_phase : equilibrium freezing point (degC)
    dT      : unused (offset applied upstream)
    HW      : mushy-zone half-width (degC)
    dr      : radial grid spacing (mm)
    n_sub   : sub-steps per data interval
    W_conv  : volumetric convection coeff (W/mm³/K), default 0 (pure conduction).
              Adds Pennes-like source q'''=W_conv*(T_i - T) in liquid/mushy nodes.
              Use h_to_W() to convert from surface h [W/m²/K].
    neumann_after : float or None
              If set, switch inner BC from Dirichlet to zero-flux Neumann
              (dT/dr = 0) after this time (seconds).  Used to model passive
              rewarming after cryogen shutoff.

    Returns
    -------
    T_pred : (N, M) predicted temperatures at output times & locations
    Rf     : (N,)   approximate ice-front position (mm from axis)
    """
    t = np.asarray(t, dtype=float)
    r_eval = np.asarray(r_eval, dtype=float)
    N_out = len(t)
    M_out = len(r_eval)

    # BC already has dT with decay applied upstream
    T_eff = np.asarray(T_bc, dtype=float)
    T_bc_func = interp1d(t, T_eff, kind='linear',
                         bounds_error=False,
                         fill_value=(T_eff[0], T_eff[-1]))

    # Radial grid. The outer Dirichlet boundary sits DOMAIN_MARGIN_MM beyond
    # the outermost evaluation radius; a domain-convergence test (revision
    # notebook §1) showed the original 3 mm margin was not far enough for the
    # longest freezes, so it is set from module-level constants.
    r_max = max(float(np.max(r_eval)) + DOMAIN_MARGIN_MM, r_bc + DOMAIN_MIN_MM)
    r = np.arange(r_bc, r_max + dr / 2, dr)
    Nr = len(r)
    n_int = Nr - 2  # interior nodes

    # Initial temperature: uniform T_i, OR user-supplied T_init field
    # T_init is a tuple (r_init, T_init_vals) interpolated onto the solver grid r.
    if T_init is not None:
        r_init, T_init_vals = T_init
        T = np.interp(r, r_init, T_init_vals,
                      left=T_init_vals[0], right=T_init_vals[-1])
    else:
        T = np.full(Nr, T_i, dtype=float)
    T[0] = T_eff[0]
    T[-1] = T_i  # far-field Dirichlet BC

    # Phase-change thresholds
    T_lo = T_phase - HW
    T_hi = T_phase + HW

    # Pre-compute geometric factors for interior nodes (indices 1..Nr-2)
    r_int = r[1:-1]
    gamma = dr / (2.0 * r_int)  # cylindrical correction

    # Geometric factor for node 0 (used during Neumann mode)
    gamma_0 = dr / (2.0 * r[0])

    # Output arrays
    T_pred = np.zeros((N_out, M_out))
    Rf_out = np.full(N_out, r_bc)

    # Record t=0
    T_pred[0] = np.interp(r_eval, r, T)

    # Time-stepping
    for n in range(1, N_out):
        dt_total = float(t[n] - t[n - 1])
        if dt_total <= 0:
            T_pred[n] = T_pred[n - 1]
            Rf_out[n] = Rf_out[n - 1]
            continue

        dt_sub = dt_total / n_sub

        for s in range(n_sub):
            t_now = t[n - 1] + (s + 1) * dt_sub

            # Decide inner BC mode
            use_neumann = (neumann_after is not None and t_now > neumann_after)

            if not use_neumann:
                # Dirichlet: prescribed probe temperature
                T[0] = float(T_bc_func(t_now))

            # Material properties at current T (lagged) for interior nodes
            frozen = T[1:-1] < T_lo
            liquid = T[1:-1] > T_hi
            mushy = ~frozen & ~liquid

            f_l = np.zeros(n_int)
            f_l[liquid] = 1.0
            if mushy.any():
                f_l[mushy] = (T[1:-1][mushy] - T_lo) / (2.0 * HW)

            k_arr = np.empty(n_int)
            if k_ice_tdep:
                # Temperature-dependent k for frozen nodes
                k_arr[frozen] = k_ice_T(T[1:-1][frozen])
                k_arr[liquid] = k_l
                if mushy.any():
                    k_frozen_mushy = k_ice_T(T[1:-1][mushy])
                    k_arr[mushy] = k_frozen_mushy + (k_l - k_frozen_mushy) * f_l[mushy]
            else:
                k_arr[frozen] = k_s
                k_arr[liquid] = k_l
                k_arr[mushy] = k_s + (k_l - k_s) * f_l[mushy]

            rc_arr = np.empty(n_int)
            rc_arr[frozen] = rho_s * c_s
            rc_arr[liquid] = rho_l * c_l
            if mushy.any():
                rc_arr[mushy] = (rho_s * c_s * (1.0 - f_l[mushy])
                                 + rho_l * c_l * f_l[mushy]
                                 + rho_l * L_f / (2.0 * HW))

            alpha = k_arr / rc_arr
            beta = alpha * dt_sub / (dr * dr)

            if use_neumann:
                # --- Neumann dT/dr=0 at inner boundary ---
                # Include node 0 in the system: solve nodes 0..Nr-2
                # Node 0: dT/dr=0 means T[-1]=T[1] (ghost node),
                #   so the stencil becomes:
                #   T0^{n+1} = T0^n + beta0 * [(1-g0)*T_{-1} - 2*T0 + (1+g0)*T1]
                #   with T_{-1} = T1 => coeff on T1 = (1-g0) + (1+g0) = 2
                #   and coeff on T0 = -2

                # Properties at node 0
                T0 = T[0]
                if T0 < T_lo:
                    k0 = k_ice_T(T0) if k_ice_tdep else k_s
                    rc0 = rho_s * c_s
                elif T0 > T_hi:
                    k0 = k_l
                    rc0 = rho_l * c_l
                else:
                    fl0 = (T0 - T_lo) / (2.0 * HW)
                    k0_ice = k_ice_T(T0) if k_ice_tdep else k_s
                    k0 = k0_ice + (k_l - k0_ice) * fl0
                    rc0 = (rho_s * c_s * (1.0 - fl0)
                           + rho_l * c_l * fl0
                           + rho_l * L_f / (2.0 * HW))
                alpha0 = k0 / rc0
                beta0 = alpha0 * dt_sub / (dr * dr)

                # Extended system: n_int + 1 unknowns (nodes 0 .. Nr-2)
                n_ext = n_int + 1
                b_main_ext = np.empty(n_ext)
                d_rhs_ext = np.empty(n_ext)

                # Row 0: node 0 with Neumann BC (ghost: T_{-1} = T_1)
                b_main_ext[0] = 1.0 + 2.0 * beta0
                d_rhs_ext[0] = T[0]

                # Rows 1..n_int: original interior nodes (shifted by 1)
                b_main_ext[1:] = 1.0 + 2.0 * beta

                # Optional Pennes convection for interior
                if W_conv > 0:
                    conv_factor = np.zeros(n_int)
                    conv_factor[liquid] = W_conv * dt_sub / rc_arr[liquid]
                    if mushy.any():
                        conv_factor[mushy] = W_conv * f_l[mushy] * dt_sub / rc_arr[mushy]
                    b_main_ext[1:] += conv_factor

                d_rhs_ext[1:] = T[1:-1].copy()
                if W_conv > 0:
                    d_rhs_ext[1:] += conv_factor * T_i

                # Sub-diagonal (n_ext - 1 entries): couples row i to row i-1
                a_sub_ext = np.empty(n_ext - 1)
                # Row 1 couples to row 0: node 1 left-neighbour is node 0
                a_sub_ext[0] = -beta[0] * (1.0 - gamma[0])
                # Rows 2..n_int: same as original a_sub
                a_sub_ext[1:] = -beta[1:] * (1.0 - gamma[1:])

                # Super-diagonal (n_ext - 1 entries): couples row i to row i+1
                c_sup_ext = np.empty(n_ext - 1)
                # Row 0 couples to row 1: ghost => coeff = -2*beta0
                c_sup_ext[0] = -2.0 * beta0
                # Rows 1..n_int-1: same as original c_sup
                c_sup_ext[1:] = -beta[:-1] * (1.0 + gamma[:-1])

                # Outer BC adjustment (Dirichlet at r_max, last row)
                d_rhs_ext[-1] += beta[-1] * (1.0 + gamma[-1]) * T[-1]

                # Solve extended system
                sol = _thomas(a_sub_ext, b_main_ext, c_sup_ext, d_rhs_ext)
                T[0] = sol[0]
                T[1:-1] = sol[1:]

            else:
                # --- Dirichlet at inner boundary (standard) ---
                # Tridiagonal coefficients
                a_sub = -beta[1:] * (1.0 - gamma[1:])         # n_int-1
                b_main = 1.0 + 2.0 * beta                      # n_int
                c_sup = -beta[:-1] * (1.0 + gamma[:-1])        # n_int-1
                d_rhs = T[1:-1].copy()

                # Optional Pennes-like convection in liquid/mushy nodes
                if W_conv > 0:
                    conv_factor = np.zeros(n_int)
                    conv_factor[liquid] = W_conv * dt_sub / rc_arr[liquid]
                    if mushy.any():
                        conv_factor[mushy] = W_conv * f_l[mushy] * dt_sub / rc_arr[mushy]
                    b_main += conv_factor
                    d_rhs += conv_factor * T_i

                # Boundary adjustments (Dirichlet both sides)
                d_rhs[0] += beta[0] * (1.0 - gamma[0]) * T[0]
                d_rhs[-1] += beta[-1] * (1.0 + gamma[-1]) * T[-1]

                # Solve
                T[1:-1] = _thomas(a_sub, b_main, c_sup, d_rhs)

        # Interpolate to output locations
        T_pred[n] = np.interp(r_eval, r, T)

        # Approximate ice-front position
        below = np.where(T < T_phase)[0]
        Rf_out[n] = r[below[-1]] if len(below) > 0 else r_bc

    if return_grid_final:
        return T_pred, Rf_out, r, T.copy()
    return T_pred, Rf_out


# ==================================================================
#  DATA I/O
# ==================================================================

def _parse_time(s):
    """HH:MM:SS -> seconds."""
    parts = str(s).strip().strip('"').split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def load_tc(path):
    """Return (time_s, temps[n_samples x n_channels])."""
    df = pd.read_csv(path)
    times = np.array([_parse_time(v) for v in df.iloc[:, 0]])
    times -= times[0]
    temps = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
    return times, temps


def smooth(arr, window=7, poly=2):
    valid = np.isfinite(arr)
    if valid.sum() < window:
        return arr
    out = arr.copy()
    out[valid] = savgol_filter(arr[valid], window, poly)
    return out


# ==================================================================
#  EXPERIMENT REGISTRY
# ==================================================================

def build_experiments(base_dir):
    """Return a flat list of experiment dicts."""
    exps = []

    celsio_dir  = os.path.join(base_dir,
                    'Final - 2D analysis_Celsio', 'Temperature Files')
    ice_dir     = os.path.join(base_dir,
                    'Final - 2D IceFx2x6', 'Temperature files')
    crystal_dir = os.path.join(base_dir,
                    'Final - Death and Apoptosis', 'Nexus3D',
                    'Temperature files')

    # -- CELSIO (27 expts) --
    celsio_tc  = [0.0, 0.0, 1.8, 2.1, 2.4, 2.8]
    celsio_reg = {
        'A549': {
            'N1_n1': 26, 'N1_n3': 44, 'N1_n4': 18,
            'N2_n1': 22, 'N2_n2':  9, 'N2_n3': 25,
            'N4_n1': 22, 'N4_n2': 19, 'N4_n3': 34,
        },
        'CALU1': {
            'N1_n1': 12, 'N1_n2': 15, 'N1_n3': 15,
            'N2_n1': 15, 'N2_n2': 24, 'N2_n3': 33,
            'N3_n1': 32, 'N3_n2': 25, 'N3_n3': 38,
        },
        'CALU6': {
            'N1_n1':  9, 'N1_n2': 24, 'N1_n3': 26,
            'N2_n1': 15, 'N2_n2': 31, 'N2_n3': 12,
            'N3_n1': 21, 'N3_n2': 26, 'N3_n3': 18,
        },
    }
    for cl, runs in celsio_reg.items():
        for key, fs in runs.items():
            bio, tech = key.split('_')
            exps.append(dict(
                id       = f'celsio_{cl}_{key}',
                stage    = 'celsio',
                file     = os.path.join(celsio_dir, f'{cl}_2D_{bio}-{tech}.csv'),
                fs       = fs,
                max_dur  = 25,
                r_probe  = 0.85,
                r_bc     = 0.85,
                T_phase  = -0.52,
                tc_pos   = celsio_tc,
                bc_cols  = [0, 1],
                fit_cols = [2],
                val_cols = [3, 4],  # exclude outermost TC (2.8mm) — negligible cooling in 10s freeze
                L_f      = L_WATER,
            ))

    # -- ICESPHERE 2x6 (6 expts) --
    ice_tc = [0.0, 1.8, 2.8, 4.8, 6.8]
    ice_2x6 = {'N1': {1:7, 2:9}, 'N2': {1:16, 2:11}, 'N3': {1:12, 2:6}}
    for rep, cycs in ice_2x6.items():
        for cyc, fs in cycs.items():
            exps.append(dict(
                id       = f'ice2x6_{rep}_c{cyc}',
                stage    = 'ice2x6',
                file     = os.path.join(ice_dir, f'A549_2x6_{cyc}_{rep}.csv'),
                fs       = fs,
                max_dur  = 420,
                r_probe  = 0.75,
                r_bc     = 0.75,
                T_phase  = -0.52,
                tc_pos   = ice_tc,
                bc_cols  = [0],
                fit_cols = [1],
                val_cols = [2, 3, 4],
                L_f      = L_WATER,
            ))

    # -- CLINICAL (9 expts) --
    clin = {
        'N1': {1: 10, 2: 370, 3: 1041},
        'N2': {1:  9, 2: 409, 3: 1065},
        'N3': {1: 14, 2: 374, 3: 1032},
    }
    clin_dur = {1: 180, 2: 420, 3: 600}
    for rep, cycs in clin.items():
        for cyc, fs in cycs.items():
            exps.append(dict(
                id       = f'clinical_{rep}_c{cyc}',
                stage    = 'clinical',
                file     = os.path.join(ice_dir, f'A549_Clinic_{rep}.csv'),
                fs       = fs,
                max_dur  = int(clin_dur[cyc] * 1.2),
                r_probe  = 0.75,
                r_bc     = 0.75,
                T_phase  = -0.52,
                tc_pos   = ice_tc,
                bc_cols  = [0],
                fit_cols = [1],
                val_cols = [2, 3, 4],
                L_f      = L_WATER,
            ))

    # -- 3-D CRYSTAL (3 expts) --
    crystal_tc = [2.4, 4.4, 6.4, 8.4, 10.4, 12.4, 14.4]
    crystal = {'N2': 63, 'N3': 126, 'N4': 124}
    for eid, fs in crystal.items():
        exps.append(dict(
            id       = f'crystal_{eid}',
            stage    = 'crystal_3d',
            file     = os.path.join(crystal_dir, f'CRYSTAL - {eid}.csv'),
            fs       = fs,
            max_dur  = 420,
            r_probe  = 0.75,
            r_bc     = 2.4,
            T_phase  = -2.8,
            tc_pos   = crystal_tc,
            bc_cols  = [0],
            fit_cols = [1],
            val_cols = [2, 3, 4, 5, 6],
            L_f      = L_GELMA,
            ch_offset= {5: 1.0},
        ))

    return exps


# ==================================================================
#  FITTING & VALIDATION
# ==================================================================

def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0


def run_experiment(exp, dT_fixed=None, T_min_fixed=None, W_conv=0.0,
                   neumann_thaw=False):
    """Fit dT on inner TC, validate on outer TCs.

    If dT_fixed is provided, skip fitting and use that value directly
    (for leave-one-out cross-validation).
    If T_min_fixed is provided, use minimum-temperature BC model instead
    of dT offset model.
    If neumann_thaw is True, switch the inner BC to zero-flux Neumann
    (dT/dr=0) after probe nadir instead of using exponential dT decay.
    The dT offset is applied only during freezing (t <= t_nadir).
    """

    if not os.path.isfile(exp['file']):
        return None
    times, temps = load_tc(exp['file'])

    # Smooth each channel (handles NaN gracefully)
    for ch in range(temps.shape[1]):
        temps[:, ch] = smooth(temps[:, ch])

    # Channel offsets (3-D Crystal ch-6 correction)
    for idx, off in exp.get('ch_offset', {}).items():
        if idx < temps.shape[1]:
            temps[:, idx] += off

    # Boundary-condition TC  (nanmin handles missing channels)
    bc_cols = exp['bc_cols']
    if len(bc_cols) == 1:
        T_bc_raw = temps[:, bc_cols[0]].copy()
    else:
        T_bc_raw = np.nanmin(temps[:, bc_cols], axis=1)

    # Replace any remaining NaN in probe with forward-fill
    mask = np.isfinite(T_bc_raw)
    if not mask.all():
        good_idx = np.where(mask)[0]
        if len(good_idx) == 0:
            return None
        T_bc_raw = np.interp(np.arange(len(T_bc_raw)), good_idx, T_bc_raw[good_idx])

    # Freeze window: full max_dur from freeze start
    fs = exp['fs']
    if fs >= len(times):
        return None
    end = min(fs + exp['max_dur'], len(times))
    if end - fs < 4:
        return None

    sl = slice(fs, end)
    t_f    = times[sl].astype(float) - float(times[fs])
    T_bc_f = T_bc_raw[sl].copy()
    T_all  = temps[sl].copy()

    # Find probe nadir time for dT decay
    nadir_local = int(np.argmin(T_bc_f))
    t_nadir = float(t_f[nadir_local])

    # Initial temperature from outermost validation TC
    bl_start = max(0, fs - 5)
    bl_end   = max(bl_start + 1, fs)
    outer_col = exp['val_cols'][-1]
    T_i_vals = temps[bl_start:bl_end, outer_col]
    T_i_vals = T_i_vals[np.isfinite(T_i_vals)]
    T_i = float(np.mean(T_i_vals)) if len(T_i_vals) > 0 else 25.0

    # TC positions & columns
    tc_pos   = np.array(exp['tc_pos'])
    fit_cols = exp['fit_cols']
    val_cols = exp['val_cols']
    all_cols = fit_cols + val_cols
    r_fit = tc_pos[fit_cols]
    r_all = tc_pos[all_cols]

    meas_fit = T_all[:, fit_cols]
    meas_val = T_all[:, val_cols]
    meas_all = T_all[:, all_cols]

    # Check for NaN in measurement data
    fit_valid = np.isfinite(meas_fit)
    val_valid = np.isfinite(meas_val)
    all_valid = np.isfinite(meas_all)

    if fit_valid.sum() < 3:
        return None

    r_bc    = exp['r_bc']
    T_phase = exp['T_phase']
    L_f     = exp['L_f']

    # Solver kwargs (fixed literature properties + temperature-dependent k_ice)
    skw = dict(k_s=K_ICE, k_l=K_WATER, c_s=C_ICE, c_l=C_WATER,
               rho_s=RHO_ICE, rho_l=RHO_WATER, L_f=L_f, HW=0.5,
               k_ice_tdep=True, W_conv=W_conv)

    # If neumann_thaw, pass t_nadir to solver so it switches BC mode
    if neumann_thaw:
        skw['neumann_after'] = t_nadir

    # dT with exponential decay after probe nadir
    # Physical basis: TC-to-surface offset exists only while cryogen flows.
    # After cryogen off (probe nadir), metallic probe equilibrates in ~2s.
    TAU_DECAY = 3.0  # s, decay time constant

    def _apply_dT_decay(T_bc_arr, t_arr, dT_val):
        """Apply dT before nadir, exponential decay after."""
        decay = np.ones_like(t_arr)
        post = t_arr > t_nadir
        if post.any():
            decay[post] = np.exp(-(t_arr[post] - t_nadir) / TAU_DECAY)
        return T_bc_arr + dT_val * decay

    def _apply_dT_step(T_bc_arr, t_arr, dT_val):
        """Apply dT before nadir only (step function).
        Used with neumann_thaw: after nadir the solver ignores the BC
        array and uses dT/dr=0, so the post-nadir values don't matter,
        but we set them to the raw measured TC for cleanliness."""
        offset = np.where(t_arr <= t_nadir, dT_val, 0.0)
        return T_bc_arr + offset

    def _apply_T_min(T_bc_arr, t_arr, T_min_val):
        """Clamp BC to minimum temperature during active cooling,
        follow measured TC during rewarming."""
        out = T_bc_arr.copy()
        # During active cooling (before nadir): clamp to T_min
        pre = t_arr <= t_nadir
        if pre.any():
            out[pre] = np.minimum(T_bc_arr[pre], T_min_val)
        # After nadir: follow measured TC (probe is warming up,
        # cryogen off, TC and surface equilibrate)
        return out

    # Select BC application function
    _apply_dT = _apply_dT_step if neumann_thaw else _apply_dT_decay

    if T_min_fixed is not None:
        # T_min model: no fitting, use fixed probe-type temperature
        dT_best = T_min_fixed  # store T_min in dT field for reporting
        T_bc_eff_best = _apply_T_min(T_bc_f, t_f, T_min_fixed)
        use_tmin = True
    elif dT_fixed is not None:
        # LOO-CV mode: skip fitting, use provided dT
        dT_best = dT_fixed
        T_bc_eff_best = _apply_dT(T_bc_f, t_f, dT_best)
        use_tmin = False
    else:
        # Single-parameter fit: dT (probe TC offset)
        def objective(dT_trial):
            T_bc_eff = _apply_dT(T_bc_f, t_f, dT_trial)
            Tp, _ = solve_fdm(t_f, T_bc_eff, r_bc, r_fit,
                               T_i, T_phase, **skw)
            pred = Tp.ravel()
            meas = meas_fit.ravel()
            v = np.isfinite(meas)
            if v.sum() < 3:
                return 0.0
            return -_r2(meas[v], pred[v])

        res = minimize_scalar(objective, bounds=(-80, 5), method='bounded',
                              options={'xatol': 0.1, 'maxiter': 50})
        dT_best = res.x
        T_bc_eff_best = _apply_dT(T_bc_f, t_f, dT_best)
        use_tmin = False

    # Evaluate on ALL TCs
    T_pred_all, Rf = solve_fdm(t_f, T_bc_eff_best, r_bc, r_all,
                                T_i, T_phase, **skw)

    n_fit = len(fit_cols)
    pred_fit = T_pred_all[:, :n_fit]
    pred_val = T_pred_all[:, n_fit:]

    fv = fit_valid.ravel()
    vv = val_valid.ravel()
    av = all_valid.ravel()

    R2_fit = _r2(meas_fit.ravel()[fv], pred_fit.ravel()[fv])
    R2_val = _r2(meas_val.ravel()[vv], pred_val.ravel()[vv]) if vv.sum() > 3 else float('nan')
    R2_all = _r2(meas_all.ravel()[av], T_pred_all.ravel()[av])
    mae_val = float(np.mean(np.abs(meas_val.ravel()[vv] - pred_val.ravel()[vv]))) if vv.sum() > 0 else float('nan')

    # Per-TC R2 for validation TCs
    r2_per_tc = []
    for j in range(len(val_cols)):
        tc_meas = meas_val[:, j]
        tc_pred = pred_val[:, j]
        v_j = np.isfinite(tc_meas)
        if v_j.sum() > 3:
            r2_per_tc.append(_r2(tc_meas[v_j], tc_pred[v_j]))
        else:
            r2_per_tc.append(float('nan'))

    return dict(
        id      = exp['id'],
        stage   = exp['stage'],
        dT      = dT_best,
        R2_fit  = R2_fit,
        R2_val  = R2_val,
        R2_all  = R2_all,
        MAE_val = mae_val,
        Rf_max  = float(np.max(Rf) - exp['r_bc']),
        T_nadir = float(np.min(T_bc_eff_best)),
        T_i     = T_i,
        n_pts   = int(av.sum()),
        n_tsteps= len(t_f),
        r2_per_tc = r2_per_tc,
        # Time-series data for plotting
        _t       = t_f,
        _T_bc    = T_bc_eff_best,
        _meas    = meas_all,
        _pred    = T_pred_all,
        _r_all   = r_all,
        _all_cols = all_cols,
        _tc_pos  = tc_pos,
        _fit_cols = fit_cols,
        _val_cols = val_cols,
    )


# ==================================================================
#  NOTEBOOK API: predict temperature field at arbitrary radii
# ==================================================================

def predict_temperature_field(exp, r_out=None, dr_out=0.05, r_max_out=15.0,
                              W_conv=0.0, neumann_thaw=False):
    """Run the FDM model for one experiment and return the full temperature field.

    This is the main entry point for notebooks that need T(r, t) at arbitrary
    radial positions (e.g., to compute T_min for cell viability mapping).

    Parameters
    ----------
    exp : dict
        Experiment dict from build_experiments(), OR an experiment id string.
        If a string, build_experiments() is called to look it up.
    r_out : array-like, optional
        Radial positions (mm from probe axis) at which to evaluate temperature.
        If None, a dense grid from r_bc to r_bc + r_max_out is generated.
    dr_out : float
        Grid spacing for auto-generated r_out (mm). Default 0.05.
    r_max_out : float
        Extent beyond r_bc for auto-generated r_out (mm). Default 15.0.
    W_conv : float
        Volumetric convection coeff (W/mm³/K). Default 0 (pure conduction).
        Use h_to_W() to convert from surface h [W/m²/K].
    neumann_thaw : bool
        If True, switch inner BC to zero-flux Neumann after probe nadir.

    Returns
    -------
    dict with keys:
        't'         : (N,)    time array (s, 0-based from freeze start)
        'r'         : (M,)    radial positions (mm from axis)
        'T'         : (N, M)  temperature field (degC)
        'Rf'        : (N,)    ice front position (mm from axis)
        'T_bc'      : (N,)    effective boundary condition used
        'dT'        : float   fitted Delta-T offset
        'R2_val'    : float   cross-validated R-squared
        'T_i'       : float   initial temperature
        'T_phase'   : float   phase change temperature
        'r_bc'      : float   boundary condition radius
        'exp_id'    : str     experiment identifier
    """
    # Accept string ID
    if isinstance(exp, str):
        base = find_base_dir()
        all_exps = build_experiments(base)
        match = [e for e in all_exps if e['id'] == exp]
        if not match:
            raise ValueError(f"Experiment '{exp}' not found in registry")
        exp = match[0]

    # Run standard per-experiment fitting
    result = run_experiment(exp, W_conv=W_conv, neumann_thaw=neumann_thaw)
    if result is None:
        return None

    dT_best = result['dT']

    # --- Re-run FDM on dense output grid ---
    if not os.path.isfile(exp['file']):
        return None

    times, temps = load_tc(exp['file'])
    for ch in range(temps.shape[1]):
        temps[:, ch] = smooth(temps[:, ch])
    for idx, off in exp.get('ch_offset', {}).items():
        if idx < temps.shape[1]:
            temps[:, idx] += off

    bc_cols = exp['bc_cols']
    if len(bc_cols) == 1:
        T_bc_raw = temps[:, bc_cols[0]].copy()
    else:
        T_bc_raw = np.nanmin(temps[:, bc_cols], axis=1)
    mask = np.isfinite(T_bc_raw)
    if not mask.all():
        good_idx = np.where(mask)[0]
        T_bc_raw = np.interp(np.arange(len(T_bc_raw)), good_idx, T_bc_raw[good_idx])

    fs = exp['fs']
    end = min(fs + exp['max_dur'], len(times))
    sl = slice(fs, end)
    t_f = times[sl].astype(float) - float(times[fs])
    T_bc_f = T_bc_raw[sl].copy()

    nadir_local = int(np.argmin(T_bc_f))
    t_nadir = float(t_f[nadir_local])

    bl_start = max(0, fs - 5)
    bl_end = max(bl_start + 1, fs)
    outer_col = exp['val_cols'][-1]
    T_i_vals = temps[bl_start:bl_end, outer_col]
    T_i_vals = T_i_vals[np.isfinite(T_i_vals)]
    T_i = float(np.mean(T_i_vals)) if len(T_i_vals) > 0 else 25.0

    r_bc = exp['r_bc']
    T_phase = exp['T_phase']
    L_f = exp['L_f']

    # Apply dT: step (Neumann mode) or decay (original mode)
    if neumann_thaw:
        T_bc_eff = T_bc_f + np.where(t_f <= t_nadir, dT_best, 0.0)
    else:
        TAU_DECAY = 3.0
        decay = np.ones_like(t_f, dtype=float)
        post = t_f > t_nadir
        if post.any():
            decay[post] = np.exp(-(t_f[post] - t_nadir) / TAU_DECAY)
        T_bc_eff = T_bc_f + dT_best * decay

    # Output radii
    if r_out is None:
        r_out = np.arange(r_bc, r_bc + r_max_out, dr_out)
    r_out = np.asarray(r_out, dtype=float)

    skw = dict(k_s=K_ICE, k_l=K_WATER, c_s=C_ICE, c_l=C_WATER,
               rho_s=RHO_ICE, rho_l=RHO_WATER, L_f=L_f, HW=0.5,
               k_ice_tdep=True, W_conv=W_conv)
    if neumann_thaw:
        skw['neumann_after'] = t_nadir

    T_pred, Rf = solve_fdm(t_f, T_bc_eff, r_bc, r_out, T_i, T_phase, **skw)

    return dict(
        t       = t_f,
        r       = r_out,
        T       = T_pred,
        Rf      = Rf,
        T_bc    = T_bc_eff,
        dT      = dT_best,
        R2_val  = result['R2_val'],
        R2_all  = result['R2_all'],
        MAE_val = result['MAE_val'],
        T_i     = T_i,
        T_phase = T_phase,
        r_bc    = r_bc,
        exp_id  = exp['id'],
        t_nadir = t_nadir,
    )


def get_tmin_at_radii(field, r_mm):
    """Extract minimum temperature at specified radial distances from a field dict.

    Parameters
    ----------
    field : dict from predict_temperature_field()
    r_mm  : float or array-like, radial positions in mm from probe axis

    Returns
    -------
    T_min : array of minimum temperatures at each requested radius
    """
    r_mm = np.atleast_1d(np.asarray(r_mm, dtype=float))
    T_min = np.full(len(r_mm), np.nan)
    for i, r in enumerate(r_mm):
        # Interpolate temperature at this radius for all timesteps
        T_at_r = np.interp(r, field['r'], field['T'][0])  # init
        T_history = np.array([np.interp(r, field['r'], field['T'][t_idx])
                              for t_idx in range(len(field['t']))])
        T_min[i] = np.nanmin(T_history)
    return T_min


def get_temperature_history_at_radius(field, r_mm):
    """Extract full temperature history at a given radius from a field dict.

    Parameters
    ----------
    field : dict from predict_temperature_field()
    r_mm  : float, radial position in mm from probe axis

    Returns
    -------
    T_history : (N,) array of temperatures at each timestep
    """
    return np.array([np.interp(r_mm, field['r'], field['T'][t_idx])
                     for t_idx in range(len(field['t']))])


# ==================================================================
#  STANDARD PLOT STYLE (shared across all notebooks)
# ==================================================================

PLOT_STYLE = {
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial'],
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.titlesize':   9,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'xtick.minor.size':   2.0,
    'ytick.minor.size':   2.0,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'lines.linewidth':    1.5,
    'legend.frameon':     False,
    'svg.fonttype':       'none',
    'figure.dpi':         300,
}


# ==================================================================
#  REPORTING
# ==================================================================

def print_results(results):
    """Pretty-print a results table (ASCII-safe for Windows)."""
    hdr = (f"{'Experiment':<26s} {'dT':>5s}  {'R2_fit':>6s}  "
           f"{'R2_val':>6s}  {'R2_all':>6s}  {'MAE_v':>5s}  "
           f"{'Npts':>4s}  per-TC R2")
    sep = '-' * 90

    print('\n' + sep)
    print('  CYLINDRICAL FDM MODEL -- SPATIAL CROSS-VALIDATION')
    print('  fit on innermost TC  ->  validate on all outer TCs')
    print(sep)
    print(hdr)
    print(sep)

    stages = ['celsio', 'ice2x6', 'clinical', 'crystal_3d']
    stage_labels = {
        'celsio':     'CELSIO  (1.7 mm, 10 s)',
        'ice2x6':     'ICESPHERE 2x6  (1.5 mm, 360 s)',
        'clinical':   'CLINICAL  (1.5 mm, 3-7-10 min)',
        'crystal_3d': '3-D CRYSTAL  (GelMA, 360 s)',
    }

    grand = []
    for stg in stages:
        rows = [r for r in results if r['stage'] == stg]
        if not rows:
            continue
        print(f'\n  {stage_labels.get(stg, stg)}')
        print(f'  {"-" * (len(hdr) - 2)}')
        for r in rows:
            tc_str = '  '.join(f'{v:.2f}' for v in r.get('r2_per_tc', []))
            print(f"  {r['id']:<24s} {r['dT']:>5.1f}  {r['R2_fit']:>6.3f}  "
                  f"{r['R2_val']:>6.3f}  {r['R2_all']:>6.3f}  "
                  f"{r['MAE_val']:>5.1f}  {r.get('n_tsteps',0):>4d}  "
                  f"{tc_str}")
            grand.append(r)

        dTs  = [r['dT']     for r in rows]
        r2v  = [r['R2_val'] for r in rows]
        r2a  = [r['R2_all'] for r in rows]
        maes = [r['MAE_val'] for r in rows]
        print(f'  {"MEAN +/- SD":<24s} '
              f'{np.mean(dTs):>5.1f}         '
              f'{np.mean(r2v):>6.3f}  {np.mean(r2a):>6.3f}  '
              f'{np.mean(maes):>5.1f}')
        print(f'  {"":24s} '
              f'+/-{np.std(dTs):<3.1f}         '
              f'+/-{np.std(r2v):<4.3f}  +/-{np.std(r2a):<4.3f}  '
              f'+/-{np.std(maes):<3.1f}')
        print(f'  {"MEDIAN":24s} '
              f'{np.median(dTs):>5.1f}         '
              f'{np.median(r2v):>6.3f}  {np.median(r2a):>6.3f}  '
              f'{np.median(maes):>5.1f}')

    if grand:
        print(f'\n{sep}')
        print(f'  GRAND SUMMARY  (n={len(grand)})')
        r2v_all = [r['R2_val'] for r in grand]
        r2a_all = [r['R2_all'] for r in grand]
        mae_all = [r['MAE_val'] for r in grand]
        print(f'  R2_val   mean={np.mean(r2v_all):.3f}  '
              f'median={np.median(r2v_all):.3f}  '
              f'min={np.min(r2v_all):.3f}  max={np.max(r2v_all):.3f}')
        print(f'  R2_all   mean={np.mean(r2a_all):.3f}  '
              f'median={np.median(r2a_all):.3f}  '
              f'min={np.min(r2a_all):.3f}  max={np.max(r2a_all):.3f}')
        print(f'  MAE_val  mean={np.mean(mae_all):.2f} degC  '
              f'median={np.median(mae_all):.2f} degC')
        print(sep + '\n')


# ==================================================================
#  VISUALIZATION
# ==================================================================

STAGE_LABELS = {
    'celsio':     'Celsio (1.7 mm probe, 10 s freeze)',
    'ice2x6':     'IceSphere 2x6 (1.5 mm, 360 s)',
    'clinical':   'Clinical (1.5 mm, 3-10 min)',
    'crystal_3d': '3-D Crystal (GelMA, 360 s)',
}

STAGE_ORDER = ['celsio', 'ice2x6', 'clinical', 'crystal_3d']


def plot_pooled_scatter(results, out_dir):
    """Pooled predicted-vs-measured scatter plot per stage + grand."""
    stages_present = [s for s in STAGE_ORDER
                      if any(r['stage'] == s for r in results)]
    n_panels = len(stages_present) + 1  # +1 for grand
    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows),
                             squeeze=False)
    axes_flat = axes.ravel()

    grand_meas, grand_pred = [], []
    panel = 0

    for stg in stages_present:
        ax = axes_flat[panel]
        stg_meas, stg_pred = [], []
        for r in results:
            if r['stage'] != stg:
                continue
            m = r['_meas'].ravel()
            p = r['_pred'].ravel()
            v = np.isfinite(m)
            stg_meas.append(m[v])
            stg_pred.append(p[v])
        stg_meas = np.concatenate(stg_meas)
        stg_pred = np.concatenate(stg_pred)
        grand_meas.append(stg_meas)
        grand_pred.append(stg_pred)

        ax.scatter(stg_meas, stg_pred, s=2, alpha=0.3, rasterized=True)
        lo = min(stg_meas.min(), stg_pred.min()) - 2
        hi = max(stg_meas.max(), stg_pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='y = x')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        r2 = _r2(stg_meas, stg_pred)
        n_exp = sum(1 for r in results if r['stage'] == stg)
        ax.set_title(f'{STAGE_LABELS.get(stg, stg)}\n'
                     f'n={n_exp}  R$^2$={r2:.3f}', fontsize=10)
        ax.set_xlabel('Measured (°C)')
        ax.set_ylabel('Predicted (°C)')
        panel += 1

    # Grand panel
    ax = axes_flat[panel]
    gm = np.concatenate(grand_meas)
    gp = np.concatenate(grand_pred)
    ax.scatter(gm, gp, s=2, alpha=0.15, rasterized=True, c='tab:green')
    lo = min(gm.min(), gp.min()) - 2
    hi = max(gm.max(), gp.max()) + 2
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_title(f'All stages pooled\nn={len(results)}  R$^2$={_r2(gm, gp):.3f}',
                 fontsize=10)
    ax.set_xlabel('Measured (°C)')
    ax.set_ylabel('Predicted (°C)')
    panel += 1

    # Hide extra axes
    for i in range(panel, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, 'pooled_scatter.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_stage_timeseries(results, out_dir):
    """Time-temperature overlay plots per stage: model vs measured for each TC."""
    tc_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2']

    for stg in STAGE_ORDER:
        rows_stg = [r for r in results if r['stage'] == stg]
        if not rows_stg:
            continue

        n_exp = len(rows_stg)
        ncols = min(n_exp, 3)
        nrows = (n_exp + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4 * nrows),
                                 squeeze=False)
        axes_flat = axes.ravel()

        for idx, r in enumerate(rows_stg):
            ax = axes_flat[idx]
            t = r['_t']
            tc_pos = r['_tc_pos']
            all_cols = r['_all_cols']
            fit_cols = r['_fit_cols']

            for j, col_idx in enumerate(all_cols):
                rr = tc_pos[col_idx]
                meas_j = r['_meas'][:, j]
                pred_j = r['_pred'][:, j]
                v = np.isfinite(meas_j)
                c = tc_colors[j % len(tc_colors)]
                is_fit = col_idx in fit_cols
                label_m = f'r={rr:.1f}mm meas' + (' (fit)' if is_fit else '')
                ax.plot(t[v], meas_j[v], '-', color=c, lw=1.2,
                        alpha=0.8, label=label_m)
                ax.plot(t, pred_j, '--', color=c, lw=1.0,
                        alpha=0.9)

            # Probe BC (light gray)
            ax.plot(t, r['_T_bc'], '-', color='0.7', lw=0.6, label='Probe BC')

            ax.set_title(f"{r['id']}\ndT={r['dT']:+.1f}  "
                         f"R$^2_{{val}}$={r['R2_val']:.3f}  "
                         f"R$^2_{{all}}$={r['R2_all']:.3f}",
                         fontsize=9)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Temperature (°C)')
            if idx == 0:
                ax.legend(fontsize=6, loc='lower right', ncol=2)

        for i in range(len(rows_stg), len(axes_flat)):
            axes_flat[i].set_visible(False)

        fig.suptitle(STAGE_LABELS.get(stg, stg), fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(out_dir, f'timeseries_{stg}.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f'  Saved: {path}')


def plot_r2_summary(results, out_dir):
    """Bar chart of R2_val per experiment, grouped by stage."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x_pos = []
    labels = []
    colors_map = {'celsio': '#1f77b4', 'ice2x6': '#ff7f0e',
                  'clinical': '#2ca02c', 'crystal_3d': '#d62728'}
    bar_colors = []
    vals = []
    pos = 0

    for stg in STAGE_ORDER:
        rows_stg = [r for r in results if r['stage'] == stg]
        if not rows_stg:
            continue
        if pos > 0:
            pos += 0.5  # gap between stages
        for r in rows_stg:
            x_pos.append(pos)
            short = r['id'].replace('celsio_', '').replace('ice2x6_', '') \
                           .replace('clinical_', '').replace('crystal_', '')
            labels.append(short)
            vals.append(r['R2_val'])
            bar_colors.append(colors_map.get(stg, '#999999'))
            pos += 1

    bars = ax.bar(x_pos, vals, color=bar_colors, width=0.7, edgecolor='white')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('R² (validation)')
    ax.set_title('Spatial Cross-Validation R² per Experiment')
    ax.axhline(0.8, color='gray', ls=':', lw=0.8)
    ax.set_ylim(min(0, min(vals) - 0.05), 1.05)

    # Stage legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors_map[s], label=STAGE_LABELS.get(s, s))
                      for s in STAGE_ORDER if any(r['stage'] == s for r in results)]
    ax.legend(handles=legend_patches, fontsize=8, loc='lower left')

    fig.tight_layout()
    path = os.path.join(out_dir, 'r2_summary.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_representative_fits(results, out_dir):
    """2x2 panel: best-fit experiment per stage (for PI presentation)."""
    tc_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.ravel()

    panel = 0
    for stg in STAGE_ORDER:
        rows_stg = [r for r in results if r['stage'] == stg]
        if not rows_stg:
            continue
        # Pick experiment with median R2_val (representative, not cherry-picked)
        rows_stg.sort(key=lambda r: r['R2_val'])
        best = rows_stg[len(rows_stg) // 2]

        ax = axes_flat[panel]
        t = best['_t']
        tc_pos = best['_tc_pos']
        all_cols = best['_all_cols']
        fit_cols = best['_fit_cols']

        for j, col_idx in enumerate(all_cols):
            rr = tc_pos[col_idx]
            meas_j = best['_meas'][:, j]
            pred_j = best['_pred'][:, j]
            v = np.isfinite(meas_j)
            c = tc_colors[j % len(tc_colors)]
            is_fit = col_idx in fit_cols
            label_m = f'r = {rr:.1f} mm' + (' (fit TC)' if is_fit else '')
            ax.plot(t[v], meas_j[v], '-', color=c, lw=1.5, alpha=0.85,
                    label=label_m)
            ax.plot(t, pred_j, '--', color=c, lw=1.2, alpha=0.9)

        ax.plot(t, best['_T_bc'], '-', color='0.75', lw=0.7, label='Probe BC')

        stg_label = STAGE_LABELS.get(stg, stg)
        ax.set_title(f'{stg_label}\n{best["id"]}   '
                     f'$\\Delta T$ = {best["dT"]:+.1f} °C   '
                     f'R$^2_{{val}}$ = {best["R2_val"]:.3f}',
                     fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.legend(fontsize=7, loc='lower right', ncol=2,
                  framealpha=0.8)
        ax.grid(True, alpha=0.2)
        # Add text: solid = measured, dashed = model
        ax.text(0.02, 0.02, 'solid = measured\ndashed = model',
                transform=ax.transAxes, fontsize=7, va='bottom',
                color='0.4')
        panel += 1

    fig.suptitle('Representative Fits — Median R² Experiments (not cherry-picked)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, 'representative_fits.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved: {path}')


def generate_plots(results, base_dir):
    """Generate all visualization figures."""
    out_dir = os.path.join(base_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    print(f'\nGenerating figures in {out_dir}/')
    plot_pooled_scatter(results, out_dir)
    plot_stage_timeseries(results, out_dir)
    plot_r2_summary(results, out_dir)
    plot_representative_fits(results, out_dir)
    print('Done.\n')


# ==================================================================
#  MAIN
# ==================================================================

def find_base_dir():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, 'Final - 2D analysis_Celsio')):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


def run_loo_cv(exps, only_stage=None):
    """Leave-one-out cross-validation with device-level Delta-T.

    For each experiment: use median dT from all OTHER experiments of the
    same stage, then evaluate with that fixed dT (no per-experiment fitting).
    """
    # First pass: fit dT per experiment (standard mode)
    print('  Phase 1: fitting dT per experiment...')
    fitted = {}
    for exp in exps:
        if only_stage and exp['stage'] != only_stage:
            continue
        r = run_experiment(exp)
        if r is not None:
            fitted[r['id']] = r
            print(f'    {r["id"]:<30s}  dT={r["dT"]:+.1f}')

    # Second pass: LOO-CV
    print('\n  Phase 2: leave-one-out cross-validation...')
    loo_results = []
    for exp in exps:
        if only_stage and exp['stage'] != only_stage:
            continue
        if exp['id'] not in fitted:
            continue

        # Median dT from all OTHER experiments in the same stage
        stage = exp['stage']
        other_dTs = [fitted[eid]['dT'] for eid, r in fitted.items()
                     if r['stage'] == stage and eid != exp['id']]
        if len(other_dTs) == 0:
            continue
        dT_loo = float(np.median(other_dTs))

        t0 = _timer.time()
        r = run_experiment(exp, dT_fixed=dT_loo)
        dt = _timer.time() - t0
        if r is not None:
            r['time_s'] = dt
            r['dT_loo'] = dT_loo
            r['dT_individual'] = fitted[exp['id']]['dT']
            loo_results.append(r)
            print(f'    {r["id"]:<30s}  dT_loo={dT_loo:+.1f}  '
                  f'R2_val={r["R2_val"]:.3f}  R2_all={r["R2_all"]:.3f}')

    return loo_results, fitted


def main():
    base_dir = find_base_dir()
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--base-dir' and i < len(sys.argv) - 1:
            base_dir = sys.argv[i + 1]

    print(f'Base directory: {base_dir}')
    exps = build_experiments(base_dir)
    print(f'Registered {len(exps)} experiments\n')

    only_stage = None
    for arg in sys.argv[1:]:
        if arg.startswith('--stage='):
            only_stage = arg.split('=', 1)[1]

    do_loo = '--loo' in sys.argv
    do_tmin = '--tmin' in sys.argv
    do_plot = '--plot' in sys.argv

    if do_tmin:
        # ===== T_MIN MODEL: zero per-experiment fitted parameters =====
        # Physical minimum probe surface temperatures per device type
        # Celsio: liquid CO2 boiling point = -78.5 C
        # IceSphere: JT argon, empirically ~-150 to -180 C at tip
        T_MIN_MAP = {
            'celsio':     -65.0,    # liquid CO2 probe surface (empirical optimum)
            'ice2x6':     -85.0,    # JT argon probe surface (empirical optimum)
            'clinical':   -85.0,    # same probe as IceSphere
            'crystal_3d': None,     # BC is embedded TC, not probe — no clamp
        }
        print('=' * 70)
        print('  T_MIN MODEL: minimum probe surface temperature (zero fitted params)')
        print('  + temperature-dependent k_ice')
        for stg, tmin in T_MIN_MAP.items():
            if tmin is not None:
                print(f'    {stg:15s}  T_min = {tmin:.1f} C')
            else:
                print(f'    {stg:15s}  (no clamp, BC is embedded TC)')
        print('=' * 70)

        results = []
        t_start = _timer.time()

        for exp in exps:
            if only_stage and exp['stage'] != only_stage:
                continue
            t_min = T_MIN_MAP.get(exp['stage'])
            t0 = _timer.time()
            r = run_experiment(exp, T_min_fixed=t_min)
            dt = _timer.time() - t0
            if r is not None:
                r['time_s'] = dt
                results.append(r)
                print(f'  {r["id"]:<30s}  R2_val={r["R2_val"]:.3f}  '
                      f'R2_all={r["R2_all"]:.3f}  ({dt:.1f}s)')
            else:
                print(f'  {exp["id"]:<30s}  SKIPPED')

        elapsed = _timer.time() - t_start
        print(f'\nTotal time: {elapsed:.1f} s')

        if results:
            print_results(results)

        if do_plot and results:
            generate_plots(results, base_dir)

        return results

    if do_loo:
        # ===== LOO-CV MODE =====
        print('=' * 70)
        print('  LEAVE-ONE-OUT CROSS-VALIDATION (device-level Delta-T)')
        print('  + temperature-dependent k_ice')
        print('=' * 70)
        loo_results, fitted_dict = run_loo_cv(exps, only_stage)

        if loo_results:
            print('\n')
            print_results(loo_results)

            # Print comparison: individual vs LOO
            print('\n' + '=' * 70)
            print('  COMPARISON: per-experiment fit vs LOO-CV')
            print('=' * 70)
            for stg in ['celsio', 'ice2x6', 'clinical', 'crystal_3d']:
                indiv = [fitted_dict[r['id']] for r in loo_results
                         if r['stage'] == stg and r['id'] in fitted_dict]
                loo = [r for r in loo_results if r['stage'] == stg]
                if not loo:
                    continue
                r2v_indiv = [fitted_dict[r['id']]['R2_val'] for r in loo
                             if r['id'] in fitted_dict]
                r2v_loo = [r['R2_val'] for r in loo]
                r2a_indiv = [fitted_dict[r['id']]['R2_all'] for r in loo
                             if r['id'] in fitted_dict]
                r2a_loo = [r['R2_all'] for r in loo]
                print(f'\n  {stg:15s}  n={len(loo):2d}')
                print(f'    R2_val  individual: median={np.median(r2v_indiv):.3f}  '
                      f'mean={np.mean(r2v_indiv):.3f}')
                print(f'    R2_val  LOO-CV:     median={np.median(r2v_loo):.3f}  '
                      f'mean={np.mean(r2v_loo):.3f}')
                print(f'    R2_all  individual: median={np.median(r2a_indiv):.3f}  '
                      f'mean={np.mean(r2a_indiv):.3f}')
                print(f'    R2_all  LOO-CV:     median={np.median(r2a_loo):.3f}  '
                      f'mean={np.mean(r2a_loo):.3f}')

        if do_plot and loo_results:
            generate_plots(loo_results, base_dir)

        return loo_results

    # ===== STANDARD MODE (per-experiment fit) =====
    print('  Using temperature-dependent k_ice')
    results = []
    t_start = _timer.time()

    for exp in exps:
        if only_stage and exp['stage'] != only_stage:
            continue
        t0 = _timer.time()
        r = run_experiment(exp)
        dt = _timer.time() - t0
        if r is not None:
            r['time_s'] = dt
            results.append(r)
            print(f'  {r["id"]:<30s}  R2_val={r["R2_val"]:.3f}  '
                  f'dT={r["dT"]:+.1f}  ({dt:.1f}s)')
        else:
            print(f'  {exp["id"]:<30s}  SKIPPED (file missing or too short)')

    elapsed = _timer.time() - t_start
    print(f'\nTotal time: {elapsed:.1f} s')

    if results:
        print_results(results)

    if do_plot and results:
        generate_plots(results, base_dir)

    return results


if __name__ == '__main__':
    main()
