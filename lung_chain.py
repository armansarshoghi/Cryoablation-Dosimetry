"""lung_chain.py -- Lung-adapted chained 3-7-10 protocol simulator.

Standalone script (not imported by the notebook). For each clinical replicate
(N1, N2, N3), runs the full 3-cycle protocol with:
  * BC trajectory: measured TC trace from clinical_*.csv, with a per-segment
    dT shift so each freeze nadir lands at the canonical T_min (-90 +/- 15 C).
  * Rewarms: Neumann zero-flux at probe inner BC after each freeze nadir.
  * Cycles chained: end-of-cycle-N T-field becomes initial condition of cycle N+1.
  * T_i = 37 C body temperature (far-field Dirichlet).
  * Three variants:
      V0: all 3 cycles in water (lung-temp baseline; vs in-vitro at lab temp)
      VA: skip cycle 1, cycles 2+3 in water (initial T = 37 C uniform)
      VB: cycle 1 in air, cycles 2+3 in water (full chain, material switch)
  * T_min sweep: -75, -90, -105 C (uncertainty band on the canonical probe nadir).

For each (rep, variant, T_min), compute the 0 C isotherm max extent during
cycle 3, convert to prolate ellipsoid via tip_to_ellipsoid(L_active=22), and
report (long_axis, short_axis) in mm.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

import cryo_thermal as ct
from cryo_thermal import (
    solve_fdm, load_tc, smooth, build_experiments,
    K_WATER, C_WATER, RHO_WATER, K_ICE, C_ICE, RHO_ICE, L_WATER,
)

# --- Air at 37 C, 1 atm  (Incropera & DeWitt; CRC handbook) -----------------
# Solver units: mm, s, degC, kg, J, W
K_AIR     = 0.0271e-3       # W/(mm*K)
C_AIR     = 1007.0          # J/(kg*K)
RHO_AIR   = 1.139e-9        # kg/mm^3
L_AIR     = 0.0             # no phase change in dry air at -90 C
T_PHASE_AIR = -200.0        # below the model's lowest temp -> phase term vanishes

# --- Lung (IT'IS Foundation Tissue Properties Database V5.0) ----------------
# Direct extraction from Thermal_dielectric_acoustic_MR properties V5.0(ASCII).txt
# Latent heat derived from tabulated water content (Hasgall et al. 2022)
# multiplied by L_water = 334 kJ/kg.
K_LUNG_INF   = 0.179e-3      # W/(mm*K)   range 0.161-0.198 W/m/K (N=1)
C_LUNG_INF   = 3886.0        # J/(kg*K)
RHO_LUNG_INF = 394e-9        # kg/mm^3    range 255-604 kg/m^3 (N=26)
WC_LUNG_INF  = 0.272         # 27.2% water content (N=15)
L_LUNG_INF   = WC_LUNG_INF * L_WATER       # ~91 kJ/kg

K_LUNG_DEF   = 0.3874e-3     # W/(mm*K)   range 0.28-0.478 W/m/K (N=5)
C_LUNG_DEF   = 3886.0        # J/(kg*K)
RHO_LUNG_DEF = 1050e-9       # kg/mm^3
WC_LUNG_DEF  = 0.803         # 80.3% water content (N=17, range 74.9-84.4)
L_LUNG_DEF   = WC_LUNG_DEF * L_WATER       # ~268 kJ/kg

# --- Clinical protocol (verbatim from cryo_thermal.build_experiments) -------
CLIN_FS = {                  # freeze-start sample index in TC file
    'N1': {1: 10, 2: 370, 3: 1041},
    'N2': {1:  9, 2: 409, 3: 1065},
    'N3': {1: 14, 2: 374, 3: 1032},
}
CLIN_DUR = {1: 180, 2: 420, 3: 600}     # freeze duration (s) per cycle
T_BODY = 37.0
L_ACTIVE = 22.0                          # mm, probe active length

ICE_DIR = Path('Final - 2D IceFx2x6') / 'Temperature files'

# Discretisation. Module-level so the convergence tests in the revision
# notebook can vary them without touching the solver call; the defaults are
# the values used for every number in the manuscript.
R_EVAL_EXTENT = 18.0     # mm beyond r_bc covered by the output grid
DR_OUT = 0.05            # mm, output grid spacing
DR_SOLVER = 0.05         # mm, solver grid spacing
N_SUB = 4                # sub-steps per 1 Hz sample -> dt = 0.25 s

EXPS = build_experiments(str(Path('.').resolve()))
EXPS_BY_ID = {e['id']: e for e in EXPS}

# ---------------------------------------------------------------------------
def load_clinical_trace(rep: str):
    """Return (times_s, T_probe_raw_smoothed) for the full TC file of rep."""
    eid = f'clinical_{rep}_c1'
    exp = EXPS_BY_ID[eid]
    times, temps = load_tc(exp['file'])
    for ch in range(temps.shape[1]):
        temps[:, ch] = smooth(temps[:, ch])
    bc_cols = exp['bc_cols']
    if len(bc_cols) == 1:
        Tbc_raw = temps[:, bc_cols[0]].copy()
    else:
        Tbc_raw = np.nanmin(temps[:, bc_cols], axis=1)
    # Forward-fill any NaN in the probe trace
    mask = np.isfinite(Tbc_raw)
    if not mask.all():
        good = np.where(mask)[0]
        Tbc_raw = np.interp(np.arange(len(Tbc_raw)), good, Tbc_raw[good])
    t = times.astype(float) - float(times[0])
    return t, Tbc_raw


def build_cycle_window(rep: str, cyc: int, post_pad: int):
    """Slice indices for cycle `cyc`: from fs to (next fs) or to fs+dur+post_pad
    if last cycle. Returns (idx_start, idx_freeze_end, idx_window_end)."""
    fs = CLIN_FS[rep][cyc]
    freeze_end = fs + CLIN_DUR[cyc]
    if cyc < 3:
        # Window extends to the start of the next freeze (covers the rewarm).
        next_fs = CLIN_FS[rep][cyc + 1]
        window_end = next_fs
    else:
        # Final cycle: pad with post_pad seconds of measured rewarm if available.
        window_end = freeze_end + post_pad
    return fs, freeze_end, window_end


def make_segment_bc(t_seg, Tbc_raw_seg, t_freeze_dur, T_min_target):
    """Apply a per-cycle dT shift so the effective probe nadir lands at T_min.
    Use Neumann zero-flux mode after the freeze ends (BC values past freeze
    are placeholders since the solver ignores them under neumann_after).
    Returns (T_bc_eff, t_nadir_in_window)."""
    freeze_mask = t_seg <= t_freeze_dur
    raw_nadir = float(np.nanmin(Tbc_raw_seg[freeze_mask]))
    dT_shift = T_min_target - raw_nadir          # shift to land nadir on target
    # Step-shift during freeze; leave rewarm segment at raw (Neumann ignores it).
    T_bc_eff = Tbc_raw_seg + np.where(freeze_mask, dT_shift, 0.0)
    nadir_local = int(np.argmin(Tbc_raw_seg[freeze_mask]))
    t_nadir = float(t_seg[nadir_local])
    return T_bc_eff, t_nadir, dT_shift


def run_one_cycle(t_seg, T_bc_eff, t_nadir, T_init,
                  material: str, T_phase: float,
                  k_def_override: float | None = None,
                  props: dict | None = None):
    """Run solve_fdm for one cycle window. material in {'water', 'air'}.
    T_init: tuple (r_grid, T_field) from prior cycle, or None for uniform body temp.
    Returns (t_seg, r_grid, T_field_history, r_grid_final, T_grid_final).

    `props` overrides individual lung properties for uncertainty propagation.
    Recognised keys (solver units: W/(mm K), kg/mm^3, J/kg):
        k_inf, rho_inf, L_inf, k_def, rho_def, L_def
    Anything absent falls back to the IT'IS central value.
    """
    p = props or {}
    if material == 'water':
        skw = dict(k_s=K_ICE, k_l=K_WATER, c_s=C_ICE, c_l=C_WATER,
                   rho_s=RHO_ICE, rho_l=RHO_WATER, L_f=L_WATER,
                   k_ice_tdep=True)
    elif material == 'air':
        # No phase change for air; pass air properties for both s/l so the
        # mushy-zone branch (never reached because T_phase=-200) is moot.
        skw = dict(k_s=K_AIR, k_l=K_AIR, c_s=C_AIR, c_l=C_AIR,
                   rho_s=RHO_AIR, rho_l=RHO_AIR, L_f=L_AIR,
                   k_ice_tdep=False)
    elif material == 'lung_inflated':
        # Treat solid (frozen) and liquid (unfrozen) with same bulk lung props;
        # latent heat scaled by water content. k_ice_tdep off (no pure-ice).
        k_i = p.get('k_inf', K_LUNG_INF)
        rho_i = p.get('rho_inf', RHO_LUNG_INF)
        L_i = p.get('L_inf', L_LUNG_INF)
        skw = dict(k_s=k_i, k_l=k_i, c_s=C_LUNG_INF, c_l=C_LUNG_INF,
                   rho_s=rho_i, rho_l=rho_i, L_f=L_i, k_ice_tdep=False)
    elif material == 'lung_deflated':
        # Optional override on the deflated-lung thermal conductivity for
        # IT'IS literature-range sensitivity (range 0.28-0.478 W/m/K, N=5).
        k_def = p.get('k_def', k_def_override if k_def_override is not None else K_LUNG_DEF)
        rho_d = p.get('rho_def', RHO_LUNG_DEF)
        L_d = p.get('L_def', L_LUNG_DEF)
        skw = dict(k_s=k_def, k_l=k_def, c_s=C_LUNG_DEF, c_l=C_LUNG_DEF,
                   rho_s=rho_d, rho_l=rho_d, L_f=L_d, k_ice_tdep=False)
    else:
        raise ValueError(material)

    # Use a dense output grid so we can chain (T_pred[-1] becomes T_init for next).
    r_bc = 0.75
    r_eval = np.arange(r_bc, r_bc + R_EVAL_EXTENT, DR_OUT)

    out = solve_fdm(
        t=t_seg, T_bc=T_bc_eff, r_bc=r_bc, r_eval=r_eval,
        T_i=T_BODY, T_phase=T_phase,
        HW=0.5, dr=DR_SOLVER, n_sub=N_SUB,
        W_conv=0.0, neumann_after=t_nadir,
        T_init=T_init,
        return_grid_final=True,
        **skw,
    )
    T_pred, Rf, r_grid_final, T_grid_final = out
    return t_seg, r_eval, T_pred, r_grid_final, T_grid_final


# ---------------------------------------------------------------------------
def chain_protocol(rep: str, variant: str, T_min_target: float,
                    k_def_override: float | None = None,
                    props: dict | None = None):
    """Returns dict with the cycle-3 0-C-isotherm radius and ellipsoid (L, S).

    k_def_override (W/(mm*K)): if set, overrides the deflated-lung k for
    cycles 2 and 3 (IT'IS literature-range sensitivity).
    """
    t_full, Tbc_raw_full = load_clinical_trace(rep)

    # For VB, cycle 1 uses air; everything else water.
    materials = {
        'V0':  {1: 'water',         2: 'water',         3: 'water'},
        'VA':  {1: None,            2: 'water',         3: 'water'},
        'VB':  {1: 'air',           2: 'water',         3: 'water'},
        'VC1': {1: 'lung_inflated', 2: 'lung_inflated', 3: 'lung_inflated'},
        'VC2': {1: 'lung_deflated', 2: 'lung_deflated', 3: 'lung_deflated'},
        'VC3': {1: 'lung_inflated', 2: 'lung_deflated', 3: 'lung_deflated'},
    }[variant]
    phases = {'water': -0.52, 'air': T_PHASE_AIR,
              'lung_inflated': -0.52, 'lung_deflated': -0.52}

    T_init = None        # cycle 1 starts uniform at body temp
    cycle3_history = None
    T_min_chain = None                 # T_min over the FULL 3-cycle chain
    r_eval_chain = None
    per_cycle = {}                     # cumulative iceball after each cycle

    for cyc in (1, 2, 3):
        material = materials[cyc]
        if material is None:
            # Skip this cycle entirely (VA's c1).
            continue

        fs, freeze_end, window_end = build_cycle_window(rep, cyc, post_pad=240)
        # Don't run past the file
        window_end = min(window_end, len(t_full))
        sl = slice(fs, window_end)
        t_seg = t_full[sl] - t_full[fs]              # 0-based for the segment
        Tbc_raw_seg = Tbc_raw_full[sl]
        t_freeze_dur = float(t_full[freeze_end - 1] - t_full[fs])

        T_bc_eff, t_nadir, dT_shift = make_segment_bc(
            t_seg, Tbc_raw_seg, t_freeze_dur, T_min_target)

        t_out, r_eval, T_pred, r_grid_final, T_grid_final = run_one_cycle(
            t_seg, T_bc_eff, t_nadir, T_init, material, phases[material],
            k_def_override=k_def_override, props=props)

        # Track T_min across the entire chain (all three cycles)
        T_min_cyc = np.nanmin(T_pred, axis=0)
        if T_min_chain is None:
            T_min_chain  = T_min_cyc.copy()
            r_eval_chain = r_eval.copy()
        else:
            T_min_chain = np.minimum(T_min_chain, T_min_cyc)

        # Cumulative iceball after this cycle: largest distance where the
        # min temperature ever reached so far is <= 0 C. This represents
        # the maximum iceball extent observed clinically by the end of cycle N.
        d_chain = r_eval_chain - r_eval_chain[0]
        below = np.where(T_min_chain <= 0.0)[0]
        if len(below) == 0:
            rt_cyc = 0.0
        else:
            ic = below[-1]
            if ic + 1 < len(d_chain):
                d1, d2 = d_chain[ic], d_chain[ic + 1]
                T1, T2 = T_min_chain[ic], T_min_chain[ic + 1]
                rt_cyc = float(d1 + (0.0 - T1) / (T2 - T1) * (d2 - d1)) if T2 != T1 else float(d1)
            else:
                rt_cyc = float(d_chain[ic])
        c_geom = L_ACTIVE / 2.0
        if rt_cyc > 0:
            b_sq = rt_cyc * rt_cyc * (1.0 + np.sqrt(1.0 + 4.0 * c_geom * c_geom / (rt_cyc * rt_cyc))) / 2.0
            b_g = np.sqrt(b_sq); a_g = b_sq / rt_cyc
            per_cycle[f'cycle_{cyc}'] = dict(
                rt=rt_cyc, length=2.0 * a_g, width=2.0 * b_g)
        else:
            per_cycle[f'cycle_{cyc}'] = dict(
                rt=0.0, length=np.nan, width=np.nan)

        # Record cycle-3 history for iceball measurement
        if cyc == 3:
            cycle3_history = (t_out, r_eval, T_pred)

        # Hand off final T-field to the next cycle.
        # If the next cycle has different material, re-interpolation handles
        # property change (the field values are continuous; only k/rho/c flip).
        # For VB: at end of cycle 1 the field is in "air" units but the values
        # are temperatures, which transfer cleanly to water for cycle 2.
        T_init = (r_grid_final, T_grid_final)

        # When VB or VC3 transitions inflated/air -> water/deflated at end of
        # cycle 1, the alveoli flood with 37 C blood. Approximate by warming
        # any unfrozen tissue (T > 0 C) back to body temp -- perfused blood
        # replaces the air space. Any sub-zero tissue stays cold (frozen
        # alveolar surface water doesn't get displaced).
        switches_at_c1 = (
            (variant == 'VB' and cyc == 1) or
            (variant == 'VC3' and cyc == 1)
        )
        if switches_at_c1:
            T_alveolar_flood = T_grid_final.copy()
            T_alveolar_flood[T_alveolar_flood > 0.0] = T_BODY
            T_init = (r_grid_final, T_alveolar_flood)

    if cycle3_history is None:
        return None

    t_c3, r_eval, T_c3 = cycle3_history
    # 0 C isotherm max extent during cycle 3
    T_min_field = np.nanmin(T_c3, axis=0)             # min over time at each r
    r_bc = float(r_eval[0])
    d_from_surface = r_eval - r_bc
    # Largest d where T_min <= 0
    below = np.where(T_min_field <= 0.0)[0]
    if len(below) == 0:
        rt = 0.0
    else:
        i = below[-1]
        if i + 1 < len(d_from_surface):
            d1, d2 = d_from_surface[i], d_from_surface[i + 1]
            T1, T2 = T_min_field[i], T_min_field[i + 1]
            if T2 != T1:
                frac = (0.0 - T1) / (T2 - T1)
                rt = float(d1 + frac * (d2 - d1))
            else:
                rt = float(d1)
        else:
            rt = float(d_from_surface[i])

    # tip_to_ellipsoid (verbatim from Crystal_Unified.py:2632)
    c = L_ACTIVE / 2.0
    # Distance grid (from probe surface) for T_min field exposure
    d_chain = (r_eval_chain - r_eval_chain[0]) if r_eval_chain is not None else None
    if rt <= 0:
        return dict(rep=rep, variant=variant, T_min=T_min_target,
                    rt=rt, length=np.nan, width=np.nan,
                    T_min_field=T_min_chain, d_grid=d_chain,
                    per_cycle=per_cycle)
    b_sq = rt * rt * (1.0 + np.sqrt(1.0 + 4.0 * c * c / (rt * rt))) / 2.0
    b = np.sqrt(b_sq)
    a = b_sq / rt
    return dict(rep=rep, variant=variant, T_min=T_min_target,
                rt=rt, length=2.0 * a, width=2.0 * b,
                T_min_field=T_min_chain, d_grid=d_chain,
                per_cycle=per_cycle)


# ---------------------------------------------------------------------------
def main():
    REPS = ['N1', 'N2', 'N3']
    VARIANTS = ['V0', 'VA', 'VB', 'VC1', 'VC2', 'VC3']
    T_MINS = [-75.0, -90.0, -105.0]

    rows = []
    for variant in VARIANTS:
        for T_min in T_MINS:
            for rep in REPS:
                try:
                    r = chain_protocol(rep, variant, T_min)
                    if r is not None:
                        rows.append(r)
                except Exception as e:
                    print(f'  ! {variant}/T_min={T_min:.0f}/{rep}: {e}', file=sys.stderr)

    print('\n' + '=' * 78)
    print(' Per-experiment ellipsoid predictions (L = long axis, S = short axis, mm)')
    print('=' * 78)
    print(f'{"variant":<5}  {"T_min":>6}  {"rep":>3}  {"r_tip":>6}  {"L":>6}  {"S":>6}')
    print('-' * 50)
    for row in rows:
        print(f'{row["variant"]:<5}  {row["T_min"]:>+6.0f}  {row["rep"]:>3}  '
              f'{row["rt"]:>6.2f}  {row["length"]:>6.2f}  {row["width"]:>6.2f}')

    print('\n' + '=' * 78)
    print(' Cohort summary (mean across replicates +/- SEM, then range across T_min)')
    print('=' * 78)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f'\n{"variant":<5}  {"T_min":>6}  '
          f'{"L_mean+/-SEM":>16}  {"S_mean+/-SEM":>16}  '
          f'{"AR":>5}')
    print('-' * 70)
    for variant in VARIANTS:
        for T_min in T_MINS:
            sub = df[(df['variant'] == variant) & (df['T_min'] == T_min)]
            if len(sub) == 0:
                continue
            L_m = sub['length'].mean(); L_e = sub['length'].std(ddof=1) / np.sqrt(len(sub))
            S_m = sub['width'].mean();  S_e = sub['width'].std(ddof=1) / np.sqrt(len(sub))
            ar  = L_m / S_m if S_m > 0 else np.nan
            print(f'{variant:<5}  {T_min:>+6.0f}  '
                  f'{L_m:>6.2f} +/- {L_e:>4.2f}    '
                  f'{S_m:>6.2f} +/- {S_e:>4.2f}    {ar:>5.2f}')

    # Patient cohort reference (from Section 8 numerical readout)
    print('\n' + '=' * 78)
    # Orientation only; nothing here feeds the solver. The cohort is the 52
    # patients on the canonical 3-3-7-3-10 protocol.
    print(' Reference: patient cohort (N=52)  L=25.35 mm, S=15.98 mm, AR=1.59')
    print(' Reference: in-vitro c3 model      L=31.46 mm, S=22.48 mm, AR=1.40')
    print('=' * 78)


if __name__ == '__main__':
    main()
