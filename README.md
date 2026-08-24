# Cryoablation dosimetry: thermal model and dose-response

Code for "Dosimetry for Pulmonary Cryoablation: Coupling Phase-Change Thermal
Modelling with Spatially Resolved Cell Death".

Two components: a one-dimensional cylindrical finite-difference solver for the
temperature field around a cryoprobe, and hierarchical Bayesian dose-response
models relating minimum temperature to acute cell death.

## Contents

    cryo_thermal.py     Finite-difference solver. Apparent heat capacity for
                        phase change, implicit Euler, Thomas algorithm. Also
                        holds the experiment registry and the per-experiment
                        boundary-offset fit.
    lung_chain.py       Lung tissue parameterisation and the chained
                        three-cycle clinical protocol.
    dose_response.py    Hierarchical logistic dose-response in PyMC (Gaussian,
                        binomial and beta-binomial observation models), the
                        dose metrics, and the confocal ellipsoid geometry.
    analysis.ipynb      Runs all of the above and prints the reported values.
    supplementary/
      input_sensitivity.py  Latin hypercube over the tissue properties and the
                        probe temperature, with partial rank correlations.
                        Slower than the notebook; run separately.
    data/               Per-bin cell counts with their assigned minimum
                        temperatures, and the cohort ice-ball summary
                        statistics.
    Final - .../        Thermocouple recordings, one CSV per experiment, in the
                        directory layout the acquisition used.

## Running

    pip install -r requirements.txt
    jupyter lab analysis.ipynb

The notebook is stored with its outputs. Re-running it end to end takes about
20 minutes on an Intel Core Ultra 7 155H; the sampling cells account for most
of that and show no progress while they run. `input_sensitivity.py` takes about
a further 40 minutes and is not run by the notebook. No GPU is needed.

PyTensor prints a warning if a C compiler is not on the path. It is harmless
here: sampling goes through nutpie, which does not use it.

No cached intermediates are committed. Everything is recomputed from the inputs
in this repository.

## Where the reported quantities come from

| Quantity | Manuscript location | Where |
|---|---|---|
| Fitted offset, per-experiment R2 and MAE | Supp. Table S1 | notebook 1 |
| Thermocouple positions, both conventions | Supp. Table S4 | notebook 2 |
| Freeze and thaw timings | Supp. Table S3 | notebook 2 |
| Directly validated temperature range | Supp. Fig. S6 | notebook 2 |
| Bin count structure | Methods | notebook 3 |
| Dose-response parameters, plateau, steepness, midpoint | Fig. 2B-D, Results | notebook 3, 5 |
| Convergence: R-hat, effective sample size, MCSE | Supp. Table S8 | notebook 3 |
| Interval coverage, LOO, Pareto k | Supp. Fig. S5, Table S8 | notebook 4 |
| AD50 and AD95, relative and absolute | Supp. Table S6, Results | notebook 5 |
| Directional posterior probabilities | Results | notebook 5 |
| Multi-cycle survival exponent | Fig. 2E, Results | notebook 6 |
| Isodose ellipsoid dimensions and volumes | Fig. 5A | notebook 7 |
| Mahalanobis distance and equivalence test | Fig. 4, Results | notebook 8 |
| Modified Stefan number | Results | notebook 9 |
| Input sensitivity and partial rank correlations | Supp. Fig. S3 | `supplementary/input_sensitivity.py` |

Two supplementary items are not reproduced here. Supp. Table S2 (patients
outside the 95 % bivariate band) and Supp. Table S7 and Fig. S4 (the confocal
mapping) require the individual patient measurements rather than the cohort
summary statistics. Supp. Table S5 (numerical convergence of the grid, time
step and domain radius) is a one-off study of the solver rather than an
analysis of the data.

## Dose metric definitions

The fitted plateau is below 100 % in every condition, so a dose threshold can
be defined relative to that plateau or on an absolute percentage scale. The
analysis uses the relative definition throughout:

    AD_q = x0 + ln(100/q - 1) / k

so AD50 is the logistic inflection point by construction and
AD95 = x0 - 2.944/k. `dose_response.dose_metric` implements both conventions
and takes the choice as an argument. Notebook section 4 reports absolute values
alongside, with the fraction of posterior draws for which an absolute 95 %
threshold is not attainable.

## Data

`data/celsio_bins.csv` and `data/icesphere_bins.csv` hold per-bin cell counts
(number alive, number expected) with the minimum temperature assigned to each
bin by the thermal model. These are the quantities the dose-response models are
fitted to. The cell-level classification exports they were derived from are
large and are not included here.

`data/cohort_geometry.csv` holds the mean, standard deviation and covariance of
the measured orthogonal ice-ball diameters across the 52-patient cohort. These
summary statistics reproduce the Mahalanobis distance and the equivalence test
exactly; per-patient coverage requires the individual measurements, which are
not included. Cell-level imaging exports and patient-level data are available
from the corresponding author on reasonable request.

## License

MIT, see LICENSE.
