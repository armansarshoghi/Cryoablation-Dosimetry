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
    data/               Per-bin cell counts with their assigned minimum
                        temperatures.
    Final - .../        Thermocouple recordings, one CSV per experiment, in the
                        directory layout the acquisition used.

## Running

    pip install -r requirements.txt
    jupyter lab analysis.ipynb

The notebook is stored with its outputs. Re-running it end to end takes about
16 minutes on an Intel Core Ultra 7 155H; the two sampling cells account for
most of that and each shows no progress while it runs. No GPU is needed.

PyTensor prints a warning if a C compiler is not on the path. It is harmless
here: sampling goes through nutpie, which does not use it.

## Where the reported quantities come from

| Quantity | Notebook section |
|---|---|
| Fitted boundary offset, out-of-sample R2 and MAE | 1 |
| Dose-response parameters (plateau, steepness, midpoint) | 2, 3 |
| Convergence diagnostics: R-hat, effective sample size, MCSE | 2 |
| ALD50 and ALD95, relative and absolute | 3 |
| Directional posterior probabilities | 3 |
| Lung-adapted isodose ellipsoid dimensions and volumes | 4 |
| Modified Stefan number | 5 |

## Dose metric definitions

The fitted plateau is below 100 % in every condition, so a dose threshold can
be defined relative to that plateau or on an absolute percentage scale. The
analysis uses the relative definition throughout:

    ALD_q = x0 + ln(100/q - 1) / k

so ALD50 is the logistic inflection point by construction and
ALD95 = x0 - 2.944/k. `dose_response.dose_metric` implements both conventions
and takes the choice as an argument. Section 3 reports absolute values
alongside, with the fraction of posterior draws for which an absolute 95 %
threshold is not attainable.

## Data

`data/` holds per-bin cell counts (number alive, number expected) with the
minimum temperature assigned to each bin by the thermal model. These are the
quantities the dose-response models are fitted to. The cell-level
classification exports they were derived from are large and are not included
here.

Patient measurements are not included. Patient-level data are available from
the corresponding author on reasonable request.

## License

MIT, see LICENSE.
