Changes
*******

0.2.0 (2026-11)
===============

* Generalized NIPA: configurable climate index and number of phases (1-4), and SST, SLP or
  Z500 as predictor field.
* Leakage-free validation: trend removal, screening, PCA and regression are re-estimated in
  every leave-one-out fold.
* Operational forecast for a given year, split into trend and teleconnection components and
  returned with the leave-one-out skill of the model that produced it.
* Outputs renamed to match the web interface (forecast.csv, metrics.csv, hindcasts.csv,
  predictor_map.pdf, scatterplot.pdf, outputs.zip with a model/ folder).
* Checks on the analysis period against the coverage of target, predictor and index.
* SST from ERSST v6; SLP and Z500 from the NCEP/NCAR Reanalysis 1 (no longer updated after
  February 2026).
* offline_forecast.py: applies a model returned by the service offline.

0.1.0 (2023-10-11)
==================

* First release.
