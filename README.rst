=========
Albatross
=========

.. image:: https://img.shields.io/pypi/v/albatross.svg
        :target: https://pypi.python.org/pypi/albatross

.. image:: https://readthedocs.org/projects/albatross/badge/?version=latest
        :target: https://albatross.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/license/climateintelligence/albatross.svg
    :target: https://github.com/climateintelligence/albatross/blob/main/LICENSE
    :alt: GitHub license

Albatross (the bird)
  *Albatross is a bird designed for seasonal forecasting of hydroclimatic variables.*

This WPS is designed to produce deterministic forecasts of seasonal hydroclimatic variables (e.g. precipitation, temperature, streamflows,...) of a given location, leveraging teleconnections. It is built upon the Niño Index Phase Analysis (NIPA) framework by Zimmerman et al. (2016).


🔍 What NIPA Does
-----------------

NIPA classifies historical climate data (e.g., Sea Surface Temperature, SST) according to the phases of El Niño, typically using the Oceanic Niño Index (ONI). For each phase (e.g., El Niño, La Niña, Neutral), NIPA:

- Identifies relevant spatial patterns in SST fields via linear correlation masks
- Applies Principal Component Analysis (PCA) to reduce dimensionality
- Calibrates a linear regression model on the first principal component to predict hydroclimatic outcomes

✨ What’s New in Albatross
--------------------------

Albatross extends the original NIPA framework with several innovations:

- 🌍 New Teleconnection Indices: In addition to ONI, Albatross incorporates other teleconnections like the North Atlantic Oscillation (NAO) to enhance predictive skill across different regions.
- 📉 Dynamic Dimensionality Reduction: The number of PCA components retained is dynamically chosen based on the explained variance, improving robustness and interpretability.
- 🧠 Modular Architecture: Albatross is built to easily accommodate new climate indices, regions, and variables.

🔭 What’s Next
--------------

We’re actively working on expanding Albatross with more powerful and flexible tools:

- ➕ More Indices: Planned integration of additional teleconnections (e.g., Pacific Decadal Oscillation, Arctic Oscillation) to cover broader climatic regimes.
- 🔁 Nonlinear Relationships: Introduction of Mutual Information and other nonlinear dependency measures to detect subtler teleconnection patterns.
- 🤖 Machine Learning Models: Future versions will explore nonlinear models, including Random Forests, and other ML tools to improve forecast accuracy and capture complex dependencies.

🔧 Requirements to Run Albatross
--------------------------------

To use the Albatross WPS for forecasting, you need to provide:

**📄 Input File**

- A plain-text file containing monthly hydroclimatic data (e.g., precipitation, temperature, etc.) for a single location.
- The file must follow the Albatross format, structured as:
- The first line must contain the title (e.g., E-OBS_precipitation)
- The second line must include two numbers: the starting year and ending year
- The following lines must contain monthly values in row-wise format, ordered by year (12 values per year)

✅ Example file: https://github.com/climateintelligence/albatross/blob/main/albatross/data/E-OBS_precipitation_Como.txt
(You can browse from your computer or provide a direct URL, such as a raw GitHub link)


**🧾 Additional Parameters**

You must also specify the following inputs when running the WPS:

+----------------------+----------------------------------------------------------------------------------+
| Parameter            | Description                                                                      |
+======================+==================================================================================+
| Start year           | First year of the time series (e.g., 1950)                                       |
+----------------------+----------------------------------------------------------------------------------+
| End year             | Last year of the time series (e.g., 2023)                                        |
+----------------------+----------------------------------------------------------------------------------+
| Target Season        | Numeric code for the target season: 1 = JFM, ..., 12 = DJF                       |
+----------------------+----------------------------------------------------------------------------------+
| Phase mode (int)     | Whether to apply NIPA separately for El Niño and La Niña phases (1 or 2)         |
+----------------------+----------------------------------------------------------------------------------+

📘 Citation
-----------

- Zimmerman, B. G., D. J. Vimont, and P. J. Block (2016), Utilizing the state of ENSO as a means for season-ahead predictor selection, Water Resour. Res., 52, 3761–3774, doi:10.1002/2015WR017644.

Documentation
-------------

Learn more about Albatross in its official documentation at
https://albatross.readthedocs.io.

Submit bug reports, questions and feature requests at
https://github.com/climateintelligence/albatross/issues

Contributing
------------

You can find information about contributing in our `Developer Guide`_.

Please use bumpversion_ to release a new version.


License
-------

* Free software: Apache Software License 2.0
* Documentation: https://albatross.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `bird-house/cookiecutter-birdhouse`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`bird-house/cookiecutter-birdhouse`: https://github.com/bird-house/cookiecutter-birdhouse
.. _`Developer Guide`: https://albatross.readthedocs.io/en/latest/dev_guide.html
.. _bumpversion: https://albatross.readthedocs.io/en/latest/dev_guide.html#bump-a-new-version
