.. image:: https://img.shields.io/badge/license-%20MPL--v2.0-blue.svg
   :target: LICENSE

Cardiotocograpy - Fetal Heart Rate classifier
=============================================


The code in this repository is byproduct of Master's Thesis of Petri Jehkonen.

.. contents:: Table of Contents

Author
------
- Petri Jehkonen

Citation
--------

If you use this tool in a program or publication, please acknowledge its
author::

  @misc{ctg_zigzag_2020,
    author    = {Jehkonen, Petri},
    title     = {ZigZag classifier: Detecting high FHR baseline variability in CTG data},
    month     = {11},
    year      = {2020},
    publisher = {Zenodo.here},
    version   = {version.here},
    doi       = {doi.here},
    url       = {full.url.here}
  }

Input dataformat
----------------
Input data format:
- X: pd.DataFrame(cols=observations, rows=time samples).
- y: vector, where 0 means no high variability and 1 means high variability.
