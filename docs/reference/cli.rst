Command-line reference
======================

Use ``perturbo`` to fit an observed screen from an AnnData or MuData file. The :doc:`Gasperini tutorial <../tutorials/gasperini>` shows the command for a screen with multiple guides per cell. The :doc:`Replogle tutorial <../tutorials/replogle>` shows the command for cells labeled by a single perturbation.

Choosing the input and results
------------------------------

* Supply raw RNA counts with ``--input``. For a MuData input, identify the RNA modality with ``--modality-key``. If you have already subset the genes, use ``--library-size-key`` to supply the original per-cell totals.
* For one construct label per cell, give its cell annotation with ``--perturbation-key`` and identify the controls. For a guide-assignment matrix, give its modality with ``--perturbation-modality-key``. The Gasperini example also maps its two guides to their intended element.
* Supply ``--cis-pairs`` to calculate a separate adjusted p-value over a prespecified list of nearby element–gene pairs. Use the identifiers that appear in the fitted results.
* ``--out-dir`` selects the destination. The default ``fitted.h5mu`` holds the original measurements and the results together. See :doc:`reading fitted outputs <../fitted_export_pilot>` to retrieve them.

Options and defaults
--------------------

This listing comes from the installed version's ``perturbo --help``.

.. perturbo-help::
