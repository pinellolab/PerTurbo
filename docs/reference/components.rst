Fitting components and result tables
====================================

These functions expose the individual steps used by a screen analysis. Keep cell, element and gene identifiers with their arrays when handling the steps separately.

Prepare counts and fit both stages
----------------------------------

.. autofunction:: perturbo.load_controls

.. autofunction:: perturbo.load_analysis_cells

.. autoclass:: perturbo.core.CovariateTransformState

.. autoclass:: perturbo.PerTurboData

.. autoclass:: perturbo.SVIConfig

.. autofunction:: perturbo.fit_control

.. autoclass:: perturbo.ControlFit

.. autoclass:: perturbo.BaselinePosteriorSummary

.. autofunction:: perturbo.fit_perturbation_effects

.. autoclass:: perturbo.BetaFit

Read effect estimates
---------------------

.. autofunction:: perturbo.summarize_betas

.. autofunction:: perturbo.build_guide_effects_df

.. autoclass:: perturbo.PosteriorMedians

Directory bundle helpers
------------------------

.. autofunction:: perturbo.load_fit_bundle

.. autofunction:: perturbo.save_fit_bundle
