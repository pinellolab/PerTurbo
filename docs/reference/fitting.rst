Fitting and restoring a screen
==============================

Use ``fit_from_path`` for a complete screen analysis. Use the model class to fit effects in memory or explore an existing fit.

Run the file-based workflow
---------------------------

.. autofunction:: perturbo.fit_from_path

Register an in-memory MuData experiment
---------------------------------------

.. autofunction:: perturbo.setup_mudata

.. autofunction:: perturbo.get_mudata_setup

.. autoclass:: perturbo.MuDataSetup

Train, load and inspect a model
-------------------------------

.. autoclass:: perturbo.PerTurboModel
   :members: load, train, save, setup_mudata, view_anndata_setup, get_element_effects, get_guide_effects, posterior_parameter_table, posterior_medians, guide_efficacy, gene_names, element_names, guide_names
