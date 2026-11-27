Simulation and count models
===========================

For worked examples of power calculations, see the :doc:`enhancer locus <../tutorials/gasperini_power>` and :doc:`TF regulon <../tutorials/replogle_power>` tutorials.

Generate and save counts
------------------------

.. autofunction:: perturbo.simulation.sample_predictive_counts

.. autofunction:: perturbo.simulate_data_from_trained_model

.. autofunction:: perturbo.save_simulated_mudata

Count distributions
-------------------

.. autoclass:: perturbo.CensoredNegativeBinomial

.. autoclass:: perturbo.LogNormalNegativeBinomial

Baseline model callables
------------------------

These NumPyro callables share the baseline model implementation. Each fixes the count likelihood for direct use in custom inference.

``perturbo.NegBinModel``
    Negative binomial.

``perturbo.CensoredNegativeBinomialModel``
    Negative binomial with right censoring.

``perturbo.LogNormalNegativeBinomialModel``
    Negative binomial with log-normal variation.

``perturbo.MixtureNegativeBinomialModel``
    Mixture of two negative binomials.
