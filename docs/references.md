# References

## Statistical background

- [SCEPTRE improves calibration and sensitivity in single-cell CRISPR screen analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC8686614/)
  introduces conditional randomization for these screens and explains robustness
  to expression-model misspecification.
- [Robust differential expression testing for single-cell CRISPR screens at low multiplicity of infection](https://pmc.ncbi.nlm.nih.gov/articles/PMC11100084/)
  discusses the low-MOI setting, sparsity, confounding, and calibration.

These are background references, not a claim that every PerTurbo configuration
implements an identical test. See [Method](method.md) for the implemented pools
and tail approximations. Record the PerTurbo version, source tag, and full
analysis settings when reporting results.

## Software documentation

- [scverse ecosystem-package criteria](https://github.com/scverse/ecosystem-packages)
- [scvi-tools documentation](https://docs.scvi-tools.org/en/stable/)
- [pertpy documentation](https://pertpy.readthedocs.io/en/stable/)
- [JAX installation and device support](https://docs.jax.dev/en/latest/installation.html)

The scverse project is described by {cite:p}`Virshup_2023`.

```{bibliography}
:cited:
```
