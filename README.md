[![Tests][badge-tests]][link-tests]

# 🏎️ PerTurbo: Fast analysis of single-cell perturbation studies

**PerTurbo** estimates the effects of perturbations (e.g. CRISPRi, CRISPR-Cas9, ORF overexpression) on single cell phenotypes using variational inference.

Support for several types of experimental designs is planned:
- [x]  CRISPRi + RNA-seq (Perturb-Seq)
- [ ] CRISPR-Cas9 + Imaging (Optical Pooled Screens)


<!-- [![Documentation][badge-docs]][link-docs] -->

[badge-tests]: https://img.shields.io/github/actions/workflow/status/pinellolab/perturbo/test.yaml?branch=main
[link-tests]: https://github.com/pinellolab/perturbo/actions/workflows/test.yml
<!-- [badge-docs]: https://img.shields.io/readthedocs/perturbo  -->

<!-- ## Getting started -->

<!-- Documentation suspended while private, check back soon :) -->

<!-- Please refer to the [documentation][link-docs]. In particular, the -->

<!-- -   [API documentation][link-api]. -->

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

PyPI and conda releases are still in the works, but in the meantime you can install the development version directly from GitHub using pip:

```bash
pip install git+https://github.com/pinellolab/PerTurbo.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out to the author [here](https://loganblaine.com).
<!-- For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse]. -->
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/pinellolab/PerTurbo/issues
[changelog]: https://perturbo.readthedocs.io/latest/changelog.html
[link-docs]: https://perturbo.readthedocs.io
[link-api]: https://perturbo.readthedocs.io/latest/api.html
