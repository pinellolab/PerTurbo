[![Tests][badge-tests]][link-tests]

# 🏎️ PerTurbo: Fast analysis of single-cell perturbation studies

**PerTurbo** is a fast statistical package for analyzing perturbation phenotypes from single cell CRISPR screens.

For reproducibility, see []

<!-- [![Documentation][badge-docs]][link-docs] -->

[badge-tests]: https://img.shields.io/github/actions/workflow/status/pinellolab/perturbo/test.yaml?branch=main
[link-tests]: https://github.com/pinellolab/perturbo/actions/workflows/test.yml
<!-- [badge-docs]: https://img.shields.io/readthedocs/perturbo  -->

<!-- ## Getting started -->

<!-- Documentation suspended while private, check back soon :) -->

<!-- Please refer to the [documentation][link-docs]. In particular, the -->

<!-- -   [API documentation][link-api]. -->

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniforge](https://github.com/conda-forge/miniforge).

Official PyPI and conda releases are still in the works, but in the meantime you can install the developer version with one of the following options:

1. **pip (easiest, recommended for end users)**
The simplest way to install PerTurbo is using pip (ideally inside an isolated conda environment or virtualenv). Simply clone the git repo to your machine, enter the folder using your terminal, and then use `pip install -e .` to install the project and its dependencies.

2. **Hatch (slightly more complex, recommended for developers)**
Hatch is a project/dependency managment tool which can create project-specific virtual environments, similar to Poetry or pip+virtualenv. First, install [Hatch](https://hatch.pypa.io/).  You may want to further configure where Hatch installs virtual environments, [see documentation](https://hatch.pypa.io/latest/config/hatch/#environments).
Clone this git repo to your machine, enter the directory, and then run `hatch env create dev.`
This will create a Hatch virtual environment in your configured directory with all the necessary project dependencies, plus Jupyter and some other development essentials for running the test notebooks in this repo.

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
[link-pypi]: https://pypi.org/project/PerTurbo
