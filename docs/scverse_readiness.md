# scverse ecosystem readiness

This is a maintainer-facing assessment of PerTurbo against the current scverse
ecosystem package checklist. It covers the `v2-port` candidate at commit
`da6751a` (tagged `v2.0.0rc6`) and externally visible services as checked on
**2026-09-13**. It is not a submission to scverse.

## Decision

PerTurbo is technically aligned with the scverse ecosystem: it uses AnnData and
MuData as its user-facing data containers, has an MIT license, versioned source,
a substantial automated test suite, and packaging built around standard Python
tools. It is **not ready for an ecosystem listing today** because three mandatory
conditions are not currently met in published infrastructure:

1. There is no installable release in PyPI, conda-forge, or bioconda.
2. The public test workflow is disabled and therefore is not running on current
   pushes or pull requests.
3. The public API reference has not been published, and the configured Read the
   Docs project does not exist publicly. This branch adds the technical reference
   and a reproducible CPU tutorial; hosting remains a release task.

There is also release-version drift that should be resolved before publication:
the checked-out commit is tagged `v2.0.0rc6`, while
[`pyproject.toml`](../pyproject.toml) declares `2.0.0rc5`, and the newest GitHub
Release is `v2.0.0rc1`. That `v2.0.0rc1` release's publish workflow failed at the
PyPI upload step. GitHub still identifies `main` as the default branch, while
the assessed candidate is on `v2-port`; the listing should be assessed from the
final default branch rather than from a release branch alone.

## Three different goals

These goals should not be conflated:

- **scverse interoperability** means using scverse data structures and fitting
  naturally into single-cell workflows. PerTurbo already does this through
  AnnData/MuData input, backed reads, metadata registration, and result bundles.
- **Ecosystem listing** means satisfying the mandatory checklist and submitting
  a metadata pull request to the
  [scverse ecosystem-packages registry](https://github.com/scverse/ecosystem-packages).
  A listing signals that minimum requirements are met; scverse explicitly says
  that it is not an endorsement or an in-depth scientific review.
- **Core package status** is a later governance and shared-maintenance decision.
  scverse describes core tools as foundational packages placed under shared
  maintenance in the scverse GitHub organization. Its governance requires every
  core package to have a core-developer maintainer, with new core developers
  admitted through nomination and a core-team vote. Passing the ecosystem
  checklist does not confer core status. See the official
  [mission](https://scverse.org/about/mission/) and
  [governance roles](https://scverse.org/about/roles/).

The appropriate near-term goal for PerTurbo is an **ecosystem listing**. Core
membership can be discussed only if scverse and the maintainers later want
shared stewardship of the project.

## Mandatory requirements

The wording and interpretation below follow the official
[ecosystem checklist](https://github.com/scverse/ecosystem-packages#what-are-the-requirements-for-an-ecosystem-package).

| Requirement | Status | PerTurbo evidence | Work needed |
|---|---|---|---|
| Public code under an OSI-approved license | Ready | [`LICENSE`](../LICENSE) is MIT; GitHub also recognizes the repository license as MIT. | None for listing. |
| Versioned releases | Ready, with release hygiene to fix | The repository has historical releases and tags. GitHub exposes releases through `v2.0.0rc1`, and the candidate is tagged `v2.0.0rc6`. | Synchronize the package version, tag, and GitHub Release. Publish releases from one immutable tagged commit. |
| Installable from a standard registry | **Blocked** | [`https://pypi.org/pypi/perturbo/json`](https://pypi.org/pypi/perturbo/json), [conda-forge](https://api.anaconda.org/package/conda-forge/perturbo), and [bioconda](https://api.anaconda.org/package/bioconda/perturbo) all returned 404 on 2026-09-13. | Publish a tested wheel and source distribution to at least one standard registry. PyPI is the direct route because trusted publishing is already configured. |
| Automated tests cover essential functions and reasonable inputs | Ready in repository; reviewer judgment remains | There are focused tests for the public API and CLI, fitting, persistence, sparse/backed input, low- and high-MOI designs, CRT, chunking, simulation, and likelihoods under [the test suite](https://github.com/pinellolab/PerTurbo/tree/v2-port/tests). The workflow tests Python 3.11 and 3.12 plus the isolated legacy extra. | Before submission, map each exported public function/class to at least one smoke or correctness test. Add missing cases rather than relying on file count alone. |
| CI runs tests on every push or pull request | **Blocked externally** | [`.github/workflows/test.yaml`](../.github/workflows/test.yaml) defines `push` and `pull_request` triggers for `main` and `v2-port`, but the [public workflow](https://github.com/pinellolab/PerTurbo/actions/workflows/test.yaml) was `disabled_inactivity` on 2026-09-13. Its latest visible runs were scheduled runs on `main` in May 2026, not current release-candidate pushes. Build and container workflows are active, but they do not replace the test suite. | Re-enable the Test workflow, run it for the release-candidate branch and its merge PR, and require it in branch protection. To satisfy “each push” literally, remove the `push.branches` restriction or document and enforce a policy in which every change reaches an integration branch through a tested PR. Retain links to successful runs for the submission. |
| API documentation via a website or README | **Blocked** | At the audited base commit, the API page was a short list of selected names with no signatures or parameter documentation. This documentation branch adds a generated [module reference](api.md), contracts in docstrings, and examples covering the top-level exports. The checklist explicitly defines API docs as an overview of all public functions with their parameters. Sphinx autodoc is configured, but `https://perturbo.readthedocs.io/` and the Read the Docs project API returned 404 on 2026-09-13. | Generate reference pages for every supported public function and class, including signatures, parameters, returns, errors, and brief examples. Provision Read the Docs, make a warning-free build, and verify its `objects.inv`. |
| Uses scverse data structures where appropriate | Ready | AnnData and MuData are direct runtime dependencies in [`pyproject.toml`](../pyproject.toml). The CLI consumes AnnData/MuData files, `setup_mudata` records the schema, and the API accepts MuData workflows. | Document the expected modalities, layers, `obs`/`var` fields, sparse/backed behavior, and where PerTurbo writes results. |
| Submitter is an author or maintainer and agrees to listing | Pending human attestation | The package metadata identifies an author/maintainer. This checkbox is completed by the person opening the registry pull request. | Have an author or maintainer submit and attest. |
| Agreement to the scverse Code of Conduct on scverse channels | Pending human attestation | This is an agreement in the registry pull request, not something CI can verify. | The submitter should read and accept the [scverse Code of Conduct](https://scverse.org/about/code_of_conduct/). |

The test-suite assessment concerns breadth visible in the repository; it is not
a coverage percentage claim. [`.codecov.yaml`](../.codecov.yaml) currently sets
a 1% project target and disables patch coverage, while the test workflow does
not upload coverage. Raising that threshold and publishing coverage would make
the submission easier to review, although the ecosystem checklist does not
mandate a particular percentage.

## Recommended items

These improve maintainability and discoverability but are not mandatory listing
criteria.

| Recommendation | Current state | Practical next step |
|---|---|---|
| Getting-started tutorials or vignettes | A reproducible [CPU tutorial](quickstart.md) is included on this documentation branch; public hosting is pending. | Add one small CPU tutorial from AnnData/MuData setup through fitting and result interpretation, with downloadable example data or deterministic data-generation code. |
| scverse cookiecutter | Already satisfied in substance. [`.cruft.json`](../.cruft.json) records `scverse/cookiecutter-scverse` as the template. | Update selected template-derived infrastructure when useful. A migration or wholesale regeneration is unnecessary. |
| Announcement on scverse channels | Not done as part of this assessment. | Announce after users can install the exact documented release and access stable docs. |
| Invitation to the scverse GitHub organization | Optional and maintainer-specific. | Request it in the ecosystem submission only if desired. It is not core-package membership. |

The separate [scverse repo-health](https://github.com/scverse/repo-health)
checks describe infrastructure expected of repositories maintained within the
scverse organization. They are useful hardening targets, but they are not the
mandatory entry checklist for an independently maintained ecosystem package.

## Packaging and repository health

[`pyproject.toml`](../pyproject.toml) uses Hatchling with a conventional `src`
layout, declares the `perturbo` console script, supplies project URLs and
maintainer contact information, and separates documentation, CUDA, legacy, test,
and development dependencies. The build workflow constructs both distribution
formats, runs strict Twine metadata checks, installs the wheel into a clean
environment, and invokes `perturbo --help`. Those are good foundations for a
registry release.

The normal dependency set contains AnnData and MuData directly. Heavy PyTorch,
Pyro, and scvi-tools dependencies are isolated in the `legacy` extra and a CI
check verifies that a normal import does not load them. This boundary is helpful
for scverse users and should be retained.

Additional improvements are useful but do not block an ecosystem application:

- Add a documentation build to pull-request CI so a local Sphinx configuration
  cannot drift from the hosted site.
- Publish meaningful coverage reporting; the present 1% Codecov target provides
  little evidence even though the focused test suite itself is substantial.
- Add `CITATION.cff` with the preferred software citation and release DOI when
  one exists. The ecosystem checklist does not require this file.
- Refresh template-derived pre-commit hook versions selectively. The existing
  cookiecutter provenance means no template migration is needed.

## Release and interoperability issues to resolve

### Make one release identity authoritative

At the audited commit, three public identities disagree:

- Git tag and checkout: `v2.0.0rc6` at `da6751a`
- package metadata: `2.0.0rc5`
- newest GitHub Release: `v2.0.0rc1`

In addition, the public repository's default branch is `main`, whereas this
assessment's code is on `v2-port`. Merge the finished candidate into the
intended default branch, or deliberately change the default branch, before
asking reviewers to verify source, documentation, and CI.

The public [release workflow](https://github.com/pinellolab/PerTurbo/actions/runs/34542734127)
built `v2.0.0rc1` successfully but failed in
`pypa/gh-action-pypi-publish`. Before cutting another release, update the
package metadata to the intended version, build the tag locally and in CI,
correct the PyPI trusted-publisher configuration, and publish a GitHub Release
from that exact tag. Verify the result from a clean environment with the
documented prerelease command. Until then, documentation should use a source or
container installation that actually exists rather than say that
`pip install perturbo` works.

### Publish complete API documentation

The Sphinx setup already enables autodoc, autosummary, NumPy-style docstrings,
and intersphinx, so the missing reference can be added without changing the
documentation stack. A useful API reference should:

- cover every supported name in `perturbo.__all__`, grouped by setup, fitting,
  models, results, persistence, distributions, and simulation;
- show full signatures and document parameters, return types, raised errors,
  supported input shapes/dtypes, and sparse/backed constraints;
- identify which interfaces are stable and which lower-level dataclasses are
  primarily for advanced workflows;
- cross-link a short end-to-end example and the artifact/result schema; and
- build with warnings treated as errors, as the existing
  [`.readthedocs.yaml`](../.readthedocs.yaml) requests.

The Python wrapper also needs a correctness pass before being described as
“CLI-equivalent.” In [`src/perturbo/api.py`](../src/perturbo/api.py),
`fit_from_path` defaults to `step_size=0.003` and `crt=False`; the CLI defaults
to `step_size=0.01`. The facade currently omits CRT arguments when `crt=False`,
so CLI automatic CRT selection can still enable testing. Several other false
CRT booleans likewise omit flags and leave CLI defaults active. This branch
documents the actual forwarding behavior; correcting the translation belongs
in a separate, tested runtime change. Use CLI `--no-crt` to explicitly disable
testing in the meantime.

### Avoid surprising other JAX-based scverse tools

Importing `perturbo` currently calls
`jax.config.update("jax_enable_x64", True)` in
[`src/perturbo/__init__.py`](../src/perturbo/__init__.py). JAX configuration is
process-global, so this changes dtype behavior for scvi-tools, other JAX models,
and user code sharing the Python process. This is not a formal ecosystem
listing blocker, but it is an interoperability risk for notebooks and pipelines.
Prefer enabling the precision mode at a controlled execution boundary before
JAX initialization, or expose/document a supported initialization method and
state clearly when a subprocess is required.

PerTurbo does **not** need to depend on scvi-tools or pertpy to qualify. The
official checklist requires appropriate use of AnnData, MuData, or SpatialData;
it does not require one analysis package to depend on another. Keeping
`scvi-tools` confined to the historical `legacy` optional extra is reasonable,
and no pertpy dependency is needed. Interoperability should be demonstrated
through standard containers, metadata conventions, and documented outputs.

## Prioritized action plan

1. **Restore mandatory CI.** Re-enable the Test workflow, confirm successful
   Python 3.11/3.12 runs on the candidate branch and merge PR, and make that
   workflow a required check.
2. **Repair and publish the release.** Synchronize the version declaration and
   tag, fix trusted publishing, publish the wheel and source distribution, and
   verify the registry metadata and a clean install. Do not reuse or move an
   existing public tag.
3. **Complete and publish the API reference.** Generate documentation for all
   public symbols and their parameters, reconcile the Python/CLI defaults,
   create the Read the Docs project, and verify a stable documentation URL plus
   intersphinx inventory.
4. **Add a reproducible first analysis.** Provide one small CPU tutorial showing
   data registration, fitting, output files, statistical interpretation, and
   common input errors.
5. **Document in-process JAX behavior.** Remove or constrain the import-time
   global setting if feasible; until then, make the process-wide effect visible
   in installation and API guidance.
6. **Prepare the registry entry.** Once steps 1–3 are externally verifiable,
   add `packages/perturbo/meta.yaml` in a fork of the ecosystem registry and
   copy its current checklist into the pull-request description. A sensible
   initial classification is `primary_category: scRNA-seq`, `category:
   ecosystem`, with controlled tags `scRNA-seq`, `differential expression`,
   `perturbation`, `probabilistic modeling`, and `GPU acceleration`. Validate
   these values against the schema again at submission time.
7. **Submit as a maintainer.** Complete the authorship/listing and Code of
   Conduct attestations, then announce the package after the listing and install
   instructions are live.

After publication, the registry metadata can use this shape. It intentionally
omits a `version`: the registry reads that from the package index and refreshes
it daily.

```yaml
name: PerTurbo
blurb: Bayesian perturbation-effect inference for Perturb-seq
description: |
  PerTurbo estimates perturbation effects from single-cell CRISPR screens
  using a NumPyro/JAX model and conditional randomization tests. It reads
  AnnData and MuData inputs and supports sparse, backed, low- and high-MOI data.
project_home: https://github.com/pinellolab/PerTurbo
documentation_home: https://perturbo.readthedocs.io/
install:
  pypi: perturbo
primary_category: scRNA-seq
tags:
  - scRNA-seq
  - differential expression
  - perturbation
  - probabilistic modeling
  - GPU acceleration
license: MIT
language: Python
contact:
  - <maintainer-github-handle>
category: ecosystem
```

Use the current schema when submitting: required fields and controlled terms
can change, and the documentation and PyPI URLs in this sketch must resolve
before the entry is proposed.

## Source and verification record

All links in this section were accessed on **2026-09-13**.

- [Official ecosystem registry, checklist, and submission instructions](https://github.com/scverse/ecosystem-packages)
- [Official ecosystem metadata schema](https://github.com/scverse/ecosystem-packages/blob/main/scripts/src/ecosystem_scripts/schema.json)
- [scverse mission and definition of core tools](https://scverse.org/about/mission/)
- [scverse roles and core-developer governance](https://scverse.org/about/roles/)
- [scverse Code of Conduct](https://scverse.org/about/code_of_conduct/)
- [scverse cookiecutter](https://github.com/scverse/cookiecutter-scverse)
- [scverse repository-health checks](https://github.com/scverse/repo-health)
- [PerTurbo GitHub Releases](https://github.com/pinellolab/PerTurbo/releases)
- [PerTurbo GitHub Actions](https://github.com/pinellolab/PerTurbo/actions)
- [PyPI JSON endpoint for `perturbo`](https://pypi.org/pypi/perturbo/json)
- [conda-forge package endpoint for `perturbo`](https://api.anaconda.org/package/conda-forge/perturbo)
- [bioconda package endpoint for `perturbo`](https://api.anaconda.org/package/bioconda/perturbo)
- [configured Read the Docs URL](https://perturbo.readthedocs.io/)
- [Read the Docs project API](https://readthedocs.org/api/v3/projects/perturbo/)

External statuses can change independently of the repository. Recheck the
registry, workflow, release, documentation, and ecosystem listing immediately
before submission. No `perturbo` entry existed in the official ecosystem
registry tree on the access date.
