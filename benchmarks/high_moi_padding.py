"""Compare the original padding-to-zero product with this checkout's product.

Run from the checkout with PYTHONPATH=src. Uses the default JAX device and
synchronizes every timing. See docs/high_moi_padding_handoff.md for GPU commands.
This standalone handoff benchmark never runs during import or package tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy.sparse as sp

from perturbo.sparse_design import IndexedDesignMatrix, design_matrix_product


def original_product(design, coefficients):
    """The pre-fix indexed product, retained only as a benchmark control."""
    indices = jnp.maximum(design.indices, 0)
    gathered = coefficients[indices]
    weights = jnp.where(design.indices >= 0, design.values, 0).astype(coefficients.dtype)
    return jnp.sum(gathered * weights[..., None], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=8000)
    parser.add_argument("--elements", type=int, default=5000)
    parser.add_argument("--active", type=int, default=30)
    parser.add_argument("--width", type=int, default=104)
    parser.add_argument("--genes", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (0 < args.active <= args.width <= args.elements) or min(args.cells, args.genes, args.repeats) <= 0:
        parser.error("Require 0 < active <= width <= elements and positive cells, genes, and repeats.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(93)
    indices = np.full((args.cells, args.width), -1, np.int32)
    # Duplicate sampled destinations are deliberately summed in the reference.
    # This synthetic fixture isolates the operation; it is not a biological fit.
    indices[:, :args.active] = rng.integers(args.elements, size=(args.cells, args.active))
    indices[0] = rng.choice(args.elements, size=args.width, replace=False)
    valid = indices >= 0
    values = valid.astype(np.float32)
    beta = rng.normal(size=(args.elements, args.genes)).astype(np.float32)
    cotangent = rng.normal(size=(args.cells, args.genes)).astype(np.float32)
    rows = np.broadcast_to(np.arange(args.cells)[:, None], indices.shape)[valid]
    matrix = sp.csr_matrix((values[valid], (rows, indices[valid])), shape=(args.cells, args.elements))
    reference_forward = matrix.astype(np.float64) @ beta.astype(np.float64)
    reference_adjoint = matrix.astype(np.float64).T @ cotangent.astype(np.float64)
    del rows, matrix
    inputs = tuple(jnp.asarray(v) for v in (beta, cotangent, indices, values))
    jax.block_until_ready(inputs)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    report = {
        "revision": revision, "working_tree_dirty": bool(dirty), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
        "cells": args.cells, "elements": args.elements, "genes": args.genes,
        "width": args.width, "active_slots": int(valid.sum()), "padding_slots": int((~valid).sum()),
        "real_element_zero_slots": int((indices == 0).sum()), "results": {},
    }
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)
    for name, product in [("original", original_product), ("candidate", design_matrix_product)]:
        def forward(b, ct, ix, weights):
            return product(IndexedDesignMatrix(ix, weights, args.elements), b)

        def adjoint(b, ct, ix, weights):
            return jax.vjp(lambda coefficients: forward(coefficients, ct, ix, weights), b)[1](ct)[0]

        report["results"][name] = {}
        for phase, operation, reference in [
            ("forward", forward, reference_forward), ("adjoint", adjoint, reference_adjoint),
        ]:
            start = time.perf_counter()
            lowered = jax.jit(operation).lower(*inputs)
            executable = lowered.compile()
            compile_seconds = time.perf_counter() - start
            (args.out_dir / f"{name}-{phase}.stablehlo.txt").write_text(str(lowered.compiler_ir()))
            (args.out_dir / f"{name}-{phase}.hlo.txt").write_text(executable.as_text())
            jax.block_until_ready(executable(*inputs))
            timings = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                observed = jax.block_until_ready(executable(*inputs))
                timings.append(time.perf_counter() - start)
            difference = np.asarray(observed, dtype=np.float64) - reference
            error = float(np.linalg.norm(difference) / np.linalg.norm(reference))
            if not np.isfinite(error) or error >= 3e-6:
                raise AssertionError(f"{name} {phase}: relative L2 error {error} exceeds 3e-6")
            memory = executable.memory_analysis()
            result = {
                "compile_seconds": compile_seconds, "median_seconds": float(np.median(timings)),
                "timings": timings, "temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
                "relative_l2_error_vs_float64": error, "max_abs_error": float(np.abs(difference).max()),
            }
            report["results"][name][phase] = result
            print(name, phase, json.dumps(result), flush=True)
            (args.out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
