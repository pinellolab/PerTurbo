"""Numerical (not mocked) all-cells CRT parity across outer gene schedules."""
import numpy as np
import pandas as pd

from perturbo import cli
from tests.test_crt_cli_all_cells import _write_high_moi_screen


def test_independent_gene_schedule_preserves_real_crt(tmp_path, monkeypatch):
    source = tmp_path / "screen.h5mu"
    _write_high_moi_screen(source, n_cells=500, n_genes=7, n_elements=6, seed=17)
    monkeypatch.setattr(cli, "_device_memory_bytes", lambda _: 16 * 1024**3)

    def run(name, stage_width=None, crt_width=5):
        out = tmp_path / name
        argv = [
            "--input", str(source), "--out-dir", str(out),
            "--modality-key", "gene", "--perturbation-modality-key", "guide",
            "--perturbation-element-varm-key", "guide_intended_target_pairs",
            "--perturbation-element-names-uns-key", "intended_targets",
            "--library-size-key", "total_umis", "--size-factor-mode", "observed",
            "--likelihood", "nb", "--num-steps-control", "100", "--num-steps-betas", "1", "--backed",
            "--no-save-model-params", "--crt", "--crt-only", "--crt-pool", "all-cells",
            "--crt-mechanism", "propensity", "--crt-tail-families", "saddlepoint",
            "--crt-saddlepoint-only", "--crt-allow-unconverged-baseline",
            "--crt-gene-chunk-size", str(crt_width),
        ]
        if stage_width is not None:
            argv.extend(["--gene-chunk-size", str(stage_width)])
        cli.main(argv)
        return pd.read_parquet(out / "element_effects.parquet").sort_values(
            ["element", "gene"]
        ).reset_index(drop=True)

    reference = run("reference")
    for name, stage_width, crt_width in [("independent", 2, 5), ("narrow", 2, 3)]:
        frame = run(name, stage_width, crt_width)
        pd.testing.assert_frame_equal(frame[["element", "gene"]], reference[["element", "gene"]])
        for field in ("crt_saddlepoint_p_value", "crt_saddlepoint_q_value",
                      "crt_saddlepoint_log_p_value"):
            np.testing.assert_allclose(frame[field], reference[field], rtol=1e-5, atol=1e-10,
                                       equal_nan=True, err_msg=field)
        for field in ("crt_tail_failure_reason", "crt_used_chernoff", "crt_low_information",
                      "crt_observed_nonzero"):
            np.testing.assert_array_equal(frame[field], reference[field], err_msg=field)
        np.testing.assert_array_equal(frame.crt_saddlepoint_q_value < .05,
                                      reference.crt_saddlepoint_q_value < .05)
