"""Exercise public API defaults against the actual CLI parser."""
import argparse

import pytest

from perturbo import api, cli
from perturbo.crt import DEFAULT_CRT_MIN_INFORMATIVE_CELLS


def _parsed(monkeypatch, argv):
    captured = []
    parse = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def capture(self, args=None, namespace=None):
        captured.append(parse(self, args, namespace))
        raise Parsed

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(Parsed):
            cli.main(argv)
    return captured[0]


@pytest.mark.parametrize("options", [{}, {
    "step_size": 0.003,
    "crt_gene_chunk_size": 73,
    "crt_min_informative_cells": 12.0,
    "crt_all_cells_batch_support": False,
}])
def test_api_crt_defaults_and_overrides_reach_cli(monkeypatch, options):
    calls = []
    monkeypatch.setattr(api, "main", lambda argv: calls.append(argv))
    api.fit_from_path("screen.h5mu", "out", crt=True, **options)
    actual = _parsed(monkeypatch, calls[0])
    expected = _parsed(monkeypatch, ["--input", "screen.h5mu", "--out-dir", "out"])
    for name in ("step_size", "crt_gene_chunk_size", "crt_min_informative_cells", "crt_all_cells_batch_support"):
        assert getattr(actual, name) == options.get(name, getattr(expected, name))
    if not options:
        assert actual.step_size == 0.01
        assert actual.crt_gene_chunk_size == 500
        assert actual.crt_min_informative_cells == DEFAULT_CRT_MIN_INFORMATIVE_CELLS
        assert actual.crt_all_cells_batch_support is True


def test_api_does_not_enable_crt_for_annotation_options(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "main", lambda argv: calls.append(argv))
    api.fit_from_path("screen.h5mu", "out", crt=False, crt_min_informative_cells=12,
                      crt_all_cells_batch_support=False)
    assert "--crt" not in calls[0]
    assert "--crt-min-informative-cells" not in calls[0]
    assert "--no-crt-all-cells-batch-support" not in calls[0]
