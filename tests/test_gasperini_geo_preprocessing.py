from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbo.preprocessing.gasperini_geo import (
    _build_gasperini_element_gene_histogram_frames,
    build_gasperini_geo_mudata,
    summarize_gasperini_geo_inputs,
)


def test_build_gasperini_geo_mudata_at_scale(tmp_path: Path) -> None:
    _write_matrix_market(
        tmp_path / "GSE120861_at_scale_screen.exprs.mtx.gz",
        "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 3\n2 2 5\n",
        compress=True,
    )
    _write_gzip_lines(
        tmp_path / "GSE120861_at_scale_screen.genes.txt.gz",
        ["ENSG000001", "ENSG000002"],
    )
    _write_gzip_lines(
        tmp_path / "GSE120861_at_scale_screen.phenoData.txt.gz",
        [
            "sampleA cellA 10 1.0 groupA groupA AAAAAAAAAAAAAAAAAAAA 4 1 1.0 1 dirA fileA idA batch1 chipA lane1 0.01",
            "sampleA cellB 20 2.0 groupB groupB TTTTTTTTTTTTTTTTTTTT_CCCCCCCCCCCCCCCCCCCC 5 2 1.0 2 dirA fileA idB batch1 chipA lane1 0.02",
        ],
    )
    _write_gzip_lines(
        tmp_path / "GSE120861_grna_groups.at_scale.txt.gz",
        [
            "groupA\tAAAAAAAAAAAAAAAAAAAA",
            "bassik_mch\tTTTTTTTTTTTTTTTTTTTT",
            "groupB\tCCCCCCCCCCCCCCCCCCCC",
        ],
    )
    _write_lines(
        tmp_path / "GSE120861_gene_gRNAgroup_pair_table.at_scale.txt",
        [
            "\t".join(
                [
                    "gRNAgroup.chr",
                    "gRNAgroup.start",
                    "gRNAgroup.stop",
                    "gRNAgroup",
                    "general_group",
                    "chr.targetgene",
                    "start.targetgene",
                    "stop.targetgene",
                    "ENSG.targetgene",
                    "targetgene_short_name",
                    "strand.targetgene",
                    "pairs",
                ]
            ),
            "\t".join(
                [
                    "chr1",
                    "100",
                    "110",
                    "groupA",
                    "candidate",
                    "chr1",
                    "1000",
                    "1010",
                    "ENSG000001",
                    "GENEA",
                    "+",
                    "GENEA:groupA",
                ]
            ),
            "\t".join(
                [
                    "NTC",
                    "NTC",
                    "NTC",
                    "bassik_mch",
                    "NTC",
                    "chr2",
                    "2000",
                    "2010",
                    "ENSG000002",
                    "GENEB",
                    "-",
                    "GENEB:bassik_mch",
                ]
            ),
            "\t".join(
                [
                    "chr2",
                    "200",
                    "210",
                    "groupB",
                    "candidate",
                    "chr2",
                    "2000",
                    "2010",
                    "ENSG000002",
                    "GENEB",
                    "-",
                    "GENEB:groupB",
                ]
            ),
        ],
    )

    summary = summarize_gasperini_geo_inputs(tmp_path, "at_scale")
    assert summary["matrix_gene_rows"] == 2
    assert summary["matrix_cell_columns"] == 2
    assert summary["expression_matrix_is_gzip"] is True
    assert summary["guide_groups_missing_pair_metadata"] == []
    assert summary["unknown_barcode_guides"] == []

    mdata = build_gasperini_geo_mudata(tmp_path, "at_scale")
    assert mdata["gene"].shape == (2, 2)
    assert mdata["guide"].shape == (2, 3)
    assert mdata["gene"].X.dtype == np.uint8
    assert mdata["gene"].obs.columns.tolist() == [
        "sample",
        "total_umis",
        "size_factor",
        "read_count",
        "umi_count",
        "proportion",
        "guide_count",
        "id",
        "prep_batch",
        "within_batch_chip",
        "within_chip_lane",
        "percent_mito",
    ]
    assert mdata["guide"].var.loc["TTTTTTTTTTTTTTTTTTTT", "is_non_targeting"]
    assert np.array_equal(np.asarray(mdata["guide"].X.sum(axis=1)).ravel(), np.array([1.0, 2.0]))


def test_build_gasperini_geo_mudata_pilot_with_plain_text_matrix(tmp_path: Path) -> None:
    _write_matrix_market(
        tmp_path / "GSE120861_pilot_highmoi_screen.exprs.mtx.gz",
        "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 7\n2 2 9\n",
        compress=False,
    )
    _write_lines(
        tmp_path / "GSE120861_pilot_highmoi_screen.genes.txt",
        ["ENSGP00001", "ENSGP00002"],
    )
    _write_lines(
        tmp_path / "GSE120861_pilot_highmoi_screen.phenoData.txt",
        [
            "sampleP cellP1 11 1.1 pilotA pilotA GGGGGGGGGGGGGGGGGGGG 6 1 1.0 1 dirP fileP K1000 FALSE FALSE",
            "sampleP cellP2 22 2.2 pilotB pilotB NNNNNNNNNNNNNNNNNNNN_AAAAAAAAAAAAAAAAAAAA 7 2 1.0 2 dirP fileP K1000 FALSE TRUE",
        ],
    )
    _write_lines(
        tmp_path / "GSE120861_grna_groups.pilot.txt",
        [
            "random_1\tGGGGGGGGGGGGGGGGGGGG",
            "pilotB\tAAAAAAAAAAAAAAAAAAAA",
        ],
    )
    _write_lines(
        tmp_path / "GSE120861_gene_gRNAgroup_pair_table.pilot.txt",
        [
            "\t".join(
                [
                    "gRNAgroup.chr",
                    "gRNAgroup.start",
                    "gRNAgroup.stop",
                    "gRNAgroup",
                    "general_group",
                    "chr.targetgene",
                    "start.targetgene",
                    "stop.targetgene",
                    "ENSG.targetgene",
                    "targetgene_short_name",
                    "type.targetgene",
                    "strand.targetgene",
                    "pairs",
                ]
            ),
            "\t".join(
                [
                    "NTC",
                    "NTC",
                    "NTC",
                    "random_1",
                    "NTC",
                    "chr7",
                    "7000",
                    "7010",
                    "ENSGP00001",
                    "GENEP1",
                    "protein_coding",
                    "+",
                    "GENEP1:random_1",
                ]
            ),
            "\t".join(
                [
                    "chr8",
                    "800",
                    "810",
                    "pilotB",
                    "candidate",
                    "chr8",
                    "8000",
                    "8010",
                    "ENSGP00002",
                    "GENEP2",
                    "protein_coding",
                    "-",
                    "GENEP2:pilotB",
                ]
            ),
        ],
    )

    summary = summarize_gasperini_geo_inputs(tmp_path, "pilot")
    assert summary["matrix_gene_rows"] == 2
    assert summary["matrix_cell_columns"] == 2
    assert summary["expression_matrix_is_gzip"] is False
    assert summary["guide_groups_missing_pair_metadata"] == []
    assert summary["unknown_barcode_guides"] == ["NNNNNNNNNNNNNNNNNNNN"]

    with pytest.raises(ValueError, match="barcode guides absent"):
        build_gasperini_geo_mudata(tmp_path, "pilot")


def test_build_gasperini_geo_mudata_respects_requested_gene_order(tmp_path: Path) -> None:
    _write_matrix_market(
        tmp_path / "GSE120861_pilot_highmoi_screen.exprs.mtx.gz",
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 1\n2 1 2\n3 2 3\n",
        compress=False,
    )
    _write_lines(
        tmp_path / "GSE120861_pilot_highmoi_screen.genes.txt",
        ["GENE_A", "GENE_B", "GENE_C"],
    )
    _write_lines(
        tmp_path / "GSE120861_pilot_highmoi_screen.phenoData.txt",
        [
            "sampleP cellP1 11 1.1 pilotA pilotA GGGGGGGGGGGGGGGGGGGG 6 1 1.0 1 dirP fileP K1000 FALSE FALSE",
            "sampleP cellP2 22 2.2 pilotB pilotB AAAAAAAAAAAAAAAAAAAA 7 2 1.0 1 dirP fileP K1000 FALSE TRUE",
        ],
    )
    _write_lines(
        tmp_path / "GSE120861_grna_groups.pilot.txt",
        [
            "random_1\tGGGGGGGGGGGGGGGGGGGG",
            "pilotB\tAAAAAAAAAAAAAAAAAAAA",
        ],
    )
    _write_lines(
        tmp_path / "GSE120861_gene_gRNAgroup_pair_table.pilot.txt",
        [
            "\t".join(
                [
                    "gRNAgroup.chr",
                    "gRNAgroup.start",
                    "gRNAgroup.stop",
                    "gRNAgroup",
                    "general_group",
                    "chr.targetgene",
                    "start.targetgene",
                    "stop.targetgene",
                    "ENSG.targetgene",
                    "targetgene_short_name",
                    "type.targetgene",
                    "strand.targetgene",
                    "pairs",
                ]
            ),
            "\t".join(
                [
                    "NTC",
                    "NTC",
                    "NTC",
                    "random_1",
                    "NTC",
                    "chr1",
                    "100",
                    "101",
                    "GENE_A",
                    "GENEA",
                    "protein_coding",
                    "+",
                    "GENEA:random_1",
                ]
            ),
            "\t".join(
                [
                    "chr2",
                    "200",
                    "201",
                    "pilotB",
                    "candidate",
                    "chr2",
                    "200",
                    "201",
                    "GENE_B",
                    "GENEB",
                    "protein_coding",
                    "-",
                    "GENEB:pilotB",
                ]
            ),
        ],
    )

    mdata = build_gasperini_geo_mudata(tmp_path, "pilot", gene_id_subset=["GENE_C", "GENE_A"])
    assert mdata["gene"].var_names.tolist() == ["GENE_C", "GENE_A"]
    assert mdata["gene"].X.dtype == np.uint8


def test_build_gasperini_element_gene_histogram_frames_selects_guides_for_element() -> None:
    gene = ad.AnnData(
        X=np.array([[1, 0], [4, 2], [7, 1], [0, 3]], dtype=np.uint16),
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(index=["ENSG_A", "ENSG_B"]),
    )
    guide = ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.uint8,
            )
        ),
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(
            {"intended_target_name": ["ELEMENT_X", "ELEMENT_X", "ELEMENT_Y"]},
            index=["guide_a", "guide_b", "guide_c"],
        ),
    )
    mdata = md.MuData({"gene": gene, "guide": guide})

    background_frame, guide_frame = _build_gasperini_element_gene_histogram_frames(
        mdata,
        "ENSG_A",
        "ELEMENT_X",
        background_n=2,
        random_seed=0,
    )

    assert len(background_frame) == 2
    assert set(background_frame["count"].tolist()).issubset({0, 1, 4, 7})
    assert guide_frame["guide"].tolist() == ["guide_a", "guide_a", "guide_b", "guide_b"]
    assert guide_frame["count"].tolist() == [1, 4, 4, 7]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def _write_gzip_lines(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_matrix_market(path: Path, contents: str, *, compress: bool) -> None:
    if compress:
        with gzip.open(path, "wt") as handle:
            handle.write(contents)
        return
    path.write_text(contents)
