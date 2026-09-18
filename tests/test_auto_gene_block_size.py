"""The stage-two gene block is sized to the card, not hard-coded at 256."""

from perturbo.cli import _AUTO_GENE_BLOCK_MAX, _AUTO_GENE_BLOCK_MIN, _auto_gene_block_size

GIB = 1024**3


def test_low_moi_screen_keeps_the_full_block_on_a_small_card():
    # Replogle essential: one guide per cell, 310k cells, 16 GiB card -> 0.3 GiB.
    block, reason = _auto_gene_block_size(310_385, 1, 16 * GIB)
    assert block == _AUTO_GENE_BLOCK_MAX
    assert "0.30 GiB" in reason and "reported by the device" in reason


def test_the_collaborators_high_moi_screen_shrinks_to_a_block_that_fits():
    # 1,060,779 cells x 47 guides per cell on a 40 GiB card: 256 genes is
    # 47.6 GiB, the allocation that failed; the budget is a third of the card.
    block, reason = _auto_gene_block_size(1_060_779, 47, 40 * GIB)
    assert block == 64
    gather = 1_060_779 * 47 * 4 * block / GIB
    assert gather < 0.33 * 40
    assert "47 guides per cell" in reason


def test_unknown_device_memory_uses_the_conservative_fallback():
    block_known, _ = _auto_gene_block_size(1_060_779, 47, 40 * GIB)
    block_unknown, reason = _auto_gene_block_size(1_060_779, 47, None)
    assert block_unknown == block_known
    assert "assumed" in reason


def test_block_never_drops_below_the_floor_and_says_so():
    block, reason = _auto_gene_block_size(50_000_000, 200, 8 * GIB)
    assert block == _AUTO_GENE_BLOCK_MIN
    assert "--gene-chunk-size" in reason


def test_block_is_monotone_in_cells_and_guides():
    b1, _ = _auto_gene_block_size(200_000, 10, 40 * GIB)
    b2, _ = _auto_gene_block_size(2_000_000, 10, 40 * GIB)
    b3, _ = _auto_gene_block_size(2_000_000, 40, 40 * GIB)
    assert b1 >= b2 >= b3
