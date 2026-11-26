"""Benjamini–Hochberg families for a finalized element-by-gene test grid."""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from ._statistics import benjamini_hochberg_over_finite


class FdrFamilies:
    """Reuse the global correction and stream element-wise and sparse cis results."""

    def __init__(self, columns, *, primary, elements, genes, cis_pairs=None, cis_source=None):
        prefix = {"resampling": "crt_", "saddlepoint": "crt_saddlepoint_"}[primary]
        self.p = np.asarray(columns[prefix + "p_value"])
        self.q_global = np.asarray(columns[prefix + "q_value"])
        if self.p.shape != (len(elements), len(genes)) or self.q_global.shape != self.p.shape:
            raise ValueError("Test results do not match the element and gene axes")

        counts = [int(np.isfinite(row).sum()) for row in self.p]
        file = "element_effects.parquet"
        fields = {
            "p_value": {"file": file, "column": "p_value", "source_column": prefix + "p_value"},
            "q_global": {
                "file": file,
                "column": "q_global",
                "source_column": prefix + "q_value",
                "family": "global",
                "finite_p": sum(counts),
            },
            "q_element": {
                "file": file,
                "column": "q_element",
                "family": "within_element",
                "finite_p": counts,
                "element_axis": "metadata.json:perturbation_names",
            },
        }
        self.metadata = {"primary_test": primary, "adjustment": "benjamini_hochberg", "result_fields": fields}
        self.cis_indices = np.empty(0, dtype=np.intp)
        self.q_cis = np.empty(0, dtype=np.float64)
        if cis_pairs is not None:
            pairs = cis_pairs[["element", "gene"]].drop_duplicates()
            rows = pd.Index(elements).get_indexer(pairs["element"])
            cols = pd.Index(genes).get_indexer(pairs["gene"])
            matched = (rows >= 0) & (cols >= 0)
            indices = np.sort(rows[matched] * len(genes) + cols[matched])
            p_cis = self.p.flat[indices]
            finite = np.isfinite(p_cis)
            self.cis_indices = indices[finite]
            self.q_cis = benjamini_hochberg_over_finite(p_cis[finite])
            self.metadata["cis_family"] = {
                "source": str(cis_source),
                "declared_pairs": len(pairs),
                "matched_pairs": len(indices),
                "finite_p": len(self.q_cis),
            }
            if self.q_cis.size:
                fields["q_cis"] = {"file": file, "column": "q_cis", "family": "cis", "finite_p": len(self.q_cis)}

    def blocks(self, size: int) -> Iterator[dict[str, np.ndarray]]:
        """Correct each complete element once, including rows spanning Parquet blocks."""
        n_genes = self.p.shape[1]
        last_element = -1
        adjusted = np.empty(0, dtype=np.float64)
        for start in range(0, max(1, self.p.size), size):
            stop = min(start + size, self.p.size)
            q_element = np.empty(stop - start, dtype=np.float64)
            pos = start
            while pos < stop:
                element, gene = divmod(pos, n_genes)
                if element != last_element:
                    adjusted = benjamini_hochberg_over_finite(self.p[element])
                    last_element = element
                take = min(n_genes - gene, stop - pos)
                q_element[pos - start : pos - start + take] = adjusted[gene : gene + take]
                pos += take
            block = {
                "p_value": np.asarray(self.p.flat[start:stop], dtype=np.float64),
                "q_global": np.asarray(self.q_global.flat[start:stop], dtype=np.float64),
                "q_element": q_element,
            }
            if self.q_cis.size:
                lo, hi = np.searchsorted(self.cis_indices, [start, stop])
                values = np.full(stop - start, np.nan)
                values[self.cis_indices[lo:hi] - start] = self.q_cis[lo:hi]
                block["q_cis"] = values
            yield block
