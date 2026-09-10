from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GasperiniPilotMatchSummary:
    shared_guides: int
    at_scale_cells_before: int
    at_scale_cells_with_shared_guides: int
    pilot_cells_with_shared_guides: int
    at_scale_median_cells_per_shared_guide_before: float
    pilot_median_cells_per_shared_guide: float
    selected_cell_fraction: float
    selected_cells: int
    at_scale_median_cells_per_shared_guide_after: float


def build_at_scale_pilot_matched_mudata(
    at_scale: md.MuData,
    pilot: md.MuData,
    *,
    random_seed: int = 0,
) -> tuple[md.MuData, GasperiniPilotMatchSummary]:
    shared_guides = _shared_guides_in_pilot_order(at_scale["guide"], pilot["guide"])
    if not shared_guides:
        raise ValueError("No shared guides found between the at-scale and pilot guide modalities.")

    at_guide_shared = at_scale["guide"][:, shared_guides]
    pilot_guide_shared = pilot["guide"][:, shared_guides]

    at_eligible_mask = _cells_with_any_guide(at_guide_shared)
    pilot_eligible_mask = _cells_with_any_guide(pilot_guide_shared)

    at_eligible = at_guide_shared[at_eligible_mask, :]
    pilot_eligible = pilot_guide_shared[pilot_eligible_mask, :]

    at_counts_before = _guide_cell_counts(at_eligible)
    pilot_counts = _guide_cell_counts(pilot_eligible)

    target_median = float(np.median(pilot_counts))
    current_median = float(np.median(at_counts_before))

    keep_fraction = 1.0 if current_median <= 0 else min(1.0, target_median / current_median)
    selected_relative_mask, selected_fraction = _select_cells_for_target_median(
        at_eligible,
        target_median=target_median,
        base_fraction=keep_fraction,
        random_seed=random_seed,
    )

    selected_absolute_indices = np.flatnonzero(at_eligible_mask)[selected_relative_mask]
    full_cell_mask = np.zeros(at_scale.n_obs, dtype=bool)
    full_cell_mask[selected_absolute_indices] = True

    gene = at_scale["gene"][full_cell_mask, :].copy()
    guide = at_scale["guide"][full_cell_mask, shared_guides].copy()
    matched = md.MuData({"gene": gene, "guide": guide})
    matched.uns["pilot_matching"] = asdict(
        GasperiniPilotMatchSummary(
            shared_guides=len(shared_guides),
            at_scale_cells_before=at_scale.n_obs,
            at_scale_cells_with_shared_guides=int(at_eligible_mask.sum()),
            pilot_cells_with_shared_guides=int(pilot_eligible_mask.sum()),
            at_scale_median_cells_per_shared_guide_before=current_median,
            pilot_median_cells_per_shared_guide=target_median,
            selected_cell_fraction=selected_fraction,
            selected_cells=int(full_cell_mask.sum()),
            at_scale_median_cells_per_shared_guide_after=float(np.median(_guide_cell_counts(guide))),
        )
    )
    matched.uns["source"] = dict(at_scale.uns.get("source", {}))
    matched.uns["source"]["matched_to_pilot"] = True
    matched.uns["source"]["pilot_shared_guides"] = len(shared_guides)
    matched.strings_to_categoricals()
    matched.update()
    summary = GasperiniPilotMatchSummary(**matched.uns["pilot_matching"])
    return matched, summary


def write_at_scale_pilot_matched_h5mu(
    at_scale_path: str | Path,
    pilot_path: str | Path,
    output_path: str | Path,
    *,
    random_seed: int = 0,
) -> tuple[Path, GasperiniPilotMatchSummary]:
    at_scale_path = Path(at_scale_path).expanduser()
    pilot_path = Path(pilot_path).expanduser()
    output_path = Path(output_path).expanduser()

    at_scale = md.read(at_scale_path)
    pilot = md.read(pilot_path)
    matched, summary = build_at_scale_pilot_matched_mudata(
        at_scale,
        pilot,
        random_seed=random_seed,
    )
    matched.write(output_path)
    return output_path, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Subset the Gasperini at-scale MuData object to guides shared with pilot and "
            "downsample cells to match the pilot median cells-per-guide."
        )
    )
    parser.add_argument(
        "--at-scale",
        default="~/Data/gasperini_geo/GSE120861_at_scale_screen.h5mu",
        help="Path to the at-scale .h5mu file.",
    )
    parser.add_argument(
        "--pilot",
        default="~/Data/gasperini_geo/GSE120861_pilot_highmoi_screen.at_scale_genes.h5mu",
        help="Path to the pilot .h5mu file used to define the shared guides and target median.",
    )
    parser.add_argument(
        "--output",
        default="~/Data/gasperini_geo/GSE120861_at_scale_screen.pilot_shared_guides.pilot_median.h5mu",
        help="Output path for the matched at-scale .h5mu file.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed controlling the cell downsampling.")
    args = parser.parse_args(argv)

    output_path, summary = write_at_scale_pilot_matched_h5mu(
        at_scale_path=args.at_scale,
        pilot_path=args.pilot,
        output_path=args.output,
        random_seed=args.seed,
    )
    print(f"wrote {output_path}")
    print(summary)


def _shared_guides_in_pilot_order(at_scale_guide: ad.AnnData, pilot_guide: ad.AnnData) -> list[str]:
    at_scale_guides = set(at_scale_guide.var_names.astype(str))
    return [guide for guide in pilot_guide.var_names.astype(str) if guide in at_scale_guides]


def _cells_with_any_guide(guide_adata: ad.AnnData) -> np.ndarray:
    return np.asarray(guide_adata.X.sum(axis=1)).ravel() > 0


def _guide_cell_counts(guide_adata: ad.AnnData) -> np.ndarray:
    return np.asarray(guide_adata.X.sum(axis=0)).ravel()


def _select_cells_for_target_median(
    at_eligible: ad.AnnData,
    *,
    target_median: float,
    base_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, float]:
    n_cells = at_eligible.n_obs
    if n_cells == 0:
        raise ValueError("No at-scale cells contain guides shared with pilot.")

    rng = np.random.default_rng(random_seed)
    scores = rng.random(n_cells)
    candidate_fractions = sorted(
        {
            min(1.0, max(0.0, base_fraction * scale))
            for scale in (0.7, 0.85, 1.0, 1.15, 1.3)
        }
        | {min(1.0, max(0.0, base_fraction))}
    )
    if 1.0 not in candidate_fractions:
        candidate_fractions.append(1.0)

    best_mask: np.ndarray | None = None
    best_fraction = 1.0
    best_distance = np.inf
    best_selected_cells = np.inf
    for fraction in candidate_fractions:
        if fraction <= 0:
            continue
        mask = scores < fraction
        if not mask.any():
            continue
        median = float(np.median(np.asarray(at_eligible[mask, :].X.sum(axis=0)).ravel()))
        distance = abs(median - target_median)
        selected_cells = int(mask.sum())
        if distance < best_distance or (distance == best_distance and selected_cells < best_selected_cells):
            best_mask = mask
            best_fraction = fraction
            best_distance = distance
            best_selected_cells = selected_cells

    if best_mask is None:
        raise ValueError("Failed to select any cells during pilot-matching downsampling.")
    return best_mask, best_fraction
