from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    # LABELS_KEY: str = "labels"
    # CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    GENE_BY_ELEMENT_KEY: str = "tested_elements"
    GUIDE_BY_ELEMENT_KEY: str = "targeted_elements"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    PERTURBATION_KEY: str = "perturbations"
    OBSERVED_LIB_SIZE: str = "observed_lib_size"
    GENE_SUMMARY_STATS: str = "gene_summary_stats"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
