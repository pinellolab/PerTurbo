from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    # LABELS_KEY: str = "labels"
    # PROTEIN_EXP_KEY: str = "proteins"
    # CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    VAR_BY_ELEMENT_KEY: str = "tested_elements"
    PERTURB_BY_ELEMENT_KEY: str = "targeted_elements"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    # MINIFY_TYPE_KEY: str = "minify_type"
    # LATENT_QZM_KEY: str = "latent_qzm"
    # LATENT_QZV_KEY: str = "latent_qzv"
    OBSERVED_LIB_SIZE: str = "observed_lib_size"
    PERTURBATION_KEY: str = "perturbations"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
