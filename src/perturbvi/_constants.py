from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    VAR_BY_ELEMENT_KEY: str = "tested_elements"
    PERTURB_BY_ELEMENT_KEY: str = "targeted_elements"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    OBSERVED_LIB_SIZE: str = "observed_lib_size"
    PERTURBATION_KEY: str = "perturbations"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
