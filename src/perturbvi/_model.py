import logging
from typing import Dict, List, Optional

from mudata import MuData, AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.model.base import BaseModelClass, PyroSviTrainMixin, PyroSampleMixin

# from scvi.train import PyroTrainingPlan, TrainRunner
from scvi.utils._docstrings import setup_anndata_dsp

from ._module import PerturbVIPyroModule

logger = logging.getLogger(__name__)

PERTURBATION_REGISTRY_KEY = "perturbations"


class PERTURBVI(PyroSviTrainMixin, PyroSampleMixin, BaseModelClass):
    def __init__(
        self,
        mdata: MuData,
        **model_kwargs,
    ):
        super(PERTURBVI, self).__init__(mdata)

        # self.summary_stats provides information about dimensions and other tensor info
        self.module = PerturbVIPyroModule(perturbation_key=PERTURBATION_REGISTRY_KEY)

        self._model_summary_string = f"MyPyroModel Model with params:\n{self.summary_stats}"

        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def setup_anndata(
        cls,
        adata: AnnData,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Use setup_mudata instead.")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        perturbation_layer: Optional[str] = None,
        modalities: Optional[Dict[str, str]] = None,
        size_factor_key: Optional[str] = None,
        cat_cov_keys: Optional[List[str]] = None,
        cont_cov_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """%(summary_mdata)s.
        Parameters
        ----------
        %(param_mdata)s
        rna_layer
            RNA layer key. If `None`, will use `.X` of specified modality key.
        perturbation_layer
            perturbation_layer layer key. If `None`, will use `.X` of specified modality key.
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_modalities)s
        """

        setup_method_args = cls._get_setup_method_args(**locals())

        if modalities is None:
            raise ValueError("Modalities cannot be None.")
        modalities = cls._create_modalities_attr_dict(modalities, setup_method_args)

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities.batch_key,
        )

        mudata_fields = [
            batch_field,
            fields.MuDataLayerField(
                PERTURBATION_REGISTRY_KEY,
                perturbation_layer,
                mod_key=modalities.perturbation_layer,
                is_count_data=True,
                mod_required=True,
            ),
            fields.MuDataLayerField(
                REGISTRY_KEYS.X_KEY,
                rna_layer,
                mod_key=modalities.rna_layer,
                is_count_data=True,
                mod_required=True,
            ),
            fields.MuDataNumericalObsField(
                REGISTRY_KEYS.OBSERVED_LIB_SIZE,
                size_factor_key,
                mod_key=modalities.rna_layer,
                mod_required=True,

            ),
            fields.MuDataCategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY,
                cat_cov_keys,
                mod_key=modalities.cat_cov_keys,
            ),
            fields.MuDataNumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY,
                cont_cov_keys,
                mod_key=modalities.cont_cov_keys,
            ),
        ]

        adata_manager = AnnDataManager(
            fields=mudata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)
