import logging
from typing import Dict, List, Optional

import numpy as np
from mudata import AnnData, MuData
from scvi.data import AnnDataManager, fields
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.utils._docstrings import setup_anndata_dsp

from ._constants import REGISTRY_KEYS
from ._module import PerturbVIPyroModule

logger = logging.getLogger(__name__)


class PERTURBVI(PyroSviTrainMixin, PyroSampleMixin, BaseModelClass):
    def __init__(
        self,
        mdata: MuData,
        **model_kwargs,
    ):
        super(PERTURBVI, self).__init__(mdata)

        # self.summary_stats provides information about dimensions and other tensor info
        self.module = PerturbVIPyroModule(self.summary_stats)

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
        raise NotImplementedError("Not implemented: use setup_mudata instead.")

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
        %(param_modalities)s
        """

        setup_method_args = cls._get_setup_method_args(**locals())

        if modalities is None:
            raise ValueError("Modalities cannot be None.")
        modalities = cls._create_modalities_attr_dict(modalities, setup_method_args)

        # add library size if not present
        if size_factor_key is None:
            size_factor_key = "_library_size"
            lib_size = mdata[modalities.rna_layer].X.sum(axis=1)
            if not lib_size.all():
                raise ValueError("Cannot infer library size: cells with zero counts. Set size_factor_key instead.")
            mdata[modalities.rna_layer]["_library_size"] = lib_size

        # add indices to enable pyro subsampling of local vars
        mdata[modalities.rna_layer].obs = mdata[modalities.rna_layer].obs.assign(_ind_x=lambda x: np.arange(len(x)))
        index_field = fields.MuDataNumericalObsField(
            REGISTRY_KEYS.INDICES_KEY,
            "_ind_x",
            mod_key=modalities.rna_layer,
        )

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities.rna_layer,
        )

        mudata_fields = [
            index_field,
            batch_field,
            fields.MuDataLayerField(
                REGISTRY_KEYS.PERTURBATION_KEY,
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
        ]

        adata_manager = AnnDataManager(
            fields=mudata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)

