"""
A pyro model
"""
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from mudata import AnnData, MuData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass
from scvi.train import PyroTrainingPlan, TrainRunner
from scvi.utils._docstrings import setup_anndata_dsp

from ._module import PerturbVIPyroModule

logger = logging.getLogger(__name__)


class PerturbVIModel(BaseModelClass):
    def __init__(
        self,
        mdata: MuData,
        **model_kwargs,
    ):
        super(PerturbVIPyroModel, self).__init__(mdata)

        # self.summary_stats provides information about dimensions and other tensor info

        self.module = PerturbVIPyroModule()

        self._model_summary_string = f"MyPyroModel Model with params:\n{self.summary_stats}"

        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        plan_kwargs
            Keyword args for :class:`~scvi.lightning.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.lightning.Trainer`.
        """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = PyroTrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

    @classmethod
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        perturbation_layer: Optional[str] = None,
        modalities: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """%(summary_mdata)s.
        Parameters
        ----------
        %(param_mdata)s
        rna_layer
            RNA layer key. If `None`, will use `.X` of specified modality key.
        protein_layer
            Protein layer key. If `None`, will use `.X` of specified modality key.
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

        mudata_fields = batch_field

        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)

        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)
