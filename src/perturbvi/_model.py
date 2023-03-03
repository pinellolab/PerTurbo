import logging
from typing import Dict, Optional, Union

import numpy as np
from mudata import AnnData, MuData
from pyro import render_model as pyro_render_model
from scvi.data import AnnDataManager, fields
from scvi.dataloaders import AnnDataLoader, DeviceBackedDataSplitter
from scvi.model.base import BaseModelClass, PyroJitGuideWarmup, PyroSampleMixin, PyroSviTrainMixin
from scvi.train import PyroTrainingPlan
from scvi.utils._docstrings import setup_anndata_dsp

from ._constants import REGISTRY_KEYS
from ._module import PerturbVIPyroModule

logger = logging.getLogger(__name__)


class PERTURBVI(PyroSviTrainMixin, PyroSampleMixin, BaseModelClass):
    data_and_attrs = {
        REGISTRY_KEYS.X_KEY: np.float32,
        REGISTRY_KEYS.SIZE_FACTOR_KEY: np.float32,
        REGISTRY_KEYS.PERTURBATION_KEY: np.float32,
        REGISTRY_KEYS.BATCH_KEY: np.int64,
        REGISTRY_KEYS.INDICES_KEY: np.int64,
    }

    def __init__(
        self,
        mdata: MuData,
        likelihood="nb",
        fit_lib_size=False,
        **model_kwargs,
    ):
        super(PERTURBVI, self).__init__(mdata)

        # self.summary_stats provides information about dimensions and other tensor info
        self.module = PerturbVIPyroModule(
            self.summary_stats,
            likelihood=likelihood,
            fit_lib_size_effect=fit_lib_size,
        )

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
        var_by_element_key: Optional[str] = None,
        perturb_by_element_key: Optional[str] = None,
        perturbation_layer: Optional[str] = None,
        modalities: Optional[Dict[str, str]] = None,
        size_factor_key: Optional[str] = None,
        library_size_key: Optional[str] = None,
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
        if library_size_key is None:
            library_size_key = "_library_size"
            library_size = mdata[modalities.rna_layer].X.sum(axis=1)
            if not library_size.all():
                raise ValueError(
                    "Cannot infer library size: cells with zero counts. Set library_size_key manually instead."
                )
            mdata[modalities.rna_layer].obs[library_size_key] = library_size

        # add size factor if not present
        if size_factor_key is None:
            size_factor_key = "_size_factor"
            library_size = mdata[modalities.rna_layer].obs[library_size_key]
            if not library_size.all():
                raise ValueError(
                    "Cannot infer size factors from library size: cells with zero counts. Set size_factor_key manually instead."
                )
            mdata[modalities.rna_layer].obs[size_factor_key] = np.log1p(library_size)

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
                REGISTRY_KEYS.SIZE_FACTOR_KEY,
                size_factor_key,
                mod_key=modalities.rna_layer,
                mod_required=True,
            ),
        ]

        if var_by_element_key is not None:
            mudata_fields.append(
                fields.MuDataVarmField(
                    REGISTRY_KEYS.VAR_BY_ELEMENT_KEY,
                    var_by_element_key,
                    mod_key=modalities.rna_layer,
                )
            )

        if perturb_by_element_key is not None:
            mudata_fields.append(
                fields.MuDataVarmField(
                    REGISTRY_KEYS.PERTURB_BY_ELEMENT_KEY,
                    perturb_by_element_key,
                    mod_key=modalities.perturbation_layer,
                )
            ),

        adata_manager = AnnDataManager(
            fields=mudata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        lr: Optional[float] = None,
        training_plan: PyroTrainingPlan = PyroTrainingPlan,
        plan_kwargs: Optional[dict] = None,
        data_splitter_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model. Modified from scVI scBASSET implementation.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        training_plan
            Training plan :class:`~scvi.train.PyroTrainingPlan`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.PyroTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})

        if data_splitter_kwargs is None:
            data_splitter_kwargs = dict()
        if "data_and_attributes" not in data_splitter_kwargs:
            data_splitter_kwargs["data_and_attributes"] = self.data_and_attrs

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                **data_splitter_kwargs,
            )
        else:
            data_splitter = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                **data_splitter_kwargs,
            )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]

        if "callbacks" not in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] = []
        trainer_kwargs["callbacks"].append(PyroJitGuideWarmup())

        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

    def _render_pyro_model(self, model):
        loader = AnnDataLoader(
            adata_manager=self.adata_manager, indices=[1], batch_size=1, data_and_attributes=self.data_and_attrs
        )
        sample_args, sample_kwargs = self.module._get_fn_args_from_batch(next(iter(loader)))
        return pyro_render_model(
            model,
            model_args=sample_args,
            model_kwargs=sample_kwargs,
            render_distributions=True,
            render_params=True,
        )

    def render_model(self):
        return self._render_pyro_model(self.module.model)

    def render_guide(self):
        return self._render_pyro_model(self.module.guide)
