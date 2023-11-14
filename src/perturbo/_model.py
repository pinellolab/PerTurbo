import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from mudata import MuData
from pandas import DataFrame
from scipy.sparse import issparse
from scipy.stats import chi2
from scvi._types import AnnOrMuData
from scvi.data import AnnDataManager, fields
from scvi.dataloaders import DeviceBackedDataSplitter
from scvi.model.base import (
    BaseModelClass,
    PyroJitGuideWarmup,
    PyroSampleMixin,
    PyroSviTrainMixin,
)
from scvi.train import PyroTrainingPlan
from scvi.utils._docstrings import devices_dsp

from ._constants import REGISTRY_KEYS
from ._module import PerTurboPyroModule

logger = logging.getLogger(__name__)


class PERTURBO(PyroSviTrainMixin, PyroSampleMixin, BaseModelClass):
    def __init__(
        self,
        mdata: AnnOrMuData,
        likelihood: Optional[str] = "nb",
        n_factors: Optional[int] = None,
        fit_dispersion: Optional[bool] = False,
        **model_kwargs,
    ):
        super().__init__(mdata)

        # data fields that will be loaded/mini-batched into the module
        self.data_and_attrs = {
            REGISTRY_KEYS.X_KEY: np.float32,
            REGISTRY_KEYS.SIZE_FACTOR_KEY: np.float32,
            REGISTRY_KEYS.PERTURBATION_KEY: np.float32,
            REGISTRY_KEYS.BATCH_KEY: np.int64,
            REGISTRY_KEYS.INDICES_KEY: np.int64,
        }

        if "n_extra_continuous_covs" in self.summary_stats:
            self.data_and_attrs.update({REGISTRY_KEYS.CONT_COVS_KEY: np.float32})

        n_cats_per_cov = None
        if "n_extra_categorical_covs" in self.summary_stats:
            self.data_and_attrs.update({REGISTRY_KEYS.CAT_COVS_KEY: np.float32})
            n_cats_per_cov = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key

        gene_summary_stats = self.adata_manager.get_from_registry(REGISTRY_KEYS.GENE_SUMMARY_STATS)

        guide_by_element = None
        if REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY in self.adata_manager.data_registry:
            guide_by_element = self.read_varm_from_registry(REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY)

        gene_by_element = None
        if REGISTRY_KEYS.GENE_BY_ELEMENT_KEY in self.adata_manager.data_registry:
            gene_by_element = self.read_varm_from_registry(REGISTRY_KEYS.GENE_BY_ELEMENT_KEY)

        self.module = PerTurboPyroModule(
            self.summary_stats,
            gene_summary_stats=gene_summary_stats,
            guide_by_element=guide_by_element,
            gene_by_element=gene_by_element,
            likelihood=likelihood,
            n_factors=n_factors,
            fit_dispersion=fit_dispersion,
            n_cats_per_cov=n_cats_per_cov,
        )

        self._model_summary_string = f"MyPyroModel Model with params:\n{self.summary_stats}"

        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def read_varm_from_registry(self, registry_key):
        varm_field = self.adata_manager.get_from_registry(registry_key)
        if isinstance(varm_field, DataFrame):
            varm_field = varm_field.values
        if issparse(varm_field):
            varm_field = varm_field.todense()
        varm_tensor = torch.tensor(varm_field, dtype=torch.float32, requires_grad=False)
        return varm_tensor

    @classmethod
    def setup_anndata(cls):
        """Required by scvi-tools"""
        raise NotImplementedError("MuData input required, use setup_mudata.")

    @classmethod
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: Optional[str] = None,
        perturbation_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        gene_by_element_key: Optional[str] = None,
        rna_element_uns_key: Optional[str] = None,
        guide_element_uns_key: Optional[str] = None,
        guide_by_element_key: Optional[str] = None,
        library_size_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        continuous_covariates_keys: Optional[str] = None,
        categorical_covariates_keys: Optional[str] = None,
        modalities: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        """Registers data from a MuData object with the model.

        Parameters
        ----------
        mdata
            A MuData object containing the perturbations and observational data.
        rna_layer
            The key of the MuData modality containing the RNA counts
        perturbation_layer
            The key of the MuData modality containing the perturbations
        batch_key
            Key within the RNA AnnData .obs corresponding to the experimental batch
        gene_by_element_key
            .varm key within the RNA AnnData object containing a mask of which genes can be affected by which genetic elements
        guide_by_element_key
            .varm key within the perturbation AnnData object containing which perturbations target which genetic elements
        rna_element_uns_key
            .uns key within the RNA AnnData object containing names of perturbed elements (if using GENE_BY_ELEMENT_KEY),
            otherwise automatically inferred from column names if .varm object is a DataFrame
        guide_element_uns_key
            .uns key within the perturbation AnnData object containing names of perturbed elements
            (if using GUIDE_BY_ELEMENT_KEY), otherwise automatically inferred from column names if .varm object is a DataFrame
        library_size_key
            .obs key within the RNA AnnData object containing raw (not log-scaled) library size factors for each sample
        size_factor_key
            .obs key within the RNA AnnData object containing library size factors for each sample (e.g. log-library size)
        continuous_covariates_keys
            list of .obs keys within the RNA AnnData object containing other continuous covariates to be "regressed out"
        modalities
            A dict containing these same setup arguments
        kwargs
            Additional keyword arguments
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
                    "Cannot infer size factors: cells with zero library size. Set size_factor_key manually instead."
                )
            log_cpm = np.log(library_size / 1e6)
            mdata[modalities.rna_layer].obs[size_factor_key] = log_cpm - log_cpm.mean()

        # add indices to enable pyro subsampling of local vars
        mdata[modalities.rna_layer].obs = mdata[modalities.rna_layer].obs.assign(_ind_x=lambda x: np.arange(len(x)))
        index_field = fields.MuDataNumericalObsField(
            REGISTRY_KEYS.INDICES_KEY,
            "_ind_x",
            mod_key=modalities.rna_layer,
        )

        # add info for method of moments estimation of gene params
        rna_adata = mdata[modalities.rna_layer]
        mean_counts = np.mean(rna_adata.X, axis=0)
        if isinstance(mean_counts, np.matrix):  # occurs when summing sparse array
            mean_counts = mean_counts.A1
        rna_adata.var["_gene_mean"] = mean_counts
        # rna_adata.var["_gene_variance"] = np.var(rna_adata.X, axis=0).squeeze()
        gene_field = fields.MuDataNumericalJointVarField(
            REGISTRY_KEYS.GENE_SUMMARY_STATS,
            ["_gene_mean"],
            mod_key=modalities.rna_layer,
        )

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities.rna_layer,
        )

        covariates_field = fields.MuDataNumericalJointObsField(
            REGISTRY_KEYS.CONT_COVS_KEY,
            continuous_covariates_keys,
            mod_key=modalities.rna_layer,
        )

        mudata_fields = [
            index_field,
            batch_field,
            gene_field,
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

        if continuous_covariates_keys is not None:
            mudata_fields.append(covariates_field)

        if gene_by_element_key is not None:
            mudata_fields.append(
                fields.MuDataVarmField(
                    REGISTRY_KEYS.GENE_BY_ELEMENT_KEY,
                    gene_by_element_key,
                    mod_key=modalities.rna_layer,
                    is_count_data=True,
                    colnames_uns_key=rna_element_uns_key,
                )
            )

        if guide_by_element_key is not None:
            (
                mudata_fields.append(
                    fields.MuDataVarmField(
                        REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY,
                        guide_by_element_key,
                        mod_key=modalities.perturbation_layer,
                        is_count_data=True,
                        colnames_uns_key=guide_element_uns_key,
                    )
                ),
            )

        adata_manager = AnnDataManager(
            fields=mudata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: Optional[int] = None,
        accelerator: str = "cpu",
        device: Union[int, str] = "auto",
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        lr: Optional[float] = None,
        training_plan: PyroTrainingPlan = PyroTrainingPlan,
        plan_kwargs: Optional[dict] = None,
        data_splitter_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_use_gpu)s
        %(param_accelerator)s
        %(param_device)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
            sequential order of the data according to `validation_size` and `train_size` percentages.
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
        plan_kwargs = plan_kwargs if plan_kwargs is not None else {}
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})
        if data_splitter_kwargs is None:
            data_splitter_kwargs = {}
        if "data_and_attributes" not in data_splitter_kwargs:
            data_splitter_kwargs["data_and_attributes"] = self.data_and_attrs

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                accelerator=accelerator,
                device=device,
                **data_splitter_kwargs,
            )
        else:
            data_splitter = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                shuffle_set_split=shuffle_set_split,
                batch_size=batch_size,
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
            accelerator=accelerator,
            devices=device,
            **trainer_kwargs,
        )
        return runner()

    def get_element_effects(
        self,
    ):
        """Return a DataFrame summary of the effects for targeted elements on each gene"""
        if REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY in self.adata_manager.data_registry:
            element_ids = self.adata_manager.get_state_registry(REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY).column_names
        else:
            element_ids = self.adata_manager.get_state_registry(REGISTRY_KEYS.PERTURBATION_KEY).column_names
        gene_ids = self.adata_manager.get_state_registry("X").column_names
        loc_values, scale_values = self.module.guide._get_loc_and_scale("element_effects")

        if self.module.local_effects:
            # loc/scale_values are the nonzero elements of a sparse matrix of elements by genes
            i, j = self.module.element_by_gene_idx.detach().cpu().numpy().astype(int)

            # pert_ids = self.adata_manager.get_state_registry("perturbations").column_names
            element_effects = pd.DataFrame(
                {
                    "loc": loc_values.detach().cpu().numpy(),
                    "scale": scale_values.detach().cpu().numpy(),
                    "element": [element_ids[idx] for idx in i],
                    "gene": [gene_ids[idx] for idx in j],
                }
            )
        else:
            # loc/scale_values are dense matrices of elements by genes

            def make_long_df(mat, value_name):
                return (
                    pd.DataFrame(data=mat, index=element_ids, columns=gene_ids)
                    .melt(var_name="gene", value_name=value_name, ignore_index=False)
                    .reset_index(names="element")
                )

            element_effects = pd.merge(
                make_long_df(loc_values.detach().cpu().numpy(), "loc"), make_long_df(scale_values.detach().cpu().numpy(), "scale")
            )

        element_effects = element_effects.assign(
            z_value=lambda x: x["loc"] / x["scale"],
            q_value=lambda x: chi2.sf(x["z_value"] * x["z_value"], df=1),
            # is_target=lambda x: x["element"].str.split("_", expand=True)[0] == x["gene"],
        )

        return element_effects.sort_values("z_value")
