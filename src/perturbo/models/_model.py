import logging

import numba as nb
import numpy as np
import pandas as pd
import pyro.optim as optim
import scipy.sparse as sp
import torch
from mudata import AnnData, MuData
from pandas import DataFrame
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, infer_discrete
from scipy.sparse import coo_matrix, csc_matrix, issparse
from scipy.stats import chi2
from scvi._types import AnnOrMuData
from scvi.data import AnnDataManager, fields
from scvi.dataloaders import AnnDataLoader, DataSplitter, DeviceBackedDataSplitter
from scvi.model._utils import parse_device_args
from scvi.model.base import BaseModelClass, PyroJitGuideWarmup, PyroSampleMixin, PyroSviTrainMixin
from scvi.train import PyroTrainingPlan
from scvi.utils._docstrings import devices_dsp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from tqdm.auto import trange

from ._constants import REGISTRY_KEYS
from ._module import PerTurboPyroModule

logger = logging.getLogger(__name__)


def compute_element_lfc_initialization(
    X: np.ndarray | sp.spmatrix,
    guide_obs: np.ndarray | sp.spmatrix,
    guide_by_element: torch.Tensor | np.ndarray,
    control_indices: np.ndarray | None = None,
    n_cells_for_control: int = 10000,
    pseudocount: float = 0.1,
) -> np.ndarray:
    """
    Compute log fold-change (LFC) estimates for element effects on genes using matrix operations.

    For each element, computes the mean expression in cells targeted by that element versus
    baseline (control or random cells), using a pseudocount to avoid log(0).

    Parameters
    ----------
    X : np.ndarray or sp.spmatrix
        Count matrix (n_cells x n_genes).
    guide_obs : np.ndarray or sp.spmatrix
        Binary or count matrix of guides per cell (n_cells x n_guides).
    guide_by_element : torch.Tensor or np.ndarray
        Binary matrix mapping guides to elements (n_guides x n_elements).
    control_indices : np.ndarray or None
        Boolean or integer indices of control cells (baseline for LFC).
        If None, samples 1000 random cells (or all if fewer available).
    n_cells_for_control : int
        Number of random cells to sample for control if control_indices is None.
    pseudocount : float
        Pseudocount added to avoid log(0). Default: 0.1.

    Returns
    -------
    np.ndarray
        Log fold-change matrix (n_elements x n_genes). Entry [i, j] is the LFC
        of element i on gene j.
    """
    n_guides, n_elements = guide_by_element.shape
    n_genes = X.shape[1]

    # Determine baseline (control) indices
    if control_indices is None:
        n_cells = X.shape[0]
        n_control = min(n_cells_for_control, n_cells)
        control_indices = np.random.choice(n_cells, size=n_control, replace=False)
    elif isinstance(control_indices, (list, np.ndarray)):
        if len(control_indices) > 0 and control_indices.dtype == bool:
            control_indices = np.where(control_indices)[0]

    # Compute mean expression in control cells
    X_control = X[control_indices, :]  # (n_control x n_genes)
    control_mean = X_control.mean(axis=0)  # (n_genes,)

    # For each element, identify cells targeted by any guide targeting that element
    # guide_obs: (n_cells x n_guides)
    # guide_by_element: (n_guides x n_elements)
    # cells_by_element: (n_cells x n_elements) = guide_obs @ guide_by_element
    if isinstance(guide_by_element, torch.Tensor):
        guide_by_element = guide_by_element.cpu().numpy()

    guide_obs = csc_matrix(guide_obs)
    lfc = np.zeros((n_elements, n_genes), dtype=np.float32)

    print("Computing element log fold-change initializations...")
    for elem_idx in trange(n_elements):
        element_guides = guide_by_element[:, elem_idx] != 0
        guide_idx = guide_obs[:, element_guides].sum(axis=1).A1 != 0
        if guide_idx.any():
            element_mean = X[guide_idx, :].mean(axis=0)
            lfc[elem_idx, :] = np.log((element_mean + pseudocount) / (control_mean + pseudocount))
    return lfc


class PERTURBO(PyroSviTrainMixin, PyroSampleMixin, BaseModelClass):
    def __init__(
        self,
        mdata: AnnOrMuData,
        control_guides: list[int] | list[bool] | None = None,
        dispersion_smoothing: str = "none",
        smoothing_factor: float = 0.3,
        **model_kwargs,
    ):
        """
        Initialize the PERTURBO model.

        Parameters
        ----------
        mdata : AnnOrMuData
            MuData or AnnData object containing the data.
        control_guides : list or None
            List of control guide indices (optional, only used for setting initial values).
        dispersion_smoothing : str
            Smoothing method for dispersion estimation ("none", "linear", "isotonic").
        smoothing_factor : float
            Smoothing factor for dispersion smoothing.
        model_kwargs : dict
            Additional keyword arguments for the model.
        """
        super().__init__(mdata)

        # data fields that will be loaded/mini-batched into the module
        self.data_and_attrs = {
            REGISTRY_KEYS.X_KEY: np.float32,
            REGISTRY_KEYS.SIZE_FACTOR_KEY: np.float32,
            REGISTRY_KEYS.PERTURBATION_KEY: np.float32,
            REGISTRY_KEYS.BATCH_KEY: np.int64,
            REGISTRY_KEYS.INDICES_KEY: np.int64,
        }

        n_extra_continuous_covs = 0
        if "n_extra_continuous_covs" in self.summary_stats:
            n_extra_continuous_covs = self.summary_stats.n_extra_continuous_covs
            self.data_and_attrs.update({REGISTRY_KEYS.CONT_COVS_KEY: np.float32})

        n_cats_per_cov = None
        if "n_extra_categorical_covs" in self.summary_stats:
            self.data_and_attrs.update({REGISTRY_KEYS.CAT_COVS_KEY: np.float32})
            n_cats_per_cov = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key

        guide_by_element = None
        n_elements = None
        if REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY in self.adata_manager.data_registry:
            n_elements = self.summary_stats.n_targeted_elements
            guide_by_element = self.read_matrix_from_registry(REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY)

        gene_by_element = None
        if REGISTRY_KEYS.GENE_BY_ELEMENT_KEY in self.adata_manager.data_registry:
            gene_by_element = self.read_matrix_from_registry(REGISTRY_KEYS.GENE_BY_ELEMENT_KEY)

        epsilon = 1e-3
        X = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        grna_counts = self.adata_manager.get_from_registry(REGISTRY_KEYS.PERTURBATION_KEY)

        if control_guides is not None:
            if issparse(grna_counts):
                control_guide_idx = grna_counts[:, control_guides].sum(axis=1).A1 > 0
            else:
                control_guide_idx = grna_counts[:, control_guides].sum(axis=1) > 0
            X = X[control_guide_idx, :]
        self.control_guides = control_guides

        n_cells_for_init = X.shape[0]
        if n_cells_for_init == 0:
            print("Warning: No cells with control guides found for initializing dispersion parameters.")
            log_means = np.zeros(self.summary_stats.n_vars, dtype=np.float32)
            log_disp_smoothed = np.ones(self.summary_stats.n_vars, dtype=np.float32)
        else:
            # run estimate_nb_params(X) safely
            log_means, log_disp, log_disp_smoothed = estimate_nb_params(X, smoothing=dispersion_smoothing)
            log_disp_smoothed = np.where(np.isfinite(log_disp_smoothed), log_disp_smoothed, 0)
            log_disp = np.where(np.isfinite(log_disp), log_disp, log_disp_smoothed)
            if dispersion_smoothing != "none":
                log_disp_smoothed = smoothing_factor * log_disp_smoothed + (1 - smoothing_factor) * log_disp
            log_means = np.clip(log_means, a_min=np.log(1 / X.shape[0]), a_max=None)

        # if control_guides is not None and "n_factors" in model_kwargs and guide_by_element is not None:
        #     # control_guides, _ = torch.max(guide_by_element[:, control_elements], dim=-1)
        #     control_mask = self.read_matrix_from_registry(REGISTRY_KEYS.PERTURBATION_KEY)[:, control_guides].sum(dim=-1)
        #     rna_data = self.read_matrix_from_registry(REGISTRY_KEYS.X_KEY)[control_mask.bool(), :]
        #     u, s, v = torch.pca_lowrank(torch.tensor(rna_data), q=model_kwargs["n_factors"])
        #     model_kwargs["control_pcs"] = v.T.unsqueeze(dim=-2)
        # print(v)

        # Compute element LFC initialization if elements are present and not already in model_kwargs
        element_lfc_init = None
        if n_elements is not None and guide_by_element is not None and "element_effects_init" not in model_kwargs:
            # Use full data for LFC computation (not filtered by control guides)
            X_full = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
            grna_counts_full = self.adata_manager.get_from_registry(REGISTRY_KEYS.PERTURBATION_KEY)

            # Determine baseline indices: use control guides if available, else sample 1000 random cells
            if control_guides is not None:
                if issparse(grna_counts_full):
                    control_indices = grna_counts_full[:, control_guides].sum(axis=1).A1 > 0
                else:
                    control_indices = grna_counts_full[:, control_guides].sum(axis=1) > 0
            else:
                control_indices = None

            element_lfc_init = compute_element_lfc_initialization(
                X=X_full,
                guide_obs=grna_counts_full,
                guide_by_element=guide_by_element,
                control_indices=control_indices,
                pseudocount=0.1,
            )

        self.module = PerTurboPyroModule(
            n_cells=self.summary_stats.n_cells,
            n_batches=self.summary_stats.n_batch,
            n_perturbations=self.summary_stats.n_perturbations,
            n_genes=self.summary_stats.n_vars,
            n_cont_covariates=n_extra_continuous_covs,
            n_elements=n_elements,
            log_gene_mean_init=torch.tensor(log_means, dtype=torch.float32),
            log_gene_dispersion_init=torch.tensor(log_disp_smoothed, dtype=torch.float32),
            lfc_init=torch.tensor(element_lfc_init, dtype=torch.float32) if element_lfc_init is not None else None,
            guide_by_element=guide_by_element,
            gene_by_element=gene_by_element,
            # n_cats_per_cov=n_cats_per_cov,
            **model_kwargs,
        )

        self._model_summary_string = f"MyPyroModel Model with params:\n{self.summary_stats}"

        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def read_matrix_from_registry(self, registry_key: str) -> torch.Tensor:
        """
        Reads a matrix from the AnnDataManager registry and converts it to a torch.Tensor.

        Parameters
        ----------
        registry_key : str
            Key to retrieve the matrix from the registry.

        Returns
        -------
        torch.Tensor
            The matrix as a torch tensor.
        """
        data = self.adata_manager.get_from_registry(registry_key)
        if isinstance(data, DataFrame):
            data = data.values
        if issparse(data):
            data = data.todense()
        data = torch.tensor(data, dtype=torch.float32, requires_grad=False)
        return data

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnOrMuData,
        **kwargs,
    ):
        if isinstance(adata, AnnData):
            raise NotImplementedError("MuData input required, use setup_mudata.")
        else:
            cls.setup_mudata(adata, **kwargs)

    @classmethod
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: str | None = None,
        perturbation_layer: str | None = None,
        batch_key: str | None = None,
        gene_by_element_key: str | None = None,
        rna_element_uns_key: str | None = None,
        guide_element_uns_key: str | None = None,
        guide_by_element_key: str | None = None,
        library_size_key: str | None = None,
        size_factor_key: str | None = None,
        gene_mean_key: str | None = None,  # not used, supported for legacy reasons
        continuous_covariates_keys: str | None = None,
        categorical_covariates_keys: str | None = None,  #  not used, supported for legacy reasons
        modalities: dict[str, str] | None = None,
        **kwargs,
    ):
        """
        Registers data from a MuData object with the model.

        Parameters
        ----------
        mdata : MuData
            A MuData object containing the perturbations and observational data.
        rna_layer : str or None
            The key of the MuData modality containing the RNA counts.
        perturbation_layer : str or None
            The key of the MuData modality containing the perturbations.
        batch_key : str or None
            Key within the RNA AnnData .obs corresponding to the experimental batch.
        gene_by_element_key : str or None
            .varm key within the RNA AnnData object containing a mask of which genes can be affected by which genetic elements.
        rna_element_uns_key : str or None
            .uns key within the RNA AnnData object containing names of perturbed elements.
        guide_element_uns_key : str or None
            .uns key within the perturbation AnnData object containing names of perturbed elements.
        guide_by_element_key : str or None
            .varm key within the perturbation AnnData object containing which perturbations target which genetic elements.
        library_size_key : str or None
            .obs key within the RNA AnnData object containing raw library size factors for each sample.
        size_factor_key : str or None
            .obs key within the RNA AnnData object containing library size factors for each sample.
        gene_mean_key : str or None
            .var key for gene mean (legacy, for simulator).
        continuous_covariates_keys : str or None
            List of .obs keys within the RNA AnnData object containing other continuous covariates.
        modalities : dict[str, str] or None
            A dict containing these same setup arguments.
        kwargs : dict
            Additional keyword arguments.
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

        # add gene mean estimate (legacy, for simulator)
        gene_mean_key = "_gene_mean"
        rna_adata = mdata[modalities.rna_layer]
        mean_counts = np.mean(rna_adata.X, axis=0)
        if isinstance(mean_counts, np.matrix):  # occurs when summing sparse array
            mean_counts = mean_counts.A1
        rna_adata.var["_gene_mean"] = mean_counts

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

        # Check continuous covariates for potential issues
        obs_df = mdata[modalities.rna_layer].obs
        if continuous_covariates_keys is not None:
            for cov in continuous_covariates_keys:
                values = obs_df[cov].values
                unique_vals = np.unique(values)
                std = np.std(values)
                is_all_int = np.all(np.equal(np.mod(values, 1), 0))
                is_all_same = len(unique_vals) == 1
                is_binary = np.array_equal(unique_vals, [0, 1]) or np.array_equal(unique_vals, [1, 0])

                if is_all_same:
                    logger.warning(
                        f"Continuous covariate '{cov}' has the same value for all observations. "
                        "Consider removing this covariate."
                    )
                elif is_all_int and len(unique_vals) > 1 and not is_binary:
                    logger.warning(
                        f"Continuous covariate '{cov}' contains only discrete counts. "
                        "Consider applying log1p transform followed by z-scoring."
                    )
                elif std > 10 or std < 0.1:
                    logger.warning(
                        f"Continuous covariate '{cov}' has standard deviation {std:.3g}. Consider z-scoring."
                    )

        covariates_field = fields.MuDataNumericalJointObsField(
            REGISTRY_KEYS.CONT_COVS_KEY,
            continuous_covariates_keys,
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

    def pretrain(
        self,
        max_epochs: int = 100,
        indices: list[int] | list[bool] | None = None,
        num_particles: int = 1,
        accelerator: str = "cpu",
        device: int | str = "auto",
        batch_size: int = 1024,
        # early_stopping: bool = False,
        lr: float | None = 0.01,
    ):
        """
        Pretrain the model on a subset of the data.

        Parameters
        ----------
        indices : array-like
            Indices of the subset of data to use for pretraining.
        max_epochs : int
            Number of passes through the dataset.
        accelerator : str
            Accelerator type ("cpu", "gpu", etc.).
        device : int or str
            Device identifier.
        batch_size : int
            Minibatch size to use during training.
        early_stopping : bool
            Perform early stopping.
        lr : float or None
            Optimizer learning rate.
        """
        _, _, device = parse_device_args(accelerator, device, return_device="torch", validate_single_device=True)

        loader = AnnDataLoader(
            adata_manager=self.adata_manager,
            indices=indices,
            batch_size=min(batch_size, len(indices)),
            data_and_attributes=self.data_and_attrs,
        )

        if self.module.local_effects and self.module.sparse_tensors:
            zero_element_effects = torch.zeros((self.module.n_element_effects), dtype=torch.float32, device=device)
            zero_guide_efficacy = torch.zeros((self.module.n_guide_effects), dtype=torch.float32, device=device)
        else:
            zero_element_effects = torch.zeros(
                (self.module.n_elements, self.module.n_genes), dtype=torch.float32, device=device
            )
            zero_guide_efficacy = torch.full(
                (self.module.n_perturbations, self.module.n_genes), fill_value=0.5, dtype=torch.float32, device=device
            )

        self.module.to(device)
        pretrain_model = poutine.condition(
            self.module.model,
            data={
                "element_effects": zero_element_effects,
                "guide_efficacy": zero_guide_efficacy,
            },
        )
        pretrain_init_values = {
            "log_gene_mean": self.module.log_gene_mean_init.to(device),
            "log_gene_dispersion": self.module.log_gene_dispersion_init.to(device),
        }

        pretrain_guide = self.module._guide_factory(
            poutine.block(self.module.model, hide=["element_effects", "guide_efficacy"]),
            init_values=pretrain_init_values,
        )

        pretrain_guide.to(device)
        svi = SVI(
            pretrain_model,
            pretrain_guide,
            optim.Adam({"lr": lr}),
            loss=Trace_ELBO(max_plate_nesting=3, num_particles=num_particles),
        )
        losses = []

        if len(loader) == 1:
            batch = next(iter(loader))
            args, kwargs = self.module._get_fn_args_from_batch(batch)
            args = tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.to(device)

            t = trange(max_epochs, desc="Pretrain epochs", leave=True)
            for _ in t:
                loss = svi.step(*args, **kwargs)
                losses.append(loss)
                t.set_postfix(loss=loss)

        else:
            t = trange(max_epochs, desc="Pretrain epochs", leave=True)
            for _ in t:
                batch_losses = []
                for batch in loader:
                    args, kwargs = self.module._get_fn_args_from_batch(batch)
                    args = tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            kwargs[k] = v.to(device)
                    loss = svi.step(*args, **kwargs)
                    batch_losses.append(loss)
                t.set_postfix(loss=np.mean(batch_losses))
                losses.append(np.mean(batch_losses))
        return losses

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 1000,
        accelerator: str = "cpu",
        device: int | str = "auto",
        batch_size: int = 1024,
        early_stopping: bool = False,
        indices: list[int] | None = None,
        pretrain: bool = False,
        blocked_sites: list[str] | None = None,
        lr: float | None = 0.01,
        load_sparse_tensor: bool = "auto",
        training_plan: PyroTrainingPlan = PyroTrainingPlan,
        plan_kwargs: dict | None = None,
        data_splitter_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs : int
            Number of passes through the dataset.
        accelerator : str
            Accelerator type ("cpu", "gpu", etc.).
        device : int or str
            Device identifier.
        train_size : float
            Size of training set in the range [0.0, 1.0]. All cells by default.
        validation_size : float or None
            Size of the validation set. Zero cells by default.
        shuffle_set_split : bool
            Whether to shuffle indices before splitting.
        batch_size : int
            Minibatch size to use during training.
        early_stopping : bool
            Perform early stopping.
        lr : float or None
            Optimizer learning rate.
        load_sparse_tensor : bool | "auto"
            Whether to transfer data to GPU as sparse tensors (may speed up GPU transfer).
            On by default for "gpu" accelerator, otherwise off.
        training_plan : type
            Training plan class.
        plan_kwargs : dict or None
            Keyword args for the training plan.
        data_splitter_kwargs : dict or None
            Keyword args for the data splitter.
        trainer_kwargs : dict
            Other keyword args for the Trainer.

        Returns
        -------
        Any
            The result of the training runner.
        """
        _, _, torch_device = parse_device_args(accelerator, device, return_device="torch", validate_single_device=True)

        if not hasattr(self.module, "_guide") or self.module._guide is None:
            self.module._guide = self.module._guide_factory(
                self.module.model,
                init_values={
                    "log_gene_mean": self.module.log_gene_mean_init.to(torch_device),
                    "log_gene_dispersion": self.module.log_gene_dispersion_init.to(torch_device),
                    "element_effects": self.module.lfc_init.to(torch_device),
                },
            )

        # if self.module.local_effects and self.module.sparse_tensors:
        #     zero_element_effects = torch.zeros(
        #         (self.module.n_element_effects), dtype=torch.float32, device=torch_device
        #     )
        #     zero_guide_efficacy = torch.zeros((self.module.n_guide_effects), dtype=torch.float32, device=torch_device)
        # else:
        #     zero_element_effects = torch.zeros(
        #         (self.module.n_elements, self.module.n_genes), dtype=torch.float32, device=torch_device
        #     )
        #     zero_guide_efficacy = torch.full(
        #         (self.module.n_perturbations, self.module.n_genes),
        #         fill_value=0.5,
        #         dtype=torch.float32,
        #         device=torch_device,
        #     )

        if pretrain and indices is None:
            if self.control_guides is None:
                raise ValueError("Pretraining requires control_guides to be set during model initialization.")
            indices = np.where(
                self.adata_manager.get_from_registry(REGISTRY_KEYS.PERTURBATION_KEY)[:, self.control_guides].sum(axis=1)
                > 0
            )[0]
            assert isinstance(indices, np.ndarray) and len(indices) > 0, (
                "No cells with control guides found for pretraining."
            )

        if indices is not None:
            control_indices = [indices, np.setdiff1d(np.arange(self.summary_stats.n_cells), indices), None]
        else:
            control_indices = None
        # else:
        #     control_indices = None
        #     self.pretrain(
        #         max_epochs=max_epochs if pretrain_max_epochs is None else pretrain_max_epochs,
        #         indices=control_indices,
        #         accelerator=accelerator,
        #         device=device,
        #         batch_size=batch_size,
        #         lr=lr,
        #         **(pretrain_kwargs if pretrain_kwargs is not None else {}),
        #     )

        plan_kwargs = (
            plan_kwargs
            if plan_kwargs is not None
            else {"n_epochs_kl_warmup": None, "n_steps_kl_warmup": None, "scale_elbo": 1.0}
        )
        if blocked_sites is not None:
            plan_kwargs["blocked"] = blocked_sites
        if len(self.module.discrete_sites) > 0:
            plan_kwargs.update({"loss_fn": TraceEnum_ELBO(max_plate_nesting=3)})
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})
        if data_splitter_kwargs is None:
            data_splitter_kwargs = {}
        if "data_and_attributes" not in data_splitter_kwargs:
            data_splitter_kwargs["data_and_attributes"] = self.data_and_attrs
        if load_sparse_tensor == "auto":
            load_sparse_tensor = accelerator == "gpu"

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=1,
                external_indexing=control_indices,
                accelerator=accelerator,
                device=device,
                **data_splitter_kwargs,
            )
        else:
            data_splitter = DataSplitter(
                self.adata_manager,
                train_size=1,
                external_indexing=control_indices,
                batch_size=batch_size,
                load_sparse_tensor=load_sparse_tensor,
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

    def get_element_names(self) -> list:
        """
        Returns the names of the targeted elements.

        Returns
        -------
        list
            List of element names.
        """
        if REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY in self.adata_manager.data_registry:
            element_ids = self.adata_manager.get_state_registry(REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY).column_names
        else:
            element_ids = self.adata_manager.get_state_registry(REGISTRY_KEYS.PERTURBATION_KEY).column_names
        return element_ids

    def get_z_values(self, return_loc_scale=False) -> pd.DataFrame:
        """
        Return a DataFrame summary of the effects for targeted elements on each gene.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for effect location, scale, element, gene, z-value, and q-value.
        """
        element_ids = self.get_element_names()
        gene_ids = self.adata_manager.get_state_registry("X").column_names

        # Check if all element effects are factorized and raise an error if so
        if "element_effects" not in self.module.guide.median():
            raise NotImplementedError(
                "All element effects are factorized. Use 'get_factorized_element_effects' instead."
            )
        else:
            for guide in self.module.guide:
                if "element_effects" in guide.median():
                    loc_values, scale_values = guide._get_loc_and_scale("element_effects")
                    z_values = loc_values / scale_values

            # loc_values, scale_values = self.module.guide._get_loc_and_scale("element_effects")

            if hasattr(self.module, "element_by_gene_idx"):
                # loc/scale_values are the nonzero elements of a sparse matrix of elements by genes
                i, j = self.module.element_by_gene_idx.detach().cpu().numpy().astype(int)
                z_values_coo = coo_matrix(
                    (z_values.detach().cpu().numpy(), (i, j)),
                    shape=(len(element_ids), len(gene_ids)),
                )
                loc_values_coo = coo_matrix(
                    (loc_values.detach().cpu().numpy(), (i, j)),
                    shape=(len(element_ids), len(gene_ids)),
                )
                scale_values_coo = coo_matrix(
                    (scale_values.detach().cpu().numpy(), (i, j)),
                    shape=(len(element_ids), len(gene_ids)),
                )

                z_values_matrix = z_values_coo.to_csr()
                loc_values_matrix = loc_values_coo.to_csr()
                scale_values_matrix = scale_values_coo.to_csr()
            else:
                z_values_matrix = z_values.detach().cpu().numpy()
                loc_values_matrix = loc_values.detach().cpu().numpy()
                scale_values_matrix = scale_values.detach().cpu().numpy()

            z_values_df = pd.DataFrame(z_values_matrix, index=element_ids, columns=gene_ids)
            loc_values_df = pd.DataFrame(loc_values_matrix, index=element_ids, columns=gene_ids)
            scale_values_df = pd.DataFrame(scale_values_matrix, index=element_ids, columns=gene_ids)

        if return_loc_scale:
            return z_values_df, loc_values_df, scale_values_df
        return z_values_df

    def get_element_effects(self, return_long=True) -> pd.DataFrame:
        """
        Return a DataFrame summary of the effects for targeted elements on each gene.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for effect location, scale, element, gene, z-value, and q-value.
        """
        element_ids = self.get_element_names()
        gene_ids = self.adata_manager.get_state_registry("X").column_names

        def fast_long_df(mat, value_name):
            # mat: (n_elements, n_genes)
            n_elements, n_genes = mat.shape
            # Use numpy broadcasting for fast construction
            df = pd.DataFrame(
                {
                    "element": np.repeat(element_ids, n_genes),
                    "gene": np.tile(gene_ids, n_elements),
                    value_name: mat.ravel(),
                }
            )
            return df

        # Check if all element effects are factorized and raise an error if so
        if "element_effects" not in self.module.guide.median():
            logger.warning("All element effects are factorized. Using 'get_factorized_element_effects' instead.")
            pert_factors, pert_loadings = self.get_factorized_element_effects()
            element_effects = fast_long_df((pert_factors.T @ pert_loadings).values, "loc")
            element_effects["scale"] = np.nan
        else:
            for guide in self.module.guide:
                if "element_effects" in guide.median():
                    loc_values, scale_values = guide._get_loc_and_scale("element_effects")
            # loc_values, scale_values = self.module.guide._get_loc_and_scale("element_effects")

            if hasattr(self.module, "element_by_gene_idx"):
                # loc/scale_values are the nonzero elements of a sparse matrix of elements by genes
                i, j = self.module.element_by_gene_idx.detach().cpu().numpy().astype(int)
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
                element_effects = fast_long_df(loc_values.detach().cpu().numpy(), "loc")
                element_effects["scale"] = scale_values.detach().cpu().numpy().ravel()

        element_effects = element_effects.assign(
            z_value=lambda x: x["loc"] / x["scale"],
            q_value=lambda x: chi2.sf(x["z_value"] * x["z_value"], df=1),
        )

        return element_effects

    def get_map_labels(self, indices: list | None = None) -> np.ndarray:
        """
        Get MAP (maximum a posteriori) labels for the discrete latent variable "perturbed".

        Parameters
        ----------
        indices : list or None
            Indices of the data subset to use.

        Returns
        -------
        np.ndarray
            Array of MAP labels.
        """
        args, kwargs = self._get_data_subset(indices)
        self.module.guide(*args, **kwargs)
        guide_trace = poutine.trace(self.module.guide).get_trace(*args, **kwargs)  # record the globals
        trained_model = poutine.replay(self.module, trace=guide_trace)  # replay the globals
        serving_model = infer_discrete(trained_model, first_available_dim=-4, temperature=0)
        serving_model_trace = poutine.trace(serving_model).get_trace(*args, **kwargs)

        assert "perturbed" in serving_model_trace.nodes, "Only works if module has discrete latent variables"

        map_labels = serving_model_trace.nodes["perturbed"]["value"].squeeze().cpu().numpy()
        return map_labels

    def _get_data_subset(self, indices: list | None = None) -> tuple:
        """
        Get a data subset for inference.

        Parameters
        ----------
        indices : list or None
            Indices of the data subset.

        Returns
        -------
        tuple
            Tuple of (args, kwargs) for the model.
        """
        loader = AnnDataLoader(
            adata_manager=self.adata_manager,
            indices=indices,
            batch_size=len(indices) if indices is not None else len(self.adata),
            data_and_attributes=self.data_and_attrs,
        )
        return self.module._get_fn_args_from_batch(next(iter(loader)))

    # def sample_posterior(
    #     self,
    #     num_samples: int = 1,
    #     return_sites: list | None = None,
    #     accelerator: str = "auto",
    #     device: int | str = "auto",
    #     return_observed: bool = False,
    # ):
    #     _, _, device = parse_device_args(
    #         accelerator=accelerator, devices=device, return_device="torch", validate_single_device=True
    #     )

    #     args, kwargs = self._get_data_subset()
    #     args = [a.to(device) for a in args]
    #     kwargs = {k: v.to(device) for k, v in kwargs.items()}
    #     kwargs[REGISTRY_KEYS.X_KEY] = None
    #     self.to_device(device)

    #     samples = self._get_posterior_samples(
    #         args,
    #         kwargs=kwargs,
    #         num_samples=num_samples,
    #         return_sites=return_sites,
    #         return_observed=return_observed,
    #     )
    #     return samples

    def get_factorized_element_effects(self):
        element_ids = self.get_element_names()
        gene_ids = self.adata_manager.get_state_registry("X").column_names

        medians = self.module.guide.median()
        if "pert_factors" not in medians or "pert_loadings" not in medians:
            raise RuntimeError("No perturbation factors found. Use get_element_effects instead")

        pert_factors_2d = medians["pert_factors"].squeeze(-1).detach().cpu().numpy()
        assert pert_factors_2d.shape == (self.module.n_pert_factors, self.module.n_elements)
        pert_factors_df = pd.DataFrame(pert_factors_2d, columns=element_ids)
        pert_loadings_2d = medians["pert_loadings"].squeeze(-2).detach().cpu().numpy()
        assert pert_loadings_2d.shape == (self.module.n_pert_factors, self.module.n_genes)
        pert_loadings_df = pd.DataFrame(pert_loadings_2d, columns=gene_ids)
        return pert_factors_df, pert_loadings_df


def estimate_nb_params(
    X: np.ndarray | sp.spmatrix, smoothing: str = "isotonic"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate NB mean and dispersion (MoM) for each gene (column) in count matrix X.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.spmatrix
        Count matrix (cells x genes).
    smoothing : str
        Smoothing method: 'linear', 'quadratic', or 'isotonic'.

    Returns
    -------
    tuple of np.ndarray
        log_means, log_disp, log_disp_smoothed
    """
    if sp.issparse(X):
        means = np.array(X.mean(axis=0)).flatten()
        variances = np.array(X.power(2).mean(axis=0) - means**2).flatten()
    else:
        means = X.mean(axis=0)
        variances = X.var(axis=0, ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        dispersions = means**2 / (variances - means)

    dispersions[~np.isfinite(dispersions)] = np.nan
    dispersions[dispersions <= 0] = np.nan

    log_means = np.log(means + 1 / X.shape[0])  # avoid log(0)
    log_disp = np.log(dispersions)

    valid_mask = np.isfinite(log_means) & np.isfinite(log_disp)
    x_valid = log_means[valid_mask]
    y_valid = log_disp[valid_mask]

    if smoothing == "linear":
        model = LinearRegression()
        model.fit(x_valid.reshape(-1, 1), y_valid)
        log_disp_smoothed = model.predict(log_means.reshape(-1, 1))

    elif smoothing == "none":
        log_disp_smoothed = log_disp

    elif smoothing == "isotonic":
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(x_valid, y_valid)
        log_disp_smoothed = iso.predict(log_means)

    else:
        raise ValueError("smoothing must be 'linear', 'quadratic', or 'isotonic'")

    return log_means, log_disp, log_disp_smoothed
