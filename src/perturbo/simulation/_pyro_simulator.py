import numpy as np
import torch
from mudata import AnnData, MuData
from pyro.poutine import condition
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import parse_device_args

from perturbo.models._model import PERTURBO, REGISTRY_KEYS
from perturbo.models._module import PerTurboPyroModule


def simulate_data_from_trained_model(
    model: PERTURBO,
    guide_obs: torch.Tensor | np.ndarray,
    guide_by_element: torch.Tensor | np.ndarray,
    element_by_gene_lfc: torch.Tensor | np.ndarray,
    guide_efficacy: torch.Tensor | np.ndarray,
    read_depth_adjust_factor: float = 1.0,
    module_kwargs: dict | None = None,
    module_init_kwargs: dict | None = None,
    gene_indices: torch.Tensor | np.ndarray = None,
    cell_indices: torch.Tensor | np.ndarray = None,
    param_values: dict | None = None,
    accelerator: str = "auto",
    device: int | str = "auto",
    # gene_ids = , # TODO: allow subsampling based on gene ids instead of indices
) -> np.ndarray:
    """
    Simulate data from a trained PerTurbo model.

    Using MAP estimates of latent variables except where overriden by user. Supports subsampling of cells and genes, and simulation
    of unseen guides and elements with user-defined efficacies and effect sizes.

    Parameters
    ----------
    model : PERTURBO
        Model to use for simulation
    guide_obs : ArrayLike
        Observed guides for simulation (n_cells x n_guides)
    guide_by_element : ArrayLike
        Guide to element map for simulation (n_guides x n_elements)
    element_by_gene_lfc : ArrayLike
        Effect of each element on each gene for simulation (n_elements x n_genes)
    guide_efficacy : ArrayLike
        Guide efficacy values [0,1]^(n_cells,)
    module_kwargs : dict | None, optional
        Any additional keyword arguments to provide to module during simulation
    module_init_kwargs : dict | None, optional
        Any additional keyword arguments to provide to PerTurboPyroModule constructor
    gene_indices : ArrayLike, optional
        Indices of genes from original dataset to use in simulation
    param_values : dict | None, optional
        Dict of additional param values to override model MAP inferences
    accelerator : str, optional
        Accelerator to use with Lightning, by default "auto"
    device : int | str, optional
        Device for all tensors, by default "auto"

    Returns
    -------
    np.ndarray
        Simulated counts (n_cells x n_genes)
    """
    # part 1: get parameters for simulations from module and user
    _, _, device = parse_device_args(
        accelerator=accelerator, devices=device, return_device="torch", validate_single_device=True
    )

    n_guides, n_elements = guide_by_element.shape
    n_cells, n_guides = guide_obs.shape
    assert guide_obs.shape[1] == guide_by_element.shape[0]
    assert element_by_gene_lfc.shape[0] == guide_by_element.shape[1]
    assert element_by_gene_lfc is not None and guide_efficacy is not None
    guide_sites_to_discard = ["element_effects", "guide_efficacy", "guide_effects", "perturbed"]
    cell_latents = ["cell_factors"]

    # get data args for a subset of cells
    if cell_indices is None:
        if n_cells != model.module.n_cells:
            cell_indices = np.random.randint(model.module.n_cells, size=n_cells)
        else:
            cell_indices = np.arange(model.module.n_cells)

    (idx,), kwargs = _get_data_subset(model, cell_indices)

    # new cell indices should just be 1 to n_samples for subsampling purposes
    args = (torch.arange(n_cells).to(device=device),)
    assert args[0].shape == idx.shape

    # load model kwargs from data subset (e.g. covariates, size factors)
    kwargs = {k: v.to(device) for k, v in kwargs.items()}
    kwargs[REGISTRY_KEYS.PERTURBATION_KEY] = torch.tensor(guide_obs, dtype=torch.float32).to(device)
    kwargs[REGISTRY_KEYS.X_KEY] = None

    if module_init_kwargs is None:
        module_init_kwargs = {}

    if module_kwargs is not None:
        kwargs.update(module_kwargs)

    guide_by_element = torch.tensor(guide_by_element, dtype=torch.float32, device=device)

    if gene_indices is None:
        n_genes_new = model.module.n_genes
        gene_indices = slice(n_genes_new)
    else:
        gene_indices_tensor = torch.tensor(gene_indices, dtype=torch.long).to(device)
        n_genes_new = gene_indices.shape[0]

    # get MAP values for latents from guide then override with any user-provided values
    latent_vars = {k: v.to(device) for k, v in model.module.guide.median().items() if k not in guide_sites_to_discard}

    latent_vars["log_gene_mean"] += np.log(read_depth_adjust_factor)
    for param_name, param_value in latent_vars.items():
        if param_name in cell_latents:
            latent_vars[param_name] = param_value[..., idx, :]
        elif (
            len(param_value.shape) > 0
            and param_value.shape[-1] == model.module.n_genes
            and n_genes_new != model.module.n_genes
        ):
            # subset gene indices for gene-specific latents
            latent_vars[param_name] = param_value[..., gene_indices_tensor]

    if element_by_gene_lfc is not None:
        assert element_by_gene_lfc.shape[1] == n_genes_new
        element_by_gene_lfc = torch.tensor(element_by_gene_lfc, dtype=torch.float32, device=device)
        latent_vars["element_effects"] = element_by_gene_lfc

    guide_efficacy = torch.tensor(guide_efficacy, dtype=torch.float32, device=device)
    if guide_efficacy.shape == (n_guides,):
        guide_efficacy = guide_efficacy.unsqueeze(-1)
    assert guide_efficacy.shape == (n_guides, 1)
    latent_vars["guide_efficacy"] = guide_efficacy

    if param_values is not None:
        latent_vars.update(param_values)

    module_kwargs = {
        "n_batches": model.module.n_batches,
        "n_cont_covariates": model.module.n_cont_covariates - 1,  # size factor auto included
        "n_factors": model.module.n_factors,
        # "dispersion_effects": model.module.dispersion_effects,
        "likelihood": model.module.likelihood,
        "effect_prior_dist": model.module.effect_prior_dist,
        # "use_interactions": model.module.use_interactions,
        "efficiency_mode": model.module.efficiency_mode,
    }

    if module_init_kwargs is not None:
        module_kwargs.update(module_init_kwargs)

    # create new module to sample from
    module_new = PerTurboPyroModule(
        n_cells=n_cells,
        n_genes=n_genes_new,
        n_elements=n_elements,
        n_perturbations=n_guides,
        guide_by_element=guide_by_element,
        **module_kwargs,
    )
    module_new.to(device)

    # sample counts matrix from conditioned model
    conditioned_model = condition(module_new, data=latent_vars)
    sampled_counts = conditioned_model(*args, **kwargs).squeeze().detach().cpu().numpy()

    # Create an AnnData object to return
    data_registry = model.adata_manager.data_registry
    rna_key = data_registry[REGISTRY_KEYS.X_KEY].mod_key
    grna_key = data_registry[REGISTRY_KEYS.PERTURBATION_KEY].mod_key
    guide_by_element_key = (
        data_registry[REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY].attr_key
        if REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY in data_registry
        else "targeted_elements"
    )
    gene_by_element_key = "lfc"  # create new field with gnee_by_element info

    obs_new = model.adata[rna_key].obs.iloc[cell_indices, :].reset_index()
    var_new = model.adata[rna_key].var.iloc[gene_indices, :]
    rna_adata = AnnData(
        X=sampled_counts,
        obs=obs_new,
        var=var_new,
        varm={gene_by_element_key: element_by_gene_lfc.T.detach().cpu().numpy()},
    )

    grna_adata = AnnData(
        X=guide_obs,
        varm={
            guide_by_element_key: guide_by_element.detach().cpu().numpy(),
        },
    )

    mdata_new = MuData({rna_key: rna_adata, grna_key: grna_adata})
    return mdata_new


def _get_data_subset(model: PERTURBO, indices: list | None = None):
    loader = AnnDataLoader(
        adata_manager=model.adata_manager,
        indices=indices,
        batch_size=len(indices) if indices is not None else len(model.adata),
        data_and_attributes=model.data_and_attrs,
    )
    return model.module._get_fn_args_from_batch(next(iter(loader)))
