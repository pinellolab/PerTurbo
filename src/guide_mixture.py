import argparse
import math

import torch
import torch.distributions as tdist
from torch.distributions import constraints

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal


# helper for creating a matrix that is useful for dealing with the combinatorial space of guide presence/absence.
def compute_combinatorial_matrix(args):
    combinatorial_matrix = torch.zeros(args.max_targeting_cell, 2 ** args.max_targeting_cell)

    for k in range(args.max_targeting_cell):
        zeros_and_ones = torch.cat([torch.zeros(2 ** k), torch.ones(2 ** k)])
        num_repeat = combinatorial_matrix.size(-1) // (2 ** (k + 1))
        combinatorial_matrix[k] = zeros_and_ones.repeat(num_repeat)

    assert (combinatorial_matrix == 0).sum() == (combinatorial_matrix == 1).sum()
    return combinatorial_matrix


def assemble_likelihood_and_compute_kl(data, args, beta, targeting_efficiencies, cell_guide_presence_prob, nu):
    index = data['cell_gene_to_guide'].view(args.num_cells, -1)
    log_half = math.log(0.5)
    # we append a value of log(0.5) so that indices that are equal to args.num_guides get mapped to this dummy value.
    log_cell_guide_presence_prob = torch.cat([cell_guide_presence_prob.log(), log_half * torch.ones(args.num_cells, 1)], dim=-1)
    log_cell_guide_presence_prob = log_cell_guide_presence_prob.gather(-1, index).view(args.num_cells, args.num_genes, args.max_targeting_cell)
    log1p_cell_guide_presence_prob = torch.cat([torch.log1p(-cell_guide_presence_prob), log_half * torch.ones(args.num_cells, 1)], dim=-1)
    log1p_cell_guide_presence_prob = log1p_cell_guide_presence_prob.gather(-1, index).view(args.num_cells, args.num_genes, args.max_targeting_cell)

    # matrix multiplication in log probability space is multiplication in probability space.
    # basically we're computing terms of the form log(p(1-q)rs(1-t)) where pqrst are all probabilities.
    # the probabilities cell_guide_presence_prob are the variational parameters that represent *inferred* cell-level quantities.
    mix_log_probs = log_cell_guide_presence_prob @ data['combinatorial_matrix'] + log1p_cell_guide_presence_prob @ data['flip_combinatorial_matrix']
    assert mix_log_probs.shape == (args.num_cells, args.num_genes, 2 ** args.max_targeting_cell)

    # we append to beta so that indices that are equal to -1 get mapped to zero, since missing guides have no effect.
    beta = torch.cat([beta, torch.zeros(1)])[data['cell_gene_to_beta']]
    assert beta.shape == (args.num_cells, args.num_genes, args.max_targeting_cell)
    beta_sum = beta @ data['combinatorial_matrix']
    assert beta_sum.shape == (args.num_cells, args.num_genes, 2 ** args.max_targeting_cell)
    nb_logits = beta_sum - nu.log()

    # this distribution encodes the probabilities of each mixture component.
    mix_cat = tdist.Categorical(logits=mix_log_probs)
    # this distribution encodes all the mixture components for all combinations of present and "maybe active" guides.
    # note you may conceivably want a different parameterization of NegativeBinomial.
    # question: do we drop cells that have zero observed guides?
    mix_comp = tdist.NegativeBinomial(total_count=nu, logits=nb_logits)

    # form our likelihood, which is a mixture distribution with 2 ** args.max_targeting_cell components.
    lkl = dist.MixtureSameFamily(mix_cat, mix_comp)
    assert lkl.batch_shape == (args.num_cells, args.num_genes)

    # since we are being variational about the cell-level binary latent variables that encode guide presence/absence, we need to
    # include a KL divergence term that effectively regularizes cell_guide_presence_prob towards the model-side targeting_efficiencies.

    # we append a value of log(0.5) so that indices that are equal to args.num_guides get mapped to this dummy value.
    log_targeting_efficiencies = torch.cat([targeting_efficiencies.log(), torch.tensor([log_half])])
    log_targeting_efficiencies = log_targeting_efficiencies[data['cell_gene_to_guide']]
    assert log_targeting_efficiencies.shape == (args.num_cells, args.num_genes, args.max_targeting_cell)
    log1p_targeting_efficiencies = torch.cat([torch.log1p(-targeting_efficiencies), torch.tensor([log_half])])
    log1p_targeting_efficiencies = log1p_targeting_efficiencies[data['cell_gene_to_guide']]

    # compute the kl regularizer
    p = tdist.Bernoulli(probs=cell_guide_presence_prob)
    q = tdist.Bernoulli(probs=targeting_efficiencies.expand(cell_guide_presence_prob.shape))
    kl = torch.distributions.kl.kl_divergence(q, p)
    assert kl.shape == (args.num_cells, args.num_guides)

    # we need to mask out terms that correspond to guides that do not "maybe appear" in a given cell, i.e.
    # those for which the relevant entry of cell_guide_presence_prob is a phantom.
    kl = data['cell_guide_mask'] * kl

    return lkl, kl


def model(data, args):
    # controls variance of NegativeBinomial distributions
    nu = pyro.param("nu", torch.ones(1), constraint=constraints.positive)
    targeting_efficiencies = pyro.param("targeting_efficiencies", 0.5 * torch.ones(args.num_guides),
                                        constraint=constraints.unit_interval)
    with pyro.plate("total_guide_gene_interactions", data['total_guide_gene_interactions']):
        beta = pyro.sample("beta", dist.Normal(0.0, 0.1))

    # this is technically a variational parameter but we include it in the model because we're
    # effectively doing variational inference over discrete latent variables by hand.
    # (the guide handles variational inference over continuous latent variables.)
    # not all of these will be used in practice but we encode as a dense matrix for simplicity.
    cell_guide_presence_prob = pyro.param("cell_guide_presence_prob", 0.5 * torch.ones(args.num_cells, args.num_guides),
                                          constraint=constraints.unit_interval)

    lkl, kl = assemble_likelihood_and_compute_kl(data, args, beta, targeting_efficiencies, cell_guide_presence_prob, nu)

    with pyro.plate("cells", args.num_cells, dim=-2):
        with pyro.plate("genes", args.num_genes, dim=-1):
            pyro.sample("obs", lkl, obs=data['Y'])
        with pyro.plate("guides", args.num_guides, dim=-1):
            # include kl regularizer, which appears with a minus sign in the elbo
            pyro.factor("kl", -kl)


def main(args):
    # create fake data.
    # note data creation code is not written in a parallelized fashion since it's just for prototyping.

    # gene expression counts
    Y = torch.randint(500, (args.num_cells, args.num_genes))

    # for each cell c, each row of cell_gene_to_guide[c] is a set of indices, each of which points to a unique guide.
    # the special value of args.num_guides is used to encode missing guides.
    cell_gene_to_guide = torch.randint(args.num_guides, (args.num_cells, args.num_genes, args.max_targeting_cell))
    for cell in range(args.num_cells):
        gene_to_guide = cell_gene_to_guide[cell]
        for row, indices in enumerate(gene_to_guide):
            num_duplicates = len(indices) - len(torch.unique(indices))
            # replace duplicate indices with args.num_guides, which corresponds to "not a valid index".
            # in other words this is a way of creating a dataset where some genes in some cells are targeted by
            # fewer than max_targeting_cell guides. we need this represented in our dataset so we can test
            # the necessary masking machinery.
            if num_duplicates > 0:
                cell_gene_to_guide[cell, row] = torch.cat([torch.unique(indices), args.num_guides * torch.ones(num_duplicates)])
    assert (cell_gene_to_guide == args.num_guides).sum() > 0

    # we will represent beta as a flat vector to avoid instantiating unused random variables.
    # so we need to create data structures to support the relevant indexing. that data structure is cell_gene_to_beta.
    # in particular the size of beta will be total_guide_gene_interactions. in practice total_guide_gene_interactions should
    # be given by the total number of unique (guide, gene) interaction pairs. however, since this is fake data, and
    # we do not try to create totally coherent data, we just choose an arbitrary number here. of course in reality
    # cell_gene_to_guide, cell_gene_to_beta, and total_guide_gene_interactions need to be self-consistent.
    total_guide_gene_interactions = 3 * args.num_guides
    # for each cell c, each row of cell_gene_to_beta[c] is a set of indices, each of which points to a unique guide-gene interaction.
    cell_gene_to_beta = torch.randint(total_guide_gene_interactions, (args.num_cells, args.num_genes, args.max_targeting_cell))
    # randomly make some of the indices -1, which encodes missing guides.
    # this is the one place where we retain "-1" as the missing index indicator.
    cell_gene_to_beta[torch.rand(cell_gene_to_beta.shape) < 0.1] = -1

    # this controls which cells maybe have--e.g. have at least one NGS read supporting presence--each of the args.num_guides-many guides.
    # one represents "maybe present" and zero represents "definitely not present." the "maybe" is why we need to do inference
    # over discrete latent variables. obviously in practice this data structure needs to be consistent with cell_gene_to_guide.
    cell_guide_mask = torch.rand(args.num_cells, args.num_guides) < 0.5

    data = {
        'Y': Y,
        'cell_gene_to_guide': cell_gene_to_guide,
        'cell_gene_to_beta': cell_gene_to_beta,
        'cell_guide_mask': cell_guide_mask,
        'total_guide_gene_interactions': total_guide_gene_interactions,
        'combinatorial_matrix': compute_combinatorial_matrix(args),
        'flip_combinatorial_matrix': 1 - compute_combinatorial_matrix(args),
    }

    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam({'lr': 0.001}), loss=TraceMeanField_ELBO())

    for k in range(args.num_epochs):
        loss = svi.step(data, args)

        if k % 50 == 0 or k == args.num_epochs - 1:
            print("[epoch %04d] training elbo: %.6g" % (k, -loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--num-genes", default=6, type=int)
    parser.add_argument("--num-guides", default=4, type=int)
    parser.add_argument("--num-cells", default=8, type=int)
    parser.add_argument("--max-targeting-cell", default=3, type=int)
    main(parser.parse_args())
