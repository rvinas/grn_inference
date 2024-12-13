# scvi-tools tutorial: https://docs.scvi-tools.org/en/1.0.0/tutorials/notebooks/model_user_guide.html
import torch
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi import REGISTRY_KEYS
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    auto_move_data,
)
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from torch.distributions import NegativeBinomial, Normal
from torch.distributions import kl_divergence as kl
from torch.nn.functional import one_hot
from scvi.distributions import (
    ZeroInflatedNegativeBinomial,
)
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.distributions.utils import logits_to_probs


class NeuralNet(torch.nn.Module):
    def __init__(
            self,
            n_input,
            n_output,
            link_var='exp',
            hdim=256  # 128
    ):
        """
        Encodes data of ``n_input`` dimensions into a space of ``n_output`` dimensions.

        Uses a one layer fully-connected neural network with 128 hidden nodes.

        Parameters
        ----------
        n_input
            The dimensionality of the input
        n_output
            The dimensionality of the output
        link_var
            The final non-linearity
        """
        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, hdim),
            torch.nn.ReLU(),
            torch.nn.Linear(hdim, n_output),
        )
        self.transformation = None
        if link_var == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)
        elif link_var == "exp":
            self.transformation = torch.exp

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)
        if self.transformation:
            output = self.transformation(output)
        return output


class ZeroInflatedNegativeBinomialResponseModule(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    n_latent
        Dimensionality of the latent space
    """

    def __init__(
            self,
            input_dim: int,
            out_dim: int,
            n_cell_type: int,
            n_perturbation_type: int,
            n_batch: int,
            null_model = False,
            n_latent = 256,
            **kwargs
    ):
        super().__init__()
        # in the init, we create the parameters of our elementary stochastic computation unit.
        self.n_cell_type = n_cell_type
        self.n_perturbation_type = n_perturbation_type
        self.n_batch = n_batch
        self.null_model = null_model

        n_latent_covs = n_latent + self.n_cell_type + self.n_perturbation_type + self.n_batch

        # Setup the parameters of the conditional model
        self.scale_decoder =  torch.nn.Sequential(torch.nn.Linear(n_latent_covs, out_dim), torch.nn.Softmax(dim=-1))  # NeuralNet(n_latent, out_dim, "softmax")  # torch.nn.Sequential(torch.nn.Linear(input_dim, out_dim), torch.nn.Softmax(dim=-1))
        self.log_theta_decoder = torch.nn.Linear(n_latent_covs, out_dim)   # NeuralNet(n_latent, out_dim, "linear")  # torch.nn.Linear(input_dim, out_dim)
        self.dropout_decoder = torch.nn.Linear(n_latent_covs, out_dim)  # NeuralNet(n_latent, out_dim, "linear")  # torch.nn.Linear(input_dim, out_dim) # 
        # self.log_theta = torch.nn.Parameter(torch.randn(out_dim))
        self.encoder = torch.nn.Sequential(torch.nn.Linear(input_dim, n_latent), torch.nn.ReLU(), torch.nn.Linear(n_latent, n_latent), torch.nn.ReLU()) # NeuralNet(input_dim, n_latent, "none")

    def _get_inference_input(self, tensors):
        return dict()

    @auto_move_data
    def inference(self):
        return dict()

    def _get_generative_input(self, tensors, inference_outputs):
        # We extract the observed library size
        x = tensors[REGISTRY_KEYS.X_KEY]
        library = torch.sum(x, dim=1, keepdim=True)
        
        # Obtain conditions
        log_doses = tensors['log_doses']
        cell_type = one_hot(tensors['cell_type'][:, 0].long(), self.n_cell_type)
        pert_type = one_hot(tensors['perturbation_type'][:, 0].long(), self.n_perturbation_type)
        batch = None
        if REGISTRY_KEYS.BATCH_KEY in tensors:
            batch = one_hot(tensors[REGISTRY_KEYS.BATCH_KEY][:, 0].long(), self.n_batch).float()

        input_dict = dict(log_doses=log_doses,
                          cell_type=cell_type,
                          pert_type=pert_type,
                          library=library,
                          batch=batch)

        return input_dict

    @auto_move_data
    def generative(self, log_doses, cell_type, pert_type, library, batch=None, add_batch_effect=False):
        """Runs the generative model."""
        x = log_doses # torch.cat([log_doses, cell_type, pert_type], dim=-1)
        """if batch is not None:
            x = torch.cat([x, batch], dim=-1)"""
        z = self.encoder(x)

        # Null model
        if self.null_model:
            z = torch.zeros_like(z)

        z = torch.cat([z, cell_type, pert_type], dim=-1)
        if batch is not None:
            z = torch.cat([z, batch], dim=-1)

        # get the "normalized" mean of the negative binomial
        """ Old
        if self.null_model:
            z_null = torch.zeros_like(z)
            px_scale = self.scale_decoder(z_null)
        else:
            px_scale = self.scale_decoder(z)
        """
        px_scale = self.scale_decoder(z)

        # get the mean of the negative binomial
        px_rate = library * px_scale
        # get the dispersion parameter
        log_theta = self.log_theta_decoder(z)
        theta = torch.exp(log_theta)
        # theta = torch.exp(self.log_theta)
        # get the dropout logits
        gate_logits = self.dropout_decoder(z)
        return dict(px_scale=px_scale,
                    theta=theta,
                    px_rate=px_rate,
                    gate_logits=gate_logits,
                    library=library)
        
    def loss(self,
            tensors,
            inference_outputs,
            generative_outputs,
            eps=1e-6
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        theta = generative_outputs["theta"]
        gate_logits = generative_outputs["gate_logits"]

        # term 1
        log_lik = (
            ZeroInflatedNegativeBinomial(mu=px_rate, theta=theta, zi_logits=gate_logits)
            .log_prob(x)
            .sum(dim=-1)
        )

        loss = torch.mean(-log_lik)
        return LossOutput(loss=loss, reconstruction_loss=-log_lik)
    
    def approximate_expectation_log1p(self, zinb_mean, zinb_var, observed_library, library_size=1e4):
        """
        Approximation of E[log(a + bX)] via Taylor expansion. We calculate the expectation of the log1p total-count normalized data,
        i.e. we set a=1 and b to 1. In contrast to modelling log-normalized data, this approach
        allows to account for zero-inflation and over-dispersed data.
        """
        log_norm_mean = torch.log1p(zinb_mean) - zinb_var / (2 * (1 + zinb_mean) ** 2)
        return log_norm_mean
    
    def approximate_variance_log1p(self, zinb_mean, zinb_var, observed_library, library_size=1e4):
        """
        Approximation of Var[log(a + bX)] via Taylor expansion. We calculate the expectation of the log1p total-count normalized data,
        i.e. we set a=1 and b to the ratio (library_size / observed library). In contrast to modelling log-normalized data, this approach
        allows to account for zero-inflation and over-dispersed data.
        """
        log_norm_var = zinb_var / (1 + zinb_mean) ** 2
        return log_norm_var


    def predict(self, tensors, library_size, inference_kwargs, generative_kwargs):
        """
        Infers parameters of predictive distribution and draws samples from the distribution.
        
        **Important**: library_size is the desired library size. The mean and variance of the predictive ZINB distribution will be calculated
        accordingly. The log-likelihoods are computed using the observed library size
        """
        _, generative_outputs = self.forward(
            tensors=tensors,
            inference_kwargs=inference_kwargs,
            generative_kwargs=generative_kwargs,
            compute_loss=False,
        )
    
        # Get parameters
        px_rate = generative_outputs["px_rate"]
        px_scale = generative_outputs["px_scale"]
        theta = generative_outputs["theta"]
        gate_logits = generative_outputs["gate_logits"]
        observed_library = generative_outputs["library"]
        px_rate_desired_library = px_scale * library_size
        lik = ZeroInflatedNegativeBinomial(mu=px_rate,
                                           theta=theta,
                                           zi_logits=gate_logits)
        dropout_probs = logits_to_probs(gate_logits)
        zinb_mean = (1 - dropout_probs) * px_rate_desired_library
        # zinb_var = (1 - dropout_probs) * px_rate_desired_library * (px_rate_desired_library * alpha + px_rate_desired_library * dropout_probs + 1) / (px_rate_desired_library*alpha + 1)
        # zinb_var = (1 - dropout_probs) * px_rate_desired_library * (px_rate_desired_library * alpha + px_rate_desired_library * dropout_probs + 1) / (px_rate_desired_library*alpha + 1)
        zinb_var = (1 - dropout_probs) * px_rate_desired_library * (px_rate_desired_library + theta + dropout_probs*theta*px_rate_desired_library)

        # Prepare outputs
        x = tensors[REGISTRY_KEYS.X_KEY].to(px_rate.device)
        out = generative_outputs
        out['samples'] = lik.sample()
        out['log_norm_mean'] = self.approximate_expectation_log1p(zinb_mean, zinb_var, observed_library, library_size)
        out['log_norm_var'] = self.approximate_variance_log1p(zinb_mean, zinb_var, observed_library, library_size)
        out['log_likelihood'] = lik.log_prob(x)
        out['dropout_probs'] = dropout_probs
        out['zinb_mean'] = zinb_mean
        out['zinb_var'] = zinb_var
        return out

"""
Response model
"""
class ResponseModel(UnsupervisedTrainingMixin, BaseModelClass):
    """
    """

    def __init__(
            self,
            adata: AnnData,
            likelihood_fn='normal',
            n_latent: int = 32,
            null_model = False,
            **model_kwargs,
    ):
        super().__init__(adata)

        self.likelihood_fn = likelihood_fn
        """
        input_dim = adata.obsm['log_doses'].shape[-1] +\
            self.summary_stats["n_cell_type"] +\
            self.summary_stats["n_perturbation_type"] +\
            self.summary_stats["n_batch"]
        """
        input_dim = adata.obsm['log_doses'].shape[-1]

        # VAE
        self.module = ZeroInflatedNegativeBinomialResponseModule(
            input_dim=input_dim,
            out_dim=self.summary_stats["n_vars"],
            n_cell_type=self.summary_stats["n_cell_type"],
            n_perturbation_type=self.summary_stats["n_perturbation_type"],
            n_batch=self.summary_stats["n_batch"],
            null_model=null_model,
            n_latent=n_latent,
            # likelihood_fn=likelihood_fn,
            **model_kwargs,
        )

        self._model_summary_string = (
            "Response Model with the following params: \nn_latent: {}"
        ).format(
            n_latent,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
            cls,
            adata: AnnData,
            log_doses_key='log_doses',
            pert_key='perturbation_type',
            cell_type_key='cell_type',
            batch_key='batch',
            layer='counts',
            **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer),  # , is_count_data=True
            # CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
            ObsmField('log_doses', log_doses_key),
            CategoricalObsField('cell_type', cell_type_key),
            CategoricalObsField('perturbation_type', pert_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        ]

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def infer_expression(
            self,
            adata=None,
            indices=None,
            gene_list=None,
            library_size=1e4,
            batch_size=None,
    ):
        r"""Returns samples, log-likelihood, and parameters of the predictive distribution
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        weights
            Weights to use for sampling. If `None`, defaults to `"uniform"`.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        Dictionary with gene expression samples, log-likelihood, and parameters of the predictive distribution
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)

        # Perform predictions
        outs = defaultdict(lambda: [])
        output_keys = ['samples', 'zinb_mean', 'zinb_var', 'log_norm_mean', 'log_norm_var', 'px_scale', 'px_rate', 'theta', 'dropout_probs', 'log_likelihood']
        for tensors in tqdm(scdl):
            generative_kwargs = {'add_batch_effect': False}
            inference_kwargs = None
            out = self.module.predict(tensors, library_size, inference_kwargs=inference_kwargs, generative_kwargs=generative_kwargs)
            for k in output_keys:
                outs[k].append(out[k][..., gene_mask].cpu().numpy())

        # Store each element in a separate pandas dataframe
        for k, v in outs.items():
            outs[k] = pd.DataFrame(
                np.concatenate(v, axis=0),
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        return outs