from src.data import load_scTFseq

import numpy as np
from scipy.stats import chi2
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
from tqdm import tqdm
import random
import jax
import jax.numpy as jnp
import jax.scipy
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import gammaln, expit
import jaxopt
import matplotlib.pyplot as plt
import seaborn as sns
import functools

class PairwiseResponseModel:
    @staticmethod
    def he_init(m, n, key):
        initializer = jax.nn.initializers.he_uniform()
        return initializer(key, (n, m)), jnp.zeros((n,))

    @staticmethod
    def init_network_params(sizes, key):
        keys = random.split(key, len(sizes))
        return [PairwiseResponseModel.he_init(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    @staticmethod
    @functools.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
    def batched_forward(params, x, library):
        activations = x
        for w, b in params[:-1]:
            h = jnp.dot(w, activations) + b
            activations = jax.nn.swish(h)
        final_w, final_b = params[-1]
        out = jnp.dot(final_w, activations) + final_b
        
        # ZINB params
        """
        mus, thetas, pis = jnp.split(out, 3, axis=-1)
        print(mus)
        mu = expit(mus) * library  # jnp.exp(out[0])
        theta = jnp.exp(jnp.clip(thetas, max=8))
        pi = pis
        """
        # ZINB params  
        mu = expit(out[0]) * library # jnp.exp(out[0])  
        theta = jnp.exp(jnp.clip(out[1], max=8))  
        pi = out[2]  
        return mu, theta, pi

    @staticmethod
    @functools.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
    def batched_multidim_forward(params, x, library):
        activations = x
        for w, b in params[:-1]:
            h = jnp.dot(w, activations) + b
            activations = jax.nn.swish(h)
        final_w, final_b = params[-1]
        out = jnp.dot(final_w, activations) + final_b
        
        # ZINB params
        mus, thetas, pis = jnp.split(out, 3, axis=-1)
        mu = jax.nn.softmax(mus) * library  # jnp.exp(out[0])
        theta = jnp.exp(jnp.clip(thetas, max=8))
        pi = pis
        
        return mu, theta, pi

    @staticmethod
    def log_zinb_positive(x, mu, theta, pi, eps: float = 1e-8):
        """Log likelihood (scalar) according to a zinb model. Parameters:
        * x: Data
        * mu: mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
        * theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
        * pi: logit of the dropout parameter (real support) (shape: minibatch x vars)
        * eps numerical stability constant
        Edited from scvi-tools
        """
        # Uses log(sigmoid(x)) = -softplus(-x)
        softplus_pi = jax.nn.softplus(-pi)
        log_theta_eps = jnp.log(theta + eps)
        log_theta_mu_eps = jnp.log(theta + mu + eps)
        
        # Zero-inflation case
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)
        log_prob_zero = jax.nn.softplus(pi_theta_log) - softplus_pi
        zero_case = jnp.where(x == 0, log_prob_zero, 0.0)

        log_prob_nb = (
            -softplus_pi
            + pi_theta_log
            + x * (jnp.log(mu + eps) - log_theta_mu_eps)
            + gammaln(x + theta)
            - gammaln(theta)
            - gammaln(x + 1)
        )
        non_zero_case = jnp.where(x > 0, log_prob_nb, 0.0)
        res = zero_case + non_zero_case
        return res

    @staticmethod
    @jax.jit
    def objective(params, x, y, library):
        mu, theta, pi = PairwiseResponseModel.batched_forward(params, x, library)
        ll = PairwiseResponseModel.log_zinb_positive(y, mu, theta, pi)
        return -jnp.mean(ll)
    
    @staticmethod
    @jax.jit
    def multidim_objective(params, x, y, library):
        mu, theta, pi = PairwiseResponseModel.batched_multidim_forward(params, x, library)
        ll = PairwiseResponseModel.log_zinb_positive(y, mu, theta, pi)
        return -jnp.mean(jnp.sum(ll, axis=-1))
    
    @staticmethod
    def approximate_expectation_log1p(zinb_mean, zinb_var):
        """
        Approximation of E[log(a + bX)] via Taylor expansion. We calculate the expectation of the log1p total-count normalized data,
        i.e. we set a=1 and b to 1. In contrast to modelling log-normalized data, this approach
        allows to account for zero-inflation and over-dispersed data.
        """
        log_norm_mean = np.log1p(zinb_mean) - zinb_var / (2 * (1 + zinb_mean) ** 2)
        return log_norm_mean

    @staticmethod
    def approximate_variance_log1p(zinb_mean, zinb_var):
        """
        Approximation of Var[log(a + bX)] via Taylor expansion. We calculate the expectation of the log1p total-count normalized data,
        i.e. we set a=1 and b to the ratio (library_size / observed library). In contrast to modelling log-normalized data, this approach
        allows to account for zero-inflation and over-dispersed data.
        """
        log_norm_var = zinb_var / (1 + zinb_mean) ** 2
        return log_norm_var

    def __init__(self, hdim=4, n_out=1, max_iter=1000):
        super().__init__()
        self.n_out = n_out
        layer_sizes = [1, hdim, 3*n_out]
        self.params = PairwiseResponseModel.init_network_params(layer_sizes, random.key(0))
        self.max_iter=max_iter
        self.fitted = False

    def fit(self, x, y, library, method='GradientDescent'):
        objective = PairwiseResponseModel.objective
        if self.n_out > 1:
            objective = PairwiseResponseModel.multidim_objective
        loss_and_grads = jax.value_and_grad(objective)
        jit_loss_and_grads = jax.jit(loss_and_grads)

        # Choose solver
        if method == 'LBFGS':
            solver = jaxopt.LBFGS(fun=jit_loss_and_grads,
                              value_and_grad=True,
                              maxiter=self.max_iter)
        elif method == 'GradientDescent':
            solver = jaxopt.GradientDescent(fun=jit_loss_and_grads,
                                value_and_grad=True,
                                maxiter=self.max_iter)
        else:
            raise ValueError(f'Invalid method: {method}')

        sol = solver.run(self.params, x=x, y=y, library=library)
        self.params = sol.params
        self.fitted = True

    def predict(self, x, library, y=None, desired_library=1e6):
        # assert self.fitted, 'Model not fitted'
        out = {}

        # Get parameters
        if self.n_out > 1:
            mu, theta, pi = PairwiseResponseModel.batched_multidim_forward(self.params, x, library)
        else:
            mu, theta, pi = PairwiseResponseModel.batched_forward(self.params, x, library)
            mu = mu[:, None]
            theta = theta[:, None]
            pi = pi[:, None]
        dropout_prob = expit(pi)
        out['mu'] = mu
        out['theta'] = theta
        out['dropout_probs'] = dropout_prob

        # Get distribution mean and variance
        mu_desired_library = mu * desired_library / library[:, None]
        out['zinb_mean'] = (1 - dropout_prob) * mu_desired_library
        out['zinb_var'] = (1 - dropout_prob) * mu_desired_library * (mu_desired_library + theta + dropout_prob*theta*mu_desired_library)
        out['log_norm_mean'] = PairwiseResponseModel.approximate_expectation_log1p(out['zinb_mean'], out['zinb_var'])
        out['log_norm_var'] = PairwiseResponseModel.approximate_variance_log1p(out['zinb_mean'], out['zinb_var'])

        # Get log-likelihood
        if y is not None:
            ll = PairwiseResponseModel.log_zinb_positive(y, mu, theta, pi)
            out['log_likelihood'] = ll

        return out
    
    def plot_response(self, x, y, library, out_idx=0, plot_y_counts=True, desired_library=1e6, cmap=plt.get_cmap('tab10')):
        # assert self.fitted, 'Model not fitted'

        # Get parameters
        out = self.predict(x, library, y=y, desired_library=desired_library)

        # Plot
        dose = x.ravel()
        y_ = y
        if len(y.shape) > 1:
            y_ = y[:, out_idx]
        if plot_y_counts:
            X = (y_ * desired_library / library)
            X_pred_ = out['zinb_mean'][:, out_idx]
            X_std_ = np.sqrt(out['zinb_var'][:, out_idx])
            norm_str = 'counts'
        else:
            X = np.log1p(y_ * desired_library / library)
            X_pred_ = out['log_norm_mean'][:, out_idx]
            X_std_ = np.sqrt(out['log_norm_var'][:, out_idx])
            norm_str = 'normalized expression'

        # plt.title(f'{TF} dose vs {symbol} expression')
        plt.scatter(dose, X, s=5, c='lightgray')
        plt.grid(linestyle='dotted')

        # Predictions
        idxs_ = np.argsort(dose)
        x_ = dose[idxs_]
        y_ = X_pred_[idxs_].ravel()
        s_ = X_std_[idxs_].ravel()
        plt.plot(x_, y_, label='Pred mean', color=cmap(0))
        plt.fill_between(x_, np.clip(y_ - s_, 0, None), y_ + s_, color=cmap(0), alpha=.1)

        # GT mean
        idxs = np.argsort(dose)
        x_ = dose[idxs]
        y_ = gaussian_filter1d(X[idxs], sigma=50)
        sns.lineplot(x=x_, y=y_, label='GT mean', color=cmap(2))

        # plt.xlabel(f'{TF} dose')
        # plt.ylabel(f'{TG}{norm_str}')
        plt.legend()



if __name__ == '__main__':
    TF = 'Pparg'
    TG = 'Foxo1'

    # Prepare data
    adata_tf = adata[adata.obs['TF'] == TF]
    x = np.array(adata_tf.obs['Dose'].values)[:, None] # np.random.uniform(0, 10, size=100)  
    y = np.array(adata_tf[:, TG].layers['counts'].ravel()) # Replace with actual data
    log_y_norm = np.array(adata_tf[:, TG].X.ravel()) # Replace with actual data

    # Init network
    layer_sizes = [1, 8, 3]
    params = init_network_params(layer_sizes, random.key(0))
    batched_forward = vmap(forward, in_axes=(None, 0))

    solver = jaxopt.LBFGS(fun=jax.value_and_grad(objective), value_and_grad=True, maxiter=5000)

    # solver = jaxopt.ScipyMinimize(fun=jax.value_and_grad(objective), method="L-BFGS-B", value_and_grad=True, maxiter=500)
    sol = solver.run(params, x=x, y=y)

    library = 1e4  # adata_tf.layers['counts'].sum(axis=-1)
    desired_library = 1e4
    mu, theta, pi = batched_forward(sol.params, x)
    dropout_prob = expit(pi)

    mu_desired_library = mu * desired_library / library
    zinb_mean = (1 - dropout_prob) * mu_desired_library
    zinb_var = (1 - dropout_prob) * mu_desired_library * (mu_desired_library + theta + dropout_prob*theta*mu_desired_library)

    log_norm_mean = approximate_expectation_log1p(zinb_mean, zinb_var)
    log_norm_var = approximate_variance_log1p(zinb_mean, zinb_var)