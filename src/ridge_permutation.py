from sklearn.linear_model import Ridge
import numpy as np
from tqdm import tqdm
import pandas as pd

def model_coefs(x, y):
    model = Ridge()
    model.fit(x, y)
    return model.coef_

def ridge_permutation_testing(adata, n_perm=100):
    x_onehot = adata.obsm['log_doses']  # Assumes one-hot log-dose representation of TFs
    batch_onehot = adata.obsm['batch_onehot']  # One-hot representation of batches
    y = adata.X  # Assumes log-normalized data
    x = np.concatenate((x_onehot, batch_onehot), axis=-1)

    # Calculate coefficients
    true_coefs = model_coefs(x, y)

    # Permutation testing
    print('Performing permutation testing ...')
    n_tfs = x_onehot.shape[-1]
    perm_coefs = np.zeros((n_perm, y.shape[-1], x.shape[-1]))
    x_perm = x.copy()
    for n in tqdm(range(n_perm)):
        # Randomly permute log-doses, keep batch information intact
        perm_idxs = np.random.permutation(x_perm.shape[0])
        x_perm[:, :n_tfs] = x_perm[perm_idxs, :n_tfs]
        perm_coefs[n, ...] = model_coefs(x_perm, y)
    
    # Compute and store p-values
    conditions_dict = adata.uns['conditions_dict']
    t = (np.abs(true_coefs)[None,...] <= np.abs(perm_coefs))[..., :n_tfs]
    p = t.mean(axis=0)
    p_df = pd.DataFrame(p, index=adata.var['Symbol'], columns=list(conditions_dict.keys())[:n_tfs])
    true_coefs_df = pd.DataFrame(true_coefs[..., :n_tfs], index=adata.var['Symbol'], columns=list(conditions_dict.keys())[:n_tfs])
    return true_coefs_df, perm_coefs, p_df
