import scanpy as sc
import pandas as pd
import numpy as np

def load_scTFseq(data_dir='/mlbio_scratch/vinas/scTFseq',
                 file_name = 'C3H10_10X_all_exps_merged_genefiltered_integrated_functional.h5ad',
                 ensemble_mapping_file_name='Mus_musculus.ENS96.csv',
                 split_combinatorial=False,
                 n_top_genes=2000):
    # Load data
    adata = sc.read_h5ad(f'{data_dir}/{file_name}')
    adata.obs['log_dose'] = adata.obs['Dose']
    adata.obs['control'] = adata.obs['TF'].str.startswith('D0')
    adata.layers['counts'] = adata.X

    # Normalize
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select HVGs
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata,
                                    n_top_genes=2000,
                                    batch_key='orig.ident',
                                    subset=True)

    # Map ENSEMBL IDs to names
    mm10_df = pd.read_csv(f'{data_dir}/{ensemble_mapping_file_name}', index_col=0)
    mm10_df = mm10_df.set_index('gene_id')
    found_ids = np.intersect1d(adata.var.index, mm10_df.index)
    adata = adata[:, found_ids]
    adata.var = mm10_df.loc[adata.var.index]
    adata.var['Symbol'] = adata.var['gene_name']

    # Prepare data
    # Categories
    control_states = ['D0', 'D0_confluent']
    reference_cell_types = ['Adipo_ref', 'Myo_ref']
    combinatorial_TFs = ['Cebpa-Mycn', 'Pparg-Runx2', 'Mycn-Runx2', 'Mycn-Myog', 'Cebpa-Myog', 'Cebpa-Pparg', 'Mycn-Pparg']
    special_categories = combinatorial_TFs + control_states + reference_cell_types
    unique_TFs = np.setdiff1d(adata.obs['TF'].unique(), special_categories)
    conditions_dict = {k: i for i, k in enumerate(sorted(unique_TFs) + special_categories)}
    conditions_dict_inv = {i: k for k, i in conditions_dict.items()}
    adata.uns['conditions_dict'] = conditions_dict
    adata.uns['conditions_dict_inv'] = conditions_dict_inv

    # Prepare data
    adata.obs['cell_type'] = [t if t in reference_cell_types else 'eMSC' for t in adata.obs['TF'].values]
    adata.obs['perturbation_type'] = [t if t in control_states else 'perturbed' for t in adata.obs['TF'].values]

    # Map doses to one-hot representation
    condition_int = [conditions_dict[tf] for tf in adata.obs['TF']]
    condition_onehot = np.eye(len(conditions_dict))[condition_int]
    condition_onehot_value = adata.obs['log_dose'].values[..., None]
    condition_onehot_value[adata.obs['TF'].isin(control_states + reference_cell_types)] = 1
    x_onehot = condition_onehot * condition_onehot_value

    # Establish combinatorial perturbation representation
    if split_combinatorial:
        x_onehot = x_onehot[..., :len(unique_TFs)]
        for cp in combinatorial_TFs:
            tfs = cp.split('-')
            m = adata.obs['TF'].values == cp
            for tf in tfs:
                idx = conditions_dict[tf]
                x_onehot[m, idx] = condition_onehot_value[m, 0]
    adata.obsm['log_doses'] = x_onehot

    # One-hot encoding of batches
    batch_dict = {k: i for i, k in enumerate(sorted(adata.obs['orig.ident'].unique()))}
    batch_int = [batch_dict[b] for b in adata.obs['orig.ident']]
    batch_onehot = np.eye(len(batch_dict))[batch_int]
    adata.obsm['batch_onehot'] = batch_onehot
    adata.uns['batch_dict'] = batch_dict

    return adata