import celloracle as co
import numpy as np

def infer_cell_oracle_grn(adata, base_GRN, values_key = '-logp'):
    # Reference Cell Oracle for GRN inference: https://github.com/morris-lab/CellOracle/issues/58
    # Create Net class object. This is the core engine in the Oracle object
    net = co.Net(gene_expression_matrix=adata.to_df(), # Input gene expression matrix as data frame
                TFinfo_matrix=base_GRN, # Input base GRN
                verbose=True)

    # Do GRN calculation
    net.fit_All_genes(bagging_number=20,
                    alpha=10,
                    verbose=True,
                    n_jobs=16)

    # Get result
    net.updateLinkList(verbose=True)
    inference_result = net.linkList#.copy()
    
    # Densify matrix
    oracle_df = inference_result[['source', 'target', values_key]].pivot(index='target', columns='source', values=values_key)
    oracle_df = oracle_df.fillna(0)
    
    return oracle_df