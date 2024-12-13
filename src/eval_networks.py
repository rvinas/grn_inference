import scanpy as sc
import pandas as pd
import numpy as np
import decoupler as dc
import warnings
from pandas.errors import SettingWithCopyWarning


def load_regulons(organism='mouse', min_refs=2):
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        mouse_regulons = dc.get_collectri(organism=organism, split_complexes=False)
    mouse_regulons.loc[:, 'n_refs'] = mouse_regulons['PMID'].str.split(';').apply(len)
    filtered_mouse_regulons = mouse_regulons
    if min_refs is not None:
        filtered_mouse_regulons = mouse_regulons[mouse_regulons['n_refs'] >= min_refs]
    return filtered_mouse_regulons

def mm10_stem_cells_chip_atlas(data_dir='/mlbio_scratch/vinas/chip-atlas'):
    df = pd.read_csv(f'{data_dir}/mm10_stem_cells_chip_atlas_network.csv', index_col=0)
    df['source'] = df['TF_name']
    df['target'] = df['TG_name']
    return df