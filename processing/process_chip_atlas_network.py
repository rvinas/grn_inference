import pandas as pd

# Set bedtools path
import sys
sys.path.append('/home/vinas/bedtools')

import pybedtools
from pybedtools import BedTool
from pathlib import Path
import argparse
import os

data_dir = '/mlbio_scratch/vinas/chip-atlas'
pybedtools.set_tempdir('/mlbio_scratch/vinas/tmp')

parser = argparse.ArgumentParser()
parser.add_argument('--genome', type=str, default='mm10')
parser.add_argument('--cell_type', type=str, default=None)
parser.add_argument('--cell_type_class', type=str, default=None)
parser.add_argument('--track_type_class', type=str, default='TFs and others')
args = parser.parse_args()

if __name__ == '__main__':
    # Make directories
    Path(f'{data_dir}/peaks').mkdir(parents=True, exist_ok=True)
    Path(f'{data_dir}/intersect').mkdir(parents=True, exist_ok=True)
    Path(f'{data_dir}/network').mkdir(parents=True, exist_ok=True)

    # Load metadata, ref: https://github.com/inutano/chip-atlas/wiki#experimentList_schema
    print('Loading metadata...')
    experiment_df = pd.read_csv(f'{data_dir}/experimentList.tab',
                            delimiter='\t',
                            header=None,
                            usecols=list(range(15)))
    experiment_df.columns = ['exp_id', 'genome', 'track_type_class', 'track_type',
                            'cell_type_class', 'cell_type', 'cell_type_description',
                            'processing_logs', 'processing_logs_bisulfite_seq',
                            'title', 'metadata_1', 'metadata_2', 'metadata_3', 'metadata_4', 'metadata_5']
    
    # Filter metadata
    df = experiment_df
    output_name = ''
    if args.genome is not None:
        df = df[df['genome'] == args.genome]
    if args.track_type_class is not None:
        df = df[df['track_type_class'] == args.track_type_class]

    ct_string = ''
    if args.cell_type_class is not None:
        df = df[df['cell_type_class'] == args.cell_type_class]
        ct_string += '-' + args.cell_type_class
    if args.cell_type is not None:
        df = df[df['cell_type'] == args.cell_type]
        cell_type = args.cell_type.replace('/', '.')
        ct_string += '-' + cell_type   
    output_name = f'{args.genome}{ct_string}'
    print(f'Output name: {output_name}')

    # Select unique experiment IDs from ChIP data
    if not os.path.exists(f'{data_dir}/peaks/{output_name}.bed'):
        print('Filtering data...')
        unique_exp_ids = df['exp_id'].unique()
        chip = BedTool(f'{data_dir}/allPeaks_light.mm10.05.bed.gz')
        chip_filtered = chip.filter(lambda x: x[3] in unique_exp_ids)
        chip_filtered = chip_filtered.saveas(f'{data_dir}/peaks/{output_name}.bed')

    # Expand genomic coordinates by different window sizes
    for window in [1000, 2000, 5000, 10000]:
        print(f'Expanding genomic coordinates, window size: {window}...')
        file_path = f'{data_dir}/Mus_musculus_TSS+-{window}_3col.ENS96.tsv'
        if not os.path.exists(file_path):
            mm10_df = pd.read_csv(f'{data_dir}/Mus_musculus.ENS96.csv', index_col=0)
            mm10_df['chr'] = 'chr'+mm10_df.index
            mm10_df[f'TSS-{window}'] = mm10_df[f'chromStart']-window
            mm10_df[f'TSS-{window}'] = mm10_df[f'TSS-{window}'].clip(lower=0)
            mm10_df[f'TSS+{window}'] = mm10_df[f'chromStart']+window
            mm10_df.set_index('chr')[[f'TSS-{window}', f'TSS+{window}', 'gene_id']].to_csv(file_path, header=None, sep='\t')

    # Intersect
    print('Intersecting ...')
    genes_df = pd.read_csv(f'{data_dir}/Mus_musculus.ENS96.csv')
    genes_df_ = genes_df.set_index('gene_id')
    chip_filtered = BedTool(f'{data_dir}/peaks/{output_name}.bed')
    for window in [1000, 2000, 5000, 10000]:
        intersect_name = f'{output_name}_TSS+-{window}'
        if not os.path.exists(f'{data_dir}/intersect/{intersect_name}.bed'):
            genes = BedTool(f'{data_dir}/Mus_musculus_TSS+-{window}_3col.ENS96.tsv')
            mapped_genes = chip_filtered.intersect(genes, wo=True).moveto(f'{data_dir}/intersect/{intersect_name}.bed')

        # Format and include TF/TG names
        matched_df = pd.read_csv(f'{data_dir}/intersect/{intersect_name}.bed', delimiter='\t', header=None)
        matched_df.columns = ['chr_1', 'start_1', 'end_1', 'exp_id', 'MACS2', 'chr_2', 'start_2', 'end_2', 'TG', 'overlap']
        df_ = df.set_index('exp_id').loc[matched_df['exp_id']]
        matched_df['TF_name'] = df_['track_type'].values
        matched_df['cell_type'] = df_['cell_type'].values
        matched_df['TG_name'] = genes_df_.loc[matched_df['TG']]['gene_name'].values
        matched_df.to_csv(f'{data_dir}/network/{intersect_name}.csv')
    print('Done\n')