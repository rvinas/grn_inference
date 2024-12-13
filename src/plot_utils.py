import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, PrecisionRecallDisplay
from src.eval_utils import align_pred_gt

def plot_expression_curve(TF, gene_idx, adata_hvg, raw_counts=False,
                          mean_layer_key='mean', var_layer_key='var',
                          single_batch=True):
    cmap = plt.get_cmap('tab10')
    label = 'Train'

    symbol = adata_hvg.var.iloc[gene_idx]["Symbol"]
    m = adata_hvg.obs['TF'] == TF
    dose = adata_hvg[m].obs['Dose'].values
    X = adata_hvg[m, :].X
    X_norm = X.sum(axis=-1)
    X = X[:, gene_idx]

    # Select majoritary batch
    m_single_batch = m
    if single_batch:
        maj_batch = adata_hvg[m].obs['batch'].value_counts().sort_values(ascending=False).index[0]
        m_batch = adata_hvg.obs['batch'] == maj_batch
        m_single_batch = m & m_batch

    # Predictions
    X_pred = adata_hvg[m, gene_idx].layers[mean_layer_key].ravel()
    X_std = np.sqrt(adata_hvg[m, gene_idx].layers[var_layer_key].ravel())
    if raw_counts:
        # TODO
        X = np.log1p(1e4 * X / X_norm)

    # Plot
    plt.subplot(1, 2, 1)
    plt.title(f'{TF} dose vs {symbol} expression')
    plt.scatter(dose, X, s=5, label=label, c='lightgray')
    plt.grid(linestyle='dotted')
    
    # Predictions
    dose_ = adata_hvg[m_single_batch].obs['Dose']
    X_pred_ = adata_hvg[m_single_batch, gene_idx].layers[mean_layer_key].ravel()
    X_std_ = np.sqrt(adata_hvg[m_single_batch, gene_idx].layers[var_layer_key].ravel())
    idxs_ = np.argsort(dose_.values)
    x_ = dose_[idxs_]
    y_ = X_pred_[idxs_].ravel()
    s_ = X_std_[idxs_].ravel()
    plt.plot(x_, y_, label='Pred mean', color=cmap(0))
    plt.fill_between(x_, np.clip(y_ - s_, 0, None), y_ + s_, color=cmap(0), alpha=.1)
    # plt.scatter(x, y, s=5, label='Pred', color=cmap(0))
    
    # GT mean
    # y = X[idxs].ravel()
    idxs = np.argsort(dose)
    x = dose[idxs]
    y = gaussian_filter1d(X[idxs], sigma=50)
    # plt.plot(x, y, label='GT mean', color=cmap(2))
    sns.lineplot(x=x, y=y, label='GT mean', color=cmap(2))
    
    plt.xlabel(f'{TF} dose')
    plt.ylabel(f'{symbol} normalized expression')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title(f'{TF} dose vs predicted {symbol} expression')

    idxs = np.argsort(dose)
    x = dose[idxs]
    y = X_pred[idxs].ravel()
    s = X_std[idxs].ravel()
    # plt.plot(x, y, label='Test pred', color=cmap(0))
    # plt.fill_between(x, y - s, y + s, color=cmap(0), alpha=.1)
    plt.scatter(dose, X_pred, s=5, label='Pred', color=cmap(0))
    plt.xlabel(f'{TF} dose')
    plt.ylabel(f'{symbol} normalized expression')
    plt.show()


def plot_precision_recall_curve(pred_dfs, gt_df, tfs=None, title=''):
    fig, ax = plt.subplots(figsize=(17, 5))
    for k, pred_df in pred_dfs.items():
        pred_df_, gt_df_ = align_pred_gt(pred_df, gt_df, tfs=tfs)
        
        # Select specific TFs
        if tfs is not None:
            pred_df_ = pred_df_[tfs]
            gt_df_ = gt_df_[tfs]

        # TODO: Option to exclude self edges?
        y_true = gt_df_.abs().values.ravel()
        y_scores = pred_df_.values.ravel()
        auprc = average_precision_score(y_true, y_scores)
        auprc_ratio = auprc / y_true.mean()
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # plt.plot(recall, precision, label=label)

        # plt.figure(figsize=(17, 5))
        display = PrecisionRecallDisplay(
            recall=recall,
            precision=precision,
            prevalence_pos_label=y_true.mean()
        )
        display.plot(ax=ax, name=f'{k}. AUPRC: {auprc:.2f}. AUPRC ratio: {auprc_ratio:.2f}')  # , color="gold"
        display.figure_.set_size_inches(7, 6)  # Adjust the size

    tfs_string = ''
    if tfs is not None:
        tfs_string = '. TFs: ' + ','.join(tfs)
    plt.title(f'Precision-recall curve{tfs_string}\n {title}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axhline(y=y_true.mean(), color='gray', linestyle='--', label='Random', linewidth=0.5)

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    plt.ylim((0, 1))
    plt.grid(False)

    # plt.legend()
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
            fancybox=True, shadow=True, ncol=1)

    # print('AUPRC', auprc, 'AUPRC ratio', auprc/y_true.mean())
    # return auprc, auprc_ratio