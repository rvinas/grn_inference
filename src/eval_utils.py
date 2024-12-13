import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from itertools import product, permutations, combinations, combinations_with_replacement
from tqdm import tqdm

def align_pred_gt(pred_df, gt_df, tfs=None):
    """
    Densifies ground truth dataframe and aligns the rows and columns
    pred_df: Pandas dataframe with rows representing TFs and columns representing candidate TGs.
             The entries of this matrix contain the predicted scores and higher scores imply higher
             chances of a regulatory interaction
    gt_df: Sparse pandas dataframe with ground truth network. Contains two columns: source and target
    """
    # Subset predictions to TFs and TGs in ground-truth
    tfs = sorted(gt_df['source'].unique())
    ctgs = sorted(gt_df['target'].unique())
    pred_df_ = pred_df.loc[ctgs, tfs]

    # Convert sparse ground-truth DF into dense adjacency matrix
    gt_df.loc[:, 'ones'] = 1
    gt_df_ = gt_df[['source', 'target', 'ones']].pivot(index='target', columns='source', values='ones')  # values='MACS2'
    gt_df_ = gt_df_.loc[ctgs, tfs]
    gt_df_ = gt_df_.fillna(0)
    
    return pred_df_, gt_df_

def calculate_auprc(pred_df, gt_df, tfs=None):
    """
    Calculates AUPRC and AUPRC ratio wrt random predictor
    pred_df: Pandas dataframe with rows representing TFs and columns representing candidate TGs.
             The entries of this matrix contain the predicted scores and higher scores imply higher
             chances of a regulatory interaction
    gt_df: Sparse pandas dataframe with ground truth network. Contains two columns: source and target
    """
    # TODO: Option to exclude self edges?
    pred_df_, gt_df_ = align_pred_gt(pred_df, gt_df, tfs=tfs)
    
    # Select specific TFs
    if tfs is not None:
        pred_df_ = pred_df_[tfs]
        gt_df_ = gt_df_[tfs]

    # Calculate AUPRC
    y_true = gt_df_.abs().values.ravel()
    y_scores = pred_df_.values.ravel()
    auprc = average_precision_score(y_true, y_scores)
    auprc_ratio = auprc / y_true.mean()
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # print('AUPRC', auprc, 'AUPRC ratio', auprc/y_true.mean())
    return auprc, auprc_ratio


def computeScores(trueEdgesDF, predEdgeDF, 
                  directed = True, selfEdges = True):
    # Ref Beeline: https://github.com/Murali-group/Beeline/blob/master/BLEval/computeAUC.py
    '''        
    Computes precision-recall and ROC curves
    using scikit-learn for a given set of predictions in the 
    form of a DataFrame.
    

    :param trueEdgesDF:   A pandas dataframe containing the true classes.The indices of this dataframe are all possible edgesin a graph formed using the genes in the given dataset. This dataframe only has one column to indicate the classlabel of an edge. If an edge is present in the reference network, it gets a class label of 1, else 0.
    :type trueEdgesDF: DataFrame
        
    :param predEdgeDF:   A pandas dataframe containing the edge ranks from the prediced network. The indices of this dataframe are all possible edges.This dataframe only has one column to indicate the edge weightsin the predicted network. Higher the weight, higher the edge confidence.
    :type predEdgeDF: DataFrame
    
    :param directed:   A flag to indicate whether to treat predictionsas directed edges (directed = True) or undirected edges (directed = False).
    :type directed: bool
    :param selfEdges:   A flag to indicate whether to includeself-edges (selfEdges = True) or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool
        
    :returns:
            - prec: A list of precision values (for PR plot)
            - recall: A list of precision values (for PR plot)
            - fpr: A list of false positive rates (for ROC plot)
            - tpr: A list of true positive rates (for ROC plot)
            - AUPRC: Area under the precision-recall curve
            - AUROC: Area under the ROC curve
    '''

    if directed:        
        # Initialize dictionaries with all 
        # possible edges
        if selfEdges:
            possibleEdges = list(product(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         repeat = 2))
        else:
            possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         r = 2))
        
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        
        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth
        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
                   (trueEdgesDF['Gene2'] == key.split('|')[1])])>0:
                    TrueEdgeDict[key] = 1
                
        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])]
            if len(subDF)>0:
                PredEdgeDict[key] = np.abs(subDF.EdgeWeight.values[0])

    # if undirected
    else:
        
        # Initialize dictionaries with all 
        # possible edges
        if selfEdges:
            possibleEdges = list(combinations_with_replacement(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        else:
            possibleEdges = list(combinations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}

        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth

        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[((trueEdgesDF['Gene1'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene2'] == key.split('|')[1])) |
                              ((trueEdgesDF['Gene2'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene1'] == key.split('|')[1]))]) > 0:
                TrueEdgeDict[key] = 1  

        # Compute PredEdgeDict Dictionary
        # from predEdgeDF

        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[((predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])) |
                              ((predEdgeDF['Gene2'] == key.split('|')[0]) &
                               (predEdgeDF['Gene1'] == key.split('|')[1]))]
            if len(subDF)>0:
                PredEdgeDict[key] = max(np.abs(subDF.EdgeWeight.values))

                
                
    # Combine into one dataframe
    # to pass it to sklearn
    outDF = pd.DataFrame([TrueEdgeDict,PredEdgeDict]).T
    outDF.columns = ['TrueEdges','PredEdges']
    
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)
    
    return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr)