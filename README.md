# Gene regulatory network inference

### Installation
We recommend using Python >= 3.10. The file `requirements.txt` contains the library versions of our environment. To install all the packages, run `pip install -r requirements.txt`.

---

### Performing gene regulatory inference using scTF-seq data

The notebook `benchmark_LRZINB_CellOracle.ipynb` demonstrates how to use our approach to infer gene regulatory networks using scTF-seq data. In this notebook, we benchmark the following baselines:
1. Ridge regression
2. Likelihood ratio of probabilistic zero-inflated negative binomial model
3. CellOracle (work in progress)

We prune a [base mouse gene regulatory network from CellOracle](https://morris-lab.github.io/CellOracle.documentation/notebooks/04_Network_analysis/Network_analysis_with_Paul_etal_2015_data.html?highlight=load_mouse_scatac_atlas_base_grn) using the interactions inferred by the methods above. We evaluate performance using the area under the precision-recall curve (AUPRC), using two ground-truth networks: 
1. Literature-derived network from the [CollecTRI](https://github.com/saezlab/CollecTRI) database. This resource collects mouse regulatory interactions from 12 publicly available databases.
2. Ground-truth networks specific to C3H/10T1/2, embryonic fibroblasts, and pluripotent stem-cell network derived from [ChIP Atlas](https://github.com/inutano/chip-atlas/wiki). The code to process this network is located in `processing/process_chip_atlas_network.py`.

---

### Next steps

#### Modelling
- [ ] **Implement cascade perturbation modelling.** Run cascade perturbation modelling to 1) infer regulatory interactions for unperturbed transcription factors and 2) potentially improve prediction performance of downstream nodes.
- [ ] **Condition model on endogenous TF expression.** The expression of downstream target genes not only depends on TF dose, but also on the endogenous expression of the TF.
- [ ] **Constructing a base GRN using ATAC-seq specific to multipotent stromal cells.** We could then follow the CellOracle tutorial (see [here](https://morris-lab.github.io/CellOracle.documentation/tutorials/base_grn.html#option1-preprocessing-scatac-seq-data)) to construct the base GRN. Bonus points if you have ideas on how to improve CellOracle or SCENIC+ in this aspect.

#### Evaluation
- [ ] **Running baseline GRN inference methods.** It would be great to compare our current approach to existing methods including CellOracle (CellOracle: done).
- [ ] **Alternative evaluation and interpretation of results.** We now evaluate GRN reconstruction performance using a ground-truth network. Can we assess if the models' predictions extrapolate beyond scTF-seq? In other words, can we validate our inferred network using an external dataset? Once cascade perturbation modelling is implemented, assess regulatory inference performance for unperturbed transcription factors. Incorporate our model to the GRETA ([paper](https://www.biorxiv.org/content/10.1101/2024.12.20.629764v1.full.pdf), [code](https://github.com/saezlab/greta)) benchmark.
- [ ] **Downstream applications.** What are some cool downstream applications that this method can enable?
