<div align="center">
    <img src="ITMOF lab.png" alt="ITMOF lab logo" width="400" height="auto"/>
</div>

# Prediction of metal-organic frameworks with phase transition via machine learning

## Project Structure Overview

### Preprocessing Directory
The `[preprocessing](/preprocessing/)` section is dedicated to the initial data preparation stages for the Prediction of Phase Transitions in Metal-Organic Frameworks (PPTiM). It encompasses scripts crucial for downloading essential resources such as the QMOF Database, Zeo++, and MOFid, setting the foundation for further analysis and model training.

### Models Directory
Within the `[models](/models/)` directory, you will find the core computational models developed for this project. This includes an autoencoder designed to reduce the dimensionality of feature data, alongside classifiers tasked with predicting phase transitions in MOFs, showcasing the project's approach to leveraging machine learning for material science.

### Database Directory
The `[database](/database/)` directory houses the PPTiM Database, a meticulously curated collection of MOFs known to exhibit phase transitions. This database not only includes CIF files detailing the structural data of these MOFs but also features manually labeled data pertaining to their phase transitions, serving as a valuable resource for research in this domain.

### Knn Prediction Directory
The `[knn_prediction](/knn_prediction/)` directory contains the data of MOFs from the QMOF Database labaled by knn prediction, trained by PPTiM Database.

### Order of execution
To navigate the project's computational pipeline effectively, follow the order of execution outlined below:
1. [for_zeopp](for_zeopp.py) - Prepares data for Zeo++ computation.
2. [cif2vec_qmof](/preprocessing/cif2vec_qmof.ipynb) - Processes CIF files from the QMOF Database using the cif2vec approach.
3. [cif2vec_main](cif2vec_main.ipynb) - Processes CIF files from the PPTiM Database using the cif2vec approach.
4. [cif2vec_t_sol](cif2vec_t_sol.ipynb)
5. [pipeline](pipeline.ipynb) - Orchestrates the overall data processing and model training pipeline.
6. [Figures](Figure_metrics.ipynb) - Conducts figures for metrics.

## Acknowledgments and Citations

For researchers utilizing the PPTiM Database or methodologies developed within this project, please reference the associated publications.

- Under review: "Prediction of Metal-Organic Frameworks With Phase Transition via Machine Learning"

- Under review: "Phase Change Metal-Organic Frameworks: Current state and Application"

## Contact

For further information, inquiries, or interest in collaboration you can reach the corresponding author's details are available [Vladimir Shirobokov](https://physics.itmo.ru/en/personality/vladimir_shirobokov) and [Grigori Karsakov](https://physics.itmo.ru/ru/personality/grigoriy_karsakov).

## Licensing

This project and all associated code and data are distributed under the GPL-2.0 license

