# Preprocessing

!! rename Preprocessing

## Structure

### vectorization of QMOF database

- [load_qmof](/preprocessing/qmof/load_qmof.ipynb) - loading QMOF database
- [QMOF Mordred](/preprocessing/qmof/get_mordred_qmof.ipynb) - extract Mordred data
- [QMOF node](/preprocessing/qmof/get_node_qmof.ipynb) - extract Node vectorization
- [cif2vec_qmof](cif2vec_qmof.ipynb)
  - Node encoding
  - concatenation and cleaning
  - fitting `Preprocessing model`
- [QMOF dataset](/preprocesing/datasets/) -saved datasets for training

### vectorization of main database

- [getting mofid](/preprocesing/mofid)
  - [get mofid](/preprocessing/mofid/get_mofid.py) - function that saves mofid data
  - [mofid data](/preprocessing/mofid/mofid_data_.csv) - file with extracted mofid data for main database
- [getting zeo](/preprocesing/zeopp)
  <!-- - [zeo data](/preprocessing/zeopp/README.md) - description of using Zeo++ -->
  - [zeo data](/preprocessing/zeopp/zeo_data_.csv) - file with extracted Zeo++ data for main database
- [cif2vec_main](cif2vec_main.ipynb) - getting MOF vector representation
  - extract Mordred data
  - Node encoding
  - concatenation and cleaning
- [Preprocessing model](preproc_model.py) - cleaning vector representation of `Nan` values and normalization
- [main dataset](/preprocessing/datasets/main_dataset.csv)
- [2pt structures](/preprocessing/datasets/t_solvent.npy) - structures that have both types of phase transmission
