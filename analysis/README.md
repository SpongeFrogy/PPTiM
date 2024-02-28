# Analysis of Linker Lengths and Bonding Types

This document outlines the methodology and resources used for examining the linker lengths and types of bonding within the QMOF (Quantum Metal-Organic Framework) database. Our analysis focuses on two main aspects: the length of linkers and the type of bonding present in these materials.

## Linker Lengths Analysis
The `linker_lengths` directory houses data on the length of linkers of QMOF database divided by knn prediction. 
Notebook [linker_lengths.ipynb](linker_lengths.ipynb) contains the code with computing and analysis of the length of linkers. This notebook is instrumental in understanding the methodologies employed in analyzing the dimensions of linkers across various MOFs


## Bonding Type Analysis
The examination of bonding types within the QMOF database is detailed in the [bonding_type.ipynb](bonding_type.ipynb) notebook. This resource is pivotal for filtering and scrutinizing the bonding types present, facilitating a deeper comprehension of the structural intricacies of MOFs. 

### Detailed Information on Bonding Types:
- **Hinge Bonding**: Characterized by a singular bond, either coordination or covalent, that connects a metal atom to an organic linker. This type of bonding plays a crucial role in the flexibility and functionality of certain MOFs.
- **Rigid (Double) Bonding**: Distinguished by the presence of two bonds, be they coordination or covalent, linking a metal atom to an organic linker. A common example of rigid bonding is observed in carboxylate groups from organic linkers, such as terephthalic acid, engaging in binding with a metal atom, thereby contributing to the structural stability of the MOF.

The dataset [type_of_bonding.csv](type_of_bonding.csv) encompasses computed information regarding the types of bonding encountered. 

The algorithm for computing bonding types is currently under development as part of a new project initiative. This endeavor aims to refine our understanding and categorization of bonding types within MOFs. For those interested in exploring the computational code or contributing to the project, we encourage reaching out to the [author](https://physics.itmo.ru/en/personality/vladimir_shirobokov) for more information and potential collaboration opportunities.
