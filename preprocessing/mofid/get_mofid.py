from typing import List
from mofid.id_constructor  import extract_fragments
import fnmatch
import pandas as pd

def get_mofid(cif_paths: List[str]) -> pd.DataFrame:
    """get "node", "linker", "cat", "mofkey" and save to csv

    Args:
        cif_paths (List[str]): paths for cif files

    Returns:
        pd.DataFrame: mofid_data
    """
    mofid_data = pd.DataFrame(columns=["node", "linker", "cat", "mofkey"])
    for file in cif_paths:
        if fnmatch.fnmatch(file, '*.cif'):
            mofid_data.loc[file[:-4]] = extract_fragments(file, "o")
    mofid_data.to_csv("mofid_data.csv")
    return mofid_data