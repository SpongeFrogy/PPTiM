import os
from typing import List, Iterable
from mofid.id_constructor  import extract_fragments
import fnmatch
import pandas as pd
# from fire import Fire

def get_mofid(cif_paths: Iterable[str]) -> pd.DataFrame:
    """get "node", "linker", "cat", "mofkey" and save to csv

    Args:
        cif_paths (Iterable[str]): paths for cif files

    Returns:
        pd.DataFrame: mofid_data
    """
    mofid_data = pd.DataFrame(columns=["node", "linker", "cat", "mofkey"])
    for file in cif_paths:
        if fnmatch.fnmatch(file, '*.cif'):
            mofid_data.loc[file[:-4]] = extract_fragments(file, "o")
    mofid_data.to_csv("mofid_data.csv")
    return mofid_data

def main() -> List[str]:
    files = list(os.walk('.'))[0][-1]
    get_mofid(files)
    # print(files)

if __name__ == "__main__":
    main()
    


        

    