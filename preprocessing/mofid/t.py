import pandas as pd
import json


file = "mofid_data.csv"

df = pd.read_csv(file, index_col=0)
new_ = pd.DataFrame(columns=df.columns)
mapping = json.load(open("../../names_mapping.json"))
print(len(mapping))
for name in df.index:
    c_name = name[5:]+".cif"
    print(c_name)
    if c_name in mapping.values():
        print("yes")
        new_.loc[c_name[:-4]] = df.loc[name]

curr_set = set([s[5:] + ".cif" for s in df.index])
print("___")
for name_100 in mapping.values():
    if name_100 not in curr_set:
        print(name_100)

new_.to_csv("mofid_data_.csv")
