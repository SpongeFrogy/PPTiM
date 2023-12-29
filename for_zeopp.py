import os 
import re 
import shutil
import json
import pandas as pd

""" This code copy cifs in another folder for zeo++ and MOFid calculation, in parellel removing special characters from cif names """

# Path to the script 
PATH = os.path.dirname(os.path.realpath(__file__))

# Path to the cifs folder
DATABASE_PATH = os.path.join(PATH, 'database')
CIFS_PATH = os.path.join(DATABASE_PATH, 'cifs')

def remove_special_characters(input_string):
    pattern = re.compile('[^a-zA-Z0-9\.]+')
    return pattern.sub('', input_string)

def cif_folder_for_zeopp():
    # Copy all cifs by the name and folder number in main.csv and t_solvent.csv 
    # to new folder in PATH but without special characters in name of cif files
    # and create json file with mapping of names

    # Create new folder if not exist
    NEW_CIFS_PATH = os.path.join(PATH, 'cifs')
    if not os.path.exists(NEW_CIFS_PATH):
        os.mkdir(NEW_CIFS_PATH)

    # Open main.csv and t_solvent.csv
    try:
        main = pd.read_csv(os.path.join(DATABASE_PATH, 'main.csv'),
                            delimiter=';',
                            header=0)

        t_solvent = pd.read_csv(os.path.join(DATABASE_PATH, 't_solvent.csv'),
                            delimiter=';',
                            header=0)
    except:
        print('Error: check existing of csv files')
        exit()

    main = pd.concat([main, t_solvent], ignore_index=True).sort_values(by=['Folder num'])
    cif_names = main['CIF name'].tolist() 
    print(len(cif_names))
    # Create json for names mapping
    names_mapping = {}

    # Copy all cifs
    for folders in os.listdir(CIFS_PATH): 
        for cif in os.listdir(os.path.join(CIFS_PATH, folders)):
            # Check if cif file
            if cif.endswith('.cif') and cif in cif_names:
                print('Before cif name: ', cif)
                # Remove special characters
                new_cif_name = remove_special_characters(cif)
                print('After cif name: ', new_cif_name)
                # Copy
                shutil.copy(os.path.join(CIFS_PATH, folders, cif), os.path.join(NEW_CIFS_PATH, new_cif_name))
                # Add to json
                names_mapping[cif] = new_cif_name

                cif_names.remove(cif)
    
    # Save json
    with open(os.path.join(PATH, 'names_mapping.json'), 'w') as f:
        json.dump(names_mapping, f)

    # Check if all cifs are copied
    if len(cif_names) != 0:
        print('Error: not all cifs are copied')
        print(cif_names)
    

if __name__ == '__main__':
    # Check existing of main and t_solvent csv files
    if not os.path.exists(os.path.join(DATABASE_PATH,'main.csv')) or not os.path.exists(os.path.join(DATABASE_PATH,'t_solvent.csv')):
        # Execute create_main.py if not exist
        os.system('python  database\create_main.py')

    cif_folder_for_zeopp()
        