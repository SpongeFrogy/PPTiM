### Script for creating main.csv ###
import pandas as pd
import os

# Path to the script 
PATH = os.path.dirname(os.path.realpath(__file__))
CSV_PATH = os.path.join(PATH, 'csv')


if __name__ == '__main__':
    
    # Check existing of csv folder
    if not os.path.exists(CSV_PATH):
        print('Error: csv folder does not exist')
        exit()

    # Open csv files 
    try:
        init_and_final = pd.read_csv(os.path.join(CSV_PATH, 'init_and_final_structure.csv'),
                                    delimiter=';',
                                    index_col=0,
                                    header=0)

        only_init = pd.read_csv(os.path.join(CSV_PATH, 'only_init_structure.csv'),
                                    delimiter=';',
                                    index_col=0,
                                    header=0)
    except:
        print('Error: check existing of csv files')
        exit()

    # From init_and_final 
    # Simple rule, take all init structre and add all final structure if they are reversible 
    # And add this structure to only init structure

    # Create main dataframe
    main = pd.DataFrame(columns=['Folder num','CIF name', 'Stimuli'])

    # Add structure to main from only_init
    for index, row in only_init.iterrows():
        # Add structure to main
        # use concat because append is deprecated
        main = pd.concat([main, pd.DataFrame({'Folder num': row['Folder num'],
                                            'CIF name': row['CIF_init'],
                                                'Stimuli': row['Stimuli']},
                                                index=[0])],
                                                    ignore_index=True)

    # Add structure to main from init_and_final
    for index, row in init_and_final.iterrows():
        # Add init structure to main
        main = pd.concat([main, pd.DataFrame({'Folder num': row['Folder num'],
                                            'CIF name': row['CIF_init'],
                                                'Stimuli': row['Stimuli']},
                                                index=[0])],
                                                    ignore_index=True)
        # Check if reversible
        if row['Reversible'] == 'yes':
            # Add final structure to init_and_final
            main = pd.concat([main, pd.DataFrame({'Folder num': row['Folder num'],
                                                'CIF name': row['CIF_final'],
                                                    'Stimuli': row['Stimuli']},
                                                    index=[0])],
                                                        ignore_index=True)

    # Sort by folder num
    main = main.sort_values(by=['Folder num'])

    # Delete duplicates
    main = main.drop_duplicates(subset=['CIF name'], keep='first')

    # Remove rows to another dataframe if stimuli is 'T, solvent'
    # Create another dataframe
    t_solvent = main[main['Stimuli'] == 'T, solvent']
    # Remove rows from main
    main = main[main['Stimuli'] != 'T, solvent']

    # Save main and t_solvent to csv
    main.to_csv(os.path.join(PATH, 'main.csv'), sep=';', index=False)
    t_solvent.to_csv(os.path.join(PATH, 't_solvent.csv'), sep=';', index=False)