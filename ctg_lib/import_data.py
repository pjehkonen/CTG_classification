from pathlib import Path
import pandas as pd


def import_data(pdg, myEnv):
    normal_df = pd.read_feather(Path(myEnv.input_dir, 'normal_df.feather'))
    salt_df = pd.read_feather(Path(myEnv.input_dir, 'salt_df.feather'))
    if pdg:
        print("Importing Feather dataframe for Normal and Saltatory cases")
        print("Information of normal_df.info()")
        print(normal_df.info())
        print("Header of normal_df")
        print(normal_df.head())
        print("Information of saltatory.info()")
        print(salt_df.info())
        print("Header of salt_df")
        print(salt_df.head())

    return normal_df, salt_df
