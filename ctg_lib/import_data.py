from pathlib import Path
import pandas as pd

def import_data(pdg, myEnv):
    normal_df = pd.read_feather(Path(myEnv.input_dir,'normal_df.feather'))
    salt_df = pd.read_feather(Path(myEnv.input_dir,'salt_df.feather'))
    if pdg:
        print("here we are")
    return normal_df, salt_df
