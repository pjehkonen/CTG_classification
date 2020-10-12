from ctg_lib import ctg_path_env
from ctg_lib import ctg_time
from set_env_dirs import in_triton

import os
import sys
from pathlib import Path

def setup_env(pdg, input_dir=None, output_dir=None):

    if in_triton.in_triton():
        print("Assuming working in Triton")
        inTriton = True
    else:
        print("Assuming working in local computer")
        inTriton = False

    if inTriton:
        sys.path.append('/scratch/cs/salka/PJ_SALKA/CTG_classification/ctg_lib')
        print("lib appended to Triton path")

    myEnv = ctg_path_env.CTGPaths()

    if input_dir is None:
        myEnv.input_dir = Path(myEnv.base_dir, 'saltatory_and_non_saltatory_vectors')
    else:
        myEnv.input_dir = Path(myEnv.base_dir, input_dir)

    if output_dir is None:
        this_run_text = "default_output_at_"+ctg_time.now_time_string()
        myEnv.output_dir = Path(myEnv.results_dir, this_run_text)
    else:
        myEnv.output_dir = Path(myEnv.results_dir, output_dir)

    myEnv.log_dir = Path(myEnv.output_dir, 'log')

    # Creating output directory
    try:
        myEnv.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("ERROR::Creation of the directory {} failed".format(str(myEnv.output_dir)))
    else:
        print("Successfully created the directory {}".format(str(myEnv.output_dir)))

    # Creating log directory
    try:
        myEnv.log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("ERROR::Creation of the directory {} failed".format(str(myEnv.log_dir)))
    else:
        print("Successfully created (or directory existed) the directory {}".format(str(myEnv.log_dir)))

    dirs_created, missing_dirs = myEnv.all_directories_exist(pdg)

    if dirs_created:
        print("All directories exist, we are good")
    else:
        print("Problem with directories:")
        for directory in missing_dirs:
            print("{} is missing".format(directory))
        print('\n\n ERROR:::: Input Directories missing, terminating')
        sys.exit("ERROR:: script directories either inputs or outputs are missing TERMINATING")

    if pdg:
        print("Printing deeper debugging information")
        print("inTriton is ",inTriton)
        print("myEnv is:\n",myEnv)
    else:
        print("Minimum debug output")

    return inTriton, myEnv
