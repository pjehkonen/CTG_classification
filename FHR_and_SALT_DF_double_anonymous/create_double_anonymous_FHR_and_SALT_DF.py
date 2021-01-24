import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from set_env_dirs import setup_env
from set_env_dirs import setup_log
from set_env_dirs import in_triton
from ctg_lib import import_data
from ctg_lib.ctg_time import now_time_string



if in_triton.in_triton():
    sys.path.append('/scratch/cs/salka/PJ_SALKA/CTG_classification/ctg_lib')
    print("lib appended to Triton path")


out_dir = "DBL_ANON_FHR_SALT"
PrintDebuggingInfo = True
start_time = now_time_string()

operating_in_triton, my_env = setup_env.setup_env(PrintDebuggingInfo, output_dir=out_dir, log_start=start_time)

normal_df, salt_df = import_data.import_data(False, my_env)

print(normal_df.head())
print(salt_df.head())
print("moi")

print("exit")

