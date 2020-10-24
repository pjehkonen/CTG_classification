import os
import git
import logging
import sys
from pathlib import Path


def setup_log(pdg, myEnv, start_time):
    os_info = os.uname()
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha


    logging.basicConfig(filename='{}/{}.log'.format(str(Path(myEnv.log_dir, start_time)), start_time),
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Started at {}'.format(start_time))
    logging.info('SRC file   {}'.format(sys.argv[0]))
    logging.info('GIT HASH   {}'.format(sha))
    logging.info('Worker     {}'.format(str(os_info.nodename)))

    if pdg:
        print('Log file is in  :{}/log_{}.log'.format(str(myEnv.log_dir), start_time))
        
    return logging.getLogger()