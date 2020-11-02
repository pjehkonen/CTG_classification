import platform
import os
from pathlib import Path


# Class to work as directory element

def path_content(directory, suffix):
    file_generator = directory.glob('*.' + suffix)
    input_files = [f.name for f in file_generator]
    return sorted(input_files)


def in_triton():
    if os.path.exists('/scratch'):
        return True
    else:
        return False


class CTGPaths:
    """
    Class containing directory information as Path-objects.

    Adapts either to Triton environment or local computer environment by
    identifying if /scratch directory exists in the root.

    Attributes
    ----------
    i_dir : Path-object to input directory
    o_dir : Path-object to output directory
    l_dir : Path object to log directory
    f_dir : Path object ot feature directory

    Methods
    -------
    hdf_dir_content()
        Prints content of HDF directory
    info_dir_content()
        Prints contents of INFO directory
    """

    def __init__(self, i_dir=None, o_dir=None, l_dir=None, f_dir=None):

        self.__worker = platform.node()

        if in_triton():
            print("ctg_path_env::CTGPaths ()   Assuming working in Triton")
            self.base_dir = Path('/scratch/cs/salka/DATA2h/')
            self.scratch_dir = Path('/scratch/cs/salka/scratch')
            self.feature_dir = Path('/scratch/cs/salka/Features')
            self.results_dir = Path('/scratch/cs/salka/Results')
        else:
            print("ctg_path_env::CTGPaths ()   Assuming working in local computer")
            self.base_dir = Path('/media/jehkonp1/SecureLacie/DATA2h')
            self.scratch_dir = Path('/media/jehkonp1/SecureLacie/DATA2h/Scratch')
            self.feature_dir = Path(self.base_dir, 'Features')
            self.results_dir = self.scratch_dir

        self.csv_dir = Path(self.base_dir, 'CSV')
        self.hdf_dir = Path(self.base_dir, 'HDF')
        self.info_dir = Path(self.base_dir, 'info')
        self.sets_dir = Path(self.base_dir, 'sets/all_pids_divided_in')
        self.pids_dir = Path(self.hdf_dir, 'pids')

        if i_dir is not None:
            self.input_dir = i_dir
        else:
            self.input_dir = Path()

        if o_dir is not None:
            self.output_dir = o_dir
        else:
            self.output_dir = Path()

        if l_dir is not None:
            self.log_dir = l_dir
        else:
            self.log_dir = Path()

        if f_dir is not None:
            self.feature_dir = f_dir

    @property
    def worker(self):
        return self.__worker

    @worker.setter
    def worker(self, name):
        self.__worker = name

    @property
    def base_dir(self):
        return self.__base_dir

    @base_dir.setter
    def base_dir(self, base_dir):
        self.__base_dir = base_dir

    @property
    def scratch_dir(self):
        return self.__scratch_dir

    @scratch_dir.setter
    def scratch_dir(self, scratch_dir):
        self.__scratch_dir = scratch_dir

    @property
    def csv_dir(self):
        return self.__csv_dir

    @csv_dir.setter
    def csv_dir(self, csv_dir):
        self.__csv_dir = csv_dir

    @property
    def hdf_dir(self):
        return self.__hdf_dir

    @hdf_dir.setter
    def hdf_dir(self, hdf_dir):
        self.__hdf_dir = hdf_dir

    @property
    def info_dir(self):
        return self.__info_dir

    @info_dir.setter
    def info_dir(self, info_dir):
        self.__info_dir = info_dir

    @property
    def sets_dir(self):
        return self.__sets_dir

    @sets_dir.setter
    def sets_dir(self, sets_dir):
        self.__sets_dir = sets_dir

    @property
    def pids_dir(self):
        return self.__pids_dir

    @pids_dir.setter
    def pids_dir(self, pids_dir):
        self.__pids_dir = pids_dir

    @property
    def output_dir(self):
        return self.__output_dir

    @output_dir.setter
    def output_dir(self, output_dir):
        self.__output_dir = output_dir

    @property
    def log_dir(self):
        return self.__log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        self.__log_dir = log_dir

    @property
    def input_dir(self):
        return self.__input_dir

    @input_dir.setter
    def input_dir(self, input_dir):
        self.__input_dir = input_dir

    @property
    def feature_dir(self):
        return self.__feature_dir

    @feature_dir.setter
    def feature_dir(self, feature_dir):
        self.__feature_dir = feature_dir

    @property
    def results_dir(self):
        return self.__results_dir

    @results_dir.setter
    def results_dir(self, results_dir):
        self.__results_dir = results_dir

    def all_directories_exist(self, pdg=False):
        nothing_missing = True
        items = dir(self)
        dir_items = [word for word in items if 'dir' in word and not word.startswith('_')]
        missing_dirs = []

        if self.base_dir.exists():
            pass
        else:
            missing_dirs.append('base_dir')
            nothing_missing = False

        if self.output_dir.exists():
            pass
        else:
            missing_dirs.append('output_dir')
            nothing_missing = False

        if self.scratch_dir.exists():
            pass
        else:
            missing_dirs.append('scratch_dir')
            nothing_missing = False

        if self.csv_dir.exists():
            pass
        else:
            missing_dirs.append('csv_dir')
            nothing_missing = False

        if self.hdf_dir.exists():
            pass
        else:
            missing_dirs.append('csv_dir')
            nothing_missing = False

        if self.feature_dir.exists():
            pass
        else:
            missing_dirs.append('feature_dir')
            nothing_missing = False

        if self.sets_dir.exists():
            pass
        else:
            missing_dirs.append('sets_dir')
            nothing_missing = False

        if self.info_dir.exists():
            pass
        else:
            missing_dirs.append('info_dir')
            nothing_missing = False

        if self.input_dir.exists():
            pass
        else:
            missing_dirs.append('input_dir')
            nothing_missing = False

        if self.pids_dir.exists():
            pass
        else:
            missing_dirs.append('pids_dir')
            nothing_missing = False

        if self.log_dir.exists():
            pass
        else:
            missing_dirs.append('log_dir')
            nothing_missing = False

        if self.results_dir.exists():
            pass
        else:
            missing_dirs.append('results_dir')
            nothing_missing = False

        if pdg:
            for directory in missing_dirs:
                print("{} is missing".format(directory))

        return nothing_missing, missing_dirs

    def input_dir_files(self, suffix=None, pdg=False):
        if suffix is None:
            input_file_generator = self.input_dir.glob('*')
        else:
            input_file_generator = self.input_dir.glob('*.' + suffix)
        input_files = [f.name for f in input_file_generator]
        input_files = sorted(input_files)
        if pdg:
            print("Total number of input files is: {}".format(len(input_files)))

        return input_files

    def __str__(self):
        text = "Worker is       {} \n".format(self.worker)
        text += "Base dir is     {} \n".format(self.base_dir)
        text += "Info dir is     {} \n".format(self.info_dir)
        text += "Scratch dir is  {} \n".format(self.scratch_dir)
        text += "Features dir is {} \n".format(self.feature_dir)
        text += "Results dir is  {} \n".format(self.results_dir)
        text += "CSV dir is      {} \n".format(self.csv_dir)
        text += "HDF dir is      {} \n".format(self.hdf_dir)
        text += "PIDs dir is     {} \n".format(self.pids_dir)
        text += "SETs dir is     {} \n\n".format(self.sets_dir)
        text += "Input dir is    {} \n".format(self.input_dir)
        text += "Output dir is   {} \n".format(self.output_dir)
        text += "LOG dir is      {} \n".format(self.log_dir)
        return text


class DirectoryMissingError(Exception):
    pass


# Run MAIN to learn and debug the code. In production initialized ctg_paths object
if __name__ == '__main__':
    myEnv = CTGPaths("/my/input", "/my/output", "/my/log")
    print(myEnv)
    myEnv.input_dir_files("csv")
    raise DirectoryMissingError('Hulabula')
