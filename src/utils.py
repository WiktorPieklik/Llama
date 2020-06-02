import inspect
import json
from pathlib import Path


def json_file_to_dict(json_path: str) -> dict:
    with open(json_path, "r") as json_file:
        return json.load(json_file)


def get_path_relative_to_caller(fpath: str):
    """ Converts specified path to absolute, relative to the caller module directory.

    Parameters
    ----------
    fpath : str
        Path relative to the caller module directory.

    Returns
    -------
    abs_fpath : str
        Absolute path to the file pointed by `fpath` from the caller module.
    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    caller_module_filename = module.__file__
    abs_fpath = str((Path(caller_module_filename).parent / fpath).resolve())
    return abs_fpath
