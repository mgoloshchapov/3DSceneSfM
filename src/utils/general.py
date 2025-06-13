import numpy as np
from typing import Union, List
from pathlib import Path


def arr_to_str(a: np.ndarray):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


def get_filenames(directory: Union[str, Path]) -> List[str]:
    """
    Returns a list of filenames in the specified directory.

    Args:
        directory (Union[str, Path]): Path to the target directory.

    Returns:
        List[str]: A list containing filenames.
    """
    directory = Path(directory)
    filenames = [file for file in directory.iterdir() if file.is_file()]
    return filenames


if __name__ == "__main__":
    print(get_filenames("data/test/church/images/"))
