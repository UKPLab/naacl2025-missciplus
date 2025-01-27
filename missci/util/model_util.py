from os import listdir
from os.path import join


def get_single_directory_checkpoint(directory: str) -> str:
    assert len(listdir(directory)) == 1
    return join(directory, listdir(directory)[0])
