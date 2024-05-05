from os import listdir
from pathlib import Path

def listdir_nohidden(path: Path | str) -> list[str]:
    """
    List files in a directory excluding hidden files.
    """
    return [file for file in listdir(path) if not file.startswith(".")]
