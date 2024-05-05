from os import listdir
from pathlib import Path

def listdir_nohidden(path: Path | str) -> list[str]:
    """
    List files in a directory excluding hidden files.
    """
    return [file for file in listdir(path) if not file.startswith(".")]

def lazy_stoi(s: str) -> int:
    """
    Convert a string to an integer.
    """
    return int(s) if s.replace(",", "").isdigit() else s

def lazy_stof(s: str) -> float:
    """
    Convert a string to a float.
    """
    return float(s) if s.replace(",", "").replace(".", "", 1).isdigit() else s
