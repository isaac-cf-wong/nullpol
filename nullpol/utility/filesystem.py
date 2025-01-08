from pathlib import Path


def get_absolute_path(file_path):
    """Get absolute path.
    
    Parameters
    ----------
    file_path: str
        Path to the file.

    Returns
    -------
    str:
        Absolute path.
    """
    return str(Path(file_path).resolve())

def get_file_extension(file_path):
    """Get the file extension.

    Parameters
    ----------
    file_path: str
        Path to the file.

    Returns
    -------
    str:
        Extension of the file.
    """
    return Path(file_path).suffix

def is_file(file_path):
    """Check if the file exists.

    Parameters
    ----------
    file_path: str
        Path to the file.

    Returns
    -------
    bool:
        True if the file exists, False otherwise.
    """
    return Path(file_path).is_file()