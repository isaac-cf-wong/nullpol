from pathlib import Path


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