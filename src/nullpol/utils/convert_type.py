from typing import Union


def convert_string_to_float(string: Union[str, None]) -> Union[float, None]:
    """Convert string to float.

    Args:
        string (str): string.

    Returns:
        float: float. None if string is None.
    """
    if string is None:
        return None
    return float(string)


def convert_string_to_int(string: Union[str, None]) -> Union[int, None]:
    """Convert string to int.

    Args:
        string (str): string.

    Returns:
        int: int. None if string is None.
    """
    if string is None:
        return None
    return int(string)


def convert_string_to_bool(string: Union[str, None]) -> Union[bool, None]:
    """Convert string to bool.

    Args:
        string (str): string.

    Returns:
        bool: bool. None if string is None.
    """
    if string is None:
        return None
    return bool(string)
