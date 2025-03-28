from __future__ import annotations

from bilby_pipe.bilbyargparser import BilbyConfigFileParser


def read_config(filename: str) -> tuple[dict, dict, dict, dict]:
    """Read configuration file.

    Args:
        filename (str): File name.

    Returns:
        tuple: A tuple containing:
            - dict: Arguments and values.
            - dict: Line numbers of the arguments.
            - dict: Comments.
            - dict: Inline comments.
    """
    parser = BilbyConfigFileParser
    with open(filename) as f:
        items, numbers, comments, inline_comments = parser.parse(f)
    return items, numbers, comments, inline_comments
