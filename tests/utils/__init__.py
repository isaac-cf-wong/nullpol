from __future__ import annotations

from .test_convert_type import (
    test_convert_string_to_bool,
    test_convert_string_to_float,
    test_convert_string_to_int,
)
from .test_error import (
    test_nullpol_error_creation,
    test_nullpol_error_inheritance,
    test_nullpol_error_raising,
)
from .test_filesystem import test_get_file_extension, test_is_file
from .test_log import (
    test_setup_logger_basic,
    test_setup_logger_with_file_output,
    test_setup_logger_with_log_level,
)

__all__ = [
    "test_convert_string_to_bool",
    "test_convert_string_to_float",
    "test_convert_string_to_int",
    "test_get_file_extension",
    "test_is_file",
    "test_nullpol_error_creation",
    "test_nullpol_error_inheritance",
    "test_nullpol_error_raising",
    "test_setup_logger_basic",
    "test_setup_logger_with_file_output",
    "test_setup_logger_with_log_level",
]
