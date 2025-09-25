from __future__ import annotations

from .test_error import (
    test_nullpol_error_creation,
    test_nullpol_error_inheritance,
    test_nullpol_error_raising,
)
from .test_filesystem import test_get_file_extension, test_is_file
from .test_imports import (
    test_import_function_builtin,
    test_import_function_from_math,
    test_import_function_from_pathlib,
    test_import_function_invalid_format,
    test_import_function_invalid_function,
    test_import_function_invalid_module,
    test_import_function_standard_library,
)
from .test_json_utils import (
    test_json_loads_with_none_array,
    test_json_loads_with_none_basic,
    test_json_loads_with_none_boolean_values,
    test_json_loads_with_none_empty_objects,
    test_json_loads_with_none_invalid_json,
    test_json_loads_with_none_mixed_nulls,
    test_json_loads_with_none_nested,
    test_json_loads_with_none_numeric_values,
    test_json_loads_with_none_with_none_values,
)
from .test_log import (
    test_setup_logger_basic,
    test_setup_logger_with_file_output,
    test_setup_logger_with_log_level,
)

__all__ = [
    "test_get_file_extension",
    "test_import_function_builtin",
    "test_import_function_from_math",
    "test_import_function_from_pathlib",
    "test_import_function_invalid_format",
    "test_import_function_invalid_function",
    "test_import_function_invalid_module",
    "test_import_function_standard_library",
    "test_is_file",
    "test_json_loads_with_none_array",
    "test_json_loads_with_none_basic",
    "test_json_loads_with_none_boolean_values",
    "test_json_loads_with_none_empty_objects",
    "test_json_loads_with_none_invalid_json",
    "test_json_loads_with_none_mixed_nulls",
    "test_json_loads_with_none_nested",
    "test_json_loads_with_none_numeric_values",
    "test_json_loads_with_none_with_none_values",
    "test_nullpol_error_creation",
    "test_nullpol_error_inheritance",
    "test_nullpol_error_raising",
    "test_setup_logger_basic",
    "test_setup_logger_with_file_output",
    "test_setup_logger_with_log_level",
]
