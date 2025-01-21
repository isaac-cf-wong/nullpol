import os
import shutil
import pkg_resources
import argparse
from pathlib import Path
from ..utility import (logger,
                       NullpolError)


def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Copy asimov YAML files to the current directory or a specified location."
    )

    parser.add_argument(
        '-o', '--outdir',
        default=os.getcwd(),
        help="Destination directory (default: current directory)."
    )

    # Parse arguments
    args = parser.parse_args()

    # Path to the data file within the package
    analysis_defaults_path = pkg_resources.resource_filename('nullpol.asimov', 'analysis_defaults.yaml')
    nullpol_analysis_path = pkg_resources.resource_filename('nullpol.asimov', 'nullpol_analysis.yaml')

    dest_dir = Path(args.outdir)
    if dest_dir.is_dir():
        analysis_defaults_dest_path = dest_dir/'analysis_defaults.yaml'
        nullpol_analysis_dest_path = dest_dir/'nullpol_analysis.yaml'
    else:
        raise NullpolError(f'{dest_dir} is not a directory.')
    # Copy the file
    shutil.copy(analysis_defaults_path, analysis_defaults_dest_path)
    logger.info(f"Copied '{analysis_defaults_path}' to '{analysis_defaults_dest_path}'.")
    shutil.copy(nullpol_analysis_path, nullpol_analysis_dest_path)
    logger.info(f"Copied '{nullpol_analysis_path}' to '{nullpol_analysis_dest_path}'.")
