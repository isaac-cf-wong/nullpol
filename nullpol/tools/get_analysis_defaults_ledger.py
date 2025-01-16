import os
import shutil
import pkg_resources
import argparse
from nullpol.utility import logger

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Copy packaged data to the current directory or a specified location."
    )

    parser.add_argument(
        '-o', '--output',
        default=os.getcwd(),
        help="Destination directory or file path (default: current directory)."
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Path to the data file within the package
    data_path = pkg_resources.resource_filename('nullpol.asimov', 'analysis_defaults.yaml')
    
    # Determine destination path
    if os.path.isdir(args.output):
        destination_file = "analysis_defaults.yaml"
        destination_path = os.path.join(args.output, destination_file)
    else:
        destination_path = args.output
    
    # Copy the file
    shutil.copy(data_path, destination_path)
    logger.info(f"Copied '{data_path}' to '{destination_path}'.")