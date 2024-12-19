from bilby_pipe.data_analysis import DataAnalysisInput as BilbyDataAnalysisInput
from ..utility import logger

class DataAnalysisInput(BilbyDataAnalysisInput):
    """Handles user-input for the data analysis script.
    """
    def __init__(self, args, unknown_args, test=False):
        """Initializer.

        Args:
            args (tuple): A tuple of arguments.
            unknown_args (tuple): A tuple of unknown arguments.
            test (bool, optional): _description_. Defaults to False.
        """
        logger.info(f"Command line arguments: {args}")

        # Generic initialisation
        self.meta_data = dict()
        self.result = None

        # Read the other arguments
        for name in dir(args):
            if not name.startswith("_"):
                setattr(self, name, getattr(args, name, None))
