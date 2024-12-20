import sys
import os
from bilby_pipe.parser import create_parser
from bilby_pipe.utils import nonestr
from bilby_pipe.bilbyargparser import (BilbyArgParser,
                                       HyphenStr)
from ..utility import logger
from .._version import __version__


def write_to_file(
    self,
    filename,
    args=None,
    overwrite=False,
    include_description=False,
    exclude_default=False,
    comment=None,
):
    if os.path.isfile(filename) and not overwrite:
        logger.warning(f"File {filename} already exists, not writing to file.")
    with open(filename, "w") as ff:
        if include_description:
            print(
                f"## This file was written with nullpol version {__version__}\n",
                file=ff,
            )
        if isinstance(comment, str):
            print("#" + comment + "\n", file=ff)
        for group in self._action_groups[2:]:
            print("#" * 80, file=ff)
            print(f"## {group.title}", file=ff)
            if include_description:
                print(f"# {group.description}", file=ff)
            print("#" * 80 + "\n", file=ff)
            for action in group._group_actions:
                if include_description:
                    print(f"# {action.help}", file=ff)
                dest = action.dest
                hyphen_dest = HyphenStr(dest)
                if isinstance(args, dict):
                    if action.dest in args:
                        value = args[dest]
                    elif hyphen_dest in args:
                        value = args[hyphen_dest]
                    else:
                        value = action.default
                else:
                    value = getattr(args, dest, action.default)

                if exclude_default and value == action.default:
                    continue
                self.write_comment_if_needed(hyphen_dest, ff)
                self.write_line(hyphen_dest, value, ff)
            print("", file=ff)

BilbyArgParser.write_to_file = write_to_file

def create_nullpol_parser(top_level=True):
    """Create the nullpol_pipe parser

    Parameters
    ----------
    top_level: bool, optional
        If true, the top-level parser is created. If false, a subparser is
        created. Default is True.
    usage: str, optional
        The usage string to display. Default is None.

    Returns
    -------
    parser: BilbyArgParser instance
        Argument parser
    """
    def remove_argument(parser, arg):
        action_to_remove = None
        for action in parser._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                parser._remove_action(action)
                action_to_remove = action
                break
        # Remove from _option_string_actions
        if action_to_remove:
            for option_string in action_to_remove.option_strings:
                if option_string in parser._option_string_actions:
                    del parser._option_string_actions[option_string]            
        for action in parser._action_groups:
            for group_action in action._group_actions:
                opts = group_action.option_strings
                if (opts and opts[0] == arg) or group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    return                    
                
    def add_argument_to_group(parser, group_name, *args, **kwargs):
        # Locate the group by name
        for grp in parser._action_groups:
            if grp.title == group_name:
                grp.add_argument(*args, **kwargs)
                return
        raise ValueError(f"Argument group '{group_name}' not found")                

    parser = create_parser(top_level=top_level)
    remove_argument(parser, "--distance-marginalization")
    remove_argument(parser, "--distance-marginalization-lookup-table")
    remove_argument(parser, "--phase-marginalization")
    remove_argument(parser, "--time-marginalization")
    remove_argument(parser, "--jitter-time")
    remove_argument(parser, "--likelihood-type")
    remove_argument(parser, "--roq-folder")
    remove_argument(parser, "--roq-linear-matrix")
    remove_argument(parser, "--roq-quadratic-matrix")
    remove_argument(parser, "--roq-weights")
    remove_argument(parser, "--roq-weight-format")
    remove_argument(parser, "--roq-scale-factor")
    remove_argument(parser, "--fiducial-parameters")
    remove_argument(parser, "--update-fiducial-parameters")
    remove_argument(parser, "--epsilon")
    remove_argument(parser, "--default-prior")
    remove_argument(parser, "--version")
    add_argument_to_group(parser, "Likelihood arguments", "--likelihood-type", default="Chi2TimeFrequencyLikelihood", help=("The likelihood. Can be one of [Chi2TimeFrequencyLikelihood,] "
                                                                                                                            "or python path to a bilby likelihood class available in the users installation."
                                                                                                                            "If `zero` is given, a testing ZeroLikelihood is used which always "
                                                                                                                            "return zero."))
    add_argument_to_group(parser, "Likelihood arguments", "--calibration-psd-lookup-table", type=nonestr, help=("Dictionary of calibration PSD lookup files for use with calibration "
                                                                                                                "marginalization/the precomputed model. If these files don't "
                                                                                                                "exist, they will be generated from the passed uncertainties."))
    add_argument_to_group(parser, "Likelihood arguments", "--polarization-modes", type=nonestr, default="pc", help=("Polarization modes. A token consists of labels for the polarization modes. "
                                                                                                                    "`plus`: `p`, `cross`: `c`, `breathing`: `b`, `longitudinal`: `l`, "
                                                                                                                    "`vector_x`: `x`, `vector_y`: `y`. "
                                                                                                                    "For example, `pc` means plus and cross polarization modes."))
    add_argument_to_group(parser, "Likelihood arguments", "--polarization-basis", type=nonestr, default="pc", help=("Polarization basis. A token consists of labels for the polarization bases. "
                                                                                                                    "`plus`: `p`, `cross`: `c`, `breathing`: `b`, `longitudinal`: `l`, "
                                                                                                                    "`vector_x`: `x`, `vector_y`: `y`. "
                                                                                                                    "For example, `pc` means plus and cross polarization modes as the bases."))
    add_argument_to_group(parser, "Likelihood arguments", "--wavelet-frequency-resolution", type=float, default=32., help="Frequency resolution in Hz in the time-frequency domain.")
    add_argument_to_group(parser, "Likelihood arguments", "--wavelet-nx", type=float, default=4., help="Sharpness of the wavelet.")
    add_argument_to_group(parser, "Likelihood arguments", "--simulate-psd-nsample", type=int, default=10, help="Number of samples to estimate the PSD in the time-frequency domain.")
    add_argument_to_group(parser, "Prior arguments", "--default-prior", type=str, default="PolarizationPriorDict", help="The name of the prior set to base the prior on.")
    clustering_parser = parser.add_argument_group(
        title="Time-frequency clustering",
        description="The configuration of time-frequency clustering.",
    )
    clustering_parser.add("--time-frequency-clustering-method", type=nonestr, help=("Method to perform clustering. Can be one of [`data`, "
                                                                                    "`maxL`, `maP`, `random`]."))
    clustering_parser.add("--time-frequency-clustering-pe-samples-filename", type=nonestr, help=("If `maxL`, `maxP` or `random` is chosen in --time-frequency-clustering-method, "
                                                                                                 "provide the path to the bilby result file."))
    clustering_parser.add('--time-frequency-threshold', type=float, default=0.9, help="Quantile threshold to filter the excess power.")
    clustering_parser.add('--time-frequency-time-padding', type=float, default=0.1, help="Time padding in second to pad on both sides of the cluster.")
    clustering_parser.add('--time-frequency-skypoints', type=int, default=100, help="Number of skypoints to compute the sky-maximized energy map.")
    parser.add("--version", action="version", version=f"%(prog)s={__version__}")
    return parser
    
def main():
    filename = sys.argv[1]
    if filename in ["-h", "--help"]:
        logger.info("Write a default config.ini file to the specified filename.")
        logger.info("Example usage: $ nullpol_pipe_write_default_ini config.ini")
        sys.exit()
    else:
        parser = create_nullpol_parser()
        logger.info(f"Default config file written to {os.path.abspath(filename)}")
        parser.write_to_file(
            filename=filename, overwrite=True, include_description=True
        )
