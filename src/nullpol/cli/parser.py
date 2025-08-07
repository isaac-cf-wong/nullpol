from __future__ import annotations

import os
import sys

import bilby_pipe.utils
from bilby_pipe.bilbyargparser import BilbyArgParser, HyphenStr
from bilby_pipe.parser import create_parser
from bilby_pipe.utils import nonestr

from .. import __version__
from ..utils import logger

bilby_pipe.utils.logger = logger


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


def create_nullpol_parser(top_level: bool = True) -> BilbyArgParser:
    """Create the nullpol_pipe parser.

    Args:
        top_level (bool, optional): If true, the top-level parser is created. If false, a subparser is
            created. Defaults to True.

    Returns:
        BilbyArgParser: Argument parser instance.
    """

    def _remove_argument(parser, arg):
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

    def _add_argument_to_group(parser, group_name, *args, **kwargs):
        # Locate the group by name
        for grp in parser._action_groups:
            if grp.title == group_name:
                grp.add_argument(*args, **kwargs)
                return
        raise ValueError(f"Argument group '{group_name}' not found")

    parser = create_parser(top_level=top_level)
    _remove_argument(parser, "--coherence-test")
    _remove_argument(parser, "--calibration-marginalization")
    _remove_argument(parser, "--calibration-lookup-table")
    _remove_argument(parser, "--number-of-response-curves")
    _remove_argument(parser, "--distance-marginalization")
    _remove_argument(parser, "--distance-marginalization-lookup-table")
    _remove_argument(parser, "--phase-marginalization")
    _remove_argument(parser, "--time-marginalization")
    _remove_argument(parser, "--jitter-time")
    _remove_argument(parser, "--likelihood-type")
    _remove_argument(parser, "--roq-folder")
    _remove_argument(parser, "--roq-linear-matrix")
    _remove_argument(parser, "--roq-quadratic-matrix")
    _remove_argument(parser, "--roq-weights")
    _remove_argument(parser, "--roq-weight-format")
    _remove_argument(parser, "--roq-scale-factor")
    _remove_argument(parser, "--fiducial-parameters")
    _remove_argument(parser, "--update-fiducial-parameters")
    _remove_argument(parser, "--epsilon")
    _remove_argument(parser, "--default-prior")
    _remove_argument(parser, "--version")
    _add_argument_to_group(
        parser,
        "Likelihood arguments",
        "--likelihood-type",
        default="FractionalProjectionTimeFrequencyLikelihood",
        help=(
            "The likelihood. Can be one of [Chi2TimeFrequencyLikelihood, GaussianTimeFrequencyLikelihood, FractionalProjectionTimeFrequencyLikelihood, zero] "
            "or python path to a bilby likelihood class available in the users installation."
            "If `zero` is given, a testing ZeroLikelihood is used which always "
            "return zero."
        ),
    )
    _add_argument_to_group(
        parser,
        "Likelihood arguments",
        "--polarization-modes",
        action="append",
        help=(
            "Polarization models. A token consists of labels for the polarization modes. "
            "`plus`: `p`, `cross`: `c`, `breathing`: `b`, `longitudinal`: `l`, "
            "`vector_x`: `x`, `vector_y`: `y`. "
            "For example, [pc, pcb] represents running over pc and pcb models."
            "pc represents only running the pc model."
        ),
    )
    _add_argument_to_group(
        parser,
        "Likelihood arguments",
        "--polarization-basis",
        action="append",
        help=(
            "Polarization basis. A token consists of labels for the polarization bases. "
            "`plus`: `p`, `cross`: `c`, `breathing`: `b`, `longitudinal`: `l`, "
            "`vector_x`: `x`, `vector_y`: `y`. "
            "For example, [p, b] represents using basis p for the first model, and basis b for the second model."
            "For a single run, you can also specify a single label e.g. `p` to indicate the basis."
        ),
    )
    _add_argument_to_group(
        parser,
        "Likelihood arguments",
        "--wavelet-frequency-resolution",
        type=float,
        default=16.0,
        help="Frequency resolution in Hz in the time-frequency domain.",
    )
    _add_argument_to_group(
        parser, "Likelihood arguments", "--wavelet-nx", type=float, default=4.0, help="Sharpness of the wavelet."
    )
    _add_argument_to_group(
        parser,
        "Prior arguments",
        "--default-prior",
        type=str,
        default="PolarizationPriorDict",
        help="The name of the prior set to base the prior on.",
    )
    _add_argument_to_group(
        parser,
        "Calibration arguments",
        "--calibration-correction-type",
        type=nonestr,
        default="data",
        help=(
            "Type of calibration correction: can be either `data` or `template`. "
            "See https://bilby-dev.github.io/bilby/api/bilby.gw.detector.calibration.html "
            "for more information."
        ),
    )
    clustering_parser = parser.add_argument_group(
        title="Time-frequency clustering",
        description="The configuration of time-frequency clustering.",
    )
    clustering_parser.add(
        "--time-frequency-clustering-method",
        type=nonestr,
        help=(
            "Method to perform clustering. Can be one of [`data`, "
            "`injection`, `injection_parameters_file`, `maxL`, `maP`, `random`]."
        ),
    )
    clustering_parser.add(
        "--time-frequency-clustering-injection-parameters-filename",
        type=nonestr,
        help=(
            "If `injection_parameters_file` is chosen in --time-frequency-clustering-method, "
            "provide the path to the injection parameters file."
        ),
    )
    clustering_parser.add(
        "--time-frequency-clustering-pe-samples-filename",
        type=nonestr,
        help=(
            "If `maxL`, `maxP` or `random` is chosen in --time-frequency-clustering-method, "
            "provide the path to the bilby result file."
        ),
    )
    clustering_parser.add(
        "--time-frequency-clustering-threshold", type=float, default=1.0, help="Threshold to filter the excess power."
    )
    clustering_parser.add(
        "--time-frequency-clustering-threshold-type", type=str, default="variance", help="Type of threshold."
    )
    clustering_parser.add(
        "--time-frequency-clustering-time-padding",
        type=float,
        default=0.1,
        help="Time padding in second to pad on both sides of the cluster.",
    )
    clustering_parser.add(
        "--time-frequency-clustering-frequency-padding",
        type=float,
        default=1,
        help="Frequency padding in Hz to pad on both sides of the cluster.",
    )
    clustering_parser.add(
        "--time-frequency-clustering-skypoints",
        type=int,
        default=100,
        help="Number of skypoints to compute the sky-maximized energy map.",
    )
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
        parser.write_to_file(filename=filename, overwrite=True, include_description=True)
