from bilby_pipe.parser import create_parser
from bilby_pipe.utils import (parse_args,
                              get_command_line_arguments,
                              get_outdir_name,
                              tcolors,
                              logger
                              )
from bilby_pipe.job_creation import generate_dag
from bilby_pipe.main import (perform_runtime_checks,
                             write_complete_config_file)
from bilby_pipe.main import MainInput as BilbyMainInput
from ..utility import log_version_information


class MainInput(BilbyMainInput):
    def __init__(self, args, unknown_args):
        self.known_args = args
        self.unknown_args = unknown_args

        self.run_local = args.local
        # Read the other arguments
        for name in dir(args):
            if not name.startswith("_"):
                setattr(self, name, getattr(args, name, None))
        if self.injection:
            self.check_injection()
        self.mpi_samplers = ["pymultinest"]
        self.use_mpi = (self.sampler in self.mpi_samplers) and (self.request_cpus > 1)
        if self.create_plots:
            for plot_attr in [
                "calibration",
                "corner",
                "marginal",
                "skymap",
                "waveform",
                "format",
            ]:
                attr = f"plot_{plot_attr}"
                setattr(self, attr, getattr(args, attr))
        self.extra_lines = []
        self.requirements = []
        

def main():
    """ Top-level interface for bilby_pipe """
    parser = create_parser(top_level=True)
    args, unknown_args = parse_args(get_command_line_arguments(), parser)

    # Check and sort outdir
    args.outdir = args.outdir.replace("'", "").replace('"', "")
    if args.overwrite_outdir is False:
        args.outdir = get_outdir_name(args.outdir)

    log_version_information()
    inputs = MainInput(args, unknown_args)
    perform_runtime_checks(inputs, args)
    inputs.pretty_print_prior()
    write_complete_config_file(parser, args, inputs)
    generate_dag(inputs)

    if len(unknown_args) > 0:
        msg = [tcolors.WARNING, f"Unrecognized arguments {unknown_args}", tcolors.END]
        logger.warning(" ".join(msg))