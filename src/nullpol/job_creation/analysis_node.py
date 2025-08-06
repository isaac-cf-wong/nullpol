from __future__ import annotations

import os

from bilby_pipe.job_creation.node import Node
from bilby_pipe.job_creation.nodes.analysis_node import touch_checkpoint_files


class AnalysisNode(Node):
    """HTCondor job node for polarization-aware parameter estimation analysis.

    Extends bilby_pipe's Node class to handle nullpol-specific analysis jobs
    with polarization mode configurations. Creates and manages the HTCondor
    job for running parameter estimation on gravitational wave data with
    specific polarization basis and derived modes.

    Args:
        inputs: Configuration object containing analysis parameters.
        generation_node: Parent data generation node this analysis depends on.
        detectors (list[str]): List of detector names to analyze.
        sampler (str): MCMC sampler to use for parameter estimation.
        parallel_idx (int): Index for parallel processing.
        dag: Parent DAG object.
        polarization_modes (str): Polarization modes to include in analysis.
        polarization_basis (str): Polarization basis modes for the analysis.

    Attributes:
        polarization_modes (str): Polarization modes being analyzed.
        polarization_basis (str): Basis polarization modes.
        base_job_name (str): Base name for the HTCondor job incorporating
            detectors and polarization configuration.
    """

    def __init__(
        self, inputs, generation_node, detectors, sampler, parallel_idx, dag, polarization_modes, polarization_basis
    ):
        super().__init__(inputs=inputs, retry=3)
        self.polarization_modes = polarization_modes
        self.polarization_basis = polarization_basis
        self.dag = dag
        self.generation_node = generation_node
        self.detectors = detectors
        self.parallel_idx = parallel_idx
        self.request_cpus = inputs.request_cpus

        data_label = generation_node.job_name
        base_name = data_label.replace("generation", "analysis")
        self.base_job_name = f"{base_name}_{''.join(detectors)}_{self.polarization_modes}_{self.polarization_basis}"
        if parallel_idx != "":
            self.job_name = f"{self.base_job_name}_{parallel_idx}"
        else:
            self.job_name = self.base_job_name
        self.label = self.job_name

        if self.inputs.use_mpi:
            self.setup_arguments(parallel_program=self._get_executable_path(self.inputs.analysis_executable))

        else:
            self.setup_arguments()

        self.arguments.add("polarization-modes", self.polarization_modes)
        self.arguments.add("polarization-basis", self.polarization_basis)

        if self.inputs.transfer_files or self.inputs.osg:
            data_dump_file = generation_node.data_dump_file
            input_files_to_transfer = (
                [
                    str(data_dump_file),
                    str(self.inputs.complete_ini_file),
                ]
                + touch_checkpoint_files(
                    os.path.join(inputs.outdir, "result"),
                    self.job_name,
                    inputs.sampler,
                    inputs.result_format,
                )
                + inputs.additional_transfer_paths
            )
            self.extra_lines.extend(
                self._condor_file_transfer_lines(
                    input_files_to_transfer,
                    [self._relative_topdir(self.inputs.outdir, self.inputs.initialdir)],
                )
            )
            self.arguments.add("outdir", os.path.relpath(self.inputs.outdir))

        for det in detectors:
            self.arguments.add("detectors", det)
        self.arguments.add("label", self.label)
        self.arguments.add("data-dump-file", generation_node.data_dump_file)
        self.arguments.add("sampler", sampler)
        if self.parallel_idx and self.inputs.sampling_seed:
            self.arguments.add(
                "sampling-seed",
                str(int(self.inputs.sampling_seed) + int(self.parallel_idx[3:])),
            )

        self.extra_lines.extend(self._checkpoint_submit_lines())

        self.process_node()
        self.job.add_parent(generation_node.job)

    @property
    def polarization_modes(self):
        return self._polarization_modes

    @polarization_modes.setter
    def polarization_modes(self, polarization_modes):
        self._polarization_modes = polarization_modes.replace('"', "").replace("'", "")

    @property
    def polarization_basis(self):
        return self._polarization_basis

    @polarization_basis.setter
    def polarization_basis(self, polarization_basis):
        self._polarization_basis = polarization_basis.replace('"', "").replace("'", "")

    @property
    def executable(self):
        if self.inputs.use_mpi:
            return self._get_executable_path("mpiexec")
        elif self.inputs.analysis_executable:
            return self._get_executable_path(self.inputs.analysis_executable)
        else:
            return self._get_executable_path("nullpol_pipe_analysis")

    @property
    def request_memory(self):
        return self.inputs.request_memory

    @property
    def log_directory(self):
        return self.inputs.data_analysis_log_directory

    @property
    def result_file(self):
        return f"{self.inputs.result_directory}/{self.job_name}_result.{self.inputs.result_format}"

    @property
    def slurm_walltime(self):
        """Default wall-time for base-name"""
        # Seven days
        return self.inputs.scheduler_analysis_time
