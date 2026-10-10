"""Adapt bilby_pipe's shared DAG builder for nullpol's Slurm workflows."""

from __future__ import annotations

import os
import subprocess

from bilby_pipe.job_creation.dag import Dag as BilbyDag
from bilby_pipe.job_creation.slurm import SubmitSLURM as BilbySubmitSLURM

from ...utils import NullpolError, logger

GENERATION_EXECUTABLE = "nullpol_pipe_generation"


class SubmitSLURM(BilbySubmitSLURM):
    """Adapt log paths and local generation before upstream Slurm submission."""

    @staticmethod
    def _output_name_from_dag(extra_lines):
        """Parse output paths independently of upstream submit-line spacing."""
        for line in extra_lines:
            key, separator, value = line.partition("=")
            if separator and key.strip() == "output":
                return value.strip().replace("_$(Cluster)", "").replace("_$(Process)", "")
        raise NullpolError("Slurm job is missing its output path")

    def run_local_generation(self):
        """Run every generation job and retain dependencies if one fails."""
        for node in list(self.dag.nodes):
            # Match on the executable: node names start with the user's label.
            if os.path.basename(node.executable) != GENERATION_EXECUTABLE:
                continue
            # Run the job's own Slurm script so modules, scheduler-env and quoting match the scheduled job.
            # -e stops it when a setup step such as sourcing scheduler-env fails.
            script = self._write_individual_processes(node.name, node.executable, node.args[0].arg)
            try:
                subprocess.run(["/bin/bash", "-e", script], check=True)  # noqa: S603
            except subprocess.CalledProcessError as error:
                raise NullpolError(
                    f"Local generation job {node.name} failed with exit code {error.returncode}: {script}"
                ) from error
            for other_node in self.dag.nodes:
                if node in other_node.parents:
                    other_node.parents.remove(node)
            self.dag.nodes.remove(node)


class Dag(BilbyDag):
    """Use upstream job creation with nullpol's Slurm submission adapter."""

    def __init__(self, inputs):
        """Keep numeric CPU counts in the intermediate pycondor jobs.

        This sets ``inputs.htcondor_strict_cpu_request`` on the object passed in,
        because the job nodes read it from the same inputs. ``generate_dag``
        passes a private copy.
        """
        # HTCondor's dynamic CPU expression cannot describe a Slurm allocation.
        if not inputs.htcondor_strict_cpu_request:
            logger.warning(
                "Ignoring htcondor-strict-cpu-request = False for scheduler = slurm: "
                "Slurm jobs request exactly request-cpus cores."
            )
            inputs.htcondor_strict_cpu_request = True
        super().__init__(inputs)

    def build_slurm_submit(self):
        """Use upstream Slurm scripts after any local generation jobs finish."""
        slurm = SubmitSLURM(self)
        if self.inputs.local_generation:
            slurm.run_local_generation()
        slurm.write_master_slurm()
