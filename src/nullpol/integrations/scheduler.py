"""Choose the DAG builder for the configured scheduler."""

from __future__ import annotations

from bilby_pipe.job_creation.dag import Dag

from .slurm import Dag as SlurmDag


def create_dag(inputs):
    """Create the DAG builder for ``inputs.scheduler``.

    Args:
        inputs: Parsed job-creation inputs.

    Returns:
        Dag: The Slurm adapter for ``scheduler = slurm``, otherwise bilby_pipe's ``Dag``.
    """
    if inputs.scheduler.lower() == "slurm":
        return SlurmDag(inputs)
    return Dag(inputs)
