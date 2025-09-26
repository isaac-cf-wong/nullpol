# pylint: disable=duplicate-code  # Legitimate CLI argument parsing patterns shared across modules
"""HTCondor job node for nullpol data generation."""
from __future__ import annotations

from bilby_pipe.job_creation.nodes.generation_node import GenerationNode as Node


class GenerationNode(Node):
    """HTCondor job node for nullpol data generation.

    Extends bilby_pipe's GenerationNode to use nullpol-specific data generation
    executable. Handles data preparation including strain data loading, PSD
    estimation, and time-frequency filter creation for polarization analysis.

    Args:
        inputs: Configuration object containing generation parameters.
        trigger_time (float): GPS time of the trigger event.
        idx (int): Index for parallel processing or multiple events.
        dag: Parent DAG object.
        parent (Node, optional): Parent node this generation depends on.
            Used to ensure cached files are built only once.

    Note:
        Uses 'nullpol_pipe_generation' executable instead of the standard
        bilby_pipe generation executable to handle polarization-specific
        data preparation.
    """

    def __init__(self, inputs, trigger_time, idx, dag, parent=None):
        super().__init__(inputs=inputs, trigger_time=trigger_time, idx=idx, dag=dag, parent=parent)

    @property
    def executable(self):
        """Path to the nullpol data generation executable.

        Returns:
            str: Path to 'nullpol_pipe_generation' executable.
        """
        return self._get_executable_path("nullpol_pipe_generation")
