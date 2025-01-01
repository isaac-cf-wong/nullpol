"""Bilby Pipeline specification."""

import glob
import os
import re
import subprocess
import configparser

import time

import pathlib
import pkg_resources
import asimov
from asimov import config
from asimov.pipeline import Pipeline, PipelineException, PipelineLogger, PESummaryPipeline


class Nullpol(asimov.pipeline.Pipeline):
    """
    The nullpol pipeline.

    Parameters
    ----------
    production : :class:`asimov.Production`
       The production object.
    category : str, optional
        The category of the job.
        Defaults to "C01_offline".
    """

    name = "nullpol"
    STATUS = {"wait", "stuck", "stopped", "running", "finished"}

    @property
    def config_template(self):
        return pkg_resources.resource_filename('nullpol.asimov', 'nullpol.ini')

    def __init__(self, production, category=None):
        super(Nullpol, self).__init__(production, category)
        self.logger.info("Using the nullpol pipeline")

        if not production.pipeline.lower() == "nullpol":
            raise PipelineException

    def build_dag(self, psds=None, user=None, clobber_psd=False, dryrun=False):
        """
        Construct a DAG file in order to submit a production to the
        condor scheduler using nullpol_pipe.

        Parameters
        ----------
        production : str
           The production name.
        psds : dict, optional
           The PSDs which should be used for this DAG. If no PSDs are
           provided the PSD files specified in the ini file will be used
           instead.
        user : str
           The user accounting tag which should be used to run the job.
        dryrun: bool
           If set to true the commands will not be run, but will be printed to standard output. Defaults to False.

        Raises
        ------
        PipelineException
           Raised if the construction of the DAG fails.
        """

        cwd = os.getcwd()

        self.logger.info(f"Working in {cwd}")

        if self.production.event.repository:
            ini = self.production.event.repository.find_prods(
                self.production.name, self.category
            )[0]
            ini = os.path.join(cwd, ini)
        else:
            ini = f"{self.production.name}.ini"

        if self.production.rundir:
            rundir = self.production.rundir
        else:
            rundir = os.path.join(
                os.path.expanduser("~"),
                self.production.event.name,
                self.production.name,
            )
            self.production.rundir = rundir

        if "job label" in self.production.meta:
            job_label = self.production.meta["job label"]
        else:
            job_label = self.production.name

        command = [
            os.path.join(config.get("pipelines", "environment"), "bin", "nullpol_pipe"),
            ini,
            "--label",
            job_label,
            "--outdir",
            f"{os.path.abspath(self.production.rundir)}",
        ]

        if "accounting group" in self.production.meta:
            command += [
                "--accounting",
                f"{self.production.meta['scheduler']['accounting group']}",
            ]
        else:
            self.logger.warning(
                "This nullpol Job does not supply any accounting"
                " information, which may prevent it running"
                " on some clusters."
            )

        if dryrun:
            print(" ".join(command))
        else:
            self.logger.info(" ".join(command))
            pipe = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            out, err = pipe.communicate()
            self.logger.info(out)

            if err or "DAG generation complete, to submit jobs" not in str(out):
                self.production.status = "stuck"
                self.logger.error(err)
                raise PipelineException(
                    f"DAG file could not be created.\n{command}\n{out}\n\n{err}",
                    production=self.production.name,
                )
            else:
                time.sleep(10)
                return PipelineLogger(message=out, production=self.production.name)