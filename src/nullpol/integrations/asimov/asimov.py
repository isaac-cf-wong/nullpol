"""Nullpol Pipeline specification."""

from __future__ import annotations

import glob
import os
import subprocess
import time

from importlib.resources import files
from asimov import config  # pylint: disable=import-error
from asimov.pipeline import Pipeline, PipelineException, PipelineLogger  # pylint: disable=import-error
from asimov.pipelines.bilby import Bilby  # pylint: disable=import-error

from .pesummary import PESummaryPipeline


class Nullpol(Bilby):
    """
    The nullpol pipeline.

    Args:
        production (asimov.Production): The production object.
        category (str, optional): The category of the job.
            Defaults to "C01_offline".
    """

    name = "nullpol"
    STATUS = {"wait", "stuck", "stopped", "running", "finished"}

    @property
    def config_template(self):
        return str(files("nullpol.integrations.asimov.templates") / "nullpol.ini")

    def __init__(self, production, category=None):
        Pipeline.__init__(self, production=production, category=category)
        self.logger.info("Using the nullpol pipeline")

        if not production.pipeline.lower() == "nullpol":
            raise PipelineException(f"Pipeline {production.pipeline.lower()} " "is not recognized.")

    def detect_completion(self):
        """
        Check for the production of the posterior file to signal that the job has completed.
        """
        self.logger.info("Checking if the nullpol job has completed")
        results_dir = glob.glob(f"{self.production.rundir}/result")
        # config = self.read_ini(self.config_file_path)
        expected_number_of_result_files = len(self.production.meta["likelihood"]["polarization modes"])
        if len(results_dir) > 0:  # dynesty_merge_result.json
            results_files = glob.glob(os.path.join(results_dir[0], "*merge*_result.hdf5"))
            results_files += glob.glob(os.path.join(results_dir[0], "*merge*_result.json"))
            self.logger.debug(f"results files {results_files}")
            if len(results_files) == expected_number_of_result_files:
                self.logger.info(f"{len(results_files)} result files found, the job is finished.")
                return True
            if len(results_files) > 0:
                self.logger.info(
                    f"{len(results_files)} < {expected_number_of_result_files} result files found, the job is not finished."
                )
                return False
            self.logger.info("No results files found.")
            return False
        self.logger.info("No results directory found")
        return False

    def subrun_samples(self, subrun_label, absolute=False):
        """
        Collect the combined samples file for PESummary.
        """

        if absolute:
            rundir = os.path.abspath(self.production.rundir)
        else:
            rundir = self.production.rundir
        self.logger.info(f"Rundir for samples: {rundir}")
        return glob.glob(os.path.join(rundir, "result", f"*_{subrun_label}_merge*_result.hdf5")) + glob.glob(
            os.path.join(rundir, "result", f"*_{subrun_label}_merge*_result.json")
        )

    def after_completion(self):
        post_pipeline = PESummaryPipeline(production=self.production)
        self.logger.info("Job has completed. Running PE Summary.")
        cluster = post_pipeline.submit_dag()
        self.production.meta["job id"] = int(cluster)
        self.production.status = "processing"
        self.production.event.update_data()

    @property
    def config_file_path(self):
        """Path of the configuration file.

        Returns:
            str: Path of the configuration file.
        """
        cwd = os.getcwd()
        if self.production.event.repository:
            ini = self.production.event.repository.find_prods(self.production.name, self.category)[0]
            ini = os.path.join(cwd, ini)
        else:
            ini = f"{self.production.name}.ini"
        return ini

    def build_dag(self, psds=None, user=None, clobber_psd=False, dryrun=False):  # pylint: disable=unused-argument
        """
        Construct a DAG file in order to submit a production to the
        condor scheduler using nullpol_pipe.

        Args:
            production (str): The production name.
        psds (dict, optional): The PSDs which should be used for this DAG. If no PSDs are
           provided the PSD files specified in the ini file will be used
           instead.
        user (str): The user accounting tag which should be used to run the job.
        dryrun (bool): If set to true the commands will not be run, but will be printed to
           standard output. Defaults to False.

        Raises:
            PipelineException: Raised if the construction of the DAG fails.
        """

        cwd = os.getcwd()

        self.logger.info(f"Working in {cwd}")

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
            self.config_file_path,
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
            return None

        self.logger.info(" ".join(command))
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as pipe:
            out, err = pipe.communicate()
        self.logger.info(out)

        expected_out = "DAG generation complete, to submit jobs"
        if err or expected_out not in str(out):
            self.production.status = "stuck"
            self.logger.error(err)
            raise PipelineException(
                (f"DAG file could not be created.\n" f"{command}\n{out}\n\n{err}"),
                production=self.production.name,
            )
        time.sleep(10)
        return PipelineLogger(message=out, production=self.production.name)
