from __future__ import annotations

import os

import htcondor
from asimov import config, utils
from asimov.pipeline import PostPipeline


class PESummaryPipeline(PostPipeline):
    """
    A postprocessing pipeline add-in using PESummary.
    """

    name = "PESummary"
    def submit_dag(self, dryrun=False):
        """
        Run PESummary on the results of this job.
        """

        psds = {ifo: os.path.abspath(psd) for ifo, psd in self.production.psds.items()}

        polarization_modes = self.production.meta['likelihood']['polarization modes']
        polarization_basis = self.production.meta['likelihood']['polarization basis']
        number_of_subruns = len(polarization_modes)

        if "calibration" in self.production.meta["data"]:
            calibration = [
                (
                    os.path.abspath(
                        os.path.join(self.production.repository.directory, cal)
                    )
                    if not cal[0] == "/"
                    else cal
                )
                for cal in self.production.meta["data"]["calibration"].values()
            ]
        else:
            calibration = None

        configfile = self.production.event.repository.find_prods(
            self.production.name, self.category
        )[0]
        command = [
            "--webdir",
            os.path.join(
                config.get("project", "root"),
                config.get("general", "webroot"),
                self.production.event.name,
                self.production.name,
                "pesummary",
            ),
            "--labels",
        ]
        for i in range(number_of_subruns):
            command += [f'{self.production.name}_{polarization_modes[i]}_{polarization_basis[i]}']
        command += ["--gw",
                    "--f_low",
                    str(min(self.production.meta["quality"]["minimum frequency"].values())),
        ]
        command = [
            "--webdir",
            os.path.join(
                config.get("project", "root"),
                config.get("general", "webroot"),
                self.production.event.name,
                self.production.name,
                "pesummary",
            ),
            "--labels",
            self.production.name,
            "--gw",
            "--f_low",
            str(min(self.production.meta["quality"]["minimum frequency"].values())),
        ]
        if "skymap samples" in self.meta:
            command += [
                "--nsamples_for_skymap",
                str(
                    self.meta["skymap samples"]
                ),  # config.get('pesummary', 'skymap_samples'),
            ]

        if "multiprocess" in self.meta:
            command += ["--multi_process", str(self.meta["multiprocess"])]

        if "regenerate" in self.meta:
            command += ["--regenerate", " ".join(self.meta["regenerate posteriors"])]

        # Config file
        command += ["--config"]
        config_filename = os.path.join(
                self.production.event.repository.directory, self.category, configfile
            )
        for i in range(number_of_subruns):
            command += [config_filename]
        # Samples
        command += ["--samples"]
        for i in range(number_of_subruns):
            command += self.production.pipeline.subrun_samples(subrun_label=f'{polarization_modes[i]}_{polarization_basis[i]}',
                                                               absolute=True)
        # Calibration information
        if calibration:
            command += ["--calibration"]
            for i in range(number_of_subruns):
                command += calibration
        # PSDs
        command += ["--psd"]
        for i in range(number_of_subruns):
            for key, value in psds.items():
                command += [f"{key}:{value}"]

        if "keywords" in self.meta:
            for key, argument in self.meta["keywords"]:
                if argument is not None and len(key) > 1:
                    command += [f"--{key}", f"{argument}"]
                elif argument is not None and len(key) == 1:
                    command += [f"-{key}", f"{argument}"]
                else:
                    command += [f"{key}"]

        with utils.set_directory(self.production.rundir):
            with open(f"{self.production.name}_pesummary.sh", "w") as bash_file:
                bash_file.write(
                    f"{config.get('pesummary', 'executable')} " + " ".join(command)
                )

        self.logger.info(
            f"PE summary command: {config.get('pesummary', 'executable')} {' '.join(command)}"
        )

        if dryrun:
            print("PESUMMARY COMMAND")
            print("-----------------")
            print(" ".join(command))

        submit_description = {
            "executable": config.get("pesummary", "executable"),
            "arguments": " ".join(command),
            "output": f"{self.production.rundir}/pesummary.out",
            "error": f"{self.production.rundir}/pesummary.err",
            "log": f"{self.production.rundir}/pesummary.log",
            "request_cpus": self.meta["multiprocess"],
            "environment": "HDF5_USE_FILE_LOCKING=FAlSE OMP_NUM_THREADS=1 OMP_PROC_BIND=false",
            "getenv": "CONDA_EXE,USER,LAL*,PATH",
            "batch_name": f"PESummary/{self.production.event.name}/{self.production.name}",
            "request_memory": "8192MB",
            # "should_transfer_files": "YES",
            "request_disk": "8192MB",
            "+flock_local": "True",
            "+DESIRED_Sites": htcondor.classad.quote("nogrid"),
        }

        if "accounting group" in self.meta:
            submit_description["accounting_group_user"] = config.get("condor", "user")
            submit_description["accounting_group"] = self.meta["accounting group"]
        else:
            self.logger.warning(
                "This PESummary Job does not supply any accounting"
                " information, which may prevent it running on"
                " some clusters."
            )

        if dryrun:
            print("SUBMIT DESCRIPTION")
            print("------------------")
            print(submit_description)

        if not dryrun:
            hostname_job = htcondor.Submit(submit_description)

            with utils.set_directory(self.production.rundir):
                with open("pesummary.sub", "w") as subfile:
                    subfile.write(hostname_job.__str__())

            try:
                # There should really be a specified submit node, and if there is, use it.
                schedulers = htcondor.Collector().locate(
                    htcondor.DaemonTypes.Schedd, config.get("condor", "scheduler")
                )
                schedd = htcondor.Schedd(schedulers)
            except Exception:
                # If you can't find a specified scheduler, use the first one you find
                schedd = htcondor.Schedd()
            with schedd.transaction() as txn:
                cluster_id = hostname_job.queue(txn)

        else:
            cluster_id = 0

        return cluster_id
