"""Check nullpol's scheduler adapter against the installed bilby_pipe package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nullpol.cli.main import main
from nullpol.integrations.slurm.dag import SubmitSLURM


def _build_workflow(tmp_path, monkeypatch, *, scheduler="slurm", strict_cpu_request=False, request_cpus=4):
    config = tmp_path / "config.ini"
    outdir = tmp_path / "outdir"
    config.write_text(
        "label = scheduler_test\n"
        f"outdir = {outdir}\n"
        f"scheduler = {scheduler}\n"
        "accounting = test.accounting\n"
        "detectors = [H1, L1, V1]\n"
        "polarization-modes = pc\n"
        "polarization-basis = p\n"
        "gaussian-noise = True\n"
        "n-simulation = 1\n"
        "trigger-time = 1126259462.4\n"
        "transfer-files = False\n"
        f"request-cpus = {request_cpus}\n"
        f"htcondor-strict-cpu-request = {strict_cpu_request}\n"
        "scheduler-analysis-time = 02:00:00\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(sys, "argv", ["nullpol_pipe", str(config)])
    main()
    return outdir / "submit"


@pytest.mark.parametrize("scheduler", ["slurm", "condor"])
@pytest.mark.parametrize("strict_cpu_request", [True, False])
@pytest.mark.parametrize("request_cpus", [1, 4])
def test_scheduler_cpu_requests(tmp_path, monkeypatch, scheduler, strict_cpu_request, request_cpus):
    """Generate submission files with numeric Slurm CPUs and unchanged Condor behavior."""
    submit_dir = _build_workflow(
        tmp_path,
        monkeypatch,
        scheduler=scheduler,
        strict_cpu_request=strict_cpu_request,
        request_cpus=request_cpus,
    )
    if scheduler == "slurm":
        master = (submit_dir / "slurm_scheduler_test_master.sh").read_text()
        assert f"--ntasks-per-node={request_cpus}" in master
        assert "--ntasks-per-node=Cpus" not in master
        assert "--time=02:00:00" in master
        assert "--dependency=afterok:" in master
        assert "--output= " not in master
        assert "--error= " not in master
        analysis = next(submit_dir.glob("*analysis*_pc_p.sh")).read_text()
        assert "--polarization-modes pc" in analysis
        assert "--polarization-basis p" in analysis
        assert "--data-dump-file" in analysis
        assert "TARGET.Cpus" not in analysis
    else:
        dynamic_cpus = not strict_cpu_request and request_cpus > 1
        analysis = next(submit_dir.glob("*analysis*_pc_p.submit")).read_text()
        expected_cpus = "Cpus" if dynamic_cpus else str(request_cpus)
        assert f"request_cpus = {expected_cpus}" in analysis
        dag = (submit_dir / "dag_scheduler_test.submit").read_text()
        assert ("$$([TARGET.Cpus])" in dag) is dynamic_cpus


@pytest.mark.parametrize(
    "line",
    ["output = logs/job_$(Cluster)_$(Process).out", "output  =   logs/job_$(Cluster)_$(Process).out  "],
)
def test_slurm_output_path(line):
    """Handle the output-line formats from bilby_pipe 1.9 and 1.10."""
    assert SubmitSLURM._output_name_from_dag(["output_other = ignored", line]) == "logs/job.out"


def test_local_generation_runs_all_nodes(tmp_path):
    """Multiple generation jobs all finish before their dependencies are removed."""
    marker = tmp_path / "generated"
    script = tmp_path / "generate.py"
    script.write_text(
        f"with open({str(marker)!r}, 'a') as stream:\n    stream.write('generated\\n')\n", encoding="utf-8"
    )
    generation_nodes = [
        SimpleNamespace(
            name=f"test_generation_{index}",
            executable=sys.executable,
            args=[SimpleNamespace(arg=str(script))],
            parents=[],
        )
        for index in range(2)
    ]
    analysis = SimpleNamespace(name="test_analysis", parents=list(generation_nodes))
    slurm = object.__new__(SubmitSLURM)
    slurm.dag = SimpleNamespace(nodes=[*generation_nodes, analysis])

    slurm.run_local_generation()

    assert marker.read_text().splitlines() == ["generated", "generated"]
    assert slurm.dag.nodes == [analysis]
    assert analysis.parents == []


def test_local_generation_failure_keeps_dependencies():
    """A failed generation job must prevent submission of its analysis jobs."""
    generation = SimpleNamespace(
        name="test_generation", executable=sys.executable, args=[SimpleNamespace(arg='-c "raise SystemExit(1)"')]
    )
    analysis = SimpleNamespace(name="test_analysis", parents=[generation])
    slurm = object.__new__(SubmitSLURM)
    slurm.dag = SimpleNamespace(nodes=[generation, analysis])

    with pytest.raises(subprocess.CalledProcessError):
        slurm.run_local_generation()

    assert slurm.dag.nodes == [generation, analysis]
    assert analysis.parents == [generation]
