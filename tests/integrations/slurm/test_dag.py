"""Check nullpol's scheduler adapter against the installed bilby_pipe package."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nullpol.cli.main import main
from nullpol.integrations.slurm.dag import SubmitSLURM
from nullpol.utils import NullpolError


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


def _generation_executable(tmp_path, body):
    """Write a stand-in ``nullpol_pipe_generation`` executable."""
    executable = tmp_path / "nullpol_pipe_generation"
    executable.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    executable.chmod(0o755)
    return str(executable)


def _submit_slurm(tmp_path, nodes, scheduler_env=None):
    """Build a Slurm adapter around a stand-in DAG."""
    slurm = object.__new__(SubmitSLURM)
    slurm.dag = SimpleNamespace(nodes=nodes)
    slurm.submit_dir = str(tmp_path)
    slurm.scheduler_env = scheduler_env
    slurm.scheduler_module = None
    return slurm


def test_local_generation_runs_all_nodes(tmp_path):
    """Multiple generation jobs all finish before their dependencies are removed."""
    marker = tmp_path / "generated"
    executable = _generation_executable(
        tmp_path, "import sys\nwith open(sys.argv[1], 'a') as stream:\n    stream.write(sys.argv[2] + '\\n')\n"
    )
    generation_nodes = [
        SimpleNamespace(
            name=f"test_data{index}_generation",
            executable=executable,
            args=[SimpleNamespace(arg=f"'{marker}' 'config {index}.ini'")],
            parents=[],
        )
        for index in range(2)
    ]
    analysis = SimpleNamespace(name="test_analysis", executable=sys.executable, parents=list(generation_nodes))
    slurm = _submit_slurm(tmp_path, [*generation_nodes, analysis])

    slurm.run_local_generation()

    assert marker.read_text().splitlines() == ["config 0.ini", "config 1.ini"]
    assert slurm.dag.nodes == [analysis]
    assert analysis.parents == []


def test_local_generation_ignores_label_matches(tmp_path):
    """A label containing ``_generation`` must not mark analysis jobs as generation jobs."""
    marker = tmp_path / "ran"
    analysis = SimpleNamespace(
        name="test_generation_run_data0_analysis",
        executable=sys.executable,
        args=[SimpleNamespace(arg=f"-c \"open('{marker}', 'w')\"")],
        parents=[],
    )
    slurm = _submit_slurm(tmp_path, [analysis])

    slurm.run_local_generation()

    assert not marker.exists()
    assert slurm.dag.nodes == [analysis]


def test_local_generation_failure_keeps_dependencies(tmp_path):
    """A failed generation job must prevent submission of its analysis jobs."""
    generation = SimpleNamespace(
        name="test_data0_generation",
        executable=_generation_executable(tmp_path, "raise SystemExit(3)\n"),
        args=[SimpleNamespace(arg="config.ini")],
    )
    analysis = SimpleNamespace(name="test_analysis", executable=sys.executable, parents=[generation])
    slurm = _submit_slurm(tmp_path, [generation, analysis])

    with pytest.raises(NullpolError, match="test_data0_generation failed with exit code 3"):
        slurm.run_local_generation()

    assert slurm.dag.nodes == [generation, analysis]
    assert analysis.parents == [generation]


def test_local_generation_uses_scheduler_env(tmp_path):
    """Local generation runs the job as its Slurm script does: scheduler-env sourced, executable run by python."""
    marker = tmp_path / "generated"
    environment = tmp_path / "activate"
    environment.write_text(
        f"export PATH={Path(sys.executable).parent}{os.pathsep}$PATH\nexport NULLPOL_TEST_ENV=activated\n",
        encoding="utf-8",
    )
    executable = tmp_path / "nullpol_pipe_generation"
    # The shebang is unusable, so the job only runs when invoked through the environment's python.
    executable.write_text(
        "#!/does/not/exist/python\nimport os, sys\nopen(sys.argv[1], 'w').write(os.environ['NULLPOL_TEST_ENV'])\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    generation = SimpleNamespace(
        name="test_data0_generation", executable=str(executable), args=[SimpleNamespace(arg=f"'{marker}'")], parents=[]
    )
    slurm = _submit_slurm(tmp_path, [generation], scheduler_env=str(environment))

    slurm.run_local_generation()

    assert marker.read_text() == "activated"
    assert slurm.dag.nodes == []


def test_slurm_overrides_loose_cpu_request(tmp_path, monkeypatch, caplog):
    """Slurm forces numeric CPU requests and says so when the user asked otherwise."""
    with caplog.at_level("WARNING"):
        _build_workflow(tmp_path, monkeypatch, strict_cpu_request=False)

    assert "Ignoring htcondor-strict-cpu-request = False" in caplog.text
