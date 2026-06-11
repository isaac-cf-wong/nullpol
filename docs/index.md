# nullpol

**A package to perform model-independent polarization tests of
gravitational-wave signals.**

`nullpol` is a Python package designed to facilitate model-agnostic tests of
gravitational-wave polarizations. Built upon the robust parameter estimation
capabilities of the [bilby](https://lscsoft.docs.ligo.org/bilby/) framework,
nullpol seamlessly integrates with
[bilby_pipe](https://lscsoft.docs.ligo.org/bilby_pipe/master/index.html) to
automate the workflow, enabling efficient and reproducible analyses.

## Quick links

- [Installation](installation.md)
- [Injections](injections.md)
- [Examples](examples/scalar-tensor.md)
- [API Reference](reference/index.md)
- [Citing nullpol](citing.md)

## Command-line tools

```bash
nullpol_create_injection --help
nullpol_create_time_frequency_filter_from_sample --help
nullpol_pipe --help
nullpol_pipe_analysis --help
nullpol_pipe_generation --help
nullpol_pipe_write_default_ini --help
nullpol_get_asimov_yaml --help
```
