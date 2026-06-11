# Injections

The usage of the package is similar to bilby_pipe. To define a set of
injections, please refer to the
[bilby_pipe injections documentation](https://lscsoft.docs.ligo.org/bilby_pipe/master/injections.html)
for details. This page collects typical nullpol usage patterns; see the
[examples](examples/scalar-tensor.md) section for a complete walkthrough.

## Creating injection files

Use `nullpol_create_injection` or the bilby_pipe injection utilities with
nullpol-specific waveform and source models. The scalar-tensor example
demonstrates generating an injection file and configuring a bilby_pipe run.
