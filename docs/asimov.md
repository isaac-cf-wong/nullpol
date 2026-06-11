# Asimov integration

nullpol registers an Asimov pipeline and hooks for running analyses within the
[Asimov](https://asimov.docs.ligo.org/) workflow. Install the optional
dependency with `pip install nullpol[asimov]`.

## Getting started

1. Create a project:

    ```bash
    asimov init "Test project"
    ```

2. Generate a default analysis configuration with `nullpol_get_asimov_yaml`.

See the API reference for
[`nullpol.integrations.asimov`](reference/nullpol/integrations/asimov/index.md).
