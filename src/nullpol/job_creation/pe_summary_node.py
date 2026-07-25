import copy
import re

from bilby_pipe.job_creation.nodes import PESummaryNode


class NullpolPESummaryNode(PESummaryNode):
    """PESummaryNode that:

    - Passes --psd and --calibration from the bilby_pipe config to the
      summarypages command (bilby_pipe's PESummaryNode has access to these
      for HTCondor file-transfer but does not add them as command arguments,
      so PESummary cannot generate PSD or calibration plots without this).

    - Shortens the auto-generated labels from bilby_pipe's full run ID
      (e.g. nullpol_data0_..._analysis_H1L1V1_pc_p_merge) to just the
      polarization mode and basis (e.g. pc_p).
    """

    def __init__(self, inputs, merged_node_list, generation_node_list, dag):
        self._n_results = len(merged_node_list)

        short_labels = [
            re.sub(r'^.*_analysis_[A-Z0-9]+_(.+)_merge$', r'\1', node.label)
            for node in merged_node_list
        ]

        # Shallow-copy inputs so we don't mutate the shared object, then
        # replace summarypages_arguments with a new dict that includes labels.
        inputs = copy.copy(inputs)
        base_args = inputs.summarypages_arguments or {}
        inputs.summarypages_arguments = {**base_args, "labels": short_labels}

        super().__init__(inputs, merged_node_list, generation_node_list, dag)

    def process_node(self):
        psd_dict = self.inputs.psd_dict
        if psd_dict:
            psd_str = " ".join(f"{ifo}:{path}" for ifo, path in psd_dict.items())
            self.arguments.add("psd", (" " + psd_str) * self._n_results)

        cal_dict = self.inputs.spline_calibration_envelope_dict
        if cal_dict:
            cal_str = " ".join(f"{ifo}:{path}" for ifo, path in cal_dict.items())
            self.arguments.add("calibration", (" " + cal_str) * self._n_results)

        super().process_node()
