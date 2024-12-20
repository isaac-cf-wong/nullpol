from bilby_pipe.job_creation.nodes.analysis_node import AnalysisNode as Node


class AnalysisNode(Node):
    def __init__(self, inputs, generation_node, detectors, sampler, parallel_idx, dag):
        super(AnalysisNode, self).__init__(inputs=inputs,
                                           generation_node=generation_node,
                                           detectors=detectors,
                                           sampler=sampler,
                                           parallel_idx=parallel_idx,
                                           dag=dag)
        
    @property
    def executable(self):
        if self.inputs.use_mpi:
            return self._get_executable_path("mpiexec")
        elif self.inputs.analysis_executable:
            return self._get_executable_path(self.inputs.analysis_executable)
        else:
            return self._get_executable_path("nullpol_pipe_analysis")