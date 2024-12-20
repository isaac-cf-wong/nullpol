from bilby_pipe.job_creation.nodes.generation_node import GenerationNode as Node


class GenerationNode(Node):
    def __init__(self, inputs, trigger_time, idx, dag, parent=None):
        super(GenerationNode, self).__init__(inputs=inputs,
                                             trigger_time=trigger_time,
                                             idx=idx,
                                             dag=dag,
                                             parent=parent)
    
    @property
    def executable(self):
        return self._get_executable_path("nullpol_pipe_generation")