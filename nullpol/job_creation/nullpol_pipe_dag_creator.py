import copy
from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.job_creation.bilby_pipe_dag_creator import (get_trigger_time_list,
                                                            get_detectors_list,
                                                            get_parallel_list)
from bilby_pipe.job_creation.nodes import (MergeNode,
                                           FinalResultNode,
                                           PlotNode,
                                           PostProcessSingleResultsNode,
                                           PESummaryNode,
                                           PostProcessAllResultsNode)
from bilby_pipe.job_creation.overview import create_overview
from bilby_pipe.utils import get_colored_string
import bilby_pipe.utils
from .generation_node import GenerationNode
from .analysis_node import AnalysisNode
from ..utility import logger


bilby_pipe.utils.logger = logger

def get_detectors_list(inputs):
    detectors_list = []
    detectors_list.append(inputs.detectors)
    return detectors_list

def generate_dag(inputs):
    """Core logic setting up parent-child structure between nodes"""
    inputs = copy.deepcopy(inputs)
    dag = Dag(inputs)
    trigger_times = get_trigger_time_list(inputs)

    # Iterate over all generation nodes and store them in a list
    generation_node_list = []
    for idx, trigger_time in enumerate(trigger_times):
        kwargs = dict(trigger_time=trigger_time, idx=idx, dag=dag)
        if idx > 0:
            # Make all generation nodes depend on the 0th generation node
            # Ensures any cached files (e.g. the distance-marginalization
            # lookup table) are only built once.
            kwargs["parent"] = generation_node_list[0]
            # The start time is automatically set based on the trigger time
            # when working with real data
            # and so needs to be reset after each generation node is created
            if hasattr(inputs, "_start_time"):
                del inputs._start_time
        generation_node = GenerationNode(inputs, **kwargs)
        generation_node_list.append(generation_node)
    
    detectors_list = get_detectors_list(inputs)
    parallel_list = get_parallel_list(inputs)
    merged_node_list = []
    all_parallel_node_list = []
    for generation_node in generation_node_list:
        for detectors in detectors_list:
            parallel_node_list = []
            for parallel_idx in parallel_list:
                analysis_node = AnalysisNode(
                    inputs,
                    generation_node=generation_node,
                    detectors=detectors,
                    parallel_idx=parallel_idx,
                    dag=dag,
                    sampler=inputs.sampler,
                )
            parallel_node_list.append(analysis_node)
            all_parallel_node_list.append(analysis_node)

        if len(parallel_node_list) == 1:
            merged_node_list.append(analysis_node)
        else:
            merge_node = MergeNode(
                inputs=inputs,
                parallel_node_list=parallel_node_list,
                detectors=detectors,
                dag=dag,
            )
            merged_node_list.append(merge_node)

    plot_nodes_list = []
    for merged_node in merged_node_list:
        if inputs.final_result:
            FinalResultNode(inputs, merged_node, dag=dag)
        if inputs.plot_node_needed:
            plot_nodes_list.append(PlotNode(inputs, merged_node, dag=dag))
        if inputs.single_postprocessing_executable:
            PostProcessSingleResultsNode(inputs, merged_node, dag=dag)

    if inputs.create_summary:
        PESummaryNode(inputs, merged_node_list, generation_node_list, dag=dag)
    if inputs.postprocessing_executable is not None:
        PostProcessAllResultsNode(inputs, merged_node_list, dag)

    THRESHOLD = 21
    npar = len(all_parallel_node_list)
    if npar > THRESHOLD and inputs.osg is False:
        msg = (
            f"You are requesting {npar} analysis jobs, this work would be "
            "better suited to the IGWN-grid. See "
            "https://lscsoft.docs.ligo.org/bilby_pipe/master/osg.html "
            "for further information"
        )
        logger.warning(get_colored_string(msg))

    dag.build()
    create_overview(
        inputs,
        generation_node_list,
        all_parallel_node_list,
        merged_node_list,
        plot_nodes_list,
    )
