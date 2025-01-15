def fill_in_pol_specific_metadata(analysis, corresponding_analysis):
    """
    For pol analysis, fill in fields in metadata other than result list.

    Parameters
    ==========
    analysis: Asimov production for given event
        corresponding_analysis - equivalent subanalysis as stored in cbcflow

    Returns
    =======
    analysis_output: dict
        Dictionary used to update cbcflow with new information.
    """
    analysis_output = {}
    analysis_output["AnalysisSoftware"] = str(analysis.pipeline)
    analysis_output["Description"] = f"Polarization analyses"
    return analysis_output