from bilby_tgr.asimov.tgrflow import Applicator as BilbyTGRApplicator
from bilby_tgr.asimov.tgrflow import Collector as BilbyTGRCollector
from .utility import def fill_in_pol_specific_metadata


class Collector(BilbyTGRCollector):
    def __init__(self, ledger):
        super(Collector, self).__init__(ledger=ledger)


class Applicator(BilbyTGRApplicator):
    def __init__(self, ledger):
        super(Applicator, self).__init__(ledger=ledger)

    def _fill_in_analysis_specific_metadata(self, analysis, corresponding_analysis):
        """
        For given analysis subtype, fill in fields in metadata other than result list.

        Parameters
        ==========
        analysis: asimov production for given event
            Corresponding_analysis - equivalent subanalysis as stored in cbcflow.

        Returns
        =======
        analysis_output: dict
            Dictionary used to update cbcflow with new information.
        """
        if analysis.meta["tgr schema section"] == "PolAnalyses":
            return fill_in_pol_specific_metadata(analysis, corresponding_analysis)
        else:
            return dict()
