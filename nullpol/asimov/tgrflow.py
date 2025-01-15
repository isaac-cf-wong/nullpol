# from bilby_tgr.asimov.tgrflow import Applicator as BilbyTGRApplicator
# from bilby_tgr.asimov.tgrflow import Collector as BilbyTGRCollector
import cbcflow
from asimov.event import Event
from asimov import config
import os
import glob
import shutil
from .utility import (fill_in_pol_specific_metadata,
                      bilby_config_to_asimov,
                      deep_update)
from ..utility import logger


class Collector:
    status_map = {
        "ready": "unstarted",
        "processing": "running",
        "running": "running",
        "stuck": "running",
        "restart": "running",
        "stopped": "cancelled",
        "finished": "running",
        "uploaded": "complete",
    }

    supported_pipelines = ["nullpol"]

    tgr_schema_sections = [
        "BHMAnalyses",
        "EchoesCWBAnalyses",
        "FTIAnalyses",
        "IMRCTAnalyses",
        "LOSAAnalyses",
        "MDRAnalyses",
        "ModeledEchoesAnalyses",
        "PCATGRAnalyses",
        "POLAnalyses",
        "PSEOBRDAnalyses",
        "PYRINGAnalyses",
        "QNMRationalFilterAnalyses",
        "ResidualsAnalyses",
        "SIMAnalyses",
        "SMAAnalyses",
        "SSBAnalyses",
        "TIGERAnalyses",
        "UnmodeledEchoesAnalyses"]

    def __init__(self, ledger):
        """
        Collect data from the asimov ledger and write it to a CBCFlow library.
        """
        hook_data = ledger.data["hooks"]["postmonitor"]["tgrflow"]
        self.library = cbcflow.core.database.LocalLibraryDatabase(
            hook_data["library location"]
        )
        self.library.git_pull_from_remote(automated=True)
        self.schema_section = "TestingGR"
        self.ledger = ledger

    def run(self):
        """
        Run the hook.
        """

        for event in self.ledger.get_event():
            # Do setup for the event
            output = {}
            output[self.schema_section] = {}
            metadata = cbcflow.get_superevent(
                event.meta["ligo"]["sname"], library=self.library
            )

            for analysis in event.productions:
                # logic for connecting subanalysis type to results already in cbcflow
                try:
                    analysis_schema_section = analysis.meta["tgr schema section"]
                except KeyError:
                    analysis_schema_section = None
                if analysis_schema_section in self.tgr_schema_sections:
                    metadata_tgr_analysis_subtypes = metadata["TestingGR"][analysis_schema_section]
                    metadata_tgr_analysis_subtypes_uids = cbcflow.core.utils.get_uids_from_object_array(
                        metadata_tgr_analysis_subtypes
                    )
                    analysis_subtype_uid = self._sort_analysis_by_subtype(analysis)
                    if analysis_subtype_uid in metadata_tgr_analysis_subtypes_uids:
                        corresponding_analysis_subtype = metadata_tgr_analysis_subtypes[
                            metadata_tgr_analysis_subtypes_uids.index(analysis_subtype_uid)
                        ]
                    else:
                        corresponding_analysis_subtype = None
                    output[self.schema_section][analysis_schema_section] = [self._fill_in_analysis_specific_metadata(
                        analysis, corresponding_analysis_subtype)]
                    output[self.schema_section][analysis_schema_section][0]["UID"] = analysis_subtype_uid

                    # if the pipeline is supported, grab result metadata
                    if str(analysis.pipeline).lower() in self.supported_pipelines:
                        # check if there is already a corresponding result in cbcflow
                        if corresponding_analysis_subtype is None:
                            corresponding_analysis = None
                        else:
                            metadata_pe_results = corresponding_analysis_subtype["Results"]
                            metadata_pe_results_uids = cbcflow.core.utils.get_uids_from_object_array(
                                metadata_pe_results
                            )
                            if analysis.name in metadata_pe_results_uids:
                                corresponding_analysis = metadata_pe_results[
                                    metadata_pe_results_uids.index(analysis.name)
                                ]
                            else:
                                corresponding_analysis = None
                        # analysis output is what goes into the Result part of the schema
                        analysis_output = self._get_pe_result_from_production(analysis, corresponding_analysis)
                        output[self.schema_section][analysis_schema_section][0]['Results'] = [analysis_output]
                        metadata.update(output)
                        # Note that Asimov *should* write to main, unlike most other processes
                        metadata.write_to_library(
                            message="Analysis run update by asimov", branch_name="main"
                        )

                    else:
                        logger.info(
                            f"Pipeline {analysis.pipeline} is not supported by tgrflow"
                        )
                        logger.info(
                            "If this is a mistake, please contact the cbcflow developers to add support."
                        )
                elif analysis_schema_section is None:
                    logger.info(
                        f"Production {analysis.name} has no information about the type of TGR analysis. Skipping."
                    )
                else:
                    logger.info(
                        f"The section {analysis_schema_section} not recognized as part of TestingGR cbcflow schema."
                    )
        self.library.git_push_to_remote()

    def _get_pe_result_from_production(self, analysis, corresponding_analysis):
        """
        Get the Result field for PE analysis in cbcflow form given asimov analysis.
        Most analyses copied it from cbc result, so it should be the same for most analyses.
        For bilby based pipelines, no changes apart from adding the pipeline name to if statement.

        input:
        analysis - asimov production for given event
        corresponding_analysis - the equivalent PE analysis as stored in cbcflow

        output:
        analysis_output - dictionary used to update cbcflow with new information
        """
        analysis_output = {}
        analysis_output["UID"] = analysis.name

        analysis_output["InferenceSoftware"] = str(analysis.pipeline)
        if analysis.status.lower() in self.status_map.keys():
            analysis_output["RunStatus"] = self.status_map[
                analysis.status.lower()
            ]
        if "waveform" in analysis.meta:
            if "approximant" in analysis.meta["waveform"]:
                analysis_output["WaveformApproximant"] = str(
                    analysis.meta["waveform"]["approximant"]
                )

        try:
            ini = analysis.pipeline.production.event.repository.find_prods(
                analysis.pipeline.production.name,
                analysis.pipeline.category,
            )[0]
            analysis_output["ConfigFile"] = {}
            analysis_output["ConfigFile"]["Path"] = ini
        except IndexError:
            logger.warning("Could not find ini file for this analysis")

        analysis_output["Notes"] = []

        if analysis.comment is not None:
            # We only want to add the comment to the notes if it doesn't already exist
            if corresponding_analysis is None:
                analysis_output["Notes"].append(analysis.comment)
            elif analysis.comment not in corresponding_analysis["Notes"]:
                analysis_output["Notes"].append(analysis.comment)

        if analysis.review.status:
            if analysis.review.status.lower() == "approved":
                analysis_output["ReviewStatus"] = "pass"
            elif analysis.review.status.lower() == "rejected":
                analysis_output["ReviewStatus"] = "fail"
            elif analysis.review.status.lower() == "deprecated":
                analysis_output["Deprecated"] = True
            messages = sorted(
                analysis.review.messages, key=lambda k: k.timestamp
            )
            if len(messages) > 0:
                if corresponding_analysis is None:
                    analysis_output["Notes"].append(
                        f"{messages[0].timestamp:%Y-%m-%d}: {messages[0].message}"
                    )
                elif (
                    f"{messages[0].timestamp:%Y-%m-%d}: {messages[0].message}"
                    in corresponding_analysis["Notes"]
                ):
                    analysis_output["Notes"].append(
                        f"{messages[0].timestamp:%Y-%m-%d}: {messages[0].message}"
                    )

        if analysis.finished:
            # Get the results
            results = analysis.pipeline.collect_assets()

            if str(analysis.pipeline).lower() in ["nullpol"]:
                # If it's bilby, we need to parse out which of possibly multiple merge results we want
                analysis_output["ResultFile"] = {}
                if len(results["samples"]) == 0:
                    logger.warning(
                        "Could not get samples from Bilby analysis, even though run is nominally finished!"
                    )
                elif len(results["samples"]) == 1:
                    # If there's only one easy enough
                    analysis_output["ResultFile"]["Path"] = results[
                        "samples"
                    ][0]
                else:
                    # If greater than one, we will try to prefer the hdf5 results
                    hdf_results = [
                        x
                        for x in results["samples"]
                        if "hdf5" in x or "h5" in x
                    ]
                    if len(hdf_results) == 0:
                        # If there aren't any, this implies we have more than one result,
                        # and they are all jsons
                        # Choose the 1st result
                        logger.warning(
                            "No hdf5 results were found, but more than one json result is present -\
                                    grabbing the first result."
                        )
                        analysis_output["ResultFile"]["Path"] = results["samples"][0]
                    elif len(hdf_results) == 1:
                        # If there's only one hdf5, then we can proceed smoothly
                        analysis_output["ResultFile"]["Path"] = hdf_results[0]
                    elif len(hdf_results) > 1:
                        # This is the same issue as described above, just with all hdf5s instead
                        logger.warning(
                            "Multiple merge_result hdf5s returned from Bilby analysis -\
                                    grabbing the first result."
                        )
                        analysis_output["ResultFile"]["Path"] = hdf_results[0]
                    if analysis_output["ResultFile"] == {}:
                        # Cleanup if we fail to get any results
                        analysis_output.pop("ResultFile")

        if analysis.status == "uploaded":
            # Next, try to get PESummary information
            pesummary_pages_dir = os.path.join(
                config.get("general", "webroot"),
                analysis.event.name,
                analysis.name,
                "pesummary",
            )
            sample_h5s = glob.glob(f"{pesummary_pages_dir}/samples/*.h5")
            if len(sample_h5s) == 1:
                # If there is only one samples h5, we're good!
                # This *should* be true, and it should normally be called "posterior_samples.h5"
                # But this may not be universal?
                analysis_output["PESummaryResultFile"] = {}
                analysis_output["PESummaryResultFile"]["Path"] = sample_h5s[
                    0
                ]
            else:
                logger.warning(
                    "Could not uniquely determine location of PESummary result samples"
                )
            if "public_html" in pesummary_pages_dir.split("/"):
                # Currently, we do the bit of trying to guess the URL ourselves
                # In the future there may be an asimov config value for this
                pesummary_pages_url_dir = (
                    cbcflow.core.utils.get_url_from_public_html_dir(
                        pesummary_pages_dir
                    )
                )
                # If we've written a result file, infer its url
                if "PESummaryResultFile" in analysis_output.keys():
                    # We want to get whatever the name we previously decided was
                    # This will only run if we did make that decision before, so we can use similar logic
                    analysis_output["PESummaryResultFile"][
                        "PublicHTML"
                    ] = f"{pesummary_pages_url_dir}/samples/{sample_h5s[0].split('/')[-1]}"
                # Infer the summary pages URL
                analysis_output[
                    "PESummaryPageURL"
                ] = f"{pesummary_pages_url_dir}/home.html"
        return analysis_output

    def _sort_analysis_by_subtype(self, analysis):
        """
        TestingGR schema differs from PE in that each entry is list of the analyses.
        It is to be associated with different kinds of test each analysis does (like coefficients in TIGER).
        This function looks at asimov ledger data to determine UID of the category each analysis falls into.

        input:
        analysis - asimov production for given event

        output:
        uid - string determining the analysis subcategory for given gr test
        """
        uid = 'default'
        return uid

    def _fill_in_analysis_specific_metadata(self, analysis, corresponding_analysis):
        """
        For given analysis subtype, fill in fields in metadata other than result list.

        input:
        analysis - asimov production for given event
        corresponding_analysis - equivalent subanalysis as stored in cbcflow

        output:
        analysis_output - dictionary used to update cbcflow with new information
        """

        if analysis.meta["tgr schema section"] == "POLAnalyses":
            return fill_in_pol_specific_metadata(analysis, corresponding_analysis)
        else:
            return dict()


"""Functionality for information which flows from cbcflow into Asimov"""


class Applicator:
    """Apply information from CBCFlow to an asimov event"""

    def __init__(self, ledger):
        hook_data = ledger.data["hooks"]["applicator"]["tgrflow"]
        self.ledger = ledger
        self.library = cbcflow.core.database.LocalLibraryDatabase(
            hook_data["library location"]
        )
        self.library.git_pull_from_remote(automated=True)
        # in case of disparity between data product in cbcflow and config,
        # choose which one to prefer (config by default)
        if 'data preference' in hook_data:
            if hook_data['data preference'] == 'cbcflow':
                self.prefer_config = False
            elif hook_data['data preference'] == 'config':
                self.prefer_config = True
            else:
                logger.warning("Unrecognized data preference applicator option. Setting to config")
                self.prefer_config = True
        else:
            self.prefer_config = True

    def run(self, sid=None):

        metadata = cbcflow.get_superevent(sid, library=self.library)
        detchar = metadata.data["DetectorCharacterization"]
        grace = metadata.data["GraceDB"]
        ifos = detchar["RecommendedDetectors"]
        participating_detectors = detchar["ParticipatingDetectors"]
        quality = {}
        max_f = quality["maximum frequency"] = {}
        min_f = quality["minimum frequency"] = {}

        # Data settings
        data = {}
        channels = data["channels"] = {}
        frame_types = data["frame types"] = {}
        # NOTE there are also detector specific quantities "RecommendedStart/EndTime"
        # but it is not clear how these should be reconciled with

        ifo_list = []
        frame_file_dict = {}
        for ifo in ifos:
            # Grab IFO specific quantities
            ifo_name = ifo["UID"]
            ifo_list.append(ifo_name)
            if "RecommendedDuration" in detchar.keys():
                data["segment length"] = int(detchar["RecommendedDuration"])
            if "RecommendedMaximumFrequency" in ifo.keys():
                max_f[ifo_name] = ifo["RecommendedMaximumFrequency"]
            if "RecommendedMinimumFrequency" in ifo.keys():
                min_f[ifo_name] = ifo["RecommendedMinimumFrequency"]
            if "RecommendedChannel" in ifo.keys():
                channels[ifo_name] = ifo["RecommendedChannel"]
            if "FrameType" in ifo.keys():
                frame_types[ifo_name] = ifo["FrameType"]
            if "FrameFile" in ifo.keys():
                frame_file_dict[ifo_name] = ifo["FrameFile"]

            if len(frame_file_dict) == 0:
                pass
            elif len(frame_file_dict) == len(ifo_list):
                data["data files"] = frame_file_dict
            else:
                logger.warning("It seems like all the detectors do not have frame files.")
                data["data files"] = frame_file_dict

        recommended_ifos_list = list()
        if ifo_list != []:
            recommended_ifos_list = ifo_list
        else:
            recommended_ifos_list = participating_detectors
            logger.info(
                "No detchar recommended IFOs provided, falling back to participating detectors"
            )

        # GraceDB Settings
        ligo = {}
        for event in grace["Events"]:
            if event["State"] == "preferred":
                ligo["preferred event"] = event["UID"]
                ligo["false alarm rate"] = event["FAR"]
                event_time = event["GPSTime"]
        ligo["sname"] = sid

        # also add the sampling rate to the output
        quality['sample rate'] = metadata.data["ParameterEstimation"]["SafeSamplingRate"]

        # delete?
        if "IllustrativeResult" in metadata.data["ParameterEstimation"]:
            ligo["illustrative result"] = metadata.data["ParameterEstimation"][
                "IllustrativeResult"
            ]

        output = {
            "name": metadata.data["Sname"],
            "quality": quality,
            "ligo": ligo,
            "data": data,
            "interferometers": recommended_ifos_list,
            "event time": event_time,
        }

        metadata_pe_results = metadata.data["ParameterEstimation"]["Results"]
        illustrative_prod = metadata.data["ParameterEstimation"]["IllustrativeResult"]
        gr_pe = {"available": False}
        for result in metadata_pe_results:
            if result["UID"] == illustrative_prod and result['RunStatus'] == 'complete':
                gr_pe = {"available": True,
                         "UID GR PE": result["UID"],
                         "result file path": "/" + result["ResultFile"]["Path"].split(':/')[-1],
                         "config file path": "/" + result["ConfigFile"]["Path"].split(':/')[-1],
                         "pesummary result path": "/" + result["PESummaryResultFile"]["Path"].split(':/')[-1]}
                if "WaveformApproximant" in result.keys():
                    quality["waveform approximant"] = result["WaveformApproximant"]
        if gr_pe['available'] is False:
            raise AttributeError(
                "IllustrativeResult not in the library or the PE run is incomplete."
            )

        # now, we need to read in the information from the config file
        # we need the reference frequency, the psd dict and the calibration dict
        config_file = gr_pe["config file path"]
        config = bilby_config_to_asimov(config_file)

        # read in the psd information if present
        if "psds" in config.keys():
            gr_pe["psds"] = config.pop('psds')
        output["gr pe info"] = gr_pe.copy()
        output["gr pe info"]['approximant'] = config['waveform']['approximant']
        # config file and cbcflow can disagree about which data to use (for example, preferred frame was updated)
        # logic below decides which version to use (should not matter when the analysis is finalized)
        if self.prefer_config:
            output['data'] = {}  # it will now be fully overridden by config
        else:
            # it will now keep values from cbcflow
            for item in ['channels', 'frame types', 'data files']:
                if item in config:
                    config.pop(item)

        output = deep_update(output, config)
        event = Event.from_dict(output)

        self.ledger.add_event(event)
        copied_data_folder = os.path.join(event.work_dir, 'PSDs')

        if not os.path.exists(copied_data_folder):
            os.makedirs(copied_data_folder)
        if 'psds' in gr_pe:
            output['psds'] = {}
            for key, value in gr_pe['psds'].items():
                filename = os.path.join(copied_data_folder, f'{key}_psd.dat')
                shutil.copy(value, filename)
                output['psds'][key] = filename
        event = Event.from_dict(output)
        self.ledger.update_event(event)