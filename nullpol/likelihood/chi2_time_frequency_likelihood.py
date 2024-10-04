from .time_frequency_likelihood import TimeFrequencyLikelihood

class Chi2TimeFrequencyLikelihood(TimeFrequencyLikelihood):
    def __init__(self, interferometers, waveform_generator=None, projector_generator=None,
                 priors=None, time_frequency_analysis_arguments={},
                 time_frequency_filter=None,
                 reference_frame="sky", time_reference="geocenter", *args, **kwargs):
        super().__init__(interferometers=interferometers,
                         waveform_generator=waveform_generator,
                         projector_generator=projector_generator,
                         priors=priors,
                         time_frequency_analysis_arguments=time_frequency_analysis_arguments,
                         time_frequency_filter=time_frequency_filter,
                         reference_frame=reference_frame,
                         time_reference=time_reference,
                         *args,
                         **kwargs)

    def log_likelihood(self):
        pass