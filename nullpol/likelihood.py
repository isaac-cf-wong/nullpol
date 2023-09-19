from bilby.core.likelihood import Likelihood


class NullStreamLikelihood(Likelihood):
    """A null stream likelihood object

    """
    def __init__(self, interferometers, projector_generator,
                 priors=None, calibration_lookup_table=None,
                 reference_frame="sky", time_reference="geocenter"):

    def __repr__(self):
        return None

    def log_likelihood(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        Pnull = self.projector_generator(self.parameters)
        z = 
