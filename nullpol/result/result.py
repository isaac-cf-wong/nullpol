from bilby.core.result import Result
from bilby.core.utils import (
    infft, check_directory_exists_and_if_not_mkdir,
    latex_plot_format, safe_file_dump, safe_save_figure,
)
import json
import os
import pickle
import numpy as np
from ..utility import logger


class PolarizationResult(Result):
    def __init__(self, **kwargs):
        super(PolarizationResult, self).__init__(**kwargs)

    def __get_from_nested_meta_data(self, *keys):
        dictionary = self.meta_data
        try:
            item = None
            for k in keys:
                item = dictionary[k]
                dictionary = item
            return item
        except KeyError:
            raise AttributeError(
                "No information stored for {}".format('/'.join(keys)))

    @property
    def sampling_frequency(self):
        """ Sampling frequency in Hertz"""
        return self.__get_from_nested_meta_data(
            'likelihood', 'sampling_frequency')

    @property
    def duration(self):
        """ Duration in seconds """
        return self.__get_from_nested_meta_data(
            'likelihood', 'duration')

    @property
    def start_time(self):
        """ Start time in seconds """
        return self.__get_from_nested_meta_data(
            'likelihood', 'start_time')

    @property
    def interferometers(self):
        """ List of interferometer names """
        return [name for name in self.__get_from_nested_meta_data(
            'likelihood', 'interferometers')]

    def detector_injection_properties(self, detector):
        """ Returns a dictionary of the injection properties for each detector

        The injection properties include the parameters injected, and
        information about the signal to noise ratio (SNR) given the noise
        properties.

        Parameters
        ==========
        detector: str [H1, L1, V1]
            Detector name

        Returns
        =======
        injection_properties: dict
            A dictionary of the injection properties

        """
        try:
            return self.__get_from_nested_meta_data(
                'likelihood', 'interferometers', detector)
        except AttributeError:
            logger.info("No injection for detector {}".format(detector))
            return None

    def plot_skymap(
            self, maxpts=None, trials=5, jobs=1, enable_multiresolution=True,
            objid=None, instruments=None, geo=False, dpi=600,
            transparent=False, colorbar=False, contour=[50, 90],
            annotate=True, cmap='cylon', load_pickle=False):
        """ Generate a fits file and sky map from a result

        Code adapted from ligo.skymap.tool.ligo_skymap_from_samples and
        ligo.skymap.tool.plot_skymap. Note, the use of this additionally
        required the installation of ligo.skymap.

        Parameters
        ==========
        maxpts: int
            Maximum number of samples to use, if None all samples are used
        trials: int
            Number of trials at each clustering number
        jobs: int
            Number of multiple threads
        enable_multiresolution: bool
            Generate a multiresolution HEALPix map (default: True)
        objid: str
            Event ID to store in FITS header
        instruments: str
            Name of detectors
        geo: bool
            Plot in geographic coordinates (lat, lon) instead of RA, Dec
        dpi: int
            Resolution of figure in fots per inch
        transparent: bool
            Save image with transparent background
        colorbar: bool
            Show colorbar
        contour: list
            List of contour levels to use
        annotate: bool
            Annotate image with details
        cmap: str
            Name of the colormap to use
        load_pickle: bool, str
            If true, load the cached pickle file (default name), or the
            pickle-file give as a path.
        """
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        try:
            from astropy.time import Time
            from ligo.skymap import io, version, plot, postprocess, bayestar, kde
            import healpy as hp
        except ImportError as e:
            logger.info("Unable to generate skymap: error {}".format(e))
            return

        check_directory_exists_and_if_not_mkdir(self.outdir)

        logger.info('Reading samples for skymap')
        data = self.posterior

        if maxpts is not None and maxpts < len(data):
            logger.info('Taking random subsample of chain')
            data = data.sample(maxpts)

        default_obj_filename = os.path.join(self.outdir, '{}_skypost.obj'.format(self.label))

        if load_pickle is False:
            pts = data[['ra', 'dec']].values
            confidence_levels = kde.Clustered2DSkyKDE

            logger.info('Initialising skymap class')
            skypost = confidence_levels(pts, trials=trials, jobs=jobs)
            logger.info('Pickling skymap to {}'.format(default_obj_filename))
            safe_file_dump(skypost, default_obj_filename, "pickle")

        else:
            if isinstance(load_pickle, str):
                obj_filename = load_pickle
            else:
                obj_filename = default_obj_filename
            logger.info('Reading from pickle {}'.format(obj_filename))
            with open(obj_filename, 'rb') as file:
                skypost = pickle.load(file)
            skypost.jobs = jobs

        logger.info('Making skymap')
        hpmap = skypost.as_healpix()
        if not enable_multiresolution:
            hpmap = bayestar.rasterize(hpmap)

        hpmap.meta.update(io.fits.metadata_for_version_module(version))
        hpmap.meta['creator'] = "nullpol"
        hpmap.meta['origin'] = 'LIGO/Virgo'
        hpmap.meta['gps_creation_time'] = Time.now().gps
        hpmap.meta['history'] = ""
        if objid is not None:
            hpmap.meta['objid'] = objid
        if instruments:
            hpmap.meta['instruments'] = instruments

        try:
            time = data['geocent_time']
            hpmap.meta['gps_time'] = time.mean()
        except KeyError:
            logger.warning('Cannot determine the event time from geocent_time')

        fits_filename = os.path.join(self.outdir, "{}_skymap.fits".format(self.label))
        logger.info('Saving skymap fits-file to {}'.format(fits_filename))
        io.write_sky_map(fits_filename, hpmap, nest=True)

        skymap, metadata = io.fits.read_sky_map(fits_filename, nest=None)
        nside = hp.npix2nside(len(skymap))

        # Convert sky map from probability to probability per square degree.
        deg2perpix = hp.nside2pixarea(nside, degrees=True)
        probperdeg2 = skymap / deg2perpix

        if geo:
            obstime = Time(metadata['gps_time'], format='gps').utc.isot
            ax = plt.axes(projection='geo degrees mollweide', obstime=obstime)
        else:
            ax = plt.axes(projection='astro hours mollweide')
        ax.grid()

        # Plot sky map.
        vmax = probperdeg2.max()
        img = ax.imshow_hpx(
            (probperdeg2, 'ICRS'), nested=metadata['nest'], vmin=0., vmax=vmax,
            cmap=cmap)

        # Add colorbar.
        if colorbar:
            cb = plot.colorbar(img)
            cb.set_label(r'prob. per deg$^2$')

        if contour is not None:
            confidence_levels = 100 * postprocess.find_greedy_credible_levels(skymap)
            contours = ax.contour_hpx(
                (confidence_levels, 'ICRS'), nested=metadata['nest'],
                colors='k', linewidths=0.5, levels=contour)
            fmt = r'%g\%%' if rcParams['text.usetex'] else '%g%%'
            plt.clabel(contours, fmt=fmt, fontsize=6, inline=True)

        # Add continents.
        if geo:
            geojson_filename = os.path.join(
                os.path.dirname(plot.__file__), 'ne_simplified_coastline.json')
            with open(geojson_filename, 'r') as geojson_file:
                geoms = json.load(geojson_file)['geometries']
            verts = [coord for geom in geoms
                     for coord in zip(*geom['coordinates'])]
            plt.plot(*verts, color='0.5', linewidth=0.5,
                     transform=ax.get_transform('world'))

        # Add a white outline to all text to make it stand out from the background.
        plot.outline_text(ax)

        if annotate:
            text = []
            try:
                objid = metadata['objid']
            except KeyError:
                pass
            else:
                text.append('event ID: {}'.format(objid))
            if contour:
                pp = np.round(contour).astype(int)
                ii = np.round(np.searchsorted(np.sort(confidence_levels), contour) *
                              deg2perpix).astype(int)
                for i, p in zip(ii, pp):
                    text.append(
                        u'{:d}% area: {:d} deg$^2$'.format(p, i))
            ax.text(1, 1, '\n'.join(text), transform=ax.transAxes, ha='right')

        filename = os.path.join(self.outdir, "{}_skymap.png".format(self.label))
        logger.info("Generating 2D projected skymap to {}".format(filename))
        safe_save_figure(fig=plt.gcf(), filename=filename, dpi=dpi)

