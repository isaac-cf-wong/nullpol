from __future__ import annotations

import json
import os
import pickle

import numpy as np
from bilby.core.result import Result
from bilby.core.utils import check_directory_exists_and_if_not_mkdir, safe_file_dump, safe_save_figure

from ..utils import logger


class PolarizationResult(Result):
    """Result class for gravitational wave polarization analysis.

    Extends bilby's Result class to handle results from polarization-specific
    parameter estimation, including access to likelihood metadata and
    polarization-aware plotting and post-processing capabilities.

    Args:
        **kwargs: Keyword arguments passed to parent Result class,
            including samples, log_evidence, priors, meta_data, etc.

    Attributes:
        Inherits all attributes from bilby.core.result.Result plus
        polarization-specific metadata access methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __get_from_nested_meta_data(self, *keys):
        """Retrieve value from nested meta_data dictionary structure.

        Navigates through a nested dictionary hierarchy using the provided
        keys to access deeply nested metadata from the likelihood object.

        Args:
            *keys: Variable number of dictionary keys forming the path
                to the desired nested value.

        Returns:
            Any: The value found at the specified nested location.

        Raises:
            AttributeError: If any key in the path is not found in the
                meta_data structure.
        """
        dictionary = self.meta_data
        try:
            item = None
            for k in keys:
                item = dictionary[k]
                dictionary = item
            return item
        except KeyError as exc:
            raise AttributeError(f"No information stored for {'/'.join(keys)}") from exc

    @property
    def sampling_frequency(self):
        """Sampling frequency in Hertz"""
        return self.__get_from_nested_meta_data("likelihood", "sampling_frequency")

    @property
    def duration(self):
        """Duration in seconds"""
        return self.__get_from_nested_meta_data("likelihood", "duration")

    @property
    def start_time(self):
        """Start time in seconds"""
        return self.__get_from_nested_meta_data("likelihood", "start_time")

    @property
    def interferometers(self):
        """List of interferometer names"""
        return list(self.__get_from_nested_meta_data("likelihood", "interferometers"))

    def detector_injection_properties(self, detector):
        """Returns a dictionary of the injection properties for each detector

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
            return self.__get_from_nested_meta_data("likelihood", "interferometers", detector)
        except AttributeError:
            logger.info(f"No injection for detector {detector}")
            return None

    def plot_skymap(
        self,
        maxpts=None,
        trials=5,
        jobs=1,
        enable_multiresolution=True,
        objid=None,
        instruments=None,
        geo=False,
        dpi=600,
        transparent=False,
        colorbar=False,
        contour=None,
        annotate=True,
        cmap="cylon",
        load_pickle=False,
    ):
        """Generate a fits file and sky map from a result

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
            Resolution of figure in dots per inch
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
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib import rcParams  # pylint: disable=import-outside-toplevel

        try:
            import healpy as hp  # pylint: disable=import-outside-toplevel
            from astropy.time import Time  # pylint: disable=import-outside-toplevel
            # pylint: disable=import-outside-toplevel  # Optional dependency for sky localization
            from ligo.skymap import (
                bayestar,
                io,
                kde,
                plot,
                postprocess,
                version,
            )  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            logger.info(f"Unable to generate skymap: error {e}")
            return

        check_directory_exists_and_if_not_mkdir(self.outdir)

        # Set default contour levels if None provided
        if contour is None:
            contour = [50, 90]

        logger.info("Reading samples for skymap")
        data = self.posterior

        if maxpts is not None and maxpts < len(data):
            logger.info("Taking random subsample of chain")
            data = data.sample(maxpts)

        default_obj_filename = os.path.join(self.outdir, f"{self.label}_skypost.obj")

        if load_pickle is False:
            pts = data[["ra", "dec"]].values
            confidence_levels = kde.Clustered2DSkyKDE

            logger.info("Initializing skymap class")
            skypost = confidence_levels(pts, trials=trials, jobs=jobs)
            logger.info(f"Pickling skymap to {default_obj_filename}")
            safe_file_dump(skypost, default_obj_filename, "pickle")

        else:
            if isinstance(load_pickle, str):
                obj_filename = load_pickle
            else:
                obj_filename = default_obj_filename
            logger.info(f"Reading from pickle {obj_filename}")
            with open(obj_filename, "rb") as file:
                skypost = pickle.load(file)
            skypost.jobs = jobs

        logger.info("Making skymap")
        hpmap = skypost.as_healpix()
        if not enable_multiresolution:
            hpmap = bayestar.rasterize(hpmap)

        hpmap.meta.update(io.fits.metadata_for_version_module(version))
        hpmap.meta["creator"] = "nullpol"
        hpmap.meta["origin"] = "LIGO/Virgo"
        hpmap.meta["gps_creation_time"] = Time.now().gps
        hpmap.meta["history"] = ""
        if objid is not None:
            hpmap.meta["objid"] = objid
        if instruments:
            hpmap.meta["instruments"] = instruments

        try:
            time = data["geocent_time"]
            hpmap.meta["gps_time"] = time.mean()
        except KeyError:
            logger.warning("Cannot determine the event time from geocent_time")

        fits_filename = os.path.join(self.outdir, f"{self.label}_skymap.fits")
        logger.info(f"Saving skymap fits-file to {fits_filename}")
        io.write_sky_map(fits_filename, hpmap, nest=True)

        skymap, metadata = io.fits.read_sky_map(fits_filename, nest=None)
        nside = hp.npix2nside(len(skymap))

        # Convert sky map from probability to probability per square degree.
        deg2perpix = hp.nside2pixarea(nside, degrees=True)
        probperdeg2 = skymap / deg2perpix

        if geo:
            obstime = Time(metadata["gps_time"], format="gps").utc.isot
            ax = plt.axes(projection="geo degrees mollweide", obstime=obstime)
        else:
            ax = plt.axes(projection="astro hours mollweide")
        ax.grid()

        # Plot sky map.
        vmax = probperdeg2.max()
        img = ax.imshow_hpx((probperdeg2, "ICRS"), nested=metadata["nest"], vmin=0.0, vmax=vmax, cmap=cmap)

        # Add colorbar.
        if colorbar:
            cb = plot.colorbar(img)
            cb.set_label(r"prob. per deg$^2$")

        if contour is not None:
            confidence_levels = 100 * postprocess.find_greedy_credible_levels(skymap)
            contours = ax.contour_hpx(
                (confidence_levels, "ICRS"), nested=metadata["nest"], colors="k", linewidths=0.5, levels=contour
            )
            fmt = r"%g\%%" if rcParams["text.usetex"] else "%g%%"
            plt.clabel(contours, fmt=fmt, fontsize=6, inline=True)

        # Add continents.
        if geo:
            geojson_filename = os.path.join(os.path.dirname(plot.__file__), "ne_simplified_coastline.json")
            with open(geojson_filename) as geojson_file:
                geoms = json.load(geojson_file)["geometries"]
            verts = [coord for geom in geoms for coord in zip(*geom["coordinates"])]
            plt.plot(*verts, color="0.5", linewidth=0.5, transform=ax.get_transform("world"))

        # Add a white outline to all text to make it stand out from the background.
        plot.outline_text(ax)

        if annotate:
            text = []
            try:
                objid = metadata["objid"]
            except KeyError:
                pass
            else:
                text.append(f"event ID: {objid}")
            if contour:
                pp = np.round(contour).astype(int)
                ii = np.round(np.searchsorted(np.sort(confidence_levels), contour) * deg2perpix).astype(int)
                for i, p in zip(ii, pp):
                    text.append(f"{p:d}% area: {i:d} deg$^2$")
            ax.text(1, 1, "\n".join(text), transform=ax.transAxes, ha="right")

        filename = os.path.join(self.outdir, f"{self.label}_skymap.png")
        logger.info(f"Generating 2D projected skymap to {filename}")
        safe_save_figure(fig=plt.gcf(), filename=filename, dpi=dpi, transparent=transparent)
