================
Open Datasets
================

Orix includes several small data sets intended specifically for testing and tutorial 
purposes within the  `:mod:orix.data` module. 

Additionaly, this file contains a list of openly available datasets hosted on Zenodo.
datasets can be downloaded using the following code:
.. code-block::

    import os, copy, zipfile, glob, pooch
    from orix.data import _fetcher


    def download_from_Zenodo(zenodo_DOI, filename, md5=None):
        """Downloads requested zenodo datasets into the local orix cache if not
        previously downlaoded, and unzips sets of data if necessary."""
        cache_path = os.sep.join([str(_fetcher.path), filename])
        url = "https://zenodo.org/record/{}/files/{}".format(zenodo_DOI, filename)
        # add local path and url to deep copy of orix's default pooch fetcher
        zenodo_fetcher = copy.deepcopy(_fetcher)
        zenodo_fetcher.urls[filename] = url
        zenodo_fetcher.registry[filename] = md5
        # Download if downloadable
        download = pooch.HTTPDownloader(progressbar=True)
        path = zenodo_fetcher.fetch(filename, downloader=download)
        if filename[-4:] == ".zip":
           zipfile.ZipFile(cache_path, 'r').extractall(zenodo_fetcher.path)
            return glob.glob(cache_path[:-4] + os.sep+"*")
        else:
            return path

Note that these works are not part of Orix themselves, and the original sources
should be properly acknowledged in any derivative work.

Users wishing to add their own datasets to this list are encouraged to open 
a related issue on the `orix github page <https://github.com/pyxem/orix/issues>`_. Please 
include the Zenodo DOI, copyright information, and preferred citation if available. 


AF96 Martensitic Steel
========================


a collection of five 2116x1048 pixel EBSD scans of AF96, originally released as 
part of the following Data in Brief:

    `Datasets acquired with correlative microscopy method for delineation of prior austenite grain boundaries and characterization of prior austenite grain size in a low-alloy high-performance steel <https://doi.org/10.1016/j.dib.2019.104471>`_

This data is under a Creative Commons licence, CC BY 4.0. Therefore, any work
using these datasets must credit the original authors, preferably by citing 
the paper listed above. Further details on the preperation of the samples 
can be found in the following publication:

    `Correlative microscopy for quantification of prior austenite grain size in AF9628 steel <https://doi.org/10.1016/j.matchar.2019.109835>`_

Copies of these datasets, as well as 40 smaller 512x512 scans taken from the larger
ones, can be found on `Zenodo <zenoodo.link>`_. These can be automatically downloaded
and converted to orix CrystalMaps as follows:
.. code-block::

    big_map_paths = download_from_Zenodo(7430395, "AF96_Large.zip", "md5:60c6eefd316e2747c721cd334ed3abaf")
    small_map_paths = download_from_Zenodo(7430395, "AF96_Small.zip", "md5:01890e210bcbc18516c571585452ed26 ")

    # load a single small map
    small_xmap = io.load(small_scan_paths[0])
    # load as single large map
    large_xmap = io.load(large_scan_paths[0])
    # load a list of all 5 large maps (can take several minutes)
    large_xmaps = [io.load(path) for path in large_scan_paths]

Include Picture?

Inconel100 3D-EBSD
========================


A collection of 117 EBSD scans, each 189x189 pixels in size. Scans were taken
from successive layers of a serially sectioned piece of Inconell 100, and used for
validation purposes for Blue Quartz's open source software package `Dream3D <http://dream3d.bluequartz.net/>`_.
This dataset was first reported on in the following publication:

    `3D reconstruction and characterization of polycrystalline microstructures using a FIB–SEM system <https://doi.org/10.1016/j.matchar.2006.01.019>`_

Additionally, Dream3D contains several tutorials for visualizing and processing this
dataset `found here <http://www.dream3d.io/2_Tutorials/EBSDReconstruction/>`_.

a copy of this dataset can be found on `Zenodo <I havent made a link yet>`_. These can be 
automatically downloaded and converted to orix CrystalMaps as follows:
.. code-block::

    in100_scan_paths = download_from_Zenodo(12345678910,'Small_IN100.zip',hash)

    # load a single map
    in100_xmap = io.load(in100_scan_paths[0])
    # load a list of all 117 maps(can take several minutes)
    in100_xmaps = [io.load(path) for path in in100_scan_paths]

Picture?
