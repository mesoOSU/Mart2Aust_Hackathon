======================
Resources
======================
NOTE TO ALL: PLEASE update this list with things you think might be helpful. 

This is a general, non-exhaustive list of useful links for information related to 
the hackathon. Below is a list of the major subject areas.

1) Main Publications
2) basic Material Science Knowledge
3) Texture specific information (ie, orientations, rotations, etc)
4) ODF Specific Information ()
5) Git, Github, workflows
6) graph cut theory
7) Baysean Statistics

********************************************************************************
Main Publications and Repos
********************************************************************************
* `Link to the OneDrive Folder of relevant publications <https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/gerlt_1_buckeyemail_osu_edu/EjbSA8hPg0VEj9OuMN3q2o0BZglPDzfV1-iPnFl9X9k4jQ?e=rQgg6n>`_


Major Publications by the original Mart2Aust and AusRecon authors

* `Using Bayesian Inference to determing the Parent-Child Orientation Relationship <https://buckeyemailosu-my.sharepoint.com/:b:/r/personal/gerlt_1_buckeyemail_osu_edu/Documents/MESO_OSU/Hackathon/Publications/Brust2020_Article_CharacterizationOfMartensiteOr.pdf?csf=1&web=1&e=BZDSBt>`_
* `Analysis of Misorientation Relationships between AUstenite Parents and Twins <https://buckeyemailosu-my.sharepoint.com/:b:/r/personal/gerlt_1_buckeyemail_osu_edu/Documents/MESO_OSU/Hackathon/Publications/Brust2019_Article_AnalysisOfMisorientationRelati.pdf?csf=1&web=1&e=6FMaM2>`_
* `Graph cut applied to Material Sciene Images <https://buckeyemailosu-my.sharepoint.com/:b:/r/personal/gerlt_1_buckeyemail_osu_edu/Documents/MESO_OSU/Hackathon/Publications/2019_application-of-the-maximum-flowminimum-cut-algorithm-to-segmentation-and-clustering-of-materials-datasets.pdf?csf=1&web=1&e=Rc5J8Z>`_
* `Probabilistic Reconstruction of Austenite Microstructure from EBSD <https://www.cambridge.org/core/journals/microscopy-and-microanalysis/article/probabilistic-reconstruction-of-austenite-microstructure-from-electron-backscatter-diffraction-observations-of-martensite/6083D1C4A478BCBEC681E2B5F0C9004C>`_

********************************************************************************
basic Material Science Knowledge
********************************************************************************
If you are new to Material Science, This is probably the Most efficient textbook
for quicly learning the material you need for this hackathon:

* `Structure of Materials <https://buckeyemailosu-my.sharepoint.com/:b:/r/personal/gerlt_1_buckeyemail_osu_edu/Documents/MESO_OSU/Hackathon/Publications/Structure+of+Materials,+AnIntroduction+to+Crystallography,+D....pdf?csf=1&web=1&e=3Tf1Us>`_

Chapters 1-3 cover the basics of Material Science, 4 & 5 introduce crystallography,
skip 6, 7 and 8 will be helpful if you need to deal with orientation transformations.

* `General overview of EBSD <http://pajarito.materials.cmu.edu/lectures/L18b-EBSD-analysis-15Apr14.pdf>`_

********************************************************************************
Texture specific information (ie, orientations, rotations, etc)
********************************************************************************
This is the goto Texture textbook, and covers orientations/rotations/misorientations
fairly well in the first 3 Chapters

* `Texture and Anisotropy <https://buckeyemailosu-my.sharepoint.com/:b:/r/personal/gerlt_1_buckeyemail_osu_edu/Documents/MESO_OSU/Hackathon/Publications/TextureAnisotropy-Kocks-Tome-Wenk-2nded.pdf?csf=1&web=1&e=2O4WTP>`_

However, for a more bite-sized digestible form organized by subject, I would 
strongly encourage the reader to check out Dr Anthony Rollett's online notes for
his 27-750 class, `which can be found here <http://pajarito.materials.cmu.edu/27750/27750.html>`_

* `3b1b video on quaterions <https://www.youtube.com/watch?v=d4EgbgTm0Bg>`_
* `Tony rollett lecture on orientations <http://pajarito.materials.cmu.edu/lectures/L20-Descriptions-of-Orientation-TugceOzturk_27750-12Apr16.pdf>`_
* `Orix <https://orix.readthedocs.io/en/stable/>`_

********************************************************************************
ODF Specific Information
********************************************************************************
Links to `NFSOFT <https://www-user.tu-chemnitz.de/~potts/nfft/nfsoft.php>`_ and 
`FFTW <http://www.fftw.org/>`_ the two major C libraries for doing spherical harmonic
ODFs the way they are done in MTEX. As a note, I believe NFSOFT should actually be
titled SOFT, and it is neither Nonuniform or FFT, and all the functions done by 
these libraries are also available in scipy, numpy, and `lie_learn <https://github.com/AMLab-Amsterdam/lie_learn/blob/master/lie_learn/spectral/SO3FFT_Naive.py>`_

* `Original Bunge Textbook on usinng spherical harmonics for an ODF <https://buckeyemailosu-my.sharepoint.com/:b:/r/personal/gerlt_1_buckeyemail_osu_edu/Documents/MESO_OSU/Hackathon/Publications/Bunge_TextureAnalysis.pdf?csf=1&web=1&e=LepioC>`_
* `Steve Niezgoda Book chapter on ODFs which is much shorter and more succinct <https://buckeyemailosu-my.sharepoint.com/:b:/r/personal/gerlt_1_buckeyemail_osu_edu/Documents/MESO_OSU/Hackathon/Publications/Estimating%20Crystallographic%20Texture%20and%20Orientation%20Statistics.pdf?csf=1&web=1&e=ZdlfJc>`_
* `Tony Rollet's lecture on the topic <http://pajarito.materials.cmu.edu/lectures/L5-Orient_Dist-27Jan14.pdf>`_
* `3b1b video on Fourer Transforms <https://www.youtube.com/watch?v=spUNpyF58BY>`_


********************************************************************************
Git, Github, workflows
********************************************************************************

* `Basic intro to git, what it is, and why version control matters <https://www.atlassian.com/git/tutorials/what-is-version-control>`_

The collaborating tab on the left is a good place to start if you are familiar 
with the basic concepts of git but have never used it in a large project.

* `Using a GitHub Project Board <https://docs.github.com/en/issues/organizing-your-work-with-project-boards/managing-project-boards/about-project-boards>`_
* `Trunk based git workflow <https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development>`_
* `Github Project tool <https://docs.github.com/en/issues/trying-out-the-new-projects-experience/about-projects>`_

********************************************************************************
Graph Cut Theory
********************************************************************************

* `A Repo I made using doing graph cut in python on HEXRD data <https://github.com/argerlt/maxflow_for_matsci>`_

it makes use of another module called maxflow, which which I tested it, was 
SIGNIFICANTLY faster than NetworkX. There are details on why this is included in
the repo.

* `PyMaxFlow Documentation, the better alternative to NetworkX <http://pmneila.github.io/PyMaxflow/maxflow.html>`_
* `Original Boykov min_cut_max_flow paper <https://www.csd.uwo.ca/~yboykov/Papers/ijcv06.pdf>`_
* `Boykov's Chapter on the subject, which I think is much more useful than the paper <https://cs.uwaterloo.ca/~yboykov/Papers/chapter_04.pdf>`_

********************************************************************************
Baysean Statistics
********************************************************************************

Talk to Steve


********************************************************************************
Misc
********************************************************************************
* `PEP-8 style guidelines <https://peps.python.org/pep-0008/>`_
* `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_
* `Github Desktop IDE <https://desktop.github.com/>`_
* `Sphinx <https://www.sphinx-doc.org/en/master/>`_ 
* `Jupyter notebook <https://jupyter.org/>`_
