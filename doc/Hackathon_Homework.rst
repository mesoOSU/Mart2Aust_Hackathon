======================
Overview
======================
This hackathon will involve several people writing code in parallel that needs
to cohesively fit together throughout the process. Additionally, it will ideally
become an addition to `Orix <https://orix.readthedocs.io/en/stable/>`_, which itself 
has over 20k downloads and is under active development. As such, it is in our best 
interest to all adhere to some common setup guidelines and coding practices, chief 
among these being the use of `git <https://git-scm.com/>`_ and 
`trunk-based version control <https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development>`_.
This means tracking issues through Github's `Project tool <https://docs.github.com/en/issues/trying-out-the-new-projects-experience/about-projects>`_,
writing well documented code, and adhering (at least mostly) to 
`PEP-8 style guidelines <https://peps.python.org/pep-0008/>`_. 

With all that in mind, this document is a step-by-step guide for setting up your 
workspace. If possible, please read through this and fully set up your workspace 
before the Hackathon begins so we can get the most out of our time in person. 
Also, if you are unfamiliar or have not worked in a git environment before, I 
encourage you to check out the links at the end of this document. 

======================
Software requirements
======================
The following is the minimum required setup for this hackathon:

1)	A `Github account <https://github.com/join>`_

2)	Python (preferably `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_) 

3)	Git (preferably with the `Github Desktop IDE <https://desktop.github.com/>`_)

4)	Matlab

======================
Setup
======================

Once the above software is installed, follow these steps to create a personal fork of this repository ready for editing.

***********************************************************************************
1)	Create a fork of Mart2Aust_Hackathon and download it with GIT
***********************************************************************************

This can be done however you like, as long as you end up with a local 
version of the repo on your computer that sees the 
MESOOSU/Mart2Aust_Hackathon repo as the upstream main. However, I made the
following GIF to show how I would encourage you to do this using Github Desktop

|How_to_gif|

.. |How_to_gif| image:: https://raw.githubusercontent.com/mesoOSU/Mart2Aust_Hackathon/master/doc/_static/img/github_setup.gif
   :width: 1250 


***********************************************************************************
2)	Add orix form your repo (NOT THE OFFICIAL ONE) to your Path
***********************************************************************************

Again, as long as you can do “from orix.crystal_map import CrystalMap” and
it imports the CrystalMap class from the orix repo, you are good. However,
I would recommend setting up a new conda environment in Anaconda and doing
a local editable pip install. This way, you can be sure you are using the
same python packages and versions as everyone else at the hackathon, and
if this code breaks something, you can nuke the environment and start from
scratch. For this method, start by opening Anaconda Prompt and creating and
activating a new environment::

	conda create --name hackathon python=3.9.12
	conda activate hackathon

Add the Spyder and Jupyter IDEs to this conda environment::

	conda install Jupyter
	conda install Spyder

then cd to the repository and do an editable pip installation::

	cd /Directory/where/Mart2Aust_Hackathon/is
	pip install –-editable .

At this point, you should be able to open up Spyder (or whatever python 
IDE you use) and import orix from anywhere.

***********************************************************************************
Notes on code linting
***********************************************************************************

Any code you commit should be readily understandable by anyone else at 
the hackathon without you having to verbally explain it. Additionally, 
orix uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to 
auto-generate User manuals from function DocStrings, so well written ones
will save you time and energy in the long run. 

Adhering to the `PEP-8 Style Guide <https://peps.python.org/pep-0008/>`_ solves both these problems. Guidelines 
can be found here ,but most Modern IDEs have some form of 'linting' built in 
that automatically provides warnings when code is not pep-8 compliant.
`Here is a link to turning on this feature in Spyder <https://stackoverflow.com/questions/59250636/how-to-turn-on-off-code-analysis-features-in-spyder-4>`_

***********************************************************************************
Notes on Git workflows
***********************************************************************************

While we will generally try to follow `trunk-based version control`, If you prefer gitflow or
some other form of git management, feel free to commit code in that way instead. 

Incomplete or requested features will be listed under the github `issues page <https://github.com/mesoOSU/Mart2Aust_Hackathon/issues>`_
If you think of a missing feature, create a new issue for it and try to verbosely describe
what is needed, what it's input and outputs should look like, etc. This will function as a 
"To-Do" List for everyone at the Hackathon.

When you write a piece of code that solves an issue, make a "pull request" and assign someone 
to review you code (ideally NOT soneone who helped write it). creating a pull request 
should auto-populate the `projects page <https://github.com/mesoOSU/Mart2Aust_Hackathon/projects/5>`_, and not only is useful for organization, but 
ensures your name (or at least your github profile) is permenantly attached to the code you
contributed

Regardless though, NO CODE SHOULD BE COMMITTED TO MAIN WITHOUT REVIEW BY SOMEONE ELSE. This
is annoying, but does significantly improve code quality and reliability.

Every new feature related to a pull request should have the following parts.

1) The actual code, properly linted
2) A DocString describing its purpose, inputs, and outputs.
3) a few Unittests for testing basic functionality.

Realistically, all these should be added for a given feature before moving onto a new  one.
However, if you have some code that is useful/important and want to add it before finishing
the polish parts (not reccomended, but sometimes necessary), just make a pull request, have
the code approved, and make a new Issue for "create unittests for *blah*"

***********************************************************************************
Creating Demos
***********************************************************************************

Sometimes, you will write a routine that really needs example code to show it off. when that
happens, create a `Jupyter notebook <https://jupyter.org/>`_ for it, and save it in the docs
folder. Every notbook in here is read by Sphinx and used to create the "Tutorial" page for 
orix (compare `this notebook <https://github.com/pyxem/orix/blob/v0.8.2/doc/crystal_map.ipynb>`_ to 
`this webpage <https://orix.readthedocs.io/en/stable/crystal_map.html>`_)
