# FREE_7773
Repo containing material for the NYU class (Master of Engineering) I teach on NLP, ML Sys etc.

This repo is a WIP: check back often for updates, documentation, new slides etc.

Last update: Fall 2021.

## Prequisites: Dependencies

Different sub-projects may have different requirements, as specified in the 
_requirements.txt_ files to be found in the various folders. We recommend using
[virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to 
keep environments isolated, i.e. creating a new environment:

`python3 -m venv venv`

then activating it and installing the required dependencies:

`source venv/bin/activate`

`pip install -r requirements.txt`

## Repo Structure

TBC

### Data
TBC


### MLSys
TBC


### Notebooks

This folder contains Python notebooks that illustrate in Python concepts discussed during the lectures.
Please note that notebooks are inherently "exploratory" in nature, so they are good for interactivity and speed but they are not always the right tool for rigorous coding. 

Note: most of the dependencies are pretty standard, but some of the "exotic" ones are added with inline 
statements to make the notebook self-contained.

### Slides

The folder contains slides discussed during the course: while they provide a guide and a general overview of the concepts, the discussions we have during lectures are very important to put the material in the right context. While numbers in the files follow the order of the lectures as they happened in class, after the first intro lecture the NLP and MLSys "curricula" relatively independent.

Note that, with time, links and references may become obsolete despite my best intentions!

### Playground

This folder contains simple throw-away scripts useful to test specific tools, like for example logging
experiments in a remote dashboard, connecting to the cloud, etc. Script-specific info are below.

##### Comet playground

The file `comet_playground.py` is a simple adaptation of Comet onboarding script for sklearn: if run correctly,
the Comet dashboard should start displaying experiments under the chosen project name.
 
Make sure to set `COMET_API_KEY` and `MY_PROJECT_NAME` as env variables before running the script.

## Acknowledgments


## Additional materials
TBC


## Contacts

TBC
