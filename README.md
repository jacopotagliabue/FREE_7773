# FREE_7773
Repo containing material for the NYU class (Master of Engineering) I teach on NLP, ML Sys etc. For context on what the class is trying to achieve and, *especially* what is NOT, please refer
to the introductory slides in the relevant folder.

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

The repo is organized by folder: each folder contains either resources - e.g. text corpora or slides - or Python programs, divided by type. 

As far as ML is concerned, language-related topics are typically covered through notebooks, MLSys-related concepts are covered through Python scripts (not surprisingly!).

### Data

The folder contains some ready-made text files to experiment with some NLP techniques: these corpora are just examples, and everything can be pretty much run in the same fashion if you swap these files (and change the appropriate variables) with other textual data you like better.


### MLSys

This folder contains script covering MLSys concepts: how to organize a ML project, how to publish a model in the cloud etc.. In particular:

* _serverless_101_ contains a vanilla AWS Lambda endpoint computing explicitely the Y value of a regression model starting from an X input provided by the client. 
* _serverless_sagemaker_ contains an AWS Lambda endpoint which uses a Sagemaker internal endpoint to serve a scikit-model, previously trained (why two endpoints? Check the slides!).
* _training_: contains a sequence of scripts taking a program training a regression model and progressively refactoring to follow industry best-practices (i.e. using Metaflow!).

For more info on each of these topics, please see the slides and the sub-sections below.

#### Serverless 101

TBC

#### Serverless Sagemaker

TBC

#### Training scripts

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

For questions, feedback, comments, please drop me a message at: `jacopo dot tagliabue at nyu.edu`.
