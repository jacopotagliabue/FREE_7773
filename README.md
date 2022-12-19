# FREE_7773
Repo containing material for the NYU class (Master of Engineering) I teach on NLP, ML Sys etc. For context on what the class is trying to achieve and, *especially* what is NOT, please refer to the slides in the relevant folder.

Last update: December 2021.

IMPORTANT: the 2022 edition is now [available](https://github.com/jacopotagliabue/MLSys-NYU-2022), with some significant changes. Check it out!

Notes:

* for unforseen issues with user permissions in the AWS Academy, the original serverless deployment we explained for MLSys could not be used. While the code is still in this repo for someone who wants to try with their own account, a local Flask app serving a model is provided as an alternative in the _project_ folder.

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

For more info on each of these topics, please see the slides and the sub-sections below; make sure you run [Metaflow tutorial](https://metaflow.org/) first if you are not familiar with Metaflow.

#### Training scripts

Progression of scripts training the same regression model on synthetica dataset in increasingly better programs, starting from a monolithic implementation and ending with a functionally equivalent DAG-based implementation. In particular: 

* you can run `create_fake_dataset.py` to generate a X,Y dataset, `regression_dataset`;
* `monolith.py` performs all operation in a long function;
* `composable.py` breaks up the monolith in smaller functions, one per core functionality, so that now `composable_script` acts as a high-level routine explicitely displaying the logical flow of the program;
* `small_flow.py` re-factores the functional components of `composable.py` into steps for a Metaflow DAG, which can be run with the usual MF syntax `python small_flow.py run`. Please note that imports of non-standard packages now happen at the relevant steps: since MF decouples code from computation, we want to make sure all steps are as self-contained as possible, dependency-wise.
* `small_flow_sagemaker.py` is the same as `small_flow.py`, but with an additional step, `deploy_model_to_sagemaker`, showing how the learned model can be first stored to S3, then used to spin up a Sagemaker endpoint, that is an internal AWS endpoint hosting automatically for us the model we just created. Serving this model is more complex than what happens in _Serverless 101_ (see below), so a second Serverless folder hosts the Sagemaker-compatible version of AWS lambda.

#### Serverless 101

The folder is a self-contained AWS Lambda that can use regression parameters learned with any of the training scripts to serve predictions from the cloud:

* `handler.py` contains the business logic, inside the `simple_regression` function. After converting a query parameter into a new _x_, we calculate _y_ using the regression equation, reading the relevant parameters from the environment (see below). 
* `serverless.yml` is a standard Serverless configuration file, which defines the GET endpoint we are asking AWS to create and run for us, and use `environment` variables to store the beta and intercept learned from training a regression model.

To deploy succeessfully, make sure to have [installed Serverless](https://www.serverless.com/framework/docs/providers/aws/guide/installation), configured with your AWS credentials. Then:

* run `small_flow.py` in the _training_ folder to obtain values for `BETA` and `INTERCEPT` (or whatever linear regression you may want to run on your dataset);
* change `BETA` and `INTERCEPT` in `serverless.yml` with the values just learned;
* `cd` into the folder and run: `serverless deploy --aws-profile myProfile`
* when deployment / update is completed, the terminal will show the cloud url where our model can be reached.

#### Serverless Sagemaker

The folder is a self-contained AWS Lambda that can use a model hosted on Sagemaker, such as the one deployed with `small_flow_sagemaker.py`, to serve prediction from the cloud. Compared to _Serverless 101_, the `handler.py` file here is not using environment variables and an explicit equation, but it is simply "passing over" the input received by the client to the internal Sagemaker endpoint hosting the model (`get_response_from_sagemaker`). 

Also in this case you need Serverless [installed and configured](https://www.serverless.com/framework/docs/providers/aws/guide/installation) to be able to deploy the lambda as a cloud endpoint: once `small_flow_sagemaker.py` is run and the Sagemaker endpoint is live, deploying the lambda itself is done with the usual commands.

Note: Sagemaker endpoints are pretty expensive - if you are not using credits, make sure to delete the endpoint when you are done with your experiments.

### Notebooks

This folder contains Python notebooks that illustrate in Python concepts discussed during the lectures. Please note that notebooks are inherently "exploratory" in nature, so they are good for interactivity and speed but they are not always the right tool for rigorous coding. 

Note: most of the dependencies are pretty standard, but some of the "exotic" ones are added with inline statements to make the notebook self-contained.

### Project

This folder contains two main files:

* `my_flow.py` is a Metaflow version of the text classification pipeline we explained in class: while not necessarily exhaustive, it contains many of the features that the final course project should display (e.g. comments, qualitative tests, etc.). The flow ends by explictely storing the artifacts from the model we just trained.
* `my_app.py` shows how to build a minimal Flask app serving predictions from the trained model. Note that the app relies on a small HTML page, while our lecture described an endpoint as a purely machine-to-machine communication (that is, outputting a JSON): both are fine for the final project, as long as you understand what the app is doing.

You can run both (`my_flow.py` first) by creating a separate environment with the provided `requirements.txt` (make sure your Metaflow setup is correct, of course).

### Slides

The folder contains slides discussed during the course: while they provide a guide and a general overview of the concepts, the discussions we have during lectures are very important to put the material in the right context After the first intro part, the NLP and MLSys "curricula" relatively independent. Note that, with time, links and references may become obsolete despite my best intentions!

### Playground

This folder contains simple throw-away scripts useful to test specific tools, like for example logging
experiments in a remote dashboard, connecting to the cloud, etc. Script-specific info are below.

##### Comet playground

The file `comet_playground.py` is a simple adaptation of Comet onboarding script for sklearn: if run correctly, the Comet dashboard should start displaying experiments under the chosen project name.
 
Make sure to set `COMET_API_KEY` and `MY_PROJECT_NAME` as env variables before running the script.

## Acknowledgments

Thanks to all outstanding people quoted and linked in the slides: this course is possible only because we truly stand on the shoulders of giants. Thanks also to:

* Meninder Purewal, for being such a great, patient, witty co-teacher;
* Patrick John Chia, for debugging sci-kit on Sagemaker and building the related flow;
* Ciro Greco, for helping with the NLP slides and greatly improving the scholarly references;
* Federico Bianchi and Tal Linzen, for sharing their wisdom in teaching NLP.

## Additional materials

The two main topics - MLSys and NLP - are huge, and we could obviously just scratch the surface. Since it is impossible to provide extensive references here, I just picked 3 great items to start:

* [Deep Learning with Python](https://www.amazon.com/Learning-Python-Second-Fran%C3%A7ois-Chollet/dp/1617296864): great practical intro to ML concepts and the basics of DL;
* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/): a great, modern book on NLP; 
* [You Don't Need a Bigger Boat](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat): our own fully open source repository showing how state-of-the-art ML systems can be built at scale, component after component.

## Contacts

For questions, feedback, comments, please drop me a message at: `jacopo dot tagliabue at nyu.edu`.
