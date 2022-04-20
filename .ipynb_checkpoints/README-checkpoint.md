![Datarock](assets/datarock_logo_2.png)


# Geospatial ML Challenges: A prospectivity analysis example


The [Transform 2022 Schedule](https://docs.google.com/spreadsheets/d/e/2PACX-1vTnJ_cTd3Y5nQPoxM-BNTHq96SobJxTliofmqLxBMFnASpKTE9JxmPoqxEnFYPLUw2ZrIiQS8o_wunC/pubhtml)

Instructors:
[Thomas Ostersen](https://www.linkedin.com/in/thomasostersen/) and
[Tom Carmichael](https://www.linkedin.com/in/thomas-carmichael-b0761242/)


## BEFORE THE TUTORIAL

Make sure you've done these things **before the tutorial on Monday**:

1. Sign-up for the [Software Underground Slack](https://softwareunderground.org/slack)
1. Join the channel `t22-mon-ml-models`. This is where **all communication will
   happen**.
1. Set up your computer ([instructions below](#setup)). We will not have time to
   solve many computer issues during the tutorial so make sure you do this
   ahead of time. If you need any help, ask at the `t22-mon-ml-models` channel on
   Slack.

## About

In this tutorial we’ll run a fairly basic random forest prospectivity analysis
workflow applied to tin-tungsten (Sn-W) deposits in northeastern Tasmania. We'll 
use open data sets provided by Mineral Resources Tasmania and Geoscience Australia, 
all of which are available to download from our [Google Drive location](https://drive.google.com/file/d/1GOwI3vlmpiEhbFVIEoAPCrJkPdfIxPhD/view?usp=sharing). The roadmap for the tutorial is as follows:

  - Load and inspect data sets
      - mineral occurrence point data sets with *geopandas*
      - gravity, magnetic and radiometric data sets with *rasterio*
  - Combine data sets to build a labeled N<sub>pixel</sub>, N<sub>layers</sub> array for model training
      - inspect differences between proximal vs. distal to mineralisation pixels      
  - Train a random forest classifier and apply to all pixels, visualise results
      - evaluate performance with a randomly selected testing subset
      - repeat with stratified classes      
  - Develop a checkerboard data selection procedure, train and evaluate models
      - discuss effects of spatially separated testing data 
  - Investigate occurrence holdout models with a spatially clustered approach

## Prerequisites

- Knowledge of Python is assumed and all coding will be done within a Jupyter notebook
- We'll use [numpy](https://numpy.org/) for data handling and [matplotlib](https://matplotlib.org/) for data visualisation
- Point data sets are handled with [geopandas](https://geopandas.org/), a [pandas](https://pandas.pydata.org/)-like library for vector GIS processing
- [Rasterio](https://rasterio.readthedocs.io/) is used to read and write gridded raster data sets
- The [scikit-learn](https://scikit-learn.org/stable/) implementation of the [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) algorithm is used for all modelling

## Setup

There are a few things you'll need to follow the tutorial:

1. A working Python installation ([Anaconda](https://www.anaconda.com/) or Miniconda)
2. The Geospatial ML tutorial *conda environment* installed
3. A web browser that works with Jupyter notebooks (basically anything except Internet Explorer)

To get things setup, please do the following.

**Windows users:** When you see "*terminal*" in the instructions,
this means the "*Anaconda Prompt*" program for you.

### Step 1

**Install a Python distribution:**

In this tutorial we will be using the [Anaconda](https://www.anaconda.com/)
Python distribution along with the `conda` package manager. If you already have
Anaconda or Miniconda installed, you can skip this step.

If not, please follow Matt Hall's video tutorial from Transform2020: [youtube instructions](https://www.youtube.com/playlist?list=PLgLft9vxdduAW-jmhYqXvtfGYJS6v2FjM)

### Step 2

**Create the `t22-mon-ml-models` conda environment:**

1. Download the `environment.yml` file from
   [here](https://github.com/Solve-Geosolutions/transform_2022/environment.yml)
   (right-click and select "Save page as" or similar)
1. Make sure that the file is called `environment.yml`. Windows sometimes adds a
   `.txt` to the end, which you should remove.
1. Open a terminal (*Anaconda Prompt* if you are running Windows). The
   following steps should be done in the terminal.
1. Navigate to the folder that has the downloaded environment file
   (if you don't know how to do this, take a moment to read [the Software
   Carpentry lesson on the Unix shell](http://swcarpentry.github.io/shell-novice/)).
1. Create the conda environment by running `conda env create --file environment.yml`
   (this will download and install all of the packages used in the tutorial).

### Step 3

**Verify that the installation works:**

1. Download the `test_install.py` script from
   [here](https://raw.githubusercontent.com/fatiando/transform21/master/test_install.py)
1. Open a terminal. The following steps should be done in the terminal.
1. Activate the environment: `conda activate t21-thurs-harmonica`
1. Navigate to the folder where you downloaded `test_install.py`
1. Run the test script: `python test_install.py`
1. You should this text in the terminal (the last part of the second line will depend on your system):
   ```
   Harmonica version: 0.2.1
   Downloading file 'south-africa-gravity.ast.xz' from 'https://github.com/fatiando/harmonica/raw/v0.2.0/data/south-africa-gravity.ast.xz' to '/home/USER/.cache/harmonica/v0.2.0'.
   ```
1. The following figure should pop up:

[![Output of `test_python.py`.](https://raw.githubusercontent.com/fatiando/transform21/master/test_install_output.png)](https://raw.githubusercontent.com/fatiando/transform21/master/test_install_output.png)

If none of these commands gives an error, then your installation should be working.
If you get any errors or the outputs look significantly different,
please let us know on Slack at `#t21-thurs-harmonica`.

### Step 4

**Start JupyterLab:**

1. **Windows users:** Make sure you set a default browser that is **not Internet Explorer**.
1. Activate the conda environment: `conda activate t21-thurs-harmonica`
1. Start the JupyterLab server: `jupyter lab`
1. Jupyter should open in your default web browser. We'll start from here in the
   tutorial and create a new notebook together.

### IF EVERYTHING ELSE FAILS

If you really can't get things to work on your computer,
you can run the code online through Google Colab (you will need a Google account).
A starter notebook that installs Harmonica can be found here:

https://swu.ng/t21-harmonica-colab

To save a copy of the Colab notebook to your own account, click on the
"Open in playground mode" and then "Save to Drive".
You might be interested in
[this tutorial](https://transform2020.sched.com/event/c7Jn/tutorial-using-python-subsurface-tools-no-install-required)
for an overview of Google Colab.

#### I don't have a Google account

If you cannot use Google Colab, a second alternative option is to use to the
Software Underground JupyterHub.
You need to sign in with your Slack credentials on this website:
https://jupyter-dev.softwareunderground.org/

For more information about the login process, please read this:
https://github.com/softwareunderground/jupyterhub-deployment/tree/first-deployment#login-process

Once you are logged in, JupyterHub will ask you to choose a server
configuration, please choose the `t21-thurs-harmonica` option.
After JupyterHub sets up an instance for you, it will prompt a JupyterLab
interface.
In order to create a new notebook for running during the tutorial, please click
the `Python [conda env:t21-thurs-harmonica]` button in the Launcher.
It will create a new notebook running the `t21-thurs-harmonica` environment, so
you don't need to install any dependency, they are already installed! 🎉

> ⚠️ The Software Undeground JupyterHub instances are still in **experimental
> phase**. You may expect some unwanted behaviour or sudden crushes. Use it
> carefully and download the notebook every once in a while to have a backup.⚠️

Thanks [Filippo Broggini](https://www.filippobroggini.com/) for setting this up!

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png