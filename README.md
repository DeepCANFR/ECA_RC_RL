## Introduction

This repository is a framework for exploring the usage of elementary cellular automata rules as a reservoir computer on reinforcement learning tasks. It was developed for [this](_blank) master thesis in collaboration with OsloMet Living Technology Lab. 

## Setup

You can use [Anaconda](https://www.anaconda.com) to install the dependencies. It uses python 3.9, pytorch for training, gymnasium for the environment, pygame for visualizations, pandas and seaborn. All of which can be installed by running the following commands in an anaconda terminal:

```bash
conda create -n env-name python=3.9
conda activate env-name
conda install pytorch
conda install gymnasium

pip install pygame==2.0.0.dev14
pip install pandas
pip install seaborn
```

## Usage

Change parameters in **config.py** and run **main.py**. This will train a model of every unique ECA rule and store them in the /models folder with a run\_name. Change BASE\_PATH and MODEL\_NAME in **testModel.py** to see the performance of a single model. Set SHOW\_RESERVOIR to True to see how the reservoir evolvs over time and press *P* to save an image. Run **testRun.py** to evaluate every model of a run, and see the results under the /runs folder. Plot the training data of a run in the **plot.ipynb** file by running the three functions: *load_data([run name])*, *to_df()* and *plot_returns()*.

![Rule 60](screenshots/rule60.png)