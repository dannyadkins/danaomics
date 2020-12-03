# danaomics: understanding mRNA degradation using Transformers
Team members: Dana Udwin, Naomi Lee, Danny Adkins

## 0. Background
Requires CometML to log experiments. Create a .comet.config file and populate it with your credentials, or contact me and I can share mine you privately.

## 1. Data
First, ensure you have the Kaggle CLI by following the steps listed at https://www.kaggle.com/docs/api.

Next, download the Stanford COVID vaccine dataset from within your project's base directory:

`kaggle competitions download -c stanford-covid-vaccine`

## 2. Training the model
The command line arguments are as follows:

`-T`: train

`-t`: test

`-s`: save model after run

`-l`: load presaved model

`python3 main.py -T` to train without saving

`python3 main.py -Ts` to train and save the model file



## 3. Evaluating the model

`python3 main.py -Tt` to train and test in the same run

`python3 main.py -lt` to test from a preloaded model file


## 4. Visualization
