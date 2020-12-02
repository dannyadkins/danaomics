# danaomics: understanding mRNA degradation using Transformers
Team members: Dana Udwin, Naomi Lee, Danny Adkins

## 0. Background
## 1. Data
First, ensure you have the Kaggle CLI by following the steps listed at https://www.kaggle.com/docs/api.

Next, download the Stanford COVID vaccine dataset from within your project's base directory:

`kaggle competitions download -c stanford-covid-vaccine`

## 2. Training/testing the model
The command line arguments are as follows:

`-T`: train

`-t`: test

`-s`: save model after run

`-l`: load presaved model

To train:
`python3 main.py -T`

To test:
`python3 main.py -t`

## 3. Evaluating the model
## 4. Visualization
