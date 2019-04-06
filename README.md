# Business Processes Predictions with Multi-Perspective A-Priori Knowledge

## Description
Continuation of [this](https://github.com/kaurjvpld/Incremental-Predictive-Monitoring-of-Business-Processes-with-A-priori-knowledge) project in which a predictive model is tasked to predict the next timesteps of an input business process log. We extended it in order to take in consideration also temporal aspects.

The input consists of categorical variables as well as real-valued timestamps, and the prediction consists of next time-step variables.

In this project, we leverage given Multi-Perspective A-priori Knowledge to improve the accuracy of the predictions.

### Predictive model
This contribution aims at improving the existing predictive model only, without improving/developing the existing prediction methods

### Inference algorithms
The project is divided into Control Flow (CF) prediction, Control Flow + Resource (CFR) prediction and Control Flow + Resource + Time (CFRT) prediction.

## Project structure

The ```src``` folder contains all the scripts used.

```experiments_runner.py``` is used to train and evaluate the predictive models.

The ```inference_algorithms``` folder contains all the inference algorithms available.

Results are saved into the ```output_files``` folder.

```shared_variables.py``` contains meta-variables used at inference-time.

```parse_results.py``` parses the results contained in ```output_files/final_experiments/results``` and plots multiple images/tables.

## Getting started
This project is intended to be self-contained, so no extra files are required.

### Prerequisites
To install the python environment for this project, refer to the [Pipenv setup guide](https://pipenv.readthedocs.io/en/latest/basics/)

### Running the algorithms
As the Java server is now integrated in the project, there is no need to start it separately.

The ```experiments_runner.py``` file contains all the code necessary to train the predictive models on each log file and evaluate them with each inference method.
