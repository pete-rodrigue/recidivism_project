# recidivism_project
North Carolina recidivism data

This repository contains all the files used to generate the report on predicting recidivism in North Carolina. 

The files in the repository are as follows:
- aequitas_analysis.py: This script runs the aequitas analysis
- best_model.py: This script takes the best model (as determined in run_pipeline.py) and creates a feature importances, precision/recall curve, and a decision tree stump
- data_explore_library.py: This script contains functions for data exploration
- data_explore_script.py: This script creates plots/tables in data exploration
- ml_functions_library.py: This script contains the functions used in the ml pipeline, to create evaluation tables
- processing_library.py: This script processes the data and creates a dataframe to be fed into the ml pipeline
- run_pipeline.py: This script runs the ml pipeline, using a set of parameters that are looped over for various different models
