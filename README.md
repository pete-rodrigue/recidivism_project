# Predicting the Risk of Recidivism among people released from prison in North Carolina
## Spring 2019
### Vedika Ahuja, Pete Rodrigue, Bhargavi Ganesh

## Paper
Our final paper summarizing the results of our models and outlining the objective of the project is [here](https://github.com/vahuja92/recidivism_project/blob/master/Reducing%20Recidivism%20in%20North%20Carolina%20Report.pdf). 

## Data 
- (North Carolina Deapartment of Public Safety Offender Public Information database)[https://webapps.doc.state.nc.us/opi/downloads.do?method=view]

## Contents of Repository 

- aequitas_analysis.py: This script runs the aequitas analysis on bias in our pipeline
- best_model.py: This script takes the best model (as determined in run_pipeline.py) and creates a feature importances, precision/recall curve, and a decision tree stump
- data_explore_library.py: This script contains functions for data exploration
- data_explore_script.py: This script creates plots/tables in data exploration
- ml_functions_library.py: This script contains the functions used in the ml pipeline, to create evaluation tables
- processing_library.py: This script processes the data and creates a dataframe to be fed into the ml pipeline
- run_pipeline.py: This script runs the ml pipeline, using a set of parameters that are looped over for various different models
