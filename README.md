# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the first project of Udacity's Machine Learning DevOps Engineer Nanodegree
The project objective is to produce production-ready clean code using best practices
The project itself aims at predicting customer churn for banking customers. This is a classification problem.
The project has the following approach:
- Load and explore dataset (EDA)
- Prepare data for training (feature engineering)
- Train two classification models (sklearn random forest and logistic regression)
- Identify most important features influencing the predictions and visualize their impact using SHAP library
- Save best models with their performance metrics

## Files and data description
Overview of the files and data present in the root directory.
The project is organized with the following directory architecture:
- Folders
    - Data      
        - eda       --> contains output of the data exploration
        - results   --> contains the dataset in csv format
    - images        --> contains model scores, confusion matrix, ROC curve
    - models        --> contains saved models in .pkl format
    - logs          --> log generated druing testing of library.py file

- project files 
    - churn_library.py
    - churn_notebook.ipnyb
    - requirements.txt

- pytest files (unit test file and configuration files)
    - test_churn_script_logging_and_tests.py  
    - pytest.ini    
    - conftest.py

    
## Running Files
- Execute the project under python 3.8 with the appropriate python packages
- Required libraries are provided in the requirements.txt file
- To run the project, execute churn_library.py script from the folder structure
    -`python churn_library.py`
- The project can also be executed using the jupyter notebook
- The project file was tested using pytest python package
    - To run the tests, type `pytest` from the main project folder in the command line
    - Project functions will be automatically tested with log file generated in the logs folder




