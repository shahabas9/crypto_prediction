import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        logging.info("Exception occured at saving the object")
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(x_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(x_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys) 
    
def load_object(file_path):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        logging.info("exception occured at loading the object")
        raise CustomException(e,sys)