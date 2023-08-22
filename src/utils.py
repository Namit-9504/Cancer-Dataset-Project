import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

def save_object(file_path,obj):
    try:
        logging.info(f'Saving the object to : {file_path}')
        
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        
        logging.info('Object successfully saved')

    except Exception as e:
        logging.info('Exception occured while saving object in utils.py file')
        raise CustomException(e, sys)
    
def evaluate_model(xtrain, ytrain, xtest, ytest, models):
    try:
        report = dict()
        tr = []
        ts = []
        tr_cv = []
        for name, model in models.items():
            # Fit the model
            model.fit(xtrain,ytrain)
            # Predict train and test data
            ypred_tr = model.predict(xtrain)
            ypred_ts = model.predict(xtest)
            # Calculating f1 score in training and testing
            tr_f1 = f1_score(ytrain,ypred_tr)
            ts_f1 = f1_score(ytest,ypred_ts)
            f1_cv = cross_val_score(model,xtrain,ytrain,cv=5,scoring='f1')
            report[name]=ts_f1
            # Appending in list format
            tr.append(tr_f1)
            ts.append(ts_f1)
            tr_cv.append(f1_cv.mean())
        # Saving the Evalutation into dataframe
        eval_dct = {'model':list(models.keys()),
                    'training_f1':tr,
                    'testing_f1':ts,
                    'training_cv5':tr_cv}
        
        eval_df = pd.DataFrame(eval_dct)
        eval_df = eval_df.sort_values(by='testing_f1',ascending=False)
        return (report,eval_df)
    
    except Exception as e:
        logging.info('Exception occured in evaluate model utils.py')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        logging.info(f'Loading File Object from : {file_path}')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in Load Object from utils.py')
        raise CustomException(e,sys)