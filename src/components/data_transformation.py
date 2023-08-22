import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass

# Sklearn Pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:    
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')

            num_pipe = Pipeline(steps=[('imputer',SimpleImputer(strategy='mean')),
                                       ('scaler',StandardScaler())])
            
            return num_pipe
        
        except Exception as e:
            logging.info('Exception occured in Data Transformation Pipeline')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Reading train and test data complete')
            logging.info(f'Train Dataframe Head : \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n {test_df.head().to_string()}')

            logging.info('Obtaining Preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column = 'target'
            drop_column = [target_column]

            # Seperating Dependent and independent features
            # Train Data
            input_feature_train_df = train_df.drop(labels=drop_column,axis=1)
            target_feature_train_df = train_df[target_column]
            # Test Data
            input_feature_test_df = test_df.drop(labels=drop_column,axis=1)
            target_feature_test_df = test_df[target_column]

            # Perform data transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Preprocessing Train and Test arrays done')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Data Transformation Complete')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Transformation stage')
            raise CustomException(e,sys)
    