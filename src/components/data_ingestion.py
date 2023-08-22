import os
import sys
from src.logger import logging
from src.exception import CustomException

import pymongo
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

# Initialize data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

# Creating Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method started')

        try:
            logging.info('Connecting to mongodb for getting the data')
            client = pymongo.MongoClient("mongodb+srv://gaikwadujg:rUns6cK8ABSmUpxs@cluster0.7chcxpg.mongodb.net/?retryWrites=true&w=majority")
            db = client['Project']
            coll = db['Cancer']
            data_dct = coll.find_one()
            X = data_dct['data']
            Y = data_dct['target']
            X = pd.DataFrame(X,columns=data_dct['feature_names'])
            Y = pd.DataFrame(Y,columns=['target'])
            df = X.join(Y)
            logging.info('Data Imported from mongodb and read as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Raw data saved in artifacts folder')

            train_set, test_set = train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            logging.info('Train Data saved in artifacts folder')

            test_set.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info('Test Data saved in artifacts folder')
            logging.info('Data ingestion completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception Occured at Data Ingestion Stage')
            raise CustomException(e,sys)