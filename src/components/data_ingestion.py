import os, sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.utils import save_object
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join(os.getcwd(),'artifacts','raw_data.csv')
    train_path = os.path.join(os.getcwd(),'artifacts','train_data.csv')
    test_path = os.path.join(os.getcwd(),'artifacts','test_data.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logger.info('Data Ingestion started...')

        try:
            df = pd.read_excel(r"D:/Data Science/Job Assignments/SunBaseData/Data/customer_churn_large_dataset.xlsx")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            df = df.drop(['CustomerID', 'Name'], axis=1)

            # Splitting into train and test data
            train, test = train_test_split(df, test_size=0.33, random_state=42)

            logger.info('Data split into train test... and saving...')

            train.to_csv(self.data_ingestion_config.train_path, index=False, header=True)

            test.to_csv(self.data_ingestion_config.test_path, index=False, header=True)

            logger.info('Data Ingestion completed.')

            return self.data_ingestion_config.train_path, self.data_ingestion_config.test_path
        
        except Exception as e:
            logger.error('Error in Data Ingestion...')
            raise CustomException(e, sys)


if __name__ == '__main__':
    DI = DataIngestion()
    DI.initiate_ingestion()