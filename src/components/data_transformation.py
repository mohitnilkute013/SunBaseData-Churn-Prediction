import os
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join(os.getcwd(), 'artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor(self):

        # Define which columns should be onehot-encoded and which should be scaled
        categorical_cols = ['Gender', 'Location']
        numerical_cols = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

        ## Numerical Pipeline
        num_pipeline=Pipeline(
            steps=[
            # ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())

            ]

        )

        # Categorigal Pipeline
        cat_pipeline=Pipeline(
            steps=[
            # ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehotencoder',OneHotEncoder(sparse=False)),
            ('scaler',StandardScaler(with_mean=False))
            ]

        )

        preprocessor=ColumnTransformer([
        ('num_pipeline',num_pipeline,numerical_cols),
        ('cat_pipeline',cat_pipeline,categorical_cols)
        ])

        return preprocessor


    def initiate_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info('Read train and test data completed')
            logger.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logger.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            # Target column
            target_column = 'Churn'
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            logger.info('Obtaining Preprocessing object...')

            preprocessor = self.get_preprocessor()
            # Preprocessing data
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            logger.info('Preprocessing completed with input training and testing dataset...')

            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            logger.info('Saving Preprocessor file...')

            save_object(
                filepath=self.transformation_config.preprocessor_path,
                obj=preprocessor
            )

            return train_arr, test_arr

        except Exception as e:
            logger.error('Error in Data Transformation.')
            raise CustomException(e, sys)


if __name__ == '__main__':

    di = DataIngestion()
    trainpath, test_path = di.initiate_ingestion()

    transformer = DataTransformation()
    print(transformer.initiate_transformation(trainpath, test_path))