import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score

from src.logger import logger
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV


def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path, exist_ok=True)

        # opening the file and storing the object in it
        with open(filepath, "wb") as file_obj:
            pickle.dump(obj, file_obj) #, protocol=pickle.HIGHEST_PROTOCOL

    except Exception as e:
        logger.error('Error in saving file object.')
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:
        report = {'Model_Name': [], "Model": [], "Acc_Score": [], "Precision_Score": [], "Recall_Score": [],
         "F1_Score": [], "ConfusionMatrix": []}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            logger.info(f'Training on {model_name}')

            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_pred = model.predict(X_test)

            # accuracy score
            test_score = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)

            logger.info(f'Training Complete... Accuracy_Score: {test_score}')

            report['Model_Name'].append(model_name)
            report['Model'].append(model)
            report['Acc_Score'].append(test_score*100)
            report['Precision_Score'].append(precision*100)
            report['Recall_Score'].append(recall*100)
            report['F1_Score'].append(f1*100)
            report["ConfusionMatrix"].append(cm)

        return report
    
    except Exception as e:
        logger.error('Error in Training')
        raise CustomException(e, sys)


def enhance_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {'Model_Name': [], "Model": [], "Acc_Score": [], "ConfusionMatrix": []}
        
        model = list(models.values())[0]
        model_name = list(models.keys())[0]

        logger.info(f'Enhancing {model_name}')

        grid_search=GridSearchCV(estimator=model,param_grid=params,cv=5, verbose=10)
        grid_search.fit(X_train,y_train)

        logger.info(f'Best Estimator: {grid_search.best_estimator_}')
        logger.info(f'Best Param: {grid_search.best_params_}')

        model.set_params(**grid_search.best_params_)

        # Train model
        model.fit(X_train, y_train)

        # Predict Testing data
        y_pred = model.predict(X_test)

        # r2 score
        test_score = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)

        logger.info(f'Training Complete... Accuracy_Score: {test_score}')

        report['Model_Name'].append(model_name)
        report['Model'].append(model)
        report['Acc_Score'].append(test_score*100)
        report["ConfusionMatrix"].append(cm)

        return report
    
    except Exception as e:
        logger.error('Error in Training')
        raise CustomException(e, sys)

def load_object(filepath):
    try:
        with open(filepath, "rb") as file_obj:
            model = pickle.load(file = file_obj)
            return model
    except Exception as e:
        logger.error('Unable to read or load the file_obj')
        raise CustomException(e, sys)