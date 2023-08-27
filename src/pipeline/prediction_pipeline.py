import sys
import os
from src.exception import CustomException
from src.logger import logger
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):

        preprocessor_path = os.path.join(os.getcwd(), 'artifacts', 'preprocessor.pkl')
        model_path=os.path.join('artifacts','model.pkl')

        self.preprocessor=load_object(preprocessor_path)
        self.model=load_object(model_path)

    def predict(self,features):
        try:
            data_scaled=self.preprocessor.transform(features)

            pred=self.model.predict(data_scaled)
            return pred
            
        except Exception as e:
            logger.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Age:int,
                 Gender:str,
                 Location:str,
                 Subscription_Length_Months:int,
                 Monthly_Bill:float,
                 Total_Usage_GB:int):
        
        self.Age=Age
        self.Gender=Gender
        self.Location=Location
        self.Subscription_Length_Months = Subscription_Length_Months
        self.Monthly_Bill = Monthly_Bill
        self.Total_Usage_GB=Total_Usage_GB
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'Gender':[self.Gender],
                'Location':[self.Location],
                'Subscription_Length_Months':[self.Subscription_Length_Months],
                'Monthly_Bill':[self.Monthly_Bill],
                'Total_Usage_GB':[self.Total_Usage_GB]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logger.info('Dataframe Gathered')
            return df
        except Exception as e:
            logger.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)


if __name__ == "__main__":
    #logger.info('Prediction Pipeline Started')
    cd = CustomData(Age=25,
    Gender='Male',
    Location='New York',
    Subscription_Length_Months=94,
    Monthly_Bill=94,
    Total_Usage_GB=10)

    cd_df = cd.get_data_as_dataframe()
    print(cd_df)

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(cd_df)

    results = round(pred[0], 2)

    print(results)