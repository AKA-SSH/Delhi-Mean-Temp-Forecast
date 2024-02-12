import sys
import pandas as pd

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file

from Scripts.feature_engineering import FeatureEngineering
from Scripts.model_training import ModelTrainer

class DataIngestion:
    def __init__(self) -> None:
        pass

    def data_ingestion(self, train_data_file_path='Data\\DailyDelhiClimateTrain.csv', test_data_file_path='Data\\DailyDelhiClimateTest.csv'):
        """
        Performs the data ingestion process for climate data.

        Parameters:
        - train_data_file_path (str): The file path to the training data in CSV format.
        - test_data_file_path (str): The file path to the testing data in CSV format.

        Raises:
        - CustomException: If an error occurs during the data ingestion process.

        Example Usage:
        ```python
        data_ingestion_object = DataIngestion()
        data_ingestion_object.data_ingestion(train_data_file_path='Data\\train.csv', test_data_file_path='Data\\test.csv')
        ```
        """
        try:
            logging.info('Data ingestion initiated')

            logging.info(f'Loading training data from {train_data_file_path}')
            train_data = pd.read_csv(train_data_file_path)

            logging.info(f'Loading testing data from {test_data_file_path}')
            test_data = pd.read_csv(test_data_file_path)

            logging.info('Setting date as index for both training and testing data')
            train_data.set_index('date', inplace=True)
            test_data.set_index('date', inplace=True)

            logging.info('Rounding numerical values to three decimal places')
            train_data = round(train_data, 3)
            test_data = round(test_data, 3)

            logging.info('Data ingestion completed')

            logging.info('Saving train and test data')
            pickle_file(object=train_data, file_name='train_data.pkl')
            pickle_file(object=test_data, file_name='test_data.pkl')
            logging.info('Data saved successfully')

        except Exception as CE:
            logging.error(f'Error during data ingestion: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)

if __name__ == '__main__':
    # Data ingestion
    logging.info('Performing data ingestion...')
    data_ingestion_object = DataIngestion()
    data_ingestion_object.data_ingestion()

    # Feature Engineering
    logging.info('Performing feature engineering...')
    feature_engineering_object = FeatureEngineering()
    feature_engineering_object.engineer_feature(train_data_file_path='artifacts\\train_data.pkl', test_data_file_path='artifacts\\test_data.pkl')

    # Model Training
    logging.info('Performing model training...')
    training_object = ModelTrainer()
    training_object.train_model(train_data_file_path='artifacts\\engineered_train_data.pkl', test_data_file_path='artifacts\engineered_test_data.pkl')
    