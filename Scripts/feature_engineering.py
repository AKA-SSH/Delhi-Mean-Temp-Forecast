import sys
import pandas as pd

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class FeatureEngineering:
    def __init__(self) -> None:
        pass

    def engineer_feature(self, train_data_file_path: str, test_data_file_path: str):
        """
        Performs feature engineering on climate data.

        Parameters:
        - train_data_file_path (str): File path to the pickled training data file.
        - test_data_file_path (str): File path to the pickled testing data file.

        Raises:
        - CustomException: If an error occurs during the feature engineering process.

        Returns:
        None

        Example Usage:
        ```python
        feature_engineering = FeatureEngineering()
        feature_engineering.engineer_feature(train_data_file_path='artifacts\\train_data.pkl', test_data_file_path='artifacts\\test_data.pkl')
        ```
        """
        try:
            logging.info('Feature engineering initiated')

            logging.info('Loading data')
            train_data = unpickle_file(train_data_file_path)
            test_data = unpickle_file(test_data_file_path)
            logging.info('Data loaded')

            logging.info('Adding lagged temperature feature (meantemp(t-1)) to data')
            train_data['meantemp (t-1)'] = train_data.meantemp.shift(1)
            test_data['meantemp (t-1)'] = test_data.meantemp.shift(1)
            
            train_data.dropna(inplace=True)
            test_data.dropna(inplace=True)

            logging.info('Feature engineering completed')

            logging.info('Saving data')
            pickle_file(object=train_data, file_name='engineered_train_data.pkl')
            pickle_file(object=test_data, file_name='engineered_test_data.pkl')
            logging.info('Engineered data saved')

        except Exception as CE:
            logging.error(f'Error during feature engineering: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)
