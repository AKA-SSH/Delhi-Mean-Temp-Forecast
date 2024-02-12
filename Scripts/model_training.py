import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class ModelTrainer:
    def __init__(self) -> None:
        pass

    def train_model(self, train_data_file_path: str, test_data_file_path: str):
        """
        Train a RandomForestRegressor model using the provided features and target data.

        Parameters:
        - train_data_file_path (str): File path to the pickled file containing the training data.
        - test_data_file_path (str): File path to the pickled file containing the testing data.

        Raises:
        - CustomException: If an error occurs during the model training process.
        """
        try:
            logging.info('Model training initiated')

            logging.info('Loading data')
            train_data = unpickle_file(file_name=train_data_file_path)
            test_data = unpickle_file(file_name=test_data_file_path)
            logging.info('Data loaded')

            logging.info('Splitting data into features and target')
            X_train, y_train = train_data.drop('meantemp', axis=1), train_data['meantemp']
            X_test, y_test = test_data.drop('meantemp', axis=1), test_data['meantemp']
            logging.info('Splitting completed')

            logging.info('Saving test data')
            pickle_file(object=X_test, file_name='test_features.pkl')
            pickle_file(object=y_test, file_name='test_target.pkl')
            logging.info('Test data saved')

            logging.info('Training RandomForestRegressor model')
            selected_parameters = {'n_estimators': 476,
                                   'max_depth': 7}
            RFR = RandomForestRegressor(**selected_parameters)
            RFR.fit(X_train, y_train)
            
            logging.info('Saving trained model')
            pickle_file(object=RFR, file_name='trained_model.pkl')
            logging.info('Model saved')

            logging.info('Model training completed')

        except Exception as e:
            logging.error(f'Error during model training: {str(e)}', exc_info=True)
            raise CustomException(e, sys)
