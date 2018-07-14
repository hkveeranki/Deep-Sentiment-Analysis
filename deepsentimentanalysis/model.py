import gc
import sys

from sklearn.metrics import precision_recall_fscore_support

from .test_results import TestResults


class Model(object):
    """
    Abstract class for all the models
    """

    def __init__(self, data_dimensions, save_path=None, batch_size=32, num_epochs=2,
                 prerequisite_save_path=None):
        """ Default Constructor

        Initialises all fields of the class with the given values

        :param save_path: if model needs to be saved give the location
        :param data_dimensions: Size of each sample in the data
        :param batch_size batch size during training
        :param num_epochs number of epochs to use for training
        :param prerequisite_save_path path to save the prerequisites

        """

        # A keras model object which can be trained
        self.model = None
        # Size of each sample in the data
        self.data_dim = data_dimensions
        # location where model needs to be saved
        self.save_path = save_path
        # Number of epochs to train
        self.epochs = num_epochs
        # Batch size to be used while training
        self._batch_size = batch_size
        # variable to hold the data whether model has been trained or not
        self._model_trained = False
        # Variable to save models required as prerequisites
        self.prerequisite_save_path = prerequisite_save_path

    def train(self, train_df):
        """ Train the model and save it to a location if requested

        This method implements the functionality of training the given model. The model
        object is assumed to be a keras model and it is trained accordingly. If a
        save_path is provided then the weights get saved at the path for the further use.

        :param train_df: data frame containing training data

       """
        x_train, y_train = self._augment_data(train_df)
        del train_df
        gc.collect()
        if not self.model:
            sys.stderr.write('Generate a model first and then call train()\n')
            sys.exit(-1)
        print(self.model.summary())
        print 'model training started'
        self.model.fit(x_train, y_train, verbose=True, epochs=self.epochs,
                       batch_size=self._batch_size)
        if self.save_path:
            self.model.save_weights(self.save_path)
        self._model_trained = True
        print 'model trained'

    def test(self, test_df, saved_weights=None):
        """ Test the trained model and return the test results

        This method implements functionality of predicting labels of the test data in the
        model and evaluate its performance. If saved_weights is provided, then weights
        loaded from the given path will be used to predict labels

        :param test_df: data frame containing testing data
        :param saved_weights: path to the saved model if the model has already been
                           trained before

        :return: TestResults object containing accuracy,f1,precision and recall scores

        """
        x_test, y_test = self._augment_data(test_df)
        del test_df
        gc.collect()
        if saved_weights:
            self.model.load_weights(saved_weights)
        elif not self._model_trained:
            sys.stderr.write('Train the model first or supply a saved weights '
                             'data\n')
            return None
        predictions = self.model.predict(x_test)
        predictions = [round(x[0]) for x in predictions]
        precision, rec, f1, _ = precision_recall_fscore_support(y_test, predictions,
                                                                average='binary')
        test_results = TestResults(precision, rec, f1)
        return test_results

    def _augment_data(self, data_df):
        """ Augment the given data as per the requirements of the model and return it

        :param data_df raw input data
        :return: augmented data

        """
        raise NotImplementedError()

    def generate_prerequisites(self, train_df, test_df):
        """Generate the prerequisite tools required for the model and save them

        This method generates models that are required as pre-requisites to the 
        corresponding model. If a path to save the model is given then it saves the model
        at the corresponding path

        :param train_df: a pandas data frame object containing the training data
        :param test_df: a pandas data frame object containing the testing data

        """
        raise NotImplementedError()

    def _save_model_weights(self):
        """Save the model to required location
        """
        self.model.save_weights(self.save_path)

    def _load_model_weights(self, saved_weights=None):
        """
        Load model_weights from the required path
        """
        weights_to_load = saved_weights or self.save_path
        self.model.load_weights(weights_to_load)

    def __generate_model(self):
        """
        Prepare the appropriate model according to data dimensions
        """
        raise NotImplementedError()
