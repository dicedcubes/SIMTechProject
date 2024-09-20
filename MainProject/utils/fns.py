"""
This file contains generic functions useful for models.

Functions
---------
preprocess_data Preprocesses the data stored in the cache, with other specifications passed in according to the model that called this.
validate        Has the model score itself on the test data and labels in the cache, and returns the model accuracy.
"""

# Imports
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(data, **kwargs):
    """
    Preprocess and return the data that is passed in. Standardize, shuffle, partition, one-hot encoding.

    Parameters
    ----------
    data            Data to be preprocessed. Should be a NumPy array.
    kwargs:
        one_hot     Boolean indicating if one-hot encoding should be performed on the data.
        standardize Boolean indicating if data should be standardized.
        shuffle     Boolean indicating if data should be shuffled.
        test_size   Ratio of data points to be partitioned for testing. Remainder set for training.
        labeled     Boolean indicating if labels are included in data and should thus be separated.

    Returns
    -------
    Processed data, according to specifications.
    """

    # Perform one-hot encoding on the data.
    if 'one_hot' in kwargs.keys() and kwargs['one_hot']:
        pass # To be implemented.

    # Mark if data should be shuffled.
    if 'shuffle' in kwargs.keys():
        shuffle = kwargs['shuffle']
    else:
        shuffle = False
    
    # Mark the ratio of test data to training data.
    if 'test_size' in kwargs.keys():
        test_size = kwargs['test_size']
    else:
        test_size = 0.2

    # Partition the data.
    train, test = train_test_split(data, test_size=test_size, shuffle=shuffle)

    if 'labeled' in kwargs.keys() and kwargs['labeled']:
        training_data = train[:, :-1]
        training_labels = train[:, -1]
        test_data = test[:, :-1]
        test_labels = test[:, -1]
    else:
        training_data = train
        training_labels = None
        test_data = test
        test_labels = None
    
    # Standardize the data.
    if 'standardize' in kwargs.keys() and kwargs['standardize']:
        scaler = StandardScaler()
        training_data = scaler.fit_transform(training_data)
        test_data = scaler.fit_transform(test_data)

    return training_data, test_data, training_labels, test_labels

def validate(model, test_data, test_labels):
    """
    Given a model, test data, and test labels, uses the model to predict labels for the test data and returns the accuracy.

    Parameters
    ----------
    model       Model for testing to be run on.
    test_data   Unlabeled test data with features matching the data the model was trained on.
    test_labels Corresponding labels for the test data.

    Returns
    -------
    Model accuracy.
    """
    
    return model.score(test_data, test_labels)