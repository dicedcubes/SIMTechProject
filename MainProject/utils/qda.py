"""
Contains all functions necessary for building and using a QDA model.
preprocess_qda  Using the preprocessing tool in fns, preprocesses the data according to the needs of QDA.
fit_qda         Creates a QDA object from the SKLearn library and fits the data to it.
test_qda        Scores the model on the cached test data and labels and returns the model accuracy.
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from . import fns

def preprocess_qda(cache, kwargs={}):
    kwargs["standardize"] = True
    kwargs["labeled"] = True
    kwargs["shuffle"] = True

    training_data, test_data, training_labels, test_labels = fns.preprocess_data(cache['raw_data'], **kwargs)

    cache["training_data"] = training_data
    cache["test_data"] = test_data
    cache["training_labels"] = training_labels
    cache["test_labels"] = test_labels
    return 0

def fit_qda(cache):
    # Create an QDA model.
    qda = QuadraticDiscriminantAnalysis()

    # Fit the cached data to the model.
    try:
        qda.fit(cache["training_data"], cache["training_labels"])
    except:
        return -1
    
    # Add the model to the cache.
    cache["qda"] = qda
    return 0

def test_qda(cache):
    # Test the model.
    return fns.validate(cache['qda'], cache['test_data'], cache['test_labels'])