from langchain_core.tools import tool
import pandas as pd
from . import lda, qda, pca

import streamlit as st
from PIL import Image

CACHE = {}

def get_cache():
    return CACHE

@tool
def load_data(filepath: str):
    """ 
    Loads data from the given filepath, and stores it in a variable named "loaded_data".
    After this tool is run, the CACHE will now contain an entry with key "loaded_data".
    Returns 0 on success.
    """
    match filepath[-3:]:
        case "csv":
            try:
                CACHE["raw_df"] = pd.read_csv(filepath)
                CACHE["features"] = CACHE["raw_df"].columns
                CACHE["raw_data"] = CACHE["raw_df"].to_numpy()
            except FileNotFoundError:
                return "File could not be read. Please check that the filepath is correct."
        case _: 
            raise Exception("File type not recognized. Supported types are: csv.")
    return 0

@tool
def data_head():
    """
    Returns the first 5 rows of the data currently loaded. Useful for retrieving basic information about the data.
    """
    try:
        return CACHE["raw_df"].head()
    except KeyError:
        return "Data head is not available, likely because data has not been loaded yet."

@tool
def run_lda(test_ratio: float = None, trim_rows: int = 0, trim_cols: int = 0):
    """
    Given that raw data is loaded, preprocesses it for LDA, then fits it to an LDA model and runs validation.

    Parameters
    ----------
    test_ratio  Ratio of samples, in decimal format, to be included in the test set.
    trim_rows   Number of rows to be trimmed, starting from the top of the dataset.
    trim_cols   Number of cols to be trimmed, starting from the left of the dataset.

    Returns
    -------
    Test accuracy for this dataset.
    """
    # Preprocess the data for LDA.
    kwargs = {
        "test_size": test_ratio,
        "trim_rows": trim_rows,
        "trim_cols": trim_cols,
    }
    if lda.preprocess_lda(CACHE, kwargs):
        return "Error with preprocessing."

    # Fit the data to an LDA model.
    if lda.fit_lda(CACHE):
        return "Error with fitting model."

    # Validate the model.
    return lda.test_lda(CACHE)

@tool
def run_qda(test_ratio: float = None, trim_rows: int = 0, trim_cols: int = 0):
    """
    Given that raw data is loaded, preprocesses it for QDA, then fits it to an QDA model and runs validation.

    Parameters
    ----------
    test_ratio  Ratio of samples, in decimal format, to be included in the test set.
    trim_rows   Number of rows to be trimmed, starting from the top of the dataset.
    trim_cols   Number of cols to be trimmed, starting from the left of the dataset.

    Returns
    -------
    Test accuracy for this dataset.
    """
    # Preprocess the data for QDA.
    kwargs = {
        "test_size": test_ratio,
        "trim_rows": trim_rows,
        "trim_cols": trim_cols,
    }
    if qda.preprocess_qda(CACHE, kwargs):
        return "Error with preprocessing."

    # Fit the data to a QDA model.
    if qda.fit_qda(CACHE):
        return "Error with fitting model."

    # Validate the model.
    return qda.test_qda(CACHE)

@tool
def run_pca(test_ratio: float = None, trim_rows: int = 0, trim_cols: int = 0, n_components: int = 3):
    """
    Unsupervised learning. Fits loaded training data for Principal Component Analysis, and transforms the test data on the model, and returns the transformed data.

    Parameters
    ----------
    test_ratio      Ratio of samples, in decimal format, to be included in the test set.
    trim_rows       Number of rows to be trimmed, starting from the top of the dataset.
    trim_cols       Number of cols to be trimmed, starting from the left of the dataset.
    n_components    Number of principal components to be included.

    Returns
    -------
    Transformed version of test data for this dataset.
    """
    kwargs = {
        "test_size": test_ratio,
        "trim_rows": trim_rows,
        "trim_cols": trim_cols,
    }
    if pca.preprocess_pca(CACHE, kwargs):
        return "Error with preprocessing."
    
    # Fit the data to a PCA model.
    if pca.fit_pca(CACHE, n_components=n_components):
        return "Error with fitting model."
    
    # Transform the test data on the model.
    return pca.transform_pca(CACHE, CACHE["test_data"])

@tool
def get_features():
    """
    Returns the features of the provided dataset.
    """
    try:
        return CACHE["features"]
    except:
        return "Features are not available, likely because data has not been loaded yet."

@tool
def set_input():
    """
    Sets the currently loaded raw data as input for prediction. Must be called before 'predict'.
    """
    CACHE['input'] = CACHE['raw_data']

@tool
def predict(model: str):
    """
    Uses the model to predict on currently loaded input data. Returns the predicted class.

    Parameters
    ----------
    model       Model to choose from. Currently supports 'lda', 'qda', and 'pca'.

    Returns
    -------
    List of predictions for each input.
    """
    try:
        CACHE["test_predictions"] = CACHE[model].predict(CACHE['input'])
        return CACHE["test_predictions"]
    except KeyError:
        return f"Unable to predict. Check that the requested model has been trained and input has been set."
    except ValueError:
        return "Unable to predict. Check that the number of features of the input matches that of the training data."

    # CACHE["test_predictions"] = CACHE[model].predict(input)
    # return CACHE["test_predictions"]

@tool
def display_image():
    """
    (TESTING) displays an image for streamlit.
    """
    image = Image.open('../../images/ref.png')
    width, height = image.size
    image = image.resize((width // 2, height // 2))
    st.sidebar.image(image, caption='Image created by DALL·E 3')
    return "Image succesfully displayed on sidebar."

tz = [load_data, data_head, run_lda, run_qda, run_pca, get_features, predict, display_image]