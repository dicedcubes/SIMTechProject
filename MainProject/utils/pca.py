"""
Contains all functions necessary for building and using a PCA model.
preprocess_pca  Using the preprocessing tool in fns, preprocesses the data according to the needs of PCA.
fit_pca         Creates a PCA object from the SKLearn library and fits the data to it.
transform_pca   Transforms the input data with the model and returns the transformed version.
"""

from sklearn.decomposition import PCA
from . import fns

def preprocess_pca(cache, kwargs={}):
    kwargs['standardize'] = True

    training_data, test_data, _, _ = fns.preprocess_data(cache['raw_data'], **kwargs)
    cache['training_data'] = training_data
    cache['test_data'] = test_data
    return 0

def fit_pca(cache, n_components: int = 3):
    # Create a PCA model.
    pca = PCA(n_components=n_components)

    # Fit the cached data to the model.
    try:
        pca.fit(cache['training_data'])
    except:
        return -1
    
    # Add the model to the cache.
    cache['pca'] = pca
    return 0

def transform_pca(cache, input):
    # Transform the loaded data based on the PCA model.
    pca = cache['pca']
    try:
        cache["pca_projection"] = pca.transform(input)
    except:
        return -1
    return cache["pca_projection"]