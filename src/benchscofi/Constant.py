#coding: utf-8

## Minimal example of model in stanscofi

from stanscofi.models import BasicModel
import numpy as np

class Constant(BasicModel):
    '''
    A example model which always predicts the positive class. It is a subclass of stanscofi.models.BasicModel (please refer to the documentation of this class for more information)

    ...

    Parameters
    ----------
    params : dict
        empty

    Attributes
    ----------
    name : str
        the name of the model
    ...
        other attributes might be present depending on the type of model

    Methods
    -------
    Same as BasicModel class
    __init__(params)
        Initialize the model with preselected parameters
    default_parameters()
        Outputs a dictionary which contains default values of parameters
    model_fit()
        Preprocess and fit the model (not implemented in BasicModel)
    model_predict_proba()
        Outputs predictions of the fitted model on test_dataset (not implemented in BasicModel)
    '''
    def __init__(self, params=None):
        '''
        Creates an instance of benchscofi.Constant

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains a key called "decision_threshold" with a float value which determines the decision threshold to label a positive class
        '''
        params = params if (params is not None) else self.default_parameters()
        super(Constant, self).__init__(params)
        self.name = "Constant"

    def default_parameters(self):
        params = {}
        return params

    def preprocessing(self, dataset, is_training=True):
        '''
        Preprocessing step, which is empty for this model.
        The general rule is that, for training:
         - if the algorithm takes as input a matrix, then consider all ratings (y in {-1,0,1})
         - if the algorithm takes as input a collection pairs (item, user), then consider only known ratings (y in {-1,1})
        For testing/predicting, regardless of the type of input considered by the algorithm, consider all ratings (y in {-1,0,1})

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert

        Returns
        ----------
        res : list of a single stanscofi.Dataset
            input to the fitting method (should be a list of all arguments)
        '''
        return [dataset]
        
    def model_fit(self, train_dataset):
        '''
        Fitting the Constant model on the training dataset (no fitting step here).

        ...

        Parameters
        ----------
        train_dataset : stanscofi.Dataset
            training dataset on which the model should fit
        '''
        pass

    def model_predict_proba(self, test_dataset):
        '''
        Making predictions using the Constant model on the testing dataset.

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            testing dataset on which the model should be validated
        '''
        scores = np.ones(test_dataset.ratings.shape)
        return scores 
