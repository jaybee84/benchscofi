#coding: utf-8

## Minimal example of model in stanscofi

from stanscofi.models import BasicModel
from benchscofi.utils import tools

class Constant(BasicModel):
    '''
    A example model which is a subclass of stanscofi.models.BasicModel (please refer to the documentation of this class for more information)

    ...

    Parameters
    ----------
    params : dict
        dictionary which contains a key called "decision_threshold" with a float value which determines the decision threshold to label a positive class

    Attributes
    ----------
    name : str
        the name of the model
    decision_threshold : float
        decision threshold to label a positive class
    ...
        other attributes might be present depending on the type of model

    Methods
    -------
    Same as BasicModel class
    __init__(params)
        Initialize the model with preselected parameters
    default_parameters()
        Outputs a dictionary which contains default values of parameters
    fit(train_dataset)
        Preprocess and fit the model (not implemented in BasicModel)
    model_predict(test_dataset)
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
        params = {"decision_threshold": 1, "random_state": 124565}
        return params

    def preprocessing(self, dataset):
        '''
        Preprocessing step, which is empty for this model 

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert

        Returns
        ----------
        dataset : stanscofi.Dataset
            dataset to convert
        '''
        return dataset
        
    def fit(self, train_dataset):
        '''
        Fitting the Constant model on the training dataset (which is empty here).

        ...

        Parameters
        ----------
        train_dataset : stanscofi.Dataset
            training dataset on which the model should fit
        '''
        pass

    def model_predict(self, test_dataset):
        '''
        Making predictions using the Constant model on the testing dataset.

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            testing dataset on which the model should be validated
        '''
        scores = tools.create_scores(self.decision_threshold, test_dataset)
        return scores
