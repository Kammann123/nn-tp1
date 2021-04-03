# Including libraries from sklearn
from sklearn.base import BaseEstimator

# Including libraries from scipy
from scipy import stats

# Including libraries from numpy
import numpy as np

def gaussian_pdf(value, parameters):
    """ Probability density function of a gaussian distributed continuous random variable.
        @param value Value where the pdf is evaluated
        @param parameters Values used to parametrize the distribution, such as the mean and the std
    """
    mean = parameters['mean']
    std = parameters['std']
    return stats.norm.pdf((value - mean) / std) / std

def exponential_pdf(value, parameters):
    """ Probability density function of an exponential distributed continuous random variable.
        @param value Value where the pdf is evaluated
        @param parameters Values used to parametrize the distribution, such as the mean
    """
    _lambda = parameters['lambda']
    return stats.expon.pdf(value * _lambda) * _lambda

class BinaryNaiveBayes(BaseEstimator):
    """ Implements the Naive Bayes classification criteria to problems with two classes, 
        allowing parametric distributions based on famous density functions.
    """
    
    # Dictionary used to map the type of distribution set in the configuration of the model
    # and the function that handles. Basically, a dispatcher of probability density functions
    supported_distributions = {
        'gaussian': gaussian_pdf,
        'exponential': exponential_pdf
    }
    
    def __init__(self, std_correction=False, filter_variables=None, variables_models=None):
        
        # Parameters of the model, contains the class distribution also known as priori probabilities,
        # and the variables parameters used to parametrize the distributions assigned to each variable
        # taken into account
        self.classes_distribution = None
        self.classes_log_distribution = None
        self.variables_distributions = None
        
        # Configuration of the model, also known as the hiper parameters, selection of the
        # model settings used to optimize according a specific performance metric
        self.std_smoothing = std_smoothing if std_smoothing is not None else 0
        self.std_correction = std_correction if std_correction is not None else False
        self.filter_variables = filter_variables
        self.variables_models = variables_models
    
    def fit(self, x_data, y_data):
        """ Fit the model with the training dataset given.
            @param x_data Matrix where the rows contain study cases and the columns contain variables or features
            @param y_data Array containing the class where the corresponding study case belong
        """
        
        # Filtering data if required
        if self.filter_variables is not None:
            x_data = x_data[:,np.array(self.filter_variables)]
        
        # Computing the probability distribution of all classes
        self.classes_distribution = np.array([len(y_data[y_data == 0]) , len(y_data[y_data == 1])])
        self.classes_distribution = self.classes_distribution / self.classes_distribution.sum()
        self.classes_log_distribution = np.log(self.classes_distribution)
        
        # Initializing parameter container
        self.variables_parameters = []
        
        # Fetch the models filtered
        models = np.array(self.variables_distributions)[np.array(self.filter_variables)]
        
        # Calculating mean and standard deviation of variables
        for variable_index in range(x_data.shape[1]):
            for class_index in range(2):
                
                # Fit the corresponding distribution assigned to the variable or feature
                if self.
                self.variables_mean[class_index][variable_index] = np.nanmean(x_data[y_data == class_index, variable_index])
                self.variables_std[class_index][variable_index] = np.nanstd(x_data[y_data == class_index, variable_index])
                
                # Apply Bessel's correction to the standard deviation error 
                if self.bessel_correction:
                    n = (y_data == class_index).sum()
                    self.variables_std[class_index][variable_index] = self.variables_std[class_index][variable_index] * np.sqrt((n) / (n-1))
            
    def predict(self, x_data):
        """ Predict the class of the given input data.
            @param x_data Matrix where the rows represent study cases and the columns contain the variables or features to analyze
        """
        
        # Filtering data if required
        if self.filter_variables is not None:
            x_data = x_data[:,np.array(self.filter_variables)]
            
        # Initialization of predictions
        predictions = np.zeros(x_data.shape[0])
        
        # Fetch the type of distributions
        distributions = np.array(self.variables_distributions)[np.array(self.filter_variables)]
        
        # Prediction for each subject
        for subject_index in range(x_data.shape[0]):
            
            # For each class (positive, negative) compute the log likelihood
            log_likelihood = np.array(
                [
                    np.log(
                        self.supported_distributions[distributions[variable_index]](
                            np.array([x_data[subject_index, variable_index] for i in range(2)]), 
                            self.variables_mean[:, variable_index], 
                            self.variables_std[:, variable_index] + self.smoothing
                        )
                    )
                    for variable_index in range(x_data.shape[1])
                ]
            ).sum(axis=0)

            # Compute the log posteriori unnormalized and predict
            log_posteriori_unnormalized = log_likelihood + self.log_priori_distribution
            predictions[subject_index] = 1 if log_posteriori_unnormalized[1] > log_posteriori_unnormalized[0] else 0
        
        # Return the predictions made by the model
        return predictions