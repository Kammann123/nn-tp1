# Including libraries from sklearn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# Including libraries from scipy
from scipy import stats

# Including libraries from numpy
import numpy as np
        
class GaussianPDF:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """ Fit the probability density function with the given data
            @param data Array of values sampled for the random variable
        """
        self.mean = np.nanmean(data)
        self.std = np.nanstd(data)
    
    def score_sample(self, value):
        """ Compute the probability density of the given values
            @param value Array of values where the pdf is evaluated
        """
        return stats.norm(self.mean, self.std).pdf(value)

class ExponentialPDF:
    def __init__(self):
        self._lambda = None

    def fit(self, data):
        """ Fit the probability density function with the given data
            @param data Array of values sampled for the random variable
        """
        self._lambda = 1 / np.nanmean(data)
    
    def score_sample(self, value):
        """ Compute the probability density of the given values
            @param value Array of values where the pdf is evaluated
        """
        return stats.expon(scale=(1 / self._lambda)).pdf(value)

class BinaryMixedNaiveBayes(BaseEstimator):
    """ Implements the Naive Bayes classification criteria to problems with two classes, 
        allowing parametric distributions based on famous density functions.
        Supports using a different implemented random distribution on each variable, where currently
        gaussian and exponential are the only options.
    """
    
    def __init__(self, filter_variables=None, variables_models=None):
        
        # Parameters of the model, contains the class distribution also known as priori probabilities,
        # and the variables parameters used to parametrize the distributions assigned to each variable
        # taken into account
        self.classes_distribution = None
        self.classes_log_distribution = None
        self.variables_distributions = None
        
        # Configuration of the model, also known as the hiper parameters, selection of the
        # model settings used to optimize according a specific performance metric
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
        
        # Initializing distribution container
        self.variables_distributions = [[None for j in range(2)] for i in range(x_data.shape[1])]
        
        # Using the variable filter to fetch the random distribution model assigned to each variable
        models = np.array(self.variables_models)[np.array(self.filter_variables)]
        
        # Calculating mean and standard deviation of variables
        for variable_index in range(x_data.shape[1]):
            for class_index in range(2):
                
                # For each variable and the class to which the conditional probability is conditioned
                # create an instance of the distribution handler
                if models[variable_index] == 'gaussian':
                    pdf = GaussianPDF()
                elif models[variable_index] == 'exponential':
                    pdf = ExponentialPDF()
                
                # Fit the distribution model
                pdf.fit(x_data[y_data == class_index, variable_index])
                
                # Loading the distribution
                self.variables_distributions[variable_index][class_index] = pdf

    def predict(self, x_data):
        """ Predict the class of the given input data.
            @param x_data Matrix where the rows represent study cases and the columns contain the variables or features to analyze
        """
        
        # Filtering data if required
        if self.filter_variables is not None:
            x_data = x_data[:,np.array(self.filter_variables)]
            
        # Initialization of predictions, and computing in each case the
        # probabilities required to obtain the highest posteriori probability
        log_posteriori_unnormalized = np.zeros((x_data.shape[0], 2))
        predictions = np.zeros(x_data.shape[0])
        for subject_index in range(x_data.shape[0]):
            
            # Calculate the posteriori probability for each class
            for class_index in range(2):
                # Compute the log likelihood
                log_likelihood = np.array(
                    [
                        np.log(
                            self.variables_distributions[variable_index][class_index].score_sample(
                                x_data[subject_index, variable_index]
                            )
                        )
                        for variable_index in range(x_data.shape[1])
                    ]
                ).sum()

                # Compute the log posteriori unnormalized and predict
                log_posteriori_unnormalized[subject_index, class_index] = log_likelihood + self.classes_log_distribution[class_index]
            
            # Make a prediction for each subject
            predictions[subject_index] = np.argmax(log_posteriori_unnormalized[subject_index, :])
        
        # Return the predictions made by the model
        return predictions