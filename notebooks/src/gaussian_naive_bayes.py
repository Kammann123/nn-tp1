# Including libraries from sklearn
from sklearn.base import BaseEstimator

# Including libraries from scipy
from scipy import stats

# Including libraries from numpy
import numpy as np

def gaussian_pdf(value, mean, std):
    """ Probability density function of a gaussian distributed continuous random variable.
        @param value Value where the pdf is evaluated
        @param mean Mean of the distribution
        @param std Standard deviation of the distribution
    """
    return stats.norm.pdf((value - mean) / std) / std

class BinaryGaussianNaiveBayes(BaseEstimator):
    """ Naive Bayes classificator for binary classes (positive and negative) using gaussian continuous random variables """
    
    def __init__(self, std_smoothing=0, std_correction=False, filter_variables=None):
        
        # Parameters of the model, contains the class distribution also known as priori probabilities,
        # and the variables parameters used to parametrize the distributions assigned to each variable
        # taken into account
        self.classes_distribution = None
        self.classes_log_distribution = None
        self.variables_mean = None
        self.variables_std = None
        
        # Configuration of the model, also known as the hiper parameters, selection of the
        # model settings used to optimize according a specific performance metric
        self.std_smoothing = std_smoothing if std_smoothing is not None else 0
        self.std_correction = std_correction if std_correction is not None else False
        self.filter_variables = filter_variables
    
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
        
        # Initializing container of mean and std values for variable's distributions
        self.variables_mean = np.zeros((2, x_data.shape[1]))
        self.variables_std = np.zeros((2, x_data.shape[1]))
        
        # Calculating mean and standard deviation of variables
        for variable_index in range(x_data.shape[1]):
            for class_index in range(2):
                # Fit the distribution
                self.variables_mean[class_index][variable_index] = np.nanmean(x_data[y_data == class_index, variable_index])
                self.variables_std[class_index][variable_index] = np.nanstd(x_data[y_data == class_index, variable_index])
                
                # Apply Bessel's correction to the standard deviation error 
                if self.std_correction:
                    n = (y_data == class_index).sum()
                    self.variables_std[class_index, variable_index] = self.variables_std[class_index, variable_index] * np.sqrt((n) / (n-1))

    def predict(self, x_data):
        """ Predict the class of the given input data.
            @param x_data Matrix where the rows represent study cases and the columns contain the variables or features to analyze
        """
        
        # Filtering data if required
        if self.filter_variables is not None:
            x_data = x_data[:,np.array(self.filter_variables)]
            
        # Initialization of predictions
        predictions = np.zeros(x_data.shape[0])
        
        # Prediction for each subject
        for subject_index in range(x_data.shape[0]):
            
            # For each class (positive, negative) compute the log likelihood
            log_likelihood = np.array(
                [
                    np.log(
                        gaussian_pdf(
                            np.array([x_data[subject_index, variable_index], x_data[subject_index, variable_index]]), 
                            self.variables_mean[:, variable_index], 
                            self.variables_std[:, variable_index] + self.std_smoothing
                        )
                    )
                    for variable_index in range(x_data.shape[1])
                ]
            ).sum(axis=0)

            # Compute the log posteriori unnormalized and predict
            log_posteriori_unnormalized = log_likelihood + self.classes_log_distribution
            predictions[subject_index] = 1 if log_posteriori_unnormalized[1] > log_posteriori_unnormalized[0] else 0
        
        # Return the predictions made by the model
        return predictions