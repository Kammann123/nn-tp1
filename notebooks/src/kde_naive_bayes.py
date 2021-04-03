# Including libraries from sklearn
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity

# Including libraries from scipy
from scipy import stats

# Including libraries from numpy
import numpy as np

class BinaryKDENaiveBayes(BaseEstimator):
    """ Naive Bayes classificator for binary classes (positive and negative) using continuous random variables with a kernel density estimator """
    
    def __init__(self, kernel='gaussian', bandwidth=0.5, filter_variables=None):
        
        # Parameters of the model, contains the class distribution also known as priori probabilities,
        # and the variables parameters used to parametrize the distributions assigned to each variable
        # taken into account
        self.classes_distribution = None
        self.classes_log_distribution = None
        self.kde = None
        
        # Configuration of the model, also known as the hiper parameters, selection of the
        # model settings used to optimize according a specific performance metric
        self.filter_variables = filter_variables
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def fit(self, x_data, y_data):
        """ Fit the model with the training dataset given.
            @param x_data Matrix where the rows contain study cases and the columns contain variables or features
            @param y_data Array containing the class where the corresponding study case belong
        """
        
        # Filtering data if required
        if self.filter_variables is not None:
            x_data = x_data[:,self.filter_variables]
        
        # Instantiating KDE objects
        self.kde = [
            KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth), 
            KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        ]
        
        # Computing the probability distribution of all classes
        self.classes_distribution = np.array([len(y_data[y_data == 0]) , len(y_data[y_data == 1])])
        self.classes_distribution = self.classes_distribution / self.classes_distribution.sum()
        self.classes_log_distribution = np.log(self.classes_distribution)
        
        # Fitting the kernel density estimators
        for class_index in range(2):
            self.kde[class_index].fit(x_data[y_data == class_index])
            
    def predict(self, x_data):
        """ Predict the class of the given input data.
            @param x_data Matrix where the rows represent study cases and the columns contain the variables or features to analyze
        """
        
        # Filtering data if required
        if self.filter_variables is not None:
            x_data = x_data[:,self.filter_variables]
            
        # Initialization of predictions
        predictions = np.zeros(x_data.shape[0])
        
        # Prediction for each subject
        for subject_index in range(x_data.shape[0]):
            
            # Computing the log likelihoods
            log_likelihood = np.array(
                [
                    self.kde[0].score(x_data[subject_index,:].reshape(1, -1)),
                    self.kde[1].score(x_data[subject_index,:].reshape(1, -1))
                ]
            )
            
            # Computing the log posteriori unnormalized
            log_posteriori_unnormalized = log_likelihood + self.classes_log_distribution
            predictions[subject_index] = 1 if log_posteriori_unnormalized[1] > log_posteriori_unnormalized[0] else 0
        
        # Return the predictions made by the model
        return predictions