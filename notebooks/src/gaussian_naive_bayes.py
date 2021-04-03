from sklearn.linear_model.base import BaseEstimator
import numpy as np

def gaussian_pdf(value, mean, std):
    return np.exp(-np.power((value - mean)/std, 2) / 2) / (std * np.sqrt(2 * np.pi))

def gaussian_log_pdf(value, mean, std):
    return (-np.power((value - mean) / std, 2) / 2) - np.log(std * np.sqrt(2 * np.pi))

class BinaryGaussianNaiveBayes(BaseEstimator):
    
    def __init__(self, smoothing=None, bessel_correction=None, filter_variables=None):
        self.variables_count = None
        self.variables_mean = None
        self.variables_std = None
        self.priori_distribution = None
        self.log_priori_distribution = None
        self.smoothing = smoothing if smoothing is not None else 0
        self.bessel_correction = bessel_correction if bessel_correction is not None else False
        self.filter_variables = filter_variables
    
    def fit(self, x_data, y_data):
        # Filtering data if required
        if self.filter_variables is not None:
            x_data = x_data[:,self.filter_variables]
        
        # Calculating priori distribution
        self.priori_distribution = np.array([len(y_data[y_data == 0]) , len(y_data[y_data == 1])])
        self.priori_distribution = self.priori_distribution / self.priori_distribution.sum()
        self.log_priori_distribution = np.log(self.priori_distribution)
        
        # Initializing vectors
        self.variables_mean = np.zeros((2, x_data.shape[1]))
        self.variables_std = np.zeros((2, x_data.shape[1]))
        
        # Calculating mean and standard deviation of variables
        for variable_index in range(x_data.shape[1]):
            for class_index in range(2):
                self.variables_mean[class_index][variable_index] = np.nanmean(x_data[y_data == class_index, variable_index])
                self.variables_std[class_index][variable_index] = np.nanstd(x_data[y_data == class_index, variable_index])
                
                # Apply Bessel's correction to the standard deviation error 
                if self.bessel_correction:
                    n = (y_data == class_index).sum()
                    self.variables_std[class_index][variable_index] = self.variables_std[class_index][variable_index] * np.sqrt((n) / (n-1))
            
    def predict(self, x_data):
        # Filtering data if required
        if self.filter_variables is not None:
            x_data = x_data[:,self.filter_variables]
            
        # Initialization of predictions
        predictions = np.zeros(x_data.shape[0])
        
        # Prediction for each subject
        for subject_index in range(x_data.shape[0]):
            log_likelihood = np.array(
                [
                    gaussian_log_pdf(
                        np.array([x_data[subject_index][variable_index], x_data[subject_index][variable_index]]), 
                        self.variables_mean[:, variable_index], 
                        self.variables_std[:, variable_index]
                    )
                    for variable_index in range(x_data.shape[1])
                ]
            ).sum(axis = 0)
            log_posteriori_unnormalized = log_likelihood + self.log_priori_distribution
            log_odds = log_posteriori_unnormalized[1] - log_posteriori_unnormalized[0]
            predictions[subject_index] = 1 if log_odds > 0 else 0
        
        # Return the predictions made by the model
        return predictions