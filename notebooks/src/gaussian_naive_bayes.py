import numpy as np

def gaussian_pdf(value, mean, std):
    return np.exp(-np.power((value - mean)/std, 2) / 2) / (std * np.sqrt(2*np.pi))

class BinaryGaussianNaiveBayes:
    def __init__(self):
        self.variables_mean = None
        self.variables_std = None
        self.priori_distribution = None
        
    
    def fit(self, x_data, y_data):
        # Calculating priori distribution
        self.priori_distribution = np.array([len(y_data[y_data == 0]) , len(y_data[y_data == 1])])
        self.priori_distribution /= self.priori_distribution.sum()
        
        # Initializing vectors
        self.variables_mean = np.zeros((2, x_data.shape[1]))
        self.variables_std = np.zeros((2, x_data.shape[1]))
        
        # Calculating mean and standard deviation of variables
        for variable_index in range(x_data.shape[1]):
            self.variables_mean[0][variable_index] = x_data[y_data == 0][variable_index].mean()
            self.variables_std[0][variable_index] = x_data[y_data == 0][variable_index].std()
            
            self.variables_mean[1][variable_index] = x_data[y_data == 1][variable_index].mean()
            self.variables_std[1][variable_index] = x_data[y_data == 1][variable_index].std()
            
    def predict(self, x_data):
        
        predictions = np.zeros(x_data.shape[0])
        
        for subject_index in range(x_data.shape[0]):
            # Posteriori 
            likelihood = np.array([gaussian_pdf(np.array([x_data[subject_index][variable_index], x_data[subject_index][variable_index]]), self.variables_mean[:][variable_index], self.variables_std[:][variable_index]) for variable_index in x_data.shape[1]]).prod()
            
            posteriori = self.priori_distribution * np.diagonal(likelihood)
            predictions[subject_index] = np.argmax(posteriori)
            
        return predictions