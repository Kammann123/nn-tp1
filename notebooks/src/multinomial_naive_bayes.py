import numpy as np

class MultinomialNaiveBayes:

    def __init__(self, alpha=None):
        """ Create an instante of the multinomial naive bayes classifier.
            @param alpha Coefficient for the laplacian smoothing, None if not desired
        """

        # Internal model information
        self.y_dist = None           # P(Y) 
        self.x_cond_dist = None      # P(X|Y)
        self.y_log_dist = None       # ln[ P(Y) ]
        self.x_log_cond_dist = None  # ln[ P(X|Y) ]

        # Hiperparameters of the model
        self.alpha = alpha if alpha is not None else 0

    def predict(self, x_data : np.array) -> np.array:
        """ Predict the category given the feature values
            @param x_data Input array or matrix, where the rows are the case and the columns the values of the features
            @return Number indicating the category to which each input belongs
        """
        
        # Verify if the model has been trained
        if self.y_dist is None or self.x_cond_dist is None or self.y_log_dist is None or self.x_log_cond_dist is None:
            raise ValueError('Model has not been trained!')
        
        # If only an array is given, reshaping to matrix of one row
        if x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)
            
        # Computing the log posteriori probabilities
        log_posteriori = np.dot(self.x_log_cond_dist, x_data.transpose()) + (self.y_log_dist.reshape(-1, 1) * np.ones(x_data.shape[0]))
    
        # Classifying according to the maximum log posteriori as criteria
        return np.array([log_posteriori[:,x_index].argmax() for x_index in range(x_data.shape[0])])

    def fit(self, x_data : np.array, y_data : np.array):
        """ Fit or train the multinomial naive bayes model with the given training data.
            @param x_data Input matrix, each row represents a training case and its columns are the feature values
            @param y_data Output array, contains the correct result of the classifier, to use for the training
        """

        # The amount of rows in the input matrix must be the same as the length of the output array
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError('Invalid shapes, y_data must have the same length as the amount of rows in the x_data parameter')

        # Estimation of the probability distribution for the categorical output variable
        self.y_dist = []
        i = 0
        while (np.array(self.y_dist).sum() < y_data.shape[0]):
            self.y_dist.append((y_data == i).sum())
            i += 1
        self.y_dist = np.array(self.y_dist) / np.array(self.y_dist).sum()
        self.y_log_dist = np.log(self.y_dist)

        # Estimation of the conditional probability distribution for the input variable given the output variable
        self.x_cond_dist = np.zeros((self.y_dist.shape[0], x_data.shape[1]), dtype=np.float64)
        for category in range(self.y_dist.shape[0]):
            unnormalized = (x_data[y_data == category][:]).sum(axis=0) + self.alpha
            self.x_cond_dist[category,:] = unnormalized / unnormalized.sum()
        self.x_log_cond_dist = np.log(self.x_cond_dist)
        