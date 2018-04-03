import numpy as np

class LinearRegression(object):
    def __init__(self):
        pass

    def feature_transform(self, X):
        """
        Appends a vector of ones for the bias term.

        Inputs:
        - X: A numpy array of shape (N, D) consisting
             of N samples each of dimension D.

        Returns:
        - X_transformed: A numpy array of shape (N, D + 1)
        """

        #############################################################################
        # TODO: Append a vector of ones across the dimension of your input data.    #
        # This accounts for the bias or the constant in your hypothesis function.   #
        #############################################################################
        X_transformed = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        return X_transformed

     
        
    def train_analytic(self, X, y):
        self.params = {}
        
        self.params['W'] = None
        #############################################################################
        # TODO: Compute for the weight vector for linear regression using the       #
        # normal equation / analytical solution.                                    #
        # Store the computed weights in self.params['W']                            #
        # Hint: lookup numpy.linalg.pinv                                            #
        #############################################################################
        pass
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def predict(self, X):
        """
        Predict values for test data using linear regression.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - y: A numpy array of shape (num_test, 1) containing predicted values for the
          test data, where y[i] is the predicted value for the test point X[i].  
        """

        W = self.params['W']
        num_test, D = X.shape

        if D != W.shape[0]:
            X = self.feature_transform(X)
        
        #############################################################################
        # TODO: Compute for the predictions of the model on new data using the      #
        # learned weight vectors.                                                   #
        #############################################################################
        prediction = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
        return prediction

    def initialize_weights(self, dim, std_dev=1e-2):
        """
        Initialize the weights of the model. The weights are initialized
        to small random values. Weights are stored in the variable dictionary
        named self.params.

        W: weight vector; has shape (D, 1)
        
        Inputs:
        - dim: (int) The dimension D of the input data.
        - std_dev: (float) Controls the standard deviation of the random values.
        """
        self.params = {}
        #############################################################################
        # TODO: Initialize the weight vector to random values with                  #
        # standard deviation determined by the parameter std_dev.                   #
        # Hint: Look up the function numpy.random.randn                             #
        #############################################################################
        self.params['W'] = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################   
    def loss(self, X, y=None):
        """
        Compute the loss and gradients for an iteration of linear regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the ground truth value for X[i].

        Returns:
        Return a tuple of:
        - loss: Loss for this batch of training samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W = self.params['W']
        N, D = X.shape

        #############################################################################
        # TODO: Compute for the prediction value given the current weight vector.   #
        # Store the result in the prediction variable                               #
        #############################################################################
        prediction = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        #############################################################################
        # TODO: Compute for the loss.                                               #
        #############################################################################
        loss = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        grads = {}
        #############################################################################
        # TODO: Compute the derivatives of the weights. Store the                   #
        # results in the grads dictionary. For example, grads['W'] should store     #
        # the gradient on W, and be a matrix of same size.                          #
        #############################################################################
        
        grads['W'] = None
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val=None, y_val=None,learning_rate=1e-3, learning_rate_decay=0.95,
            num_iters=1000, batch_size=200, verbose=False):
        """
        Train Linear Regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N, 1) containing the ground truth values.
        - X_val: A numpy array of shape (N_val, D) containing validation data.
        - y_val: A numpy array of shape (N_val, 1) containing validation ground truth values.
        - learning_rate: (float) learning rate for optimization.
        - learning_rate_decay: (float) scalar denoting the factor used to decay the learning rate.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        return a list containing the value of the loss function at each training iteration.
        """
        X = self.feature_transform(X)
        num_train, dim = X.shape
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Implement the initiaize_weights function.
        self.initialize_weights(dim)

        loss_history = []
        train_rmse = []
        val_rmse = []
        for it in range(num_iters):

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            # Hint: Look up the function numpy.random.choice                        #
            #########################################################################
            X_batch = None
            y_batch = None
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            
            loss, grads = self.loss(X=X_batch, y=y_batch)
            loss_history.append(np.squeeze(loss))

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the model (stored in the dictionary self.params)        #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            pass
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            if verbose and (it + 1) % 100 == 0:
                print('iteration %d / %d: loss %f' % (it + 1, num_iters, loss))
                

            if it % iterations_per_epoch == 0:
                learning_rate *= learning_rate_decay

                if verbose:
                    y_pred = self.predict(X_batch)
                    rmse = self.root_mean_squared_error(y_batch,y_pred)
                    train_rmse.append(rmse)

                    print("Epoch {} \t training RMSE: {:0.4f}".format(it // iterations_per_epoch, rmse))

                    if X_val is not None and y_val is not None:
                        y_pred = self.predict(X_val)
                        rmse_val = self.root_mean_squared_error(y_val,y_pred)
                        val_rmse.append(rmse_val)

                        print("\t\t validation RMSE: {:0.4f}".format(rmse_val))


        return {'loss_history':loss_history, 'train_rmse':train_rmse, 'val_rmse':val_rmse}

    
    def root_mean_squared_error(self,y,y_prediction):
        """
        Root Mean Squared Error evaluation metric

        Inputs:
        - y: A numpy array of shape (N, 1) containing the ground truth values.
        - y_prediction: A numpy array of shape (N, 1) containing the predicted values.

        Outputs:
        returns the root mean squared error of the prediction.
        """

        rmse = np.sqrt(np.mean((y-y_prediction)**2))

        return rmse

  
    
