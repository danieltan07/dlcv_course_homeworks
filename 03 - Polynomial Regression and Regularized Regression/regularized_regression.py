import numpy as np

class RegularizedRegression(object):
    def __init__(self):
        pass


    def poly_feature_transform(self,X,poly_order=1):
        """
        Transforms the input data to match the specified polynomial order.

        Inputs:
        - X: A numpy array of shape (N, D) consisting
             of N samples each of dimension D.
        - poly_order: Determines the order of the polynomial of the hypothesis function. (default is 1)

        Returns:
        - f_transform: A numpy array of shape (N, D * order + 1) representing the transformed
            features following the specified poly_order.
        """
        f_transform = X

        #############################################################################
        # TODO: Transform your inputs to the corresponding polynomial with order    #
        # given by the parameter poly_order.                                        #
        #############################################################################           

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO: Append a vector of ones across the dimension of your input data.    #
        # This accounts for the bias or the constant in your hypothesis function.   #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return f_transform

    def train_analytic(self, X, y, poly_order=1, lambda_reg=0):
        """
        Solves for the weight vector using the normal equation.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - y: A numpy array of shape (num_test, 1) containing predicted values for the
          test data, where y[i] is the predicted value for the test point X[i].  
        - poly_order: Determines the order of the polynomial of the hypothesis function. (default is 1)
        - lambda_reg: (float) Regularization strength.
        
        """
        # store the polynomial order in the object state
        self.poly_order = poly_order

        self.params = {}        
        self.params['W'] = None

        #############################################################################
        # TODO: Compute for the weight vector for linear regression using the       #
        # normal equation / analytical solution.                                    #
        # Store the computed weights in self.params['W']                            #
        # Hint: lookup numpy.linalg.pinv                                            #
        #############################################################################

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
            X = self.poly_feature_transform(X, self.poly_order)
        
        #############################################################################
        # TODO: Compute for the predictions of the model on new data using the      #
        # learned weight vectors.                                                   #
        #############################################################################

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

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
    
    def loss(self, X, y=None, lambda_reg=0.0):
        """
        Compute the loss and gradients for an iteration of linear regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the ground truth value for X[i].
        - lambda_reg: (float) Regularization strength.

        Returns:
        Return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
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

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        #############################################################################
        # TODO: Compute for the loss.                                               #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        grads = {}
        #############################################################################
        # TODO: Compute the derivatives of the weights. Store the                   #
        # results in the grads dictionary. For example, grads['W'] should store     #
        # the gradient on W, and be a matrix of same size.                          #
        #############################################################################

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads
        
        
    def train(self, X, y, poly_order=1, learning_rate=0.2, lambda_reg=0, num_iters=100, std_dev=1e-2,
            batch_size=20, verbose=False):
        """
        Train Linear Regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N, 1) containing the ground truth values.
        - poly_order: (integer) determines the polynomial order of your hypothesis function.
        - learning_rate: (float) learning rate for optimization.
        - lambda_reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - std_dev: (float) Controls the standard deviation of the random weights initialization.
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        return a list containing the value of the loss function at each training iteration.
        """
        self.poly_order = poly_order
        X = self.poly_feature_transform(X, self.poly_order)
        num_train, dim = X.shape

        # Implement the initialize_weights function.
        self.initialize_weights(dim, std_dev)

        loss_history = []
        for it in range(num_iters):

            indices = np.random.choice(num_train,batch_size,replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            loss, grads = self.loss(X_batch, y=y_batch, lambda_reg=lambda_reg)
            loss_history.append(np.squeeze(loss))

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the model (stored in the dictionary self.params)        #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))


        return loss_history

    
    
    
    
