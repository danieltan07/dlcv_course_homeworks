import numpy as np

class MultinomialLogisticRegression(object):
    def __init__(self):
        pass

    def initialize_weights(self, input_dim, num_classes, std_dev=1e-2):
        """
        Initialize the weights of the model. The weights are initialized
        to small random values. Weights are stored in the variable dictionary
        named self.params.

        W: weight vector; has shape (D, C)
        b: bias vector; has shape (C,)
        
        Inputs:
        - input_dim: (int) The dimension D of the input data.
        - num_classes: (int) The dimension C representing the number of classes.
        - std_dev: (float) Controls the standard deviation of the random values.
        """
        
        self.params = {}
        #############################################################################
        # TODO: Initialize the weight and bias.                                     #
        #############################################################################
        self.params['W'] = None
        self.params['b'] = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
        
    def softmax(self,x):
        
        """
        Compute the softmax function for each row of the input x.

        Inputs:
        - x: A numpy array of shape (N, C) containing scores for each class; there are N
          examples each of dimension C.

        Returns:
        probs: A numpy array of shape (N, C) containing probabilities for each class.
        """


        #############################################################################
        # TODO: Implement softmax which converts scores to probabilities.           #
        #############################################################################
        probs = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        ############################################################################# 

        return probs

    def predict(self, X):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (N, D) containing test data consisting
             of N samples each of dimension D.

        Returns:
        - predictions: A numpy array of shape (N,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """

        #############################################################################
        # TODO: Compute for the predictions of the model on new data using the      #
        # learned weight vectors.                                                   #
        #############################################################################
        predictions = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return predictions

    def loss(self, X, y=None, lambda_reg=0.0):
        """
        Compute the loss and gradients for an iteration of linear regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels of shape (N,) where y[i] is the ground truth value for X[i].
        - lambda_reg: Regularization strength.

        Returns:
        Return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        W, b = self.params['W'], self.params['b']
        N, D = X.shape

        #############################################################################
        # TODO: Computing the class scores for the input.                           #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        score = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        
        #############################################################################
        # TODO: Compute the loss which should include both the data loss 			#
        # and L2 regularization for W. Store the result in the variable loss, 		#
        # which should be a scalar.                           						#
        #############################################################################
        loss = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the derivatives of the weights and biases. Store the        #
        # results in the grads dictionary. For example, grads['W'] should store     #
        # the gradient on W, and be a matrix of same size.                          #
        #############################################################################
        grads['W'] = None
        grads['b'] = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads
    
    def train(self, X, y, learning_rate=1e-3, lambda_reg=1e-5, num_iters=100,
            batch_size=200, std_dev=1e-2, verbose=False):
        """
        Train a Multinomial Logistic Regression model using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        - learning_rate: (float) learning rate for optimization.
        - lambda_reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - std_dev: (float) Controls the standard deviations of the initial weights.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = len(np.unique(y.squeeze()))

        self.initialize_weights(dim, num_classes, std_dev)

        loss_history = []
        for it in range(num_iters):

            indices = np.random.choice(num_train,batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            loss, grads = self.loss(X_batch, y=y_batch, lambda_reg=lambda_reg)

            if it % 10 == 0:
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
            if verbose and (it+1) % 100 == 0:
                print('iteration %d / %d: loss %f' % (it+1, num_iters, loss))

        return loss_history




