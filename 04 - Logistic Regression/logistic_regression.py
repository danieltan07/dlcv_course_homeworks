import numpy as np

class LogisticRegression(object):
    def __init__(self):
        pass

    def initialize_weights(self, dim, std_dev=1e-2):
        """
        Initialize the weights and biases of the model. The weights are initialized
        to small random values and the biases are initialized to zero. Both are stored 
        in the dictionary named self.params.

        W: weight vector; has shape (D, 1)
        b: bias; a scalar
        
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
        self.params['b'] = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
    
    def sigmoid(self,x):
        """
        Sigmoid function that transforms the input to values ranging from 0 to 1

        Inputs:
        - x: (float) Any scalar/vector/matrix.

        Outputs:
        The sigmoid activations given the input.
        """

        #############################################################################
        # TODO: Implement the sigmoid / logistic function.                          #
        #############################################################################
        
        sig = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return sig    

    def predict(self, X):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        W, b = self.params['W'], self.params['b']

        #############################################################################
        # TODO: Compute for the predictions of the model on new data using the      #
        # learned weight vectors.                                                   #
        #############################################################################
    
        prediction = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
        return prediction

    def binary_cross_entropy(self, probs, labels):
        """
        Cross entropy for two outcomes. (Can be thought of as a measure of misclassification)

        Inputs:
        - probs: (float) probability of belonging to class 1
        - labels: (integer) ground truth label.

        Outputs:
        The sigmoid activations given the input.
        """
        #############################################################################
        # TODO: Implement binary cross entropy                                      #
        #############################################################################
        
        bce = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################



        return bce

    def loss(self, X, y=None):
        """
        Compute the loss and gradients for an iteration of logistic regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the ground truth label for X[i].

        Returns:
        Return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W, b = self.params['W'], self.params['b']
        N, D = X.shape

        #############################################################################
        # TODO: Compute for the predicted probabilities of belonging to class 1     #
        # given the current weights and bias.                                       #
        #############################################################################

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

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the derivatives of the weights and biases. Store the        #
        # results in the grads dictionary. For example, grads['W'] should store     #
        # the gradient on W, and be a matrix of same size.                          #
        #############################################################################
        

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train Logistic Regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N, 1) containing the ground truth labels.
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        # Implement the initiaize_weights function.
        self.initialize_weights(dim)

        loss_history = []
        for it in range(num_iters):

            indices = np.random.choice(num_train,batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            loss, grads = self.loss(X_batch, y=y_batch)

            if it % 100 == 0:
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




    
    



