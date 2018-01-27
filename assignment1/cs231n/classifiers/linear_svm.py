import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # C
  num_train = X.shape[0] # N

  loss = 0.0
  for i in xrange(num_train):
    Xi = X[i];
    scores = Xi.dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
        if j == y[i]:
            continue;

        margin = scores[j] - correct_class_score + 1 # note delta = 1
              
        if margin > 0:
            loss += margin
            dW[:,j] += Xi
            dW[:,y[i]] -= Xi

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW +=  2.0 * reg * np.sum(W * W)
    
  # For the comparison with the numerical difference, the following will lead to similar erro as this function
  # is also used for the numerical evaluation.
  # loss += 0.5*reg * np.sum(W * W)
  # dW +=  reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means

  Inputs and outputs are the same as svm_loss_naive.
  """
  N = X.shape[0]
  C = W.shape[1]
  delta = 1

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_score = scores[range(N), y]
  correct_class_score = np.tile(correct_class_score.reshape(-1,1), C) 
  
  margin = scores - correct_class_score + delta
    
  margin[range(N), y] = 0 # Y_i class won't contribute to the loss
  margin[margin < 0] = 0 # max function, set margin to 0 if negative
    
  loss = np.sum(margin)/N + reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  I = margin
  I[I > 0] = 1
  row_count = np.sum(I, axis=1)
  I[range(N), y] = -row_count
  
  dW = X.T.dot(I)
  dW = dW/N + 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
