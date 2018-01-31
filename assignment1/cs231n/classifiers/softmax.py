import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    D, C = W.shape
    N = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(N): # Loop through each sample
        xi = X[i, :]
        yi = y[i]
        
        score = xi.dot(W) # shape (1, C) for scores
        score -= np.max(score) # shift max score to 0
        
        sexp = np.exp(score) # exponential score
        sexp /= np.sum(sexp) # normalized exponential score
        
        loss += - math.log(sexp[yi])
        
        for j in range(C):
            dW[:, j] += sexp[j] * xi.T
            
            if j == yi:
                dW[:, j] += - xi.T
            
    # Average over all samples and add regulation term        
    loss = loss / N + reg * np.sum(W * W) 
    dW = dW / N + 2* reg * W     
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    D, C = W.shape
    N = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    F = X.dot(W) # shape (N, C) score matrix
    offSet = np.reshape(np.max(F, axis=1), (N, 1))
    F -= np.tile(offSet, (1, C)) # shift max score of each sample to zero
    
    P = np.exp(F)
    sums = np.reshape(np.sum(P, axis=1), (N, 1))
    P /= np.tile(sums, (1, C)) # normalize exp score matrix
    
    Prob = P[range(N), y] # shape (N, 1) probability vector of the true class
    
    # Total loss
    loss = np.sum(-np.log(Prob))/N + reg * np.sum(W*W)
    
    # Gradient 
    I = np.zeros(F.shape) # shape (N, C) indicator matrix for the true class label
    I[range(N), y] = np.ones(N)
    
    dW = (X.T).dot(-I + P)/N + 2*reg * W 
    # --- Analytical derivatives is given in the softmax.ipynb notebook

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

