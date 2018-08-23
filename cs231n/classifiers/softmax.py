import numpy as np
from random import shuffle
from sklearn.preprocessing import normalize

'''
Clarification beforehand:
naive version of code means doing vector operations elementwise with loops or so,
while vectorized codes do vector operations directly based on vectors.
However we do not distinguish between these to styles of coding in this version
'''

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
	
	# Initialize the loss and gradient to zero.
	"""
	loss = 0.0
	dW = np.zeros_like(W)
	"""
	# minibatch size is N, the feature vector is row vector with size 1*D
	# the label vector per minibatch is a column vector with size N*1
	# Thus the setting of some hyperparameters:
	MINIBAT_SIZE, FEATURE_LEN = X.shape
	_, CLASS_NUM = W.shape
	lambdA = 0.5	# hyperparam of regularization term in loss
	learning_rate = 1e-4
	epoch_num = 200
	reg_stg = 1e-3	# regularization strength

	for epoch in range(1, epoch_num+1):
		# Hypothesis function, note that the bias term has been added into the input X
		# which takes the form of adding one column with value of 1 to X
		# The score vectors derived from this function H stands for the unnormalized
		# log probabilities for each class
		# Which I strongly suspect that such approach may be problematic
		score = np.exp( np.dot(X, W) )
		probs = normalize(score, norm='l1')

		# Loss:
		# After dot producting X with W, we get a matrice with shape (N, C) for every example in the minibatch, 
		# we get a vector, whose value represents the scores for every class this example can be graded and 
		# thus classified
		# One particular caveat to attention is that during training process, we believe that we only train
		# W with C-1 classes rather than C classes out of the preoccupation that the final score being derived
		# from 1-sum(other scores)
		
		# data_loss = np.zeros(MINIBAT_SIZE)
		data_loss = ( -np.log(probs[range(MINIBAT_SIZE),y]) ).mean()
		# data_loss = data_loss.mean()
		reg_loss = 0.5 * lambdA * np.sum(W*W)
		loss = data_loss + reg_loss

		# compute gradient
		probs[range(MINIBAT_SIZE), y] -= 1
		dW = np.dot(X.T, probs)
		dW += reg_stg * lambdA*np.sum(W)

		# gradient update
		W += learning_rate * dW
	"""


		#############################################################################
		# TODO: Compute the softmax loss and its gradient using explicit loops.     #
		# Store the loss in loss and the gradient in dW. If you are not careful     #
		# here, it is easy to run into numeric instability. Don't forget the        #
		# regularization!                                                           #
		#############################################################################
	pass
		#############################################################################
		#                          END OF YOUR CODE                                 #
		#############################################################################

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################

	# Hyperparameters Settings:
	MINIBAT_SIZE, FEATURE_LEN = X.shape
	_, CLASS_NUM = W.shape
	lambdA = 0.5	# hyperparam of regularization term in loss
	learning_rate = 1e-4
	epoch_num = 200
	reg_stg = 1e-3	# regularization strength

	for epoch in range(1, epoch_num+1):
		# Hypothesis function, note that the bias term has been added into the input X
		# which takes the form of adding one column with value of 1 to X
		# The score vectors derived from this function H stands for the unnormalized
		# log probabilities for each class
		# Which I strongly suspect that such approach may be problematic
		score = np.exp( np.dot(X, W) )
		probs = normalize(score, norm='l1')

		# Loss:
		'''
		After dot producting X with W, we get a matrice with shape (N, C) for every example in the minibatch, 
		we get a vector, whose value represents the scores for every class this example can be graded and 
		thus classified
		One particular caveat to attention is that during training process, we believe that we only train
		W with C-1 classes rather than C classes out of the preoccupation that the final score being derived
		from 1-sum(other scores)
		'''
		# data_loss = np.zeros(MINIBAT_SIZE)
		data_loss = ( -np.log(probs[range(MINIBAT_SIZE),y]) ).mean()
		# data_loss = data_loss.mean()
		reg_loss = 0.5 * lambdA * np.sum(W*W)
		loss = data_loss + reg_loss

		# compute gradient
		probs[range(MINIBAT_SIZE), y] -= 1
		dW = np.dot(X.T, probs)
		dW += reg_stg * lambdA*np.sum(W)

		# gradient update
		W += learning_rate * dW
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW

