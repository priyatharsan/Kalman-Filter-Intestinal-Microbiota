import numpy as np
from numpy.random import multivariate_normal
from scipy import linalg

'''
    calculate the log probability density of a point in a multivariant distribution
    
    # Arguments
        x: the point at which we are estimating, 1D numpy array
        means: the mean of the multivariate normal distribution, 1D numpy array
        covars: the covariance matrix of the multivariate normal distribution, 2D numpy array
    # Return
        log_prob: the log probability density, with natural number as base
'''
def log_multivariate_normal_density(x, means, covars):
    
    # asserting the input has the right dimensions
    num_dimensions = x.shape[0]
    assert(means.shape[0] == num_dimensions)
    assert(covars.shape == (num_dimensions, num_dimensions))
    
    # forcing the covariant matrix to be symmetric
    covars = (covars + covars.T) / 2
    log_prob = - 1 / 2 * np.log(linalg.det(covars)) \
        - 1 / 2 * (x - means).T.dot(linalg.inv(covars)).dot(x - means) \
        - 1 / 2 * means.shape[0] * np.log(2 * np.pi)
    
    return log_prob

'''
    Apply masks to observations
    
    # Arguments
        X: a 2D list of observations
        mask: a 2D list of boolean array, True for unmasked (observed) values, False for unmasked (missing) values
'''
def apply_mask(X, mask):
    
    # check whether X and mask has the same number of sequences
    seq_count = len(X)
    assert(len(mask) == seq_count)
    
    for seq_idx in range(seq_count):
        # check whether each sequence and mask have the same length
        assert(len(X[seq_idx]) == len(mask[seq_idx]))
        
        # apply masks
        for time_step in range(len(X[seq_idx])):
            if not mask[seq_idx][time_step]:
                X[seq_idx][time_step] = None
'''
    Apply masks to observations
    
    # Arguments
        X: a 2D list of observations
    #Return
        mask: a 2D list of booleans
'''
def extract_mask(X):
    mask =  [[(X[seq_idx][time_step] is not None)
            for time_step in range(len(X[seq_idx]))]
            for seq_idx in range(len(X))]
    return mask

'''
    Get the dimension of X
    
    # Arguments
        X: a 2D array of measurements
    # Return
        x.shape[0]: the dimension of measurements
'''
def get_dimension(X):
    for x in X[0]:
        if x is not None:
            return x.shape[0]
    print('In the first sequence, all observations are missing')

'''
    Get the control dimension of U
    
    # Arguments
        U: a 2D array of control
    # Return
        U[0][0].shape[0]: the controal dimension of control
'''
def get_control_dimension(U):
    return U[0][0].shape[0]

'''
    Pad the measurements with most recent observations
    The argument is not changed in this process
    Assert that the first observation is not null
    
    # Arguments
        X: measurements with missing observations (if no observations missing, then the returned value will be the same with X)
    # Returns
        X_padded: the measurements where missing observations are all padded with most recent observations
'''
def pad_with_most_recent_observations(X):
    X_padded = []
    
    # pad every sequence
    for seq_idx in range(len(X)):
        X_single_seq = []
        assert(X[seq_idx][0] is not None)
        most_recent_observation = X[seq_idx][0]
        
        # find the most recent observation for every time_step
        for time_step in range(len(X[seq_idx])):
            if X[seq_idx][time_step] is not None:
                most_recent_observation = X[seq_idx][time_step]
            X_single_seq.append(np.array(most_recent_observation))
        X_padded.append(X_single_seq)
    return X_padded
