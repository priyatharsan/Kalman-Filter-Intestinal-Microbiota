"""
=====
Utils
=====
This module implements miscellaneous utility functions
"""
import numpy as np
from scipy import linalg


def log_multivariate_normal_density(x, means, covars):
    """
    Calculate the log probability density of a point in a multivariant distribution

    Parameters
    ----------
    x: [dimension] numpy array
         the point at which we are estimating the density
    means: [dimension] numpy array
        the mean of the multivariant normal distribution
    covars: [dimension, dimension] numpy array
        the covariance matrix of the multivariate normal distribution,

    Returns
    -------
    log_prob: numpy float
        the log probability density, with natural number as base
    """
    # asserting the input has the right dimensions
    num_dimensions = x.shape[0]
    assert (means.shape[0] == num_dimensions)
    assert (covars.shape == (num_dimensions, num_dimensions))

    # forcing the covariant matrix to be symmetric
    covars = (covars + covars.T) / 2
    log_prob = - 1 / 2 * np.log(linalg.det(covars)) \
               - 1 / 2 * (x - means).T.dot(linalg.inv(covars)).dot(x - means) \
               - 1 / 2 * means.shape[0] * np.log(2 * np.pi)

    return log_prob


def apply_mask(X, mask):
    """
    Apply masks (treat as missing) to measurements

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
        None for missing measurements
    mask: [seq_count] array of [seq_length] array of boolean
        'mask[seq_idx][time_step]': whether [seq_idx][time_step] is not missing
    """
    # check whether X and mask has the same number of sequences
    seq_count = len(X)
    assert (len(mask) == seq_count)

    for seq_idx in range(seq_count):
        # check whether each sequence and mask have the same length
        assert (len(X[seq_idx]) == len(mask[seq_idx]))

        # apply masks
        for time_step in range(len(X[seq_idx])):
            if not mask[seq_idx][time_step]:
                X[seq_idx][time_step] = None


def extract_mask(X):
    """
    Extract masks from measurements

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
        None for missing measurements

    Returns
    -------
    mask: [seq_count] array of [seq_length] array of boolean
        'mask[seq_idx][time_step]': whether [seq_idx][time_step] is not missing
    """
    mask = [[(X[seq_idx][time_step] is not None)
             for time_step in range(len(X[seq_idx]))]
            for seq_idx in range(len(X))]
    return mask


def get_dimension(X):
    """
    Get the dimension of the elements of X

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
        None for missing measurements

    Returns
    -------
    x.shape[0]: int
        dimension of measurement
    """
    for x in X[0]:
        if x is not None:
            return x.shape[0]
    print('In the first sequence, all measurements are missing')


def get_control_dimension(U):
    """
    Get the control dimension of U

    Parameters
    ----------
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        'U[seq_idx][time_step]' = the control of the seq_idx sequence at time time_step

    Returns
    -------
    U[0][0].shape[0]: int
        dimension of control
    """
    return U[0][0].shape[0]


def pad_with_most_recent_measurements(X):
    """
    Pad the measurements with most recent measurements
    The argument is not changed in this process
    Assert that the first measurement is not missing

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
        None for missing measurements

    Returns
    -------
    X_padded: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        'X[seq_idx][time_step]' = the latest measurement of the seq_idx sequence at time time_step
    """
    X_padded = []

    # pad every sequence
    for seq_idx in range(len(X)):
        X_single_seq = []
        assert (X[seq_idx][0] is not None)
        most_recent_measurement = X[seq_idx][0]

        # find the most recent measurement for every time_step
        for time_step in range(len(X[seq_idx])):
            if X[seq_idx][time_step] is not None:
                most_recent_measurement = X[seq_idx][time_step]
            X_single_seq.append(np.array(most_recent_measurement))
        X_padded.append(X_single_seq)
    return X_padded


def write_otu(X, otu_file, idx2bacteria, id2start_date, program_id2file_id):
    """
    Write X to the otu_file in the otu table format

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        'X[seq_idx][time_step]' = the (inferred) frequency of the seq_idx sequence at time time_step
    otu_file: String
        the file name of the otu Table file
    idx2bacteria: an array of String
        a map from a dimension in the numpy array to its corresponding 'bacteria cluster'
    id2start_date: {int: int}
        a map from patientId (starting from 0) to the starting measurement date
        relative to the reference date
    program_id2file_id: {int: int}
        a map from the index in X and U to the actual patient id in the original file
    """
    out_file = open(otu_file, 'w')

    # write the patient id line
    id_line = 'patientId\t'
    for _ in range(len(X)):
        id = program_id2file_id[_]
        id_line += (str(id) + '\t') * len(X[_])
    id_line += '\n'
    out_file.write(id_line)

    # write the time line
    time_line = 'time\t'
    for _ in range(len(X)):
        start_date = id2start_date[_]
        for time_step in range(len(X[_])):
            time_line += str(start_date + time_step) + '\t'
    time_line += '\n'
    out_file.write(time_line)

    # write the microbiiome
    for idx in range(len(idx2bacteria)):
        bacertia_name = idx2bacteria[idx]
        rel_abundance_line = bacertia_name + '\t'
        for _ in X:
            for __ in _:
                rel_abundance_line += str(__[idx]) + '\t'
        rel_abundance_line += '\n'
        out_file.write(rel_abundance_line)
    out_file.close()
