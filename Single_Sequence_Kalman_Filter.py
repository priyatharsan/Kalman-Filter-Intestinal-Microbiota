# Author: Ruiqi Zhong
# A building block of the multiple sequence Kalman Filter model

import numpy as np
import math
from numpy.random import multivariate_normal
from scipy import linalg
from utils import log_multivariate_normal_density

class SSKF:
    
    '''
        Initializer of the Single_Sequence_Kalman_Filter class
        
        # Arguments
            _dimension: dimension of x
            _control_dimension: dimension of u
            _A: transition matrix
            _B: control matrix
            _Q: transition covariance matrix
            _R: emission matrix
            _z_0_hat: the mean for the initial state as a Guassian distribution
            _p_0_hat: the covariance for the initial state as a Gaussian distribution
    '''
    def __init__(self, _dimension, _control_dimension, _z_0_hat, _P_0_hat, _A, _B, _Q, _R):
        self.dimension, self.control_dimension = _dimension, _control_dimension
        self.z_0_hat, self.P_0_hat = _z_0_hat, (_P_0_hat + _P_0_hat.T) / 2
        self.A, self.B, self.Q, self.R = _A, _B, (_Q + _Q.T) / 2, (_R + _R.T) / 2
    
    '''
        Set the observed data and control matrix for this single sequence Kalman Filter
        
        # Arguments
            measurements: single sequence observed X, equivalent to self.X
            controls: single sequence control U, equivalent to self.U
        # Dimensions
            measurements: an array seq_length of numpy array/None, each of dimension self.dimension
            controls: an array seq_length of numpy array/None, each of dimension self.control_dimension
    '''
    def pass_in_measurement_and_control(self, measurements, controls):
        # setting the X and U
        self.X = measurements
        self.U = controls
        
        # checking the dimensions of X
        for x in self.X:
            if x is not None:
                assert(x.shape[0] == self.dimension)
    
        # checking the dimensions of U
        for u in self.U:
            assert(u.shape[0] == self.control_dimension)
        
        # setting the length of the sequence
        # assert control and observations have equal length
        self.seq_length = len(self.X)
        assert(len(self.U) == self.seq_length)
        
        # number of observations in X (not None/missing)
        self.n_obs = sum([1 if x is not None else 0 for x in self.X])

    '''
        Simulate the X and Z according to its parameters
        
        # Arguments
            controls: the control matrix U
        # Return
            self.X, self.Z: the simulated observation and hidden true state
    '''
    def simulate(self, controls):
        simulated_seq_length = len(controls)
        simulated_Z, simulated_X = [], []
        
        # initial Z and initial X
        cur_Z = multivariate_normal(self.z_0_hat, self.P_0_hat)
        simulated_Z.append(cur_Z)
        cur_X = multivariate_normal(cur_Z, self.R)
        simulated_X.append(cur_X)
        
        # evolution step based on Kalman Filter model
        for time_step in range(simulated_seq_length - 1):
            cur_Z = multivariate_normal(self.A.dot(cur_Z) + self.B.dot(controls[time_step]), self.Q)
            simulated_Z.append(cur_Z)
            simulated_X.append(multivariate_normal(cur_Z, self.R))
        return simulated_Z, simulated_X
    
    '''
        Calculate the offsets introduced by B and U
        We can substract the offset from observation so that we can transform it to a
        standard Kalman Filter Inference problem without B
        
        # Returns
            offsets: the offsets introduced by B and U
    '''
    def get_offsets(self):
        # define simpler names for the parameters
        A, B, U = self.A, self.B, self.U
        offsets = [np.zeros(self.dimension)]
        cur = np.copy(offsets[0])
        for i in range(self.seq_length - 1):
            cur = A.dot(cur) + B.dot(U[i])
            offsets += [np.copy(cur)]
        return offsets

    '''
       Filtering steps of the Kalman Filter algorithm
       All the posterior distributions are Gaussian
       
       # Variable Names:
            z: mean of the Gaussian Distribution
            P: covariance of the Gaussian Distribution
            z_t_t: the mean of z_t given x1, x2 ... xt, naming convention same for other variables
            tp1: t + 1
            K: Kalman gain
    '''
    def filter(self):
        # define simpler names in this function
        A, B, U, X = self.A, self.B, self.U, self.X
        # calculate offsets due to B and U and reduce the inference to a standard Kalman Filter problem
        # without the B matrix
        self.O = self.get_offsets()
        X_prime = [(None if (X[i] is None) else (X[i] - self.O[i])) for i in range(self.seq_length)]
        
        # initialize the varaibles
        self.z_t_t_array, self.P_t_t_array, self.K_tp1 = [], [], []
        self.z_tp1_t_array, self.P_tp1_t_array = [self.z_0_hat], [self.P_0_hat]
        # define simpler names for variables
        z_t_t_array, P_t_t_array, K_tp1, z_tp1_t_array, P_tp1_t_array = \
            self.z_t_t_array, self.P_t_t_array, self.K_tp1, self.z_tp1_t_array, self.P_tp1_t_array
        
        # forward filtering step begins
        for time_step in range(self.seq_length):
            
            # filtered values
            z_t_t, P_t_t = None, None
            if X_prime[time_step] is not None:
                K_tp1.append(P_tp1_t_array[time_step].dot((linalg.pinv((P_tp1_t_array[time_step]
                                                                        + self.R)))))
                z_t_t = (z_tp1_t_array[time_step]
                         + K_tp1[time_step].dot(X_prime[time_step] - z_tp1_t_array[time_step]))
                P_t_t = P_tp1_t_array[time_step] - K_tp1[time_step].dot(P_tp1_t_array[time_step])
            
            # if data is missing, then the filtered mean/covariance is the same as the predicted ones
            else:
                K_tp1.append(np.zeros((self.dimension, self.dimension)))
                z_t_t = np.copy(z_tp1_t_array[time_step])
                P_t_t = np.copy(P_tp1_t_array[time_step])
            
            z_t_t_array.append(z_t_t)
            P_t_t_array.append(P_t_t)
            
            # predicted values
            if (time_step < self.seq_length - 1):
                z_tp1_t = A.dot(z_t_t_array[time_step])
                P_tp1_t = A.dot((P_t_t_array[time_step]).dot(A.T)) + self.Q
                z_tp1_t_array.append(z_tp1_t)
                P_tp1_t_array.append(P_tp1_t)
        
        # add the offset back
        self.predicted_Z = [(self.O[i] + z_tp1_t_array[i]) for i in range(self.seq_length)]
        self.filtered_Z = [(self.O[i] + z_t_t_array[i]) for i in range(self.seq_length)]
        
        self.predicted_P = P_tp1_t_array
        self.predicted = {'mean': self.predicted_Z, 'covariance': self.predicted_P}
        self.filtered_P = P_t_t_array
        self.filtered = {'mean': self.filtered_Z, 'covariance': self.filtered_P}

    '''
        The smooth step of the Kalman Filter algorithm
        
        # Variable Names:
            T (capital): given all the observations
    '''
    def smooth(self):
        # initialize the "backward" smoothing
        self.z_t_T_array, self.P_t_T_array, self.L_t_array = [self.z_t_t_array[-1]], [self.P_t_t_array[-1]], []
        # define simpler names in this function
        A, B, U, X = self.A, self.B, self.U, self.X
        
        # smoothing starts
        for _ in range(self.seq_length - 1):
            time_step = self.seq_length - _ - 2
            L_t = self.P_t_t_array[time_step].dot((A.T).dot(linalg.pinv(self.P_tp1_t_array[time_step + 1])))
            self.L_t_array = [L_t] + self.L_t_array
            z_t_T = self.z_t_t_array[time_step] + L_t.dot((self.z_t_T_array[0]
                                                           - self.z_tp1_t_array[time_step + 1]))
            P_t_T = self.P_t_t_array[time_step] + L_t.dot(((self.P_t_T_array[0]
                                                            - self.P_tp1_t_array[time_step + 1])).dot(L_t.T))
            self.z_t_T_array, self.P_t_T_array = [z_t_T] + self.z_t_T_array, [P_t_T] + self.P_t_T_array

        self.smoothed_Z = [(self.O[i] + self.z_t_T_array[i]) for i in range(self.seq_length)]
        self.smoothed_P = self.P_t_T_array
        self.smoothed = {'mean': self.smoothed_Z, 'covariance': self.smoothed_P}
        
    '''
        Calculate values necessary for the maximization step
        
        # Variable Names:
            covtt: covariance of z_t
            covttp1: covariance of z_t and z_tp1
            zu: the concatenation of z and u
    '''
    def calculate_covariances(self):
        self.covtt = [self.P_t_T_array[time_step] for time_step in range(self.seq_length)]
        self.covttp1 = [(self.L_t_array[time_step].dot(self.P_t_T_array[time_step + 1]))
                        for time_step in range(self.seq_length - 1)]
        self.zu = [(np.concatenate((self.smoothed_Z[time_step], self.U[time_step])))
                   for time_step in range(self.seq_length - 1)]

    '''
        Return the variables needed to maximize over z_0_hat for multiple sequences
        
        # Returns
        smoothed_Z[0]: E[z_0_T]
    '''
    def prepare_z_0_hat(self):
        return np.array(self.smoothed_Z[0])


    '''
        Return the variables needed to maximize over P_0_hat for multiple sequences
        
        # Returns
            smoothed_P[0]: E[P_0_T]
    '''
    def prepare_P_0_hat(self):
        return self.covtt[0] + np.outer(self.smoothed_Z[0] - self.z_0_hat,
                                        self.smoothed_Z[0] - self.z_0_hat)
    
    '''
        Return the variables needed to maximize over A and B for multiple sequences
        
        # Returns
            zutztp1: E[zu (dot product) ztp1]
            zutzut: E[zu (dot product) zu]
    '''
    def prepare_AB(self):
        self.zutztp1 = sum([np.concatenate((np.outer(self.smoothed_Z[i], self.smoothed_Z[i + 1])
                                            + self.covttp1[i],
                                            np.outer(self.U[i], self.smoothed_Z[i + 1])))
                            for i in range(self.seq_length - 1)])
        self.zutzut = np.zeros((self.dimension + self.control_dimension, self.dimension + self.control_dimension))
        
        for i in range(self.seq_length - 1):
            # pad the covariance term with 0 to so that dimension matches
            # covar(u, z) and covar (u, u) are 0, since u is constant
            padded_covtt = np.zeros((self.dimension + self.control_dimension, self.dimension + self.control_dimension))
            padded_covtt[:self.dimension, :self.dimension] = self.covtt[i]
            # square of expectation term + covariance
            self.zutzut += np.outer(self.zu[i], self.zu[i]) + padded_covtt
        return self.zutztp1, self.zutzut

    '''
        Return the variables needed to maximize over Q for multiple sequences
        
        # Returns
            state_cov: the sum of covariance of the state transition noise
    '''
    def prepare_Q(self):
        self.state_cov = sum([(np.outer(
                                        self.smoothed_Z[i + 1] - self.A.dot(self.smoothed_Z[i]) - self.B.dot(self.U[i]),
                                        self.smoothed_Z[i + 1] - self.A.dot(self.smoothed_Z[i]) - self.B.dot(self.U[i]))
                                        + self.A.dot(self.covtt[i]).dot(self.A.T)
                                        + self.covtt[i + 1]
                                        - self.A.dot(self.covttp1[i])
                                        - self.A.dot(self.covttp1[i]).T) for i in range(self.seq_length - 1)])
        return self.state_cov
    
    '''
        Return the variables needed to maximize over R for multiple sequences
        
        # Returns
            emission_cov: the sum of covariance of the emission/measurement noise
    '''
    def prepare_R(self):
        self.emission_cov = sum([(np.zeros((self.dimension, self.dimension)))
                                 if (self.X[i] is None)
                                 else (np.outer(self.X[i] - self.smoothed_Z[i],
                                                self.X[i] - self.smoothed_Z[i]) + self.covtt[i])
                                 for i in range(self.seq_length)])
        return self.emission_cov

    '''
        Return the log likelihood of the observations given the current parameters
        
        # Argument
            normalize: return the log likelihood normalized by the number of observations
        # Return
            seq_log_prob: the sum of log probabilities of the entire sequence
    '''
    def log_likelihood(self, normalize=True):
        seq_log_prob = 0
        for time_step in range(self.seq_length):
            predicted_mean = self.predicted_Z[time_step]
            predicted_covariance = self.predicted_P[time_step] + self.R
            if self.X[time_step] is not None:
                log_prob = log_multivariate_normal_density(
                                                    self.X[time_step],
                                                    predicted_mean,
                                                    predicted_covariance
                                                    )
                seq_log_prob += log_prob
    
        if normalize:
            return seq_log_prob / self.n_obs
        return seq_log_prob
