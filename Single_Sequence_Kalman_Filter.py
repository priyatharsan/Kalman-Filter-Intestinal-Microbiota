"""
============================================
Kalman Filter functions on a single sequence
============================================

This Module implements the SSKF (single sequence Kalman Filter) class
It supports:
i) inference (predict, filter, smooth)
ii) prepare for maximization (z_0_hat, P_0_hat, A, B, Q, R)
iii) simulation
iv) calculate log likelihood
"""
import numpy as np
from numpy.random import multivariate_normal
from scipy import linalg
from Utils import log_multivariate_normal_density

class SSKF:
    """
    Implement the single sequence Kalman Filter class
    A basic building block for the Kalma Filter related algorithms
    Most algorithms (and math) can be seen here
    """

    def __init__(self, _dimension, _control_dimension, _z_0_hat, _P_0_hat, _A, _B, _Q, _R):
        """
        Initialize the SSKF class

        Parameters
        ----------
        _dimension: int
            dimension of x (measurement)
        _control_dimension: int
            dimension of u (control)
        _z_0_hat: [dimension] numpy array
            initial mean for every sequence
        _P_0_hat: [dimension, dimension] numpy array
            initial covariance for every sequence
        _A: [dimension, dimension] numpy array
            transition matrix
        _B: [dimension, control_dimension] numpy array
            control matrix
        _Q: [dimension, dimension] numpy array
            transition covariance matrix
        _R: [dimension, dimension] numpy array
            emission covariance matrix
        """
        self.dimension, self.control_dimension = _dimension, _control_dimension
        self.z_0_hat, self.P_0_hat = _z_0_hat, (_P_0_hat + _P_0_hat.T) / 2
        self.A, self.B, self.Q, self.R = _A, _B, (_Q + _Q.T) / 2, (_R + _R.T) / 2

    def pass_in_measurement_and_control(self, measurements, controls):
        """
        Set the observed data and control matrix for this single sequence Kalman Filter

        Parameters
        ----------
        measurements: [seq_length] of ([dimension] numpy array/None)
            'measurements[time_step]' = the measurement at time time_step
            None for missing measurements
        controls: [seq_length] of ([dimension] numpy array)
            'controls[time_step]' = the control at time time_step
        """
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
        # assert control and measurements have equal length
        self.seq_length = len(self.X)
        assert(len(self.U) == self.seq_length)
        
        # number of measurements in X (not None/missing)
        self.n_obs = sum([1 if x is not None else 0 for x in self.X])

    def simulate(self, controls):
        """
        Simulate the X and Z according to its parameters

        Parameters
        ----------
        controls: [seq_length] of ([dimension] numpy array)
            'controls[time_step]' = the control at time time_step

        Returns
        -------
        Z: [seq_length] array of ([dimension] numpy array)
            'Z[time_step]' = the true value of the hidden variable of the seq_idx sequence at time time_step
        X: [seq_length] array of ([dimension] numpy array/None)
            'X[time_step]' = the measurement at time time_step
        """
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

    def get_offsets(self):
        """
        Calculate the offsets introduced by B and U
        We can substract the offset from measurement so that we can transform it to a
        standard Kalman Filter Inference problem without B

        Returns
        -------
        offsets: the offsets introduced by B and U
        """
        # define simpler names for the parameters
        A, B, U = self.A, self.B, self.U
        offsets = [np.zeros(self.dimension)]
        cur = np.copy(offsets[0])
        for i in range(self.seq_length - 1):
            cur = A.dot(cur) + B.dot(U[i])
            offsets += [np.copy(cur)]
        return offsets

    def filter(self):
        """
        Filtering steps of the Kalman Filter algorithm
        All the posterior distributions are Gaussian
        """
        # define simpler names in this function
        A, B, U, X = self.A, self.B, self.U, self.X

        # calculate offsets due to B and U and reduce the inference to a standard Kalman Filter problem
        # without the B matrix
        self.O = self.get_offsets()
        X_prime = [(None if (X[i] is None) else (X[i] - self.O[i])) for i in range(self.seq_length)]

        # z: mean of the Gaussian Distribution
        # P: covariance of the Gaussian Distribution
        # z_t_t: the mean of z_t given x1, x2 ... xt, naming convention same for other variables
        # tp1: t + 1
        # K: Kalman gain
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

    def smooth(self):
        """
        The smooth step of the Kalman Filter algorithm
        """
        # initialize the "backward" smoothing
        # T means "all the measurements"
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

    def calculate_covariances(self):
        """
        Calculate values necessary for the maximization step
        """
        # covtt: covariance of z_t
        self.covtt = [self.P_t_T_array[time_step] for time_step in range(self.seq_length)]
        # covttp1: covariance of z_t and z_tp1
        self.covttp1 = [(self.L_t_array[time_step].dot(self.P_t_T_array[time_step + 1]))
                        for time_step in range(self.seq_length - 1)]
        # zu: the concatenation of z and u
        self.zu = [(np.concatenate((self.smoothed_Z[time_step], self.U[time_step])))
                   for time_step in range(self.seq_length - 1)]

    def prepare_z_0_hat(self):
        """
        Prepare the variables needed to maximize over z_0_hat for multiple sequences
        """
        return np.array(self.smoothed_Z[0])

    def prepare_P_0_hat(self):
        """
        Prepare the variables needed to maximize over P_0_hat for multiple sequences
        """
        return self.covtt[0] + np.outer(self.smoothed_Z[0] - self.z_0_hat,
                                        self.smoothed_Z[0] - self.z_0_hat)

    def prepare_AB(self):
        """
        Prepare the variables needed to maximize over A, B for multiple sequences
        """
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

    def prepare_Q(self):
        """
        Prepare the variables needed to maximize over Q for multiple sequences
        """
        self.state_cov = sum([(np.outer(
                                        self.smoothed_Z[i + 1] - self.A.dot(self.smoothed_Z[i]) - self.B.dot(self.U[i]),
                                        self.smoothed_Z[i + 1] - self.A.dot(self.smoothed_Z[i]) - self.B.dot(self.U[i]))
                                        + self.A.dot(self.covtt[i]).dot(self.A.T)
                                        + self.covtt[i + 1]
                                        - self.A.dot(self.covttp1[i])
                                        - self.A.dot(self.covttp1[i]).T) for i in range(self.seq_length - 1)])
        return self.state_cov

    def prepare_R(self):
        """
        Prepare the variables needed to maximize over R for multiple sequences
        """
        self.emission_cov = sum([(np.zeros((self.dimension, self.dimension)))
                                 if (self.X[i] is None)
                                 else (np.outer(self.X[i] - self.smoothed_Z[i],
                                                self.X[i] - self.smoothed_Z[i]) + self.covtt[i])
                                 for i in range(self.seq_length)])
        return self.emission_cov

    def log_likelihood(self, normalize=True):
        """
        Calculate the log likelihood of the measurements given the current parameters

        Parameters
        ----------
        normalize: boolean
            whether to return the log likelihood normalized by the number of measurements

        Returns
        -------
            seq_log_prob: numpy float
                the sum of log probabilities of the entire sequence
        """
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
