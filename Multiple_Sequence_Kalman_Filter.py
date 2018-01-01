import numpy as np
import math
from numpy.random import multivariate_normal
from Single_Sequence_Kalman_Filter import SSKF
from scipy import linalg
import sys
from EM_Config import EM_Config

class MSKF:

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
    
    # Variables
        max_likelihood: the max likelihood every found
        max_likelihood_parameters: the parameters that achieve maximum likelihood till now
        SSKFs: an array of single sequence Kalman Filter that stores the data
        current_estimate: whether the sskfs stores the inferred values according to the current variables
    '''
    def __init__(self, _dimension, _control_dimension, _z_0_hat=None, _P_0_hat=None, _A=None, _B=None, _Q=None, _R=None):
        self.dimension = _dimension
        self.control_dimension = _control_dimension
        if _z_0_hat is None:
            _z_0_hat = np.zeros(self.dimension)
        if _P_0_hat is None:
            _P_0_hat = np.eye(self.dimension)
        if _A is None:
            _A = np.eye(self.dimension)
        if _B is None:
            _B = np.zeros((self.dimension, self.control_dimension))
        if _Q is None:
            _Q = np.eye(self.dimension)
        if _R is None:
            _R = np.eye(self.dimension)
        
        # setting parameters
        self.z_0_hat, self.P_0_hat, self.A, self.B, self.Q, self.R = _z_0_hat, _P_0_hat, _A, _B, _Q, _R
        
        # check shape
        assert(self.z_0_hat.shape == (self.dimension,))
        assert(self.P_0_hat.shape == (self.dimension, self.dimension))
        assert(self.A.shape == (self.dimension, self.dimension))
        assert(self.B.shape == (self.dimension, self.control_dimension))
        assert(self.Q.shape == (self.dimension, self.dimension))
        assert(self.R.shape == (self.dimension, self.dimension))
        
        self.SSKFs = []
        self.current_estimate = False

    '''
        Initialize all single sequence Kalman Filter and Set the observed data and control matrix
        
        # Arguments
            measurements: multiple sequences observed X, equivalent to self.X
            controls: multiple sequences control U, equivalent to self.U
        # Dimensions
            measurements: an array of (array seq_length of numpy array/None), each of dimension self.dimension
            controls: an array of *array seq_length of numpy array/None), each of dimension self.control_dimension
    '''
    def pass_in_measurement_and_control(self, measurements, controls):
        # check the number of sequence for control and measurements match
        self.seq_count = len(measurements)
        assert(len(controls) == self.seq_count)

        # initialize the #seq_count single sequence Kalman Filter
        # pass in measurements and controls for each of them
        for seq_idx in range(self.seq_count):
            self.SSKFs.append(SSKF(self.dimension, self.control_dimension,
                                   self.z_0_hat, self.P_0_hat, self.A, self.B, self.Q, self.R))
            self.SSKFs[seq_idx].pass_in_measurement_and_control(measurements[seq_idx], controls[seq_idx])

        # calculate relevant statistics
        self.seq_lengths = [sskf.seq_length for sskf in self.SSKFs]
        self.total_obs = sum([sskf.n_obs for sskf in self.SSKFs])
        self.total_time_steps = sum(self.seq_lengths)
        self.total_transition_count = self.total_time_steps - self.seq_count

    '''
        Estimation step for all hidden Z
        
        # Returns
            predicted: predicted values at each time step;
                        it is a dictionary with entries 'mean'/'covariance';
                        the values of the dictionary is a 2D array (seq_count * seq_length) of numpy array
            filtered: filtered values at each time step
            smoothed: smoothed values (the expectation) at each time step
    '''
    def estimate(self):
        predicted, filtered, smoothed = {'mean': [], 'covariance': []}, {'mean': [], 'covariance': []}, \
            {'mean': [], 'covariance': []}
        
        #perform inference/estimation for each single sequence Kalman filter
        for sskf in self.SSKFs:
            sskf.filter()
            sskf.smooth()
            sskf.calculate_covariances()
            
            # aggregate the result in the return dictionaries
            predicted['mean'].append(sskf.predicted['mean'])
            predicted['covariance'].append(sskf.predicted['covariance'])
            filtered['mean'].append(sskf.filtered['mean'])
            filtered['covariance'].append(sskf.filtered['covariance'])
            smoothed['mean'].append(sskf.smoothed['mean'])
            smoothed['covariance'].append(sskf.smoothed['covariance'])
        self.current_estimate = True
        return predicted, filtered, smoothed

    '''
        Calculate the log likelihood
        
        # Returns
            normalized_ll: normalized log likelihood of all the data
    '''
    def log_likelihood(self):
        if not self.current_estimate:
            self.estimate()
        multi_seq_prob = 0
        for sskf in self.SSKFs:
            multi_seq_prob += sskf.log_likelihood(normalize=False)
        normalized_ll = multi_seq_prob / self.total_obs
        return normalized_ll

    '''
        Maximization
        Update z_0_hat, P_0_hat, A, B, Q, R
    '''
    def maximize(self):
        self.update_z_0_hat()
        self.update_P_0_hat()
        self.update_AB()
        self.update_Q()
        self.update_R()
        self.current_estimate = False
    
    '''
        Calculate the z_0_hat for the maximization step and update
    '''
    def update_z_0_hat(self):
        
        # value fixed, no update
        if self.config['z_0_hat_option'] == 'fixed':
            return
    
        # argument not recognized
        if self.config['z_0_hat_option'] != 'flexible':
            print('WARNING: value %s not understood for key z_0_hat_option' % self.config['z_0_hat_option'])
        
        # update without constraint
        self.z_0_hat = sum([sskf.prepare_z_0_hat() for sskf in self.SSKFs]) / self.seq_count
        
        # copy the value to each single sequence Kalman Filter
        for sskf in self.SSKFs:
            sskf.z_0_hat = self.z_0_hat

    '''
        Calculate the P_0_hat for the maximization step and update
    '''
    def update_P_0_hat(self):
        
        # value fixed, no update
        if self.config['P_0_hat_option'] == 'fixed':
            return
        
        # update without constraint
        self.P_0_hat = sum([sskf.prepare_P_0_hat() for sskf in self.SSKFs]) / self.seq_count
        self.P_0_hat = (self.P_0_hat + self.P_0_hat.T) / 2
        
        # update under the constraint of being a diagnol matrix
        if self.config['P_0_hat_option'] == 'diag':
            self.P_0_hat = np.diag(np.diag(self.P_0_hat))
        
        # update under the constraint of being a sclar times an identity matrix
        if self.config['P_0_hat_option'] == 'scalar':
            self.P_0_hat = np.mean(np.diag(self.P_0_hat)) * np.eye(self.dimension)
        
        # argument not recognized
        if self.config['P_0_hat_option'] not in EM_Config.OPTIONS_CHOICES:
            print('WARNING: value %s not understood for key P_0_hat_option' % self.config['P_0_hat_option'])
        
        # copy the value to each single sequence Kalman Filter
        for sskf in self.SSKFs:
            sskf.P_0_hat = self.P_0_hat

    '''
        Calculate the A, B for the maximization step and update
    '''
    def update_AB(self):
        
        # value fixed, no update
        if self.config['AB_option'] == 'fixed':
            return
        
        if self.config['AB_option'] != 'flexible':
            print('WARNING: value %s not understood for key AB_option' % self.config['AB_option'])
        
        # collect the sums of some variables needed for A, B maximization
        zuztp1_sum, zutzut_sum = np.zeros((self.control_dimension + self.dimension, self.dimension)), \
                                    np.zeros((self.control_dimension + self.dimension, self.control_dimension
                                              + self.dimension))

        for seq_idx in range(self.seq_count):
            zuztp1, zutzut = self.SSKFs[seq_idx].prepare_AB()
            zuztp1_sum += zuztp1
            zutzut_sum += zutzut
        
        # calculate A, B
        best_fit_AB = zuztp1_sum.T.dot(linalg.pinv(zutzut_sum))
        updated_A = best_fit_AB[:,range(self.dimension)]
        updated_B = best_fit_AB[:,self.dimension::]
        self.A, self.B = updated_A, updated_B
        for sskf in self.SSKFs:
            sskf.A, sskf.B = self.A, self.B
    
    '''
        Calculate the Q for the maximization step and update
    '''
    def update_Q(self):
        
        # value fixed, no update
        if self.config['Q_option'] == 'fixed':
            return
        
        # update without constraint
        self.Q = sum([sskf.prepare_Q() for sskf in self.SSKFs]) / self.total_transition_count
        self.Q = (self.Q + self.Q.T) / 2
        
        # update under the constraint of being a diagnol matrix
        if self.config['Q_option'] == 'diag':
            self.Q = np.diag(np.diag(self.Q))

        # update under the constraint of being a sclar times an identity matrix
        if self.config['Q_option'] == 'scalar':
            self.Q = np.mean(np.diag(self.Q)) * np.eye(self.dimension)

        # argument not recognized
        if self.config['Q_option'] not in EM_Config.OPTIONS_CHOICES:
            print('WARNING: value %s not understood for key Q_option' % self.config['Q_option'])

        # copy the value to each single sequence Kalman Filter
        for sskf in self.SSKFs:
            sskf.Q = self.Q

    '''
        Calculate the R for the maximization step and update
    '''
    def update_R(self):
        
        # value fixed, no update
        if self.config['R_option'] == 'fixed':
            return
        
        # update without constraint
        self.R = sum([sskf.prepare_R() for sskf in self.SSKFs]) / self.total_obs
        self.R = (self.R + self.R.T) / 2
        
        # update under the constraint of being a diagnol matrix
        if self.config['R_option'] == 'diag':
            self.R = np.diag(np.diag(self.R))

        # update under the constraint of being a sclar times an identity matrix
        if self.config['R_option'] == 'scalar':
            self.R = np.mean(np.diag(self.R)) * np.eye(self.dimension)
        
        # argument not recognized
        if self.config['R_option'] not in EM_Config.OPTIONS_CHOICES:
            print('WARNING: value %s not understood for key R_option' % self.config['R_option'])
        
        # copy the value to each single sequence Kalman Filter
        for sskf in self.SSKFs:
            sskf.R = self.R

    '''
        One iteration of EM algorithm
        # Returns
            log likelihood of data
    '''
    def _em_(self):
        if not self.current_estimate:
            self.estimate()
        self.maximize()
        return self.log_likelihood()
    
    '''
        Set the EM configuration
    '''
    def set_em_config(self, config):
        self.config = config

    '''
        The actual EM algorithm implemented here
        # Argument
            verbose_level: (0) print nothing (1) print info at the end of EM (2) print info at every iteration of EM
            print_every: print the log likelihood every print_every iterations
        # Returns
            the optimized parameters
    '''
    def em(self, verbose_level=2, print_every=10):
        
        # tolerance for numerical accuracy
        epsilon = 1e-7
        warning_flag = False
        
        # check whether the configuration has been initialized
        if self.config is None:
            self.config = EM_Config(self.dimension, self.control_dimension).get_default_config()
        
        # EM iteration loop
        iteration_idx = 0
        log_likelihood_history = []
        prev_ll = float('-inf')
        for _ in range(self.config['num_iterations']):
            current_ll = self._em_()
            
            # print warning if log likelihood decreases more than 1e-7
            if current_ll + epsilon <= prev_ll:
                print('WARNING: log-likelihood decreases. Something is going wrong')
            
            # converge if the improvement in log likelihood is smaller than the threshold
            if current_ll - self.config['threshold'] <= prev_ll:
                break

            iteration_idx = _ + 1
            if verbose_level == 2 and iteration_idx % print_every == 0:
                print('EM Iteration %d: log likelihood=%.5f' % (iteration_idx, current_ll))
            log_likelihood_history.append(current_ll)
            prev_ll = current_ll

        # EM iterations end
        if verbose_level >= 1:
            print('At iteration %d EM stops, log likelihood %.5f' % (iteration_idx, prev_ll))
        if warning_flag:
            print('WARNING: errors have occurred in the EM algorithm')

        return (self.z_0_hat, self.P_0_hat, self.A, self.B, self.Q, self.R), log_likelihood_history
        
    '''
        Print all the current parameters
    '''
    def print_parameters_to_screen(self):
        print('--- Printing Parameters ---')
        print('Initial State Mean - z_0_hat')
        print(self.z_0_hat)
        print('----------')
        print('Intial State Covariance Matrix - P_0_hat')
        print(self.P_0_hat)
        print('----------')
        print('Transition Matrix - A')
        print(self.A)
        print('----------')
        print('Controal Matrix - B')
        print(self.B)
        print('----------')
        print('Transition Covariance Matrix - Q')
        print(self.Q)
        print('----------')
        print('Emission Covariance Matrix - R')
        print(self.R)
