"""
=============================================
Kalman Filter functions on multiple sequences
=============================================

This Module implements the MSKF (multiple sequence Kalman Filter) class
It supports:
i) inference (predict, filter, smooth)
ii) maximization (z_0_hat, P_0_hat, A, B, Q, R)
iii) calculate log likelihood
"""
import numpy as np
from Single_Sequence_Kalman_Filter import SSKF
from scipy import linalg
from EM_Config import EM_Config

class MSKF:
    """
    Implement the multiple sequence Kalman Filter class
    Most important functionalities are implemented in here and Single_Sequence_Kalman_Filter.py
    """

    def __init__(self, _dimension, _control_dimension, _z_0_hat=None, _P_0_hat=None, _A=None, _B=None, _Q=None, _R=None):
        """
        Initialize the MSKF class

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

    def pass_in_measurement_and_control(self, measurements, controls):
        """
        Initialize the single sequences Kalman Filter and pass in the measurements and controls

        Parameters
        ----------
        measurements: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
            'measurements[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
            None for missing measurements
        controls: [seq_count] array of [seq_length] array of ([dimension] numpy array)
            'controls[seq_idx][time_step]' = the control of the seq_idx sequence at time time_step
        """
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

    def estimate(self):
        """
        Perform estimation on every single sequence Kalman Filter and aggregate the results

        Returns
        -------
        predicted: {String: array}
            'predicted['mean'][seq_idx][time_step]' = the mean of the prediction
            of the seq_idx sequence at time time_step
            'predicted['covariance'][seq_idx][time_step]' = the covariance of prediction
            of the seq_idx sequence at time time_step
        filtered: {String: array}
            'filtered['mean'][seq_idx][time_step]' = the mean of filtered value
            of the seq_idx sequence at time time_step
            'filtered['covariance'][seq_idx][time_step]' = the covariance of filtered value
            of the seq_idx sequence at time time_step
        smoothed: {String: array}
            'smoothed['mean'][seq_idx][time_step]' = the mean of smoothed value
            of the seq_idx sequence at time time_step
            'smoothed['covariance'][seq_idx][time_step]' = the covariance of smoothed value
            of the seq_idx sequence at time time_step
        """
        predicted, filtered, smoothed = {'mean': [], 'covariance': []}, {'mean': [], 'covariance': []}, \
            {'mean': [], 'covariance': []}
        
        # perform inference/estimation for each single sequence Kalman filter
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

    def log_likelihood(self):
        """
        Calculate the log likelihood

        Returns
        -------
        normalized_ll: numpy float
            average log likelihood per data point
        """
        if not self.current_estimate:
            self.estimate()
        multi_seq_prob = 0
        for sskf in self.SSKFs:
            multi_seq_prob += sskf.log_likelihood(normalize=False)
        normalized_ll = multi_seq_prob / self.total_obs
        return normalized_ll

    """
        Maximization
        Update z_0_hat, P_0_hat, A, B, Q, R
    """
    def maximize(self):
        """
        Maximization
        Update z_0_hat, P_0_hat, A, B, Q, R
        """
        self.update_z_0_hat()
        self.update_P_0_hat()
        self.update_AB()
        self.update_Q()
        self.update_R()
        self.current_estimate = False

    def update_z_0_hat(self):
        """
        Calculate the z_0_hat for the maximization step and update
        """
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

    def update_P_0_hat(self):
        """
        Calculate the P_0_hat for the maximization step and update
        """
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

    def update_AB(self):
        """
        Calculate the A, B for the maximization step and update
        """
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

    def update_Q(self):
        """
        Calculate the Q for the maximization step and update
        """
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

    def update_R(self):
        """
        Calculate the R for the maximization step and update
        """
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

    def _em_(self):
        """
        One iteration of the EM algorithm

        Returns
        -------
        self.log_likelihood(): numpy float
            the log likelihood per data point
        """
        if not self.current_estimate:
            self.estimate()
        self.maximize()
        return self.log_likelihood()

    def set_em_config(self, config):
        """
        Set the EM configuration

        Parameters
        ----------
        config: instance of EM_Config
            the Expectation Maximization configuration
        """
        self.config = config

    def em(self, verbose_level=2, print_every=10):
        """
        The actual Expectation Maximization algorithm implemented here

        Parameters
        ----------
        verbose_level: int
            0: print nothing
            1: print summary at the end of all EM iterations
            2: print summary for every print_every EM iteration
        print_every: int
            print summary for every print_every EM iteration if verbose_level = 2

        Returns
        -------
        parameters: z_0_hat, P_0_hat, A, B, Q, R
            z_0_hat: [dimension] numpy array
                initial mean for every sequence
            P_0_hat: [dimension, dimension] numpy array
                initial covariance for every sequence
            A: [dimension, dimension] numpy array
                transition matrix
            B: [dimension, control_dimension] numpy array
                control matrix
            Q: [dimension, dimension] numpy array
                transition covariance matrix
            R: [dimension, dimension] numpy array
                emission covariance matrix
        log_likelihood_history: [num_iterations] array of numpy float
            the log likelihood at EM iteration iteration_idx
        """
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
        parameters = (self.z_0_hat, self.P_0_hat, self.A, self.B, self.Q, self.R)
        return parameters, log_likelihood_history

    def print_parameters_to_screen(self):
        """
        Print all the current parameters
        """
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
