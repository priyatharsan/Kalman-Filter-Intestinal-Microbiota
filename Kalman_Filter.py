from Kalman_Filter_API import *
from Utils import *
from EM_Config import EM_Config

class Kalman_Filter():
    '''
    An sklearn flavor wrapper
    '''
    
    def __init__(self, dimension, control_dimension):
        '''
        Initializing a Kalman Filter class of Sklearn flavor

        Parameters
        ----------
        dimension: int
            dimension of x (measurement)
        control_dimension: int
            dimension of u (control)
        '''
        self.parameters = None
        self.config = EM_Config(dimension, control_dimension).get_default_config()
    
    def fit(self, X, U, config=None, verbose_level=2, print_every=10):
        '''
        Train the parameters

        Parameters
        ----------
        X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
            'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
            None for missing measurements
        U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
            'U[seq_idx][time_step]' = the control of the seq_idx sequence at time time_step
        config: {String: String}
            specified EM config dict
        verbose_level: int
            0: print nothing
            1: print summary at the end of all EM iterations
            2: print summary for every print_every EM iteration
        print_every: int
            print summary for every print_every EM iteration if verbose_level = 2
        '''
        dimension, control_dimension = get_dimension(X), get_control_dimension(U)
        if config is not None:
            self.config = config
        self.parameters, self.ll_history = expectation_maximization(X, U, self.config, 
                                                                    verbose_level, print_every)
        
        # we can continue optimize with the trained parameters as a starting point
        z_0_hat, P_0_hat, A, B, Q, R = self.parameters
        self.config['initial_z_0_hat'] = z_0_hat
        self.config['initial_P_0_hat'] = P_0_hat
        self.config['initial_A'] = A
        self.config['initial_B'] = B
        self.config['initial_Q'] = Q
        self.config['initial_R'] = R
    
    def estimate(self, X, U):
        '''
        Infer the hidden variables

        Parameters
        ----------
        X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
            'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
            None for missing measurements
        U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
            'U[seq_idx][time_step]' = the control of the seq_idx sequence at time time_step

        Returns
        -------
        infer(X, U, self.parameters): predicted, filtered, smoothed
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
        '''
        return infer(X, U, self.parameters)