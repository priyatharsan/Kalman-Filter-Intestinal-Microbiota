from Kalman_Filter_API import *
from Utils import *
from EM_Config import EM_Config

'''
    An sklearn flavor wrapper class
'''
class Kalman_Filter():
    
    def __init__(self, dimension, control_dimension):
        self.parameters = None
        self.config = EM_Config(dimension, control_dimension).get_default_config()
    
    '''
        Train the parameters
    '''
    def fit(self, X, U, config=None, verbose_level=2, print_every=10):
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
    
    '''
        Return the inferred values
    '''
    def estimate(self, X, U):
        return infer(X, U, self.parameters)
